import random
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import sentencepiece as spm
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm


class StreamingTextDataset(IterableDataset):
    def __init__(self, files: List[str], sp_model_path: str, seq_len: int = 32):
        super().__init__()
        self.files = files
        self.seq_len = seq_len
        self.slide = seq_len // 2

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)
        self.vocab_size = self.sp.get_piece_size()

    def _file_iterator_for_worker(self) -> Iterator[str]:
        worker_info = get_worker_info()
        if worker_info is None:
            # single-process loader
            file_list = self.files
        else:
            # split files between workers
            wid = worker_info.id
            nworkers = worker_info.num_workers
            file_list = self.files[wid::nworkers]

        for p in file_list:
            yield p

    def _token_stream_from_file(self, path: str) -> Iterator[int]:
        """Yield token ids (ints) from a file, streaming line by line."""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids = self.sp.encode(line, out_type=int)
                for tid in ids:
                    yield tid

    def __iter__(self) -> Iterator[Tuple[torch.LongTensor, torch.LongTensor]]:
        for file_path in self._file_iterator_for_worker():
            token_buffer = []
            for tid in self._token_stream_from_file(file_path):
                token_buffer.append(tid)
                # when buffer large enough, emit windows
                while len(token_buffer) >= self.seq_len + 1:
                    chunk = token_buffer[: self.seq_len + 1]
                    input_ids = torch.LongTensor(chunk[:-1])
                    target_ids = torch.LongTensor(chunk[1:])
                    yield input_ids, target_ids
                    token_buffer = token_buffer[self.slide :]


class LSTMLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int = 512,
        n_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        if self.emb_dim != self.hidden_dim:
            self.projection = nn.Linear(hidden_dim, emb_dim, bias=False)
        self.output = nn.Linear(emb_dim, vocab_size)
        self.output.weight = self.embedding.weight

    def forward(self, input_ids: torch.LongTensor, hidden=None):
        emb = self.embedding(input_ids)
        out, hidden = self.lstm(emb, hidden)
        if self.emb_dim != self.hidden_dim:
            out = self.projection(out)
        logits = self.output(out)
        return logits, hidden

class LSTMModelWrapper:
    """
    Wrapper class for LSTM Language Model to provide prediction interface.

    This class encapsulates the LSTM model and provides a simplified interface
    for making predictions on token sequences.

    Attributes:
        device (str): Device to run the model on ('cuda' or 'cpu').
        model (LSTMLanguageModel): The underlying LSTM language model.
        vocab_size (int): Size of the vocabulary.
        seq_len (int): Maximum sequence length to consider for predictions.
    """

    def __init__(self, model: LSTMLanguageModel, seq_len: int = 64):
        """
        Initialize the LSTM model wrapper.

        Args:
            model: The LSTM language model to wrap.
            seq_len: Maximum sequence length for context. Defaults to 64.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.model.to(self.device)
        self.model.eval()  # set model to the evaluation mode

        self.vocab_size = self.model.vocab_size
        self.seq_len = seq_len

    def predict(self, context_tokens: List[int]) -> List[float]:
        """
        Predict probability distribution over next tokens given context.

        Args:
            context_tokens: List of token IDs representing the context.
                           If empty, returns uniform distribution.

        Returns:
            List of probabilities for each token in vocabulary.
            Length equals vocab_size.
        """
        if not context_tokens:
            return [1.0 / self.vocab_size] * self.vocab_size
        context_tokens = context_tokens[-self.seq_len :]
        input_ids = torch.LongTensor([context_tokens]).to(self.device)
        with torch.no_grad():
            logits, _ = self.model(input_ids)
            last_logits = logits[0, -1, :]
            probs = torch.softmax(last_logits, dim=0)
            return probs.cpu().tolist()


def collate_batch(
    samples: List[Tuple[torch.LongTensor, torch.LongTensor]]
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    inputs = torch.stack([s[0] for s in samples], dim=0)
    targets = torch.stack([s[1] for s in samples], dim=0)
    return inputs, targets


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer,
    device,
    clip_grad_norm: float = 1.0,
    log_steps: int = 200,
):
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    it = iter(dataloader)
    pbar = tqdm(enumerate(it), desc="train")
    for step, (inputs, targets) in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits, _ = model(inputs)
        b, s, v = logits.size()
        loss = criterion(logits.view(b * s, v), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        if step % log_steps == 0 and step > 0:
            pbar.set_postfix({"loss": total_loss / (step + 1)})

    return total_loss / (step + 1)


def read_by_index(indices: list[int], paths_file: str, offsets_file: str) -> list[str]:
    offsets = np.load(offsets_file, mmap_mode="r")
    result = []
    with open(paths_file, "rb") as f:
        for idx in indices:
            off = int(offsets[idx])
            f.seek(off)
            line = f.readline().rstrip(b"\n\r")
            result.append(line.decode("utf-8"))
    return result


def get_files_paths(batch_size: int, indices, paths_file, offsets_file):
    i = 0
    while i < len(indices) - batch_size:
        batch_indices = indices[i : i + batch_size]
        yield read_by_index(batch_indices.tolist(), paths_file, offsets_file)
        i += batch_size

    yield read_by_index(indices[i:].tolist(), paths_file, offsets_file)


def compute_val_ppl(model: nn.Module, dataloader: DataLoader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits, _ = model(inputs)
            b, s, v = logits.size()
            loss = criterion(logits.view(b * s, v), targets.view(-1))
            total_loss += loss.item()
            total_tokens += b * s
    avg_loss = total_loss / total_tokens
    ppl = float(torch.exp(torch.tensor(avg_loss)).item())
    return avg_loss, ppl


def train_model(config: dict, model_config: dict, save_path: str = "model_checkpoint.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare file list
    train_indices = np.load(config["train-indices-file"])
    val_indices = np.load(config["val-indices-file"])

    paths_file = config["train-paths-file"]
    offsets_file = config["train-offsets-file"]

    n_workers = config["n-workers"]
    n_epochs = config["n-epochs"]

    # get model hyperparameters
    tokenizer = model_config["tokenizer"]
    emb_dim = model_config["emb_dim"]
    n_layers = model_config["n_layers"]
    hidden_units = model_config["hidden_units"]
    dropout = model_config["dropout"]
    weight_decay = model_config["weight_decay"]
    batch_size = model_config["batch_size"]
    seq_len = model_config["seq_len"]
    lr = model_config["lr"]

    for files, val_files in zip(
        get_files_paths(
            len(train_indices),
            train_indices,
            paths_file=paths_file,
            offsets_file=offsets_file,
        ),
        get_files_paths(
            len(val_indices),
            val_indices,
            paths_file=paths_file,
            offsets_file=offsets_file,
        ),
    ):

        dataset = StreamingTextDataset(
            files=files, sp_model_path=tokenizer, seq_len=seq_len
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_batch,
            num_workers=n_workers,
        )

        val_dataset = StreamingTextDataset(
            files=val_files, sp_model_path=tokenizer, seq_len=seq_len
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collate_batch,
            num_workers=0,
        )

        # instantiate model
        sp_proc = spm.SentencePieceProcessor()
        sp_proc.load(tokenizer)

        model = LSTMLanguageModel(
            vocab_size=sp_proc.get_piece_size(),
            emb_dim=emb_dim,
            hidden_dim=hidden_units,
            n_layers=n_layers,
            dropout=dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # training loop (sketch)
        best_val_loss = float("inf")
        save_every = 1
        patience = 3
        epochs_without_improve = 0
        start_epoch = 1
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_ppl": [],
        }

        model_path = Path(save_path)

        if model_path.exists():
            checkpoint = torch.load(save_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            history = checkpoint["history"]
            best_val_loss = min(checkpoint["history"]["val_loss"])
            model.train()

        for epoch in range(start_epoch, n_epochs + 1):
            print(f"Training epoch {epoch}:")
            train_loss = train_epoch(model, dataloader, optimizer, device)
            print(f"Epoch {epoch} train loss: {train_loss:.4f}")

            val_loss, val_ppl = compute_val_ppl(model, val_dataloader, device)

            print(f"Epoch {epoch} val loss: {val_loss:.4f} ppl: {val_ppl:.2f}")

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_ppl"].append(val_ppl)

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab_size": sp_proc.get_piece_size(),
                "sp_model_path": tokenizer,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "history": history,
            }

            if epoch % save_every == 0:
                torch.save(checkpoint, save_path)
                print(f"Saved checkpoint to {save_path} (epoch {epoch})")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improve = 0
                best_path = save_path.replace(".pt", f"_best.pt")
                torch.save(checkpoint, best_path)
                print(f"New best val loss {best_val_loss:.4f} — saved {best_path}")
            else:
                epochs_without_improve += 1
                print(f"No improvement for {epochs_without_improve} epoch(s)")
                if epochs_without_improve >= patience:
                    print("Early stopping triggered")
                    break


def main():
    # load general project config
    with open("next_token_prediction_model/config.yml") as f:
        config = yaml.safe_load(f)

    # load model-specific config
    with open("next_token_prediction_model/model_config.yml") as f:
        model_config = yaml.safe_load(f)

    # set seed to make the model training reproducible
    seed = config.get("seed", 42)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    train_model(config, model_config)


if __name__ == "__main__":
    main()
