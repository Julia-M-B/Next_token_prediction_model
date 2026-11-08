from src.model import StreamingTextDataset, LSTMLanguageModel, train_epoch, get_files_paths, collate_batch, compute_val_ppl
import torch
from torch.utils.data import DataLoader
import yaml
import random
import numpy as np
from typing import List, Tuple
import sentencepiece as spm


def split_tuning_files_list(config_dict: dict, mix_ratio: float = 0.5) -> Tuple[List[str], List[str]]:
    file_len_normalisation_ratio = 8.5  # on average the conversation file was 8.5 times longer than written file

    tuning_files = []

    # prepare files for fine-tuning
    written_tune_indices = np.load(config_dict["tune-indices-file"])
    conversation_indices = np.load(config_dict["conversations-tune-indices-file"])

    written_paths = config_dict["train-paths-file"]
    written_offsets = config_dict["train-offsets-file"]

    conversations_paths = config_dict["conversations-tune-paths-file"]
    conversations_offsets = config_dict["conversations-tune-offsets-file"]

    for written_files, conversations_files in zip(
            get_files_paths(
                len(written_tune_indices),
                written_tune_indices,
                paths_file=written_paths,
                offsets_file=written_offsets,
            ),
            get_files_paths(
                len(conversation_indices),
                conversation_indices,
                paths_file=conversations_paths,
                offsets_file=conversations_offsets,
            ),
    ):

        print(
            f"Tuning on {len(conversation_indices)} conversations transcripts.")
        tuning_files.extend(conversations_files)

        n_written = min(len(written_tune_indices), int(file_len_normalisation_ratio * len(conversation_indices) * mix_ratio))
        print(f"Tuning on {n_written} written texts.")
        tuning_files.extend(written_files[:n_written])

    val_ratio = config_dict.get("val-ratio", 0.05)
    val_idx = max(5, int(len(tuning_files) * val_ratio))

    random.shuffle(tuning_files)

    return tuning_files[:-val_idx], tuning_files[-val_idx:]

def fine_tune_model(config_dict: dict, model_path: str, tokenizer_path: str, model_config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_workers = config_dict["n-workers"]

    # get fine-tuning configuration
    ft_epochs = config_dict.get("ft-epochs", 3)
    ft_lr = config_dict.get("ft-lr", 1e-4)

    # get model hyperparameters
    tokenizer = tokenizer_path
    emb_dim = model_config["emb_dim"]
    n_layers = model_config["n_layers"]
    hidden_units = model_config["hidden_units"]
    dropout = model_config["dropout"]
    weight_decay = model_config["weight_decay"]
    batch_size = config_dict.get("ft-batch-size", 8)
    seq_len = model_config["seq_len"]

    tune_files, val_files = split_tuning_files_list(config_dict)

    dataset = StreamingTextDataset(
        files=tune_files, sp_model_path=tokenizer, seq_len=seq_len
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

    # instantiate spm model
    sp_proc = spm.SentencePieceProcessor()
    sp_proc.load(tokenizer)

    # Load model state dict
    state_dict = torch.load(model_path, map_location=device)
    # Create model
    model = LSTMLanguageModel(
        vocab_size=sp_proc.get_piece_size(),
        emb_dim=emb_dim,
        hidden_dim=hidden_units,
        n_layers=n_layers,
        dropout=dropout,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ft_lr,
        weight_decay=weight_decay,
    )

    # training loop (sketch)
    best_val_loss = float("inf")
    save_path = "fined_tuned_model_checkpoint.pt"
    save_every = 1
    patience = 2
    epochs_without_improve = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_ppl": [],
    }

    for epoch in range(ft_epochs):
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

    # set seed to make the model training reproducible
    seed = config.get("seed", 42)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    fine_tune_model(config_dict=config,
                    model_path="model.pt",
                    tokenizer_path="spm_pl.model",
                    model_config={
                        "emb_dim": 256,
                        "n_layers": 3,
                        "hidden_units": 512,
                        "dropout": 0.2,
                        "weight_decay": 0.01,
                        "seq_len": 256,
                    })

    print("Fine tuning ended.")


if __name__ == "__main__":
    main()


