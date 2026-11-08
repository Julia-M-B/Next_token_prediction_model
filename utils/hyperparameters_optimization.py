import json
import random
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import numpy as np
import sentencepiece as spm
import torch
import yaml
from src.beam_search import WordPredictionBeamSearch
from src.model import (
    LSTMLanguageModel,
    LSTMModelWrapper,
    StreamingTextDataset,
    compute_val_ppl,
    get_files_paths,
    train_epoch,
)
from torch.utils.data import DataLoader

# capacity parameters
EMBEDDING_DIM = [256, 320]

# LSTM parameters
N_LAYERS = [2, 3]
HIDDEN_UNITS = [256, 384, 512]

# regularization parameters
DROPOUT = [0.2, 0.25, 0.3]
WEIGHT_DECAY = [1e-2, 1e-3, 1e-4]

# learning parameters
BATCH_SIZE = [64, 128]
SEQ_LEN = [64, 128, 256]
LEARNING_RATE = [1e-3, 2e-3]


@dataclass()
class Hyperparameters:
    """
    Data class storing all hyperparameters for LSTM model training.

    Attributes:
        tokenizer: Path to the SentencePiece tokenizer model file.
        emb_dim: Dimension of token embeddings.
        n_layers: Number of LSTM layers.
        hidden_units: Number of hidden units in LSTM layers.
        dropout: Dropout rate for regularization.
        weight_decay: L2 regularization coefficient.
        batch_size: Number of samples per training batch.
        seq_len: Length of input sequences.
        lr: Learning rate for optimizer.
    """

    tokenizer: str
    emb_dim: int
    n_layers: int
    hidden_units: int
    dropout: float
    weight_decay: float
    batch_size: int
    seq_len: int
    lr: float


@dataclass(order=True)
class ExperimentData:
    """
    Data class storing results from a single hyperparameter experiment.

    The class is ordered by best_val_acc5 for easy comparison of experiments.

    Attributes:
        params: Dictionary of hyperparameter values used.
        best_val_acc5: Best validation accuracy achieved during training.
        history: Dictionary containing training metrics over epochs.
        trial_params: List of parameter names that were varied in this trial.
    """

    params: dict = field(compare=False)
    best_val_acc5: float
    history: dict = field(compare=False)
    trial_params: list = field(compare=False)


def get_top_k_accuracy(
    searcher: WordPredictionBeamSearch, context: str, true_word: str, k: int = 5
) -> float:
    """
    Compute top-k accuracy for a single prediction.

    Args:
        searcher: Beam search object for word prediction.
        context: String context to predict from.
        true_word: The actual next word to predict.
        k: Number of top predictions to consider. Defaults to 5.

    Returns:
        1.0 if true_word is in top-k predictions, 0.0 otherwise.
    """
    top_words = searcher.get_top_k_words(context_text=context, k=k)
    predictions = [word for word, _, _ in top_words]
    return 1.0 if true_word in predictions else 0.0


def evaluate_top_k_words(
    searcher: WordPredictionBeamSearch, files_paths: List[str], seq_len: int, k: int = 5
) -> float:
    """
    Evaluate top-k accuracy across multiple text files.

    Uses a sliding window approach to generate context-target pairs from
    the text files and computes the proportion where the true word appears
    in the top-k predictions.

    Args:
        searcher: Beam search object for word prediction.
        files_paths: List of paths to text files for evaluation.
        seq_len: Length of context window in words.
        k: Number of top predictions to consider. Defaults to 5.

    Returns:
        Average top-k accuracy across all predictions.
        Returns 0.0 if no predictions were made.
    """
    preds_counter = 0
    acc5 = 0
    slide = int(seq_len / 2)

    for file in files_paths:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
        words = text.split()

        for i in range(0, len(words) - seq_len, slide):
            context = " ".join(words[i : i + seq_len]) + " "
            target = words[i + seq_len]
            acc5 += get_top_k_accuracy(searcher, context, target, k)
            preds_counter += 1

    return acc5 / preds_counter if preds_counter > 0 else 0


def create_and_train_model(
    train_files: List[str],
    val_files: List[str],
    hyperparameters: Hyperparameters,
    trial_params: List[str],
    n_epochs: int = 3,
    device: str = "cuda",
) -> ExperimentData:
    """
    Create, train and evaluate an LSTM language model.

    This function performs the complete training pipeline:
    1. Creates datasets and dataloaders
    2. Initializes the model and optimizer
    3. Trains for specified number of epochs
    4. Tracks training metrics and validation performance

    Args:
        train_files: List of file paths for training data.
        val_files: List of file paths for validation data.
        hyperparameters: Hyperparameter configuration for the model.
        trial_params: Names of hyperparameters being varied in grid search.
        n_epochs: Number of training epochs. Defaults to 3.
        device: Device to train on ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        ExperimentData object containing results, history and configuration.

    Note:
        The model is evaluated on validation set after each epoch.
        Best score is tracked based on validation top-5 accuracy.
    """
    train_ds = StreamingTextDataset(
        files=train_files,
        sp_model_path=hyperparameters.tokenizer,
        seq_len=hyperparameters.seq_len,
    )
    val_ds = StreamingTextDataset(
        files=val_files,
        sp_model_path=hyperparameters.tokenizer,
        seq_len=hyperparameters.seq_len,
    )

    train_dl = DataLoader(train_ds, batch_size=hyperparameters.batch_size)
    val_dl = DataLoader(val_ds, batch_size=hyperparameters.batch_size)

    sp_proc = spm.SentencePieceProcessor()
    sp_proc.load(hyperparameters.tokenizer)

    model = LSTMLanguageModel(
        vocab_size=sp_proc.get_piece_size(),
        emb_dim=hyperparameters.emb_dim,
        hidden_dim=hyperparameters.hidden_units,
        n_layers=hyperparameters.n_layers,
        dropout=hyperparameters.dropout,
    ).to(device)

    model_wrapper = LSTMModelWrapper(model=model, seq_len=hyperparameters.seq_len)

    searcher = WordPredictionBeamSearch(model=model_wrapper, tokenizer=sp_proc)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameters.lr,
        weight_decay=hyperparameters.weight_decay,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_ppl": [],
        "val_acc5": [],
    }
    best_val_acc5 = 0

    for epoch in range(1, n_epochs + 1):
        print(f"Training epoch {epoch}:")
        train_loss = train_epoch(model, train_dl, optimizer, device)
        print(f"Epoch {epoch} train loss: {train_loss:.4f}")

        val_loss, val_ppl = compute_val_ppl(model, val_dl, device)
        val_acc5 = evaluate_top_k_words(
            searcher, files_paths=val_files, seq_len=hyperparameters.seq_len
        )

        print(f"Epoch {epoch} val loss: {val_loss:.4f} ppl: {val_ppl:.2f}")
        print(f"Epoch {epoch} val acc: {val_acc5:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)
        history["val_acc5"].append(val_acc5)

        if best_val_acc5 < val_acc5:
            best_val_acc5 = val_acc5

    return ExperimentData(
        params=asdict(hyperparameters),
        best_val_acc5=best_val_acc5,
        history=history,
        trial_params=trial_params,
    )


def hyperparameters_setup_generator(
    fixed_config: Dict[str, Any], trial_config: Dict[str, List[Any]]
) -> Generator[Hyperparameters, None, None]:
    """
    Generate all combinations of hyperparameters for grid search.

    Creates Hyperparameters objects by combining fixed values with all
    possible combinations of trial values using Cartesian product.

    Args:
        fixed_config: Dictionary of fixed hyperparameter values.
        trial_config: Dictionary mapping parameter names to lists of values
                     to try in grid search.

    Yields:
        Hyperparameters object for each combination in the grid.

    Example:
        >>> fixed = {"n_layers": 3, "dropout": 0.2}
        >>> trial = {"lr": [1e-3, 1e-4], "batch_size": [32, 64]}
        >>> for hp in hyperparameters_setup_generator(fixed, trial):
        ...     # Will yield 4 combinations (2 lr × 2 batch_size)
        ...     print(hp.lr, hp.batch_size)
    """

    def _get_param_value(param_name, trial_params, trial_combination):
        """Helper to get parameter value from trial or fixed config."""
        if param_name in trial_params:
            idx = list(trial_params).index(param_name)
            return trial_combination[idx]
        fixed_val = fixed_config.get(param_name)
        if fixed_val is None:
            raise KeyError(f"Missing hyperparameter: {param_name}")
        return fixed_val

    trial_params_names = trial_config.keys()
    combinations = list(product(*trial_config.values()))

    for combination in combinations:
        yield Hyperparameters(
            tokenizer=_get_param_value("tokenizer", trial_params_names, combination),
            emb_dim=_get_param_value("emb_dim", trial_params_names, combination),
            n_layers=_get_param_value("n_layers", trial_params_names, combination),
            hidden_units=_get_param_value(
                "hidden_units", trial_params_names, combination
            ),
            dropout=_get_param_value("dropout", trial_params_names, combination),
            weight_decay=_get_param_value(
                "weight_decay", trial_params_names, combination
            ),
            batch_size=_get_param_value("batch_size", trial_params_names, combination),
            seq_len=_get_param_value("seq_len", trial_params_names, combination),
            lr=_get_param_value("lr", trial_params_names, combination),
        )


def run_grid_search(
    study_num: int,
    fixed_config: Dict[str, Any],
    trial_config: Dict[str, List[Any]],
    train_files: List[str],
    val_files: List[str],
    n_epochs: int = 3,
    device: str = "cuda",
    save_dir: str = "next_token_prediction_model/study_results",
) -> List[ExperimentData]:
    """
    Run grid search over hyperparameter combinations.

    Trains models for all combinations of trial parameters while keeping
    fixed parameters constant. Saves results to JSON files after each trial.

    Args:
        study_num: Identifier for this grid search study.
        fixed_config: Dictionary of hyperparameters to keep constant.
        trial_config: Dictionary of hyperparameters to vary, mapping names
                     to lists of values to try.
        train_files: List of file paths for training data.
        val_files: List of file paths for validation data.
        n_epochs: Number of epochs to train each model. Defaults to 3.
        device: Device to train on ('cuda' or 'cpu'). Defaults to 'cuda'.
        save_dir: Directory to save experiment results. Defaults to
                 'next_token_prediction_model/study_results'.

    Returns:
        List of ExperimentData objects, one for each hyperparameter combination.

    Side Effects:
        Creates JSON files in save_dir with results from each trial.
        File format: study_{study_num}_trial_{trial_num}.json
    """
    results = []

    for i, hyperparams in enumerate(hyperparameters_setup_generator(fixed_config, trial_config)):
        print("Trial nr", i)
        print("Hyperparameters:")
        print(asdict(hyperparams))

        experiment_result = create_and_train_model(
            train_files=train_files,
            val_files=val_files,
            hyperparameters=hyperparams,
            trial_params=list(trial_config.keys()),
            n_epochs=n_epochs,
            device=device,
        )
        results.append(experiment_result)

        if save_dir:
            file_name = f"study_{study_num}_trial_{i}.json"
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            save_path = save_path / file_name
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(asdict(experiment_result), f, indent=2)

    return results


def get_best_config(
    results: List[ExperimentData],
) -> Tuple[ExperimentData, Dict[str, Any]]:
    """
    Find the best performing configuration from grid search results.

    Extracts the configuration of trial parameters that has the highest
    value of top-5 accuracy.

    Args:
        results: List of ExperimentData objects from grid search.

    Returns:
        Tuple containing:
            - ExperimentData object with best top-5 accuracy
            - Dictionary mapping trial parameter names to their best values

    Example:
        >>> best_result, best_config = get_best_config(results)
        >>> print(f"Best accuracy: {best_result.best_val_acc5}")
        >>> print(f"Best learning rate: {best_config['lr']}")
    """
    best_result = max(results, key=lambda r: r.best_val_acc5)
    trial_params = best_result.trial_params
    params = best_result.params
    best_config = {}

    for param in trial_params:
        best_config[param] = params.get(param)

    return best_result, best_config


def run_hyperparameters_optimization(
    config: Dict[str, Any],
    model_config_path: str = "next_token_prediction_model/model_config.yml",
    save_dir: str = "next_token_prediction_model/study_results",
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run the sequence of grid search studies and save the best configuration.

    Each study fixes the best values found by the previous ones.

    Args:
        config: General project configuration.
        model_config_path: YAML file to save the best hyperparameters to.
        save_dir: Directory to save the results of every trial to.
        device: Device to train on ('cuda' or 'cpu').

    Returns:
        Dictionary with the best value of every hyperparameter.
    """
    HYPERPARAMETERS_RATIO = config.get("hyperparameters-ratio", 0.25)
    VAL_RATIO = 5e-4

    TOKENIZER = [f"{prefix}.model" for prefix in config["tokenizer-prefix"]]

    study_1 = {
        "fixed_config": {
            "n_layers": 3,
            "hidden_units": 384,
            "dropout": 0.2,
            "weight_decay": 1e-2,
            "batch_size": 128,
            "seq_len": 128,
            "lr": 1e-3,
        },
        "trial_config": {
            "tokenizer": TOKENIZER,
            "emb_dim": EMBEDDING_DIM,
        },
    }

    study_2 = {
        "fixed_config": {
            "dropout": 0.2,
            "weight_decay": 1e-2,
            "batch_size": 128,
            "seq_len": 128,
            "lr": 1e-3,
        },
        "trial_config": {
            "n_layers": N_LAYERS,
            "hidden_units": HIDDEN_UNITS,
        },
    }

    study_3 = {
        "fixed_config": {
            "batch_size": 128,
            "seq_len": 128,
            "lr": 1e-3,
        },
        "trial_config": {
            "dropout": DROPOUT,
            "weight_decay": WEIGHT_DECAY,
        },
    }

    study_4 = {
        "fixed_config": {},
        "trial_config": {
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "lr": LEARNING_RATE,
        },
    }

    final_results = {
        "fixed_config": {},
    }

    params_configs = [study_1, study_2, study_3, study_4, final_results]

    # get files paths
    written_paths_file = config["train-paths-file"]
    written_offsets_file = config["train-offsets-file"]
    train_indices = np.load(config["train-indices-file"])

    n = int(HYPERPARAMETERS_RATIO * len(train_indices))
    m = max(int(VAL_RATIO * n), 1)

    for files_paths in get_files_paths(
        batch_size=n,
        indices=train_indices[:n],
        paths_file=written_paths_file,
        offsets_file=written_offsets_file,
    ):
        train_files = files_paths[:-m]
        val_files = files_paths[-m:]

    # run grid search
    for study_num, params_config in enumerate(params_configs[:-1]):
        results = run_grid_search(
            study_num=study_num,
            fixed_config=params_config.get("fixed_config"),
            trial_config=params_config.get("trial_config"),
            train_files=train_files,
            val_files=val_files,
            device=device,
            save_dir=save_dir,
        )
        best_result, best_config = get_best_config(results)
        print("=" * 50)
        print("BEST CONFIG:")
        print(best_config)
        print("TOP 5 ACCURACY:", best_result.best_val_acc5)

        for next_config in params_configs[study_num + 1 :]:
            for param, val in best_config.items():
                next_config["fixed_config"][param] = val

    # save the final results (best hyperparameters configuration) of the run experiment
    with open(model_config_path, "w") as f_out:
        yaml.dump(final_results["fixed_config"], f_out)

    return final_results["fixed_config"]


if __name__ == "__main__":

    with open("next_token_prediction_model/config.yml", "r") as f:
        config = yaml.safe_load(f)

    # set seed to make the hyperparameters optimization reproducible
    SEED = config.get("seed", 42)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    run_hyperparameters_optimization(config)
