"""
Full pipeline of the next token prediction model: from downloading the data to
evaluating the final models on the test sets.

Steps, in order:
    download         download the Speakleash, HuggingFace and Spokes datasets
    preprocess       clean the written texts and conversations transcripts
    split            create train / val / tune / test splits
    tokenizer        train a SentencePiece tokenizer for every vocabulary size
    hyperparameters  find the best hyperparameters with grid search
    train            train the final model
    fine-tune        fine-tune the model on conversations transcripts
    evaluate         compare both models on the written and spoken test sets

Every step reads what the previous steps saved to disk, so the pipeline can be
started from (and stopped after) any step:
    python full_pipeline.py
    python full_pipeline.py --start-from split
    python full_pipeline.py --start-from train --stop-after fine-tune
"""
import argparse
import os
import random
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

warnings.filterwarnings("ignore")

# paths in config.yml are relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_CONFIG_PATH = "config/model_config.yml"
STUDY_RESULTS_DIR = "study_results"

MODEL_CHECKPOINT_PATH = "model_checkpoint.pt"
# name of the best checkpoint is set by src/model.py and src/fine_tuning.py
BEST_MODEL_CHECKPOINT_PATH = "model_checkpoint_best.pt"
BEST_FINE_TUNED_CHECKPOINT_PATH = "fined_tuned_model_checkpoint_best.pt"

# file names expected by src/model_loader.py
MODEL_PATH = "model.pt"
FINE_TUNED_MODEL_PATH = "fine_tuned_model.pt"
TOKENIZER_PATH = "spm_pl.model"

# beam search parameters used for the evaluation
BEAM_WIDTH = 25
MAX_WORD_LENGTH = 10
ALPHA = 0.2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_gpu() -> None:
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: GPU not available, training will run on CPU and be very slow.")


def load_model_config() -> dict:
    with open(MODEL_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def extract_state_dict(checkpoint_path: str, model_path: str) -> None:
    """Save only the weights from a training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    torch.save(checkpoint["model_state_dict"], model_path)
    print(f"Saved model from {checkpoint_path} to {model_path}")


# Modules are imported inside the steps, so that running only some of the steps
# doesn't require dependencies of the other ones (e.g. speakleash).
def download(config: dict) -> None:
    import pandas as pd
    from speakleash import Speakleash

    from utils.download_huggingface_data import download_hf_data
    from utils.download_speakleash_data import save_quality_docs
    from utils.download_spokes_data import (
        change_xls_to_txt,
        download_conversations_transcripts,
    )

    # downloading speakleash datasets
    Path(config["dir-path"]).mkdir(parents=True, exist_ok=True)
    sl = Speakleash(config["dir-path"])
    save_quality_docs(
        sl,
        dir_path=config["dir-path"],
        datasets_names=config["speakleash-datasets"],
        limit=config["data-limit"],
    )

    # downloading huggingface news dataset
    download_hf_data(
        url=config["huggingface-dataset-url"],
        dir_path=config["dir-path"],
        dataset_name=config["huggingface-dataset-name"],
    )

    # downloading spokes datasets
    conversations = pd.read_excel(config["conversations-id-file"])
    conversations_train = conversations[conversations["Id"].str.startswith("CBIZ")]
    conversations_test = conversations[~conversations["Id"].str.startswith("CBIZ")]

    print("Saving train conversations ... ")
    for file_path in download_conversations_transcripts(
        conversations_df=conversations_train,
        dir_path=config["dialog-dir-path"],
        dataset_name=config["conversations-train-dir-name"],
    ):
        change_xls_to_txt(file_path)

    print("Saving test conversations ... ")
    for file_path in download_conversations_transcripts(
        conversations_df=conversations_test,
        dir_path=config["dialog-dir-path"],
        dataset_name=config["conversations-test-dir-name"],
    ):
        change_xls_to_txt(file_path)


def preprocess(config: dict) -> None:
    from utils.preprocess_files import get_files_names, preprocess_files

    dir_path = config["dir-path"]
    preprocessed_dir_path = config["preprocessed-dir-path"]
    datasets = config["speakleash-datasets"] + [config["huggingface-dataset-name"]]

    conversations_dir_path = config["dialog-dir-path"]
    preprocessed_conversations_dir_path = config["preprocessed-dialog-path"]
    conversations_datasets = [
        config["conversations-train-dir-name"],
        config["conversations-test-dir-name"],
    ]

    print("Preprocessing written texts ...")
    for dataset in datasets:
        files_names = get_files_names(dir_path=dir_path, dataset_name=dataset)
        preprocess_files(
            dir_path=dir_path,
            preprocessed_dir_path=preprocessed_dir_path,
            dataset_name=dataset,
            files_names=files_names,
        )

    print("Preprocessing conversations transcripts ...")
    for c_dataset in conversations_datasets:
        c_files_names = get_files_names(
            dir_path=conversations_dir_path, dataset_name=c_dataset
        )
        preprocess_files(
            dir_path=conversations_dir_path,
            preprocessed_dir_path=preprocessed_conversations_dir_path,
            dataset_name=c_dataset,
            files_names=c_files_names,
        )


def split(config: dict) -> None:
    from utils.train_test_split import create_train_test_split

    create_train_test_split(config)


def tokenizer(config: dict) -> None:
    from src.train_tokenizer import train_tokenizers

    train_tokenizers(config)


def hyperparameters(config: dict) -> None:
    from utils.hyperparameters_optimization import run_hyperparameters_optimization

    best_config = run_hyperparameters_optimization(
        config,
        model_config_path=MODEL_CONFIG_PATH,
        save_dir=STUDY_RESULTS_DIR,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"Best hyperparameters saved to {MODEL_CONFIG_PATH}: {best_config}")


def train(config: dict) -> None:
    from src.model import train_model

    model_config = load_model_config()
    train_model(config, model_config, save_path=MODEL_CHECKPOINT_PATH)

    extract_state_dict(BEST_MODEL_CHECKPOINT_PATH, MODEL_PATH)
    shutil.copy(model_config["tokenizer"], TOKENIZER_PATH)


def fine_tune(config: dict) -> None:
    from src.fine_tuning import fine_tune_model

    fine_tune_model(
        config_dict=config,
        model_path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        model_config=load_model_config(),
    )

    extract_state_dict(BEST_FINE_TUNED_CHECKPOINT_PATH, FINE_TUNED_MODEL_PATH)


def evaluate(config: dict) -> None:
    from utils.ngram_lstm_compare import LSTMPredictor, ModelEvaluator

    seq_len = load_model_config()["seq_len"]
    models = {
        name: LSTMPredictor(
            model_dir=".",
            beam_width=BEAM_WIDTH,
            max_word_length=MAX_WORD_LENGTH,
            device="cpu",
            alpha=ALPHA,
            seq_len=seq_len,
            model_name=model_name,
        )
        for name, model_name in (
            ("lstm", MODEL_PATH),
            ("fine_tuned_lstm", FINE_TUNED_MODEL_PATH),
        )
    }

    test_sets = {
        "written": (
            config["test-indices-file"],
            config["train-paths-file"],
            config["train-offsets-file"],
        ),
        "conversations": (
            config["conversations-test-indices-file"],
            config["conversations-test-paths-file"],
            config["conversations-test-offsets-file"],
        ),
    }

    for test_set, (indices_file, paths_file, offsets_file) in test_sets.items():
        print(f"Evaluating on the {test_set} test set ...")
        output_path = f"evaluation_results_{test_set}.json"
        # ModelEvaluator accumulates results, so each test set needs a new one
        evaluator = ModelEvaluator(models)
        evaluator.evaluate_directory(
            test_indices=np.load(indices_file),
            paths_file=paths_file,
            offsets_file=offsets_file,
            output_path=output_path,
        )
        evaluator.print_comparison()
        evaluator.save_results(output_path)


STEPS = {
    "download": download,
    "preprocess": preprocess,
    "split": split,
    "tokenizer": tokenizer,
    "hyperparameters": hyperparameters,
    "train": train,
    "fine-tune": fine_tune,
    "evaluate": evaluate,
}


def parse_args() -> argparse.Namespace:
    steps = list(STEPS)
    parser = argparse.ArgumentParser(description="Run the full model training pipeline.")
    parser.add_argument("--config", default="config/config.yml",
                        help="path to the config file, relative to the project root")
    parser.add_argument("--start-from", choices=steps, default=steps[0],
                        help="first step to run")
    parser.add_argument("--stop-after", choices=steps, default=steps[-1],
                        help="last step to run")
    args = parser.parse_args()
    if steps.index(args.start_from) > steps.index(args.stop_after):
        parser.error("--start-from has to be before --stop-after")
    return args


def main():
    args = parse_args()
    check_gpu()
    os.chdir(PROJECT_ROOT)

    # loading config file
    with open(args.config) as f:
        config = yaml.safe_load(f)

    steps = list(STEPS)
    first_step = steps.index(args.start_from)
    last_step = steps.index(args.stop_after)

    for step in steps[first_step : last_step + 1]:
        print(f"{'=' * 20} {step} {'=' * 20}")
        # seeding every step makes it reproducible also when run on its own
        set_seed(config["seed"])
        STEPS[step](config)


if __name__ == "__main__":
    main()
