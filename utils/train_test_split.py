from pathlib import Path
import glob
import numpy as np
import yaml

def files_paths_generator(dir_path: str, dataset_names: list[str]):
    for dataset_name in dataset_names:
        path = Path(f"{dir_path}/{dataset_name}")
        files_names = glob.glob("*.txt", root_dir=path)
        yield [f"{dir_path}/{dataset_name}/{file_name}" for file_name in files_names]

def save_files_paths_to_file(paths_file:str, files_list: list[str]):
    with open(paths_file, "a") as f:
        for file_path in files_list:
            f.write(file_path + "\n")

def build_offsets(paths_file:str, offsets_file:str):
    offsets = []
    with open(paths_file, "rb") as f:
        off = f.tell()
        line = f.readline()
        while line:
            offsets.append(off)
            off = f.tell()
            line = f.readline()

    offsets = np.array(offsets, dtype=np.uint64)
    np.save(offsets_file, offsets, allow_pickle=False)
    print(f"Saved {len(offsets)} offsets -> {offsets_file}")
    return len(offsets)

def generate_indices(dir_path: str, dataset_names: list[str], paths_file: str, offsets_file: str, indices_file: str|None, shuffle: bool = True):
    # paths are appended per dataset, so a previous run's list has to go first
    Path(paths_file).unlink(missing_ok=True)
    for files in files_paths_generator(dir_path, dataset_names):
        save_files_paths_to_file(paths_file, files)

    n_files = build_offsets(paths_file, offsets_file)
    indices = np.arange(0, n_files, step=1)
    if shuffle:
        np.random.shuffle(indices)
    if indices_file:
        np.save(indices_file, indices)
    return indices

def create_train_test_split(config: dict) -> None:
    preprocessed_dir_path = config["preprocessed-dir-path"]
    datasets = config["speakleash-datasets"] + [config["huggingface-dataset-name"]]
    written_paths_file = config["train-paths-file"]
    written_offsets_file = config["train-offsets-file"]
    train_indices_file = config["train-indices-file"]

    val_indices_file = config["val-indices-file"]
    tune_indices_file = config["tune-indices-file"]
    test_indices_file = config["test-indices-file"]

    train_ratio = config["train-ratio"]
    val_ratio = config["val-ratio"]
    tune_ratio = config["tune-ratio"]

    print("Generating indices files for 'written' datasets ...")
    written_indices = generate_indices(dir_path=preprocessed_dir_path,
                                       dataset_names=datasets,
                                       paths_file=written_paths_file,
                                       offsets_file=written_offsets_file,
                                       indices_file=None,
                                       shuffle=True)
    n_written = len(written_indices)
    n_train = int(train_ratio * n_written)
    n_val = int(val_ratio * n_written)
    n_tune = int(tune_ratio * n_written)

    train_written = written_indices[:n_train]
    val_written = written_indices[n_train: n_train + n_val]
    tune_written = written_indices[n_train + n_val:n_train + n_val + n_tune]
    test_written = written_indices[n_train + n_val + n_tune:]

    np.save(train_indices_file, train_written)
    np.save(val_indices_file, val_written)
    np.save(tune_indices_file, tune_written)
    np.save(test_indices_file, test_written)

    preprocessed_conversations_dir_path = config["preprocessed-dialog-path"]
    conversations_tune_dataset = [config["conversations-train-dir-name"]]
    conversations_test_dataset = [config["conversations-test-dir-name"]]

    conversations_tune_paths = config["conversations-tune-paths-file"]
    conversations_test_paths = config["conversations-test-paths-file"]
    conversations_tune_offsets = config["conversations-tune-offsets-file"]
    conversations_test_offsets = config["conversations-test-offsets-file"]
    conversations_tune_indices = config["conversations-tune-indices-file"]
    conversations_test_indices = config["conversations-test-indices-file"]

    print("Generating indices file for fine-tuning `spoken` dataset ...")
    spoken_tune_indices = generate_indices(dir_path=preprocessed_conversations_dir_path,
                                           dataset_names=conversations_tune_dataset,
                                           paths_file=conversations_tune_paths,
                                           offsets_file=conversations_tune_offsets,
                                           indices_file=conversations_tune_indices,
                                           shuffle=True)

    print("Generating indices file for test `spoken` dataset ...")
    spoken_test_indices = generate_indices(dir_path=preprocessed_conversations_dir_path,
                                           dataset_names=conversations_test_dataset,
                                           paths_file=conversations_test_paths,
                                           offsets_file=conversations_test_offsets,
                                           indices_file=conversations_test_indices,
                                           shuffle=True)


if __name__ == "__main__":
    with open("next_token_prediction_model/config.yml") as f:
        config = yaml.safe_load(f)

    np.random.seed(config["seed"])

    create_train_test_split(config)
