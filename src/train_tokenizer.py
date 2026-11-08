import os
import sentencepiece as spm
import numpy as np
from tqdm import tqdm
import yaml

def read_by_index(indices: list[int], paths_file:str, offsets_file:str) -> list[str]:
    offsets = np.load(offsets_file, mmap_mode="r")
    result = []
    with open(paths_file, "rb") as f:
        for idx in indices:
            off = int(offsets[idx])
            f.seek(off)
            line = f.readline().rstrip(b"\n\r")
            result.append(line.decode("utf-8"))
    return result

def train_sentencepiece_tokenizer(paths_files: list[str], indices_list: list[list[int]],
                                  offsets_files: list[str], model_prefix: str,
                                  vocab_size: int, model_type: str = 'unigram') -> None:

    concatenated_files = f'{model_prefix}_filelist.txt'
    for paths_file, indices, offsets_file in zip(paths_files, indices_list, offsets_files):
        files = read_by_index(indices, paths_file, offsets_file)

        for file in tqdm(files, total=len(files)):
            with open(file, "r") as f_in, open(concatenated_files, 'a',
                                               encoding='utf-8') as f_out:
                f_out.write(f_in.read())

    print("Created concatenated file.")
    print("Training tokenizer ...")

    spm.SentencePieceTrainer.Train(
        input=concatenated_files,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=0.9995,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        input_sentence_size=10_000_000,
        shuffle_input_sentence=True,
        num_threads=8
    )
    os.remove(concatenated_files)


def train_tokenizers(config: dict) -> None:
    written_paths_file = config["train-paths-file"]
    written_offsets_file = config["train-offsets-file"]
    conversations_tune_paths = config["conversations-tune-paths-file"]
    conversations_tune_offsets = config["conversations-tune-offsets-file"]
    tokenizer_type = config["tokenizer-type"]

    # train different tokenizers for hyperparameter optimization
    model_prefixes = config["tokenizer-prefix"]
    vocab_sizes = config["vocab-size"]

    train_indices = np.load(config["train-indices-file"])
    np.random.shuffle(train_indices)
    n = int(config["tokenizer-ratio"] * len(train_indices))
    tokenizer_indices = train_indices[:n].tolist()
    tune_tokenizer_indices = np.load(config["conversations-tune-indices-file"]).tolist()

    for model_prefix, vocab_size in zip(model_prefixes, vocab_sizes):
        train_sentencepiece_tokenizer(paths_files=[written_paths_file, conversations_tune_paths],
                                      indices_list=[tokenizer_indices, tune_tokenizer_indices],
                                      offsets_files=[written_offsets_file, conversations_tune_offsets],
                                      model_prefix=model_prefix,
                                      vocab_size=vocab_size,
                                      model_type=tokenizer_type)


if __name__ == "__main__":
    with open("next_token_prediction_model/config.yml") as f:
        config = yaml.safe_load(f)

    np.random.seed(config["seed"])

    train_tokenizers(config)
