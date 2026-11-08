import json
import math
import os
import random

import yaml
from collections import Counter
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



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
        batch_indices = indices[i: i + batch_size]
        yield read_by_index(batch_indices.tolist(), paths_file, offsets_file)
        i += batch_size

    yield read_by_index(indices[i:].tolist(), paths_file, offsets_file)


def iter_lines(paths: List[str]):
    """
    Generator: yield one non-empty stripped line at a time across all files.
    Never loads more than one line into memory at once.
    """
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line

def welford_std(n, m2):
    return math.sqrt(m2 / n) if n > 1 else 0.0

def extract_metrics(batch_size: int, indices: list[int], paths_file: str,
                    offsets_file: str) -> Dict[str, Any]:
    """
    Compute all metrics for one corpus in a single streaming pass.
    """
    dataset_statistics = {}

    # variables per dataset
    total_sentences = 0
    total_words = 0
    total_files = 0
    avg_words_num_per_file = 0
    avg_sentences_num_per_file = 0
    avg_sentence_len = 0
    avg_word_len = 0

    # sentence-length running stats  (Welford online algorithm)
    sentence_len_m2 = 0  # for std-dev without storing all values
    sent_very_short = 0  # <= 2 tokens
    sent_very_long = 0  # >  20 tokens

    # word-length running stats  (Welford online algorithm)
    word_len_m2 = 0 # for std-dev without storing all values

    words_freqs = Counter()  # unigram counts

    repetition_count = 0
    words_pairs_count = 0
    within_sent_rep_mean = 0

    for paths in tqdm(get_files_paths(batch_size, indices, paths_file, offsets_file), total=len(indices)):

        # variables per file
        sentences_in_file = 0
        words_in_file = 0

        for line in iter_lines(paths):
            # for bigram counts and in sentence repetition
            # reset every sentence
            prev1 = prev2 = None

            within_sent_rep = 0

            words = line.split()
            n_words = len(words)
            sentences_in_file += 1
            total_sentences += 1
            words_in_file += n_words

            # Welford's online algorithm
            sentence_len_delta = n_words - avg_sentence_len
            avg_sentence_len += sentence_len_delta / total_sentences
            sentence_len_m2 += sentence_len_delta * (n_words - avg_sentence_len)
            if n_words <= 2:
                sent_very_short += 1
            if n_words > 20:
                sent_very_long += 1

            for word in words:
                word_len = len(word)
                total_words += 1

                word_len_delta = word_len - avg_word_len
                avg_word_len += word_len_delta / total_words
                word_len_m2 += word_len_delta * (word_len - avg_word_len)

                # frequencies dict
                words_freqs[word] += 1

                if prev2 is not None and prev1 is not None:
                    words_pairs_count += 1
                    if prev2 == prev1:
                        repetition_count += 1
                        within_sent_rep += 1
                        within_sent_rep_mean += (within_sent_rep - within_sent_rep_mean) / total_sentences

                prev2, prev1 = prev1, word

        total_files += batch_size
        avg_words_num_per_file += (words_in_file - avg_words_num_per_file) / total_files
        avg_sentences_num_per_file += (sentences_in_file - avg_sentences_num_per_file) / total_files

    dataset_statistics["total_files"] = total_files
    dataset_statistics["total_sentences"] = total_sentences
    dataset_statistics["total_words"] = total_words

    dataset_statistics["avg_words_num_per_file"] = avg_words_num_per_file
    dataset_statistics["avg_sentences_num_per_file"] = avg_sentences_num_per_file

    dataset_statistics["avg_sentence_len"] = avg_sentence_len
    dataset_statistics["avg_word_len"] = avg_word_len

    dataset_statistics["sentence_len_std"] = welford_std(total_sentences, sentence_len_m2)
    dataset_statistics["word_len_std"] = welford_std(total_words, word_len_m2)

    dataset_statistics["sent_very_short_ratio"] = sent_very_short / max(total_sentences, 1) * 100
    dataset_statistics["sent_very_long_ratio"] = sent_very_long / max(total_sentences, 1) * 100

    # vocabulary
    vocab = set(words_freqs.keys())
    dataset_statistics["vocab_size"] = len(vocab)
    dataset_statistics["ttr"] = len(vocab) / max(total_words, 1)


    dataset_statistics["repetition_count"] = repetition_count
    dataset_statistics["repetition_ratio"] = repetition_count / max(words_pairs_count, 1) * 100
    dataset_statistics["within_sent_rep_mean"] = within_sent_rep_mean


    dataset_statistics["top50_words"] = words_freqs.most_common(50)

    return dataset_statistics


def export_json(dataset_statistics_a: Dict, dataset_statistics_b: Dict,
                label_a: str, label_b: str,
                path: str) -> None:

    data = {
        label_a: dataset_statistics_a,
        label_b: dataset_statistics_b,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  JSON saved to: {os.path.abspath(path)}\n")



def create_latex_table(dataset_statistics_a: dict, dataset_statistics_b: dict) -> str:
    def _rel_diff(a, b):
        if a == 0:
            return 0.0
        return (b - a) / a * 100

    total_words_a = dataset_statistics_a.get("total_words", 0)
    total_words_b = dataset_statistics_b.get("total_words", 0)
    total_words_diff = _rel_diff(total_words_a, total_words_b)

    total_sentences_a = dataset_statistics_a.get("total_sentences", 0)
    total_sentences_b = dataset_statistics_b.get("total_sentences", 0)
    total_sentences_diff = _rel_diff(total_sentences_a, total_sentences_b)

    vocab_size_a = dataset_statistics_a.get("vocab_size", 0)
    vocab_size_b = dataset_statistics_b.get("vocab_size", 0)
    vocab_size_diff = _rel_diff(vocab_size_a, vocab_size_b)

    mean_word_len_a = dataset_statistics_a.get("avg_word_len", 0)
    mean_word_len_b = dataset_statistics_b.get("avg_word_len", 0)
    mean_word_len_diff = _rel_diff(mean_word_len_a, mean_word_len_b)

    mean_sen_len_a = dataset_statistics_a.get("avg_sentence_len", 0)
    mean_sen_len_b = dataset_statistics_b.get("avg_sentence_len", 0)
    mean_sen_len_diff = _rel_diff(mean_sen_len_a, mean_sen_len_b)

    repetitions_a = dataset_statistics_a.get("repetition_ratio", 0)
    repetitions_b = dataset_statistics_b.get("repetition_ratio", 0)
    repetitions_diff = _rel_diff(repetitions_a, repetitions_b)

    short_sen_a = dataset_statistics_a.get("sent_very_short_ratio", 0)
    short_sen_b = dataset_statistics_b.get("sent_very_short_ratio", 0)
    short_sen_diff = _rel_diff(short_sen_a, short_sen_b)

    long_sen_a = dataset_statistics_a.get("sent_very_long_ratio", 0)
    long_sen_b = dataset_statistics_b.get("sent_very_long_ratio", 0)
    long_sen_diff = _rel_diff(long_sen_a, long_sen_b)

    row_1 = "\\textbf{Łączna liczba słów} & " + f"{total_words_a} & {total_words_b} & {total_words_diff:.4f}\\% \\\ \\midrule"
    row_2 = "\\textbf{Łączna liczba zdań} & " + f"{total_sentences_a} & {total_sentences_b} & {total_sentences_diff:.4f}\\% \\\ \\midrule"
    row_3 = "\\textbf{Wielkość słownika} & " + f"{vocab_size_a} & {vocab_size_b} & {vocab_size_diff:.4f}\\% \\\ \\midrule"
    row_4 = "\\textbf{Średnia długość słowa} & " + f"{mean_word_len_a:.4f} & {mean_word_len_b:.4f} & {mean_word_len_diff:.4f}\\% \\\ \\midrule"
    row_5 = "\\textbf{Średnia długość zdania} & " + f"{mean_sen_len_a:.4f} & {mean_sen_len_b:.4f} & {mean_sen_len_diff:.4f}\\% \\\ \\midrule"
    row_6 = "\\textbf{Procent powtórzeń} & " + f"{repetitions_a:.4f} & {repetitions_b:.4f} & {repetitions_diff:.4f}\\% \\\ \\midrule"
    row_7 = "\\textbf{Procent krótkich zdań} & " + f"{short_sen_a:.4f} & {short_sen_b:.4f} & {short_sen_diff:.4f}\\% \\\ \\midrule"
    row_8 = "\\textbf{Procent długich zdań} & " + f"{long_sen_a:.4f} & {long_sen_b:.4f} & {long_sen_diff:.4f}\\% \\\ \\midrule"

    table = [row_1, row_2, row_3, row_4, row_5, row_6, row_7, row_8]
    return "\n".join(table)


def create_common_words_plot(common_words: list, words_num: int, corpus_type: str, save_path: str, colour: str):
    import seaborn as sns
    # set style
    sns.set_theme(
        context="paper",
        style="whitegrid"
    )



    top_common_words = sorted(common_words, key=lambda x: x[1], reverse=True)[:20]
    words = [w for w, wc in top_common_words]
    words_counts = [wc for w, wc in top_common_words]
    words_counts = [int(wc / words_num * 1_000_000) for wc in words_counts]

    plt.figure(figsize=(9, 6))
    plt.barh(words, words_counts, color=colour)
    plt.gca().invert_yaxis()  # highest value on top
    plt.xlabel("Liczba wystąpień na milion słów", fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f"Najczęstsze 20 słów w języku {corpus_type}", fontsize=16)
    plt.savefig(f"{save_path}.png")


def main():
    colour_1 = "#219CDE"
    colour_2 = "#DE6321"

    # load general project config
    with open("next_token_prediction_model/config.yml") as f:
        config = yaml.safe_load(f)


    written_tune_indices = np.load(config["tune-indices-file"])
    conversation_indices = np.load(config["conversations-tune-indices-file"])

    random.shuffle(written_tune_indices)
    cut_idx = int(8.5 * len(conversation_indices))
    written_tune_indices = written_tune_indices[:cut_idx]

    written_paths = config["train-paths-file"]
    written_offsets = config["train-offsets-file"]

    conversations_paths = config["conversations-tune-paths-file"]
    conversations_offsets = config["conversations-tune-offsets-file"]


    print(f"\n  Written corpus: {len(written_tune_indices)} file(s)")
    print(f"  Spoken corpus: {len(conversation_indices)} file(s)")

    print("\n  Analysing written corpus ...")
    written_metrics = extract_metrics(batch_size=1,
                                      indices=written_tune_indices,
                                      paths_file=written_paths,
                                      offsets_file=written_offsets)
    print("\n  Analysing spoken corpus ...")
    conversations_metrics = extract_metrics(batch_size=1,
                                            indices=conversation_indices,
                                            paths_file=conversations_paths,
                                            offsets_file=conversations_offsets)

    export_json(written_metrics,
                conversations_metrics,
                "written",
                "spoken",
                "dataset_comparison.json")


    t = create_latex_table(written_metrics, conversations_metrics)
    print("\n", t, "\n")

    written_words = written_metrics.get("top50_words")
    written_words_num = written_metrics.get("total_words")
    create_common_words_plot(written_words, written_words_num, "pisanym", "top20_written", colour_1)

    spoken_words = conversations_metrics.get("top50_words")
    spoken_words_num = conversations_metrics.get("total_words")
    create_common_words_plot(spoken_words, spoken_words_num, "mówionym", "top20_spoken", colour_2)


if __name__ == "__main__":
    main()