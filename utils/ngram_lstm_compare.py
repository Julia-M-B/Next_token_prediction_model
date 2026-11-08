import numpy as np
from typing import List, Tuple, Dict
import json
from abc import ABC, abstractmethod
import time
import csv

import yaml

try:
    import kenlm
except ImportError:
    # only NGramPredictor needs kenlm; evaluating LSTM models works without it
    kenlm = None


class NextTokenPredictor(ABC):
    @abstractmethod
    def predict(self, text: str, k: int = 10) -> List[Tuple[str, float]]:
        """Zwraca [(token, probability), ...]"""
        pass

    def get_probability(self, text: str, next_token: str) -> float:
        """Zwraca prawdopodobieństwo konkretnego tokenu"""
        pass


class NGramPredictor(NextTokenPredictor):
    def __init__(self, model_name: str, csv_path: str):
        self.model = kenlm.Model(model_name + ".binary")
        self.order = self.model.order
        self.csv_path = csv_path
        self.vocab = self._load_vocab_from_csv()

    def _load_vocab_from_csv(self) -> List[str]:
        """
        Load vocabulary from the pre-built CSV (already sorted, filtered,
        and pruned). Much faster than re-parsing the ARPA file every run.
        """
        vocab = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vocab.append(row["word"])
        return vocab

    @staticmethod
    def _split_context_and_prefix(text: str) -> Tuple[List[str], str]:
        if not text or text.endswith(" "):
            return text.split(), ""
        context_tokens = text.split()
        return context_tokens[:-1], context_tokens[-1]

    def predict(self, text: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Return the top-k next-word predictions with their log10 probabilities.

        Uses model.BaseScore(state, word, out_state) which scores only the
        new token given the current state — avoids rescoring the whole context
        for every candidate word, making it significantly faster.

        Args:
            text: The input text (may end with a space or a partial word).
            k:    Number of predictions to return.

        Returns:
            List of (word, log10_prob) sorted best-first.
        """
        context_tokens, prefix = self._split_context_and_prefix(text)

        # Build KenLM state by feeding the context tokens one by one.
        # This is the correct way to condition on history.
        state = kenlm.State()
        self.model.BeginSentenceWrite(state)
        out_state = kenlm.State()

        history = context_tokens[-(self.order - 1):]
        for token in history:
            self.model.BaseScore(state, token, out_state)
            state, out_state = out_state, state  # out_state becomes new state

        # Score each vocab word conditioned on the accumulated state.
        candidates = []
        for word in self.vocab:
            if prefix and not word.startswith(prefix):
                continue
            log_prob = self.model.BaseScore(state, word, out_state)
            candidates.append((word, log_prob))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]

    def get_probability(self, text: str, next_token: str) -> float:
        tokens = text.strip().split()
        history = tokens[-(self.order - 1):]

        # Build state by feeding history tokens one by one
        state = kenlm.State()
        self.model.BeginSentenceWrite(state)
        out_state = kenlm.State()

        for token in history:
            self.model.BaseScore(state, token, out_state)
            state, out_state = out_state, state

        log_prob = self.model.BaseScore(state, next_token, out_state)
        return 10 ** log_prob


class LSTMPredictor(NextTokenPredictor):
    def __init__(self, model_dir: str, beam_width: int = 20,
                 max_word_length: int = 10, device: str = None,
                 alpha: float = 0.0, seq_len: int = 256,
                 model_name: str = "model.pt",
                 dictionary_csv: str = None):
        from src.beam_search import create_beam_searcher
        import torch

        self.device = torch.device(device if device else "cpu")
        self.searcher = create_beam_searcher(
            model_dir=model_dir,
            beam_width=beam_width,
            max_word_length=max_word_length,
            device=device,
            alpha=alpha,
            seq_len=seq_len,
            model_name=model_name,
            # dictionary_csv=dictionary_csv
        )

    def predict(self, context: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Returns (word, probability) pairs.
        Filters out fallback predictions (probability == 0.0) so they don't
        distort accuracy and MRR metrics — fallback words were chosen by
        frequency, not by the model, so including them would misattribute
        any accidental hits to model quality.
        """
        top_words = self.searcher.get_top_k_words(context_text=context, k=k)
        predictions = [
            (word, prob)
            for word, prob, num_tokens in top_words
            if not (prob == 0.0 and num_tokens == 0)   # exclude fallback tier-3
        ]
        return predictions

    def get_probability(self, context: str, next_token: str) -> float:
        return self.searcher.get_word_probability(context, next_token)
    
    
class TransformerPredictor(NextTokenPredictor):
    def __init__(self, model_id: str, beam_width: int = 20,
                 max_word_length: int = 10, device: str = "cpu",
                 alpha: float = 0.0):
        from bielik_beam_search import create_beam_searcher
        import torch

        self.device = torch.device(device if device else "cpu")
        self.searcher = create_beam_searcher(
            model_id=model_id,
            beam_width=beam_width,
            max_word_length=max_word_length,
            device=device,
            alpha=alpha,
        )

    def predict(self, context: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Returns (word, probability) pairs.
        Filters out fallback predictions (probability == 0.0) so they don't
        distort accuracy and MRR metrics — fallback words were chosen by
        frequency, not by the model, so including them would misattribute
        any accidental hits to model quality.
        """
        top_words = self.searcher.get_top_k_words(context_text=context, k=k)
        predictions = [
            (word, prob)
            for word, prob, num_tokens in top_words
            if not (prob == 0.0 and num_tokens == 0)   # exclude fallback tier-3
        ]
        return predictions

    def get_probability(self, context: str, next_token: str) -> float:
        return self.searcher.get_word_probability(context, next_token)



class ModelEvaluator:
    def __init__(self, models: Dict[str, NextTokenPredictor], seq_len: int = 100):
        self.models = models
        self.seq_len = seq_len
        self.results = {name: {
            'accuracy@1': [],
            'accuracy@3': [],
            'accuracy@5': [],
            'accuracy@7': [],
            'mrr': [],
            'perplexity': [],
            'inference_time': [],
            'n_preds': [],
            # Tracks how often the model returned no beam predictions at all
            # (i.e. every result was a fallback) — useful for diagnosing how
            # often the dictionary filter is too aggressive.
            'fallback_rate': [],
        } for name in models.keys()}

    def context_target_pairs_generator(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        words = text.split()
        sequence = []
        for word in words:
            sequence.append(word)
            while len(sequence) >= self.seq_len + 1:
                yield " ".join(sequence[:self.seq_len]) + " ", sequence[self.seq_len]
                sequence = sequence[int(self.seq_len / 2):]

    def evaluate_file(self, file_path: str):
        preds_counter = 0
        for context, target in self.context_target_pairs_generator(file_path):
            preds_counter += 1
            for model_name, model in self.models.items():
                start_time = time.time()
                predictions = model.predict(context, k=7)
                inference_time = time.time() - start_time

                # Track whether this sample produced any real beam predictions.
                # An empty list here means everything was filtered out and the
                # fallback was used (or nothing was found at all).
                is_fallback_only = len(predictions) == 0
                self.results[model_name]['fallback_rate'].append(
                    1.0 if is_fallback_only else 0.0
                )

                if not predictions:
                    self.results[model_name]['accuracy@1'].append(0.0)
                    self.results[model_name]['accuracy@3'].append(0.0)
                    self.results[model_name]['accuracy@5'].append(0.0)
                    self.results[model_name]['accuracy@7'].append(0.0)
                    self.results[model_name]['mrr'].append(0.0)
                    self.results[model_name]['inference_time'].append(inference_time)
                    # Perplexity is skipped when there are no predictions —
                    # get_probability still works but would be inconsistent with
                    # the zero accuracy signal. The perplexity denominator
                    # therefore only counts samples with real beam predictions.
                    continue

                pred_words = [word for word, _ in predictions]

                acc1 = 1.0 if target == pred_words[0] else 0.0
                acc3 = 1.0 if target in pred_words[:3] else 0.0
                acc5 = 1.0 if target in pred_words[:5] else 0.0
                acc7 = 1.0 if target in pred_words[:7] else 0.0
                mrr = self._calculate_mrr(pred_words, target)

                prob = model.get_probability(context, target)
                log_prob = np.log(prob + 1e-10)

                self.results[model_name]['accuracy@1'].append(acc1)
                self.results[model_name]['accuracy@3'].append(acc3)
                self.results[model_name]['accuracy@5'].append(acc5)
                self.results[model_name]['accuracy@7'].append(acc7)
                self.results[model_name]['mrr'].append(mrr)
                self.results[model_name]['perplexity'].append(log_prob)
                self.results[model_name]['inference_time'].append(inference_time)

        for model_name in self.models.keys():
            self.results[model_name]['n_preds'].append(preds_counter)

    @staticmethod
    def get_files_paths(indices, paths_file, offsets_file):
        from src.model import read_by_index
        return read_by_index(indices.tolist(), paths_file, offsets_file)

    def evaluate_directory(self, test_indices: np.ndarray, paths_file: str,
                           offsets_file: str, output_path: str,
                           save_every: int = 10) -> None:
        from tqdm import tqdm
        import random

        files = self.get_files_paths(test_indices, paths_file, offsets_file)
        random.shuffle(files)
        files = files[:250]

        count = 0
        for file in tqdm(files, desc="Evaluating files"):
            self.evaluate_file(file)
            count += 1
            if count % save_every == 0:
                self.save_results(output_path)

    @staticmethod
    def _calculate_mrr(predictions: List[str], true_token: str) -> float:
        try:
            rank = predictions.index(true_token) + 1
            return 1.0 / rank
        except ValueError:
            return 0.0

    def get_summary(self) -> Dict:
        summary = {}
        for model_name, metrics in self.results.items():
            n_total = sum(metrics['n_preds'])
            fallback_rate = (np.mean(metrics['fallback_rate'])
                             if metrics['fallback_rate'] else 0.0)

            summary[model_name] = {
                'accuracy@1': np.mean(metrics['accuracy@1']) if metrics['accuracy@1'] else 0.0,
                'accuracy@3': np.mean(metrics['accuracy@3']) if metrics['accuracy@3'] else 0.0,
                'accuracy@5': np.mean(metrics['accuracy@5']) if metrics['accuracy@5'] else 0.0,
                'accuracy@7': np.mean(metrics['accuracy@7']) if metrics['accuracy@7'] else 0.0,
                'mrr': np.mean(metrics['mrr']) if metrics['mrr'] else 0.0,
                # Perplexity only counts samples with real beam predictions
                'perplexity': np.exp(-np.mean(metrics['perplexity'])) if metrics['perplexity'] else float('inf'),
                'avg_inference_time_ms': np.mean(metrics['inference_time']) * 1000,
                'fallback_rate': fallback_rate,
                'model_predictions': len(metrics['accuracy@1']),
                'total_predictions': n_total,
            }
        return summary

    def print_comparison(self):
        summary = self.get_summary()

        print("\n" + "=" * 80)
        print("MODEL COMPARISON RESULTS")
        print("=" * 80)

        for model_name, metrics in summary.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy@1:    {metrics['accuracy@1']:.4f} ({metrics['accuracy@1'] * 100:.2f}%)")
            print(f"  Accuracy@3:    {metrics['accuracy@3']:.4f} ({metrics['accuracy@3'] * 100:.2f}%)")
            print(f"  Accuracy@5:    {metrics['accuracy@5']:.4f} ({metrics['accuracy@5'] * 100:.2f}%)")
            print(f"  Accuracy@7:   {metrics['accuracy@7']:.4f} ({metrics['accuracy@7'] * 100:.2f}%)")
            print(f"  MRR:           {metrics['mrr']:.4f}")
            print(f"  Perplexity:    {metrics['perplexity']:.2f}")
            print(f"  Avg Time:      {metrics['avg_inference_time_ms']:.2f} ms")
            print(f"  Fallback rate: {metrics['fallback_rate'] * 100:.2f}%  "
                  f"(samples with no beam predictions)")
            print(f"  Model Preds:   {metrics['model_predictions']}")
            print(f"  Total Preds:   {metrics['total_predictions']}")

        print("\n" + "-" * 80)
        print("WINNER IN EACH CATEGORY:")
        print("-" * 80)

        categories = ['accuracy@1', 'accuracy@3', 'accuracy@5', 'accuracy@7', 'mrr']
        for cat in categories:
            best_model = max(summary.items(), key=lambda x: x[1][cat])
            print(f"  {cat:15s}: {best_model[0]} ({best_model[1][cat]:.4f})")

        best_perplexity = min(summary.items(), key=lambda x: x[1]['perplexity'])
        print(f"  {'perplexity':15s}: {best_perplexity[0]} ({best_perplexity[1]['perplexity']:.2f})")

        best_time = min(summary.items(), key=lambda x: x[1]['avg_inference_time_ms'])
        print(f"  {'speed':15s}: {best_time[0]} ({best_time[1]['avg_inference_time_ms']:.2f} ms)")

        best_fallback = min(summary.items(), key=lambda x: x[1]['fallback_rate'])
        print(f"  {'fallback rate':15s}: {best_fallback[0]} ({best_fallback[1]['fallback_rate'] * 100:.2f}%)")

    def save_results(self, output_path: str):
        summary = self.get_summary()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nWyniki zapisane do: {output_path}")


if __name__ == "__main__":
    with open("next_token_prediction_model/config.yml") as f:
        config = yaml.safe_load(f)

    test_indices = np.load(config["test-indices-file"])
    paths_file = config["train-paths-file"]
    offsets_file = config["train-offsets-file"]

    conversations_test_indices = np.load(config["conversations-test-indices-file"])
    conversations_paths_file = config["conversations-test-paths-file"]
    conversations_offsets_file = config["conversations-test-offsets-file"]

    ngram_model = NGramPredictor(
        model_name="model_3gram",
        csv_path="unigrams200k.csv"
    )

    lstm_model_alpha0 = LSTMPredictor(
        model_dir=".",
        beam_width=25,
        max_word_length=10,
        device="cpu",
        alpha=0.0,
        model_name="model.pt",
        dictionary_csv="unigrams200k.csv"
    )

    lstm_model_alpha2 = LSTMPredictor(
        model_dir=".",
        beam_width=25,
        max_word_length=10,
        device="cpu",
        alpha=0.2,
        model_name="model.pt",
        dictionary_csv="unigrams200k.csv"
    )

    fine_tuned_lstm_model = LSTMPredictor(
        model_dir=".",
        beam_width=25,
        max_word_length=10,
        device="cpu",
        alpha=0.2,
        model_name="fine_tuned_model.pt",
        dictionary_csv="unigrams200k.csv"
    )

    lstm_model_alpha4 = LSTMPredictor(
        model_dir=".",
        beam_width=25,
        max_word_length=10,
        device="cpu",
        alpha=0.4,
        model_name="model.pt",
        dictionary_csv="unigrams200k.csv"
    )

    lstm_model_alpha6 = LSTMPredictor(
        model_dir=".",
        beam_width=25,
        max_word_length=10,
        device="cpu",
        alpha=0.6,
        model_name="model.pt",
        dictionary_csv="unigrams200k.csv"
    )

    lstm_model_alpha8 = LSTMPredictor(
        model_dir=".",
        beam_width=25,
        max_word_length=10,
        device="cpu",
        alpha=0.8,
        model_name="model.pt",
        dictionary_csv="unigrams200k.csv"
    )

    lstm_model_alpha10 = LSTMPredictor(
        model_dir=".",
        beam_width=25,
        max_word_length=10,
        device="cpu",
        alpha=1.0,
        model_name="model.pt",
        dictionary_csv="unigrams200k.csv"
    )

    lstm_model_alpha2_beam50 = LSTMPredictor(
        model_dir=".",
        beam_width=50,
        max_word_length=10,
        device="cpu",
        alpha=0.2,
        model_name="model.pt",
        dictionary_csv="unigrams200k.csv"
    )

    lstm_model_alpha2_no_unigram = LSTMPredictor(
        model_dir=".",
        beam_width=25,
        max_word_length=10,
        device="cpu",
        alpha=0.2,
        model_name="model.pt",
        dictionary_csv=None
    )

    lstm_model_alpha2_unigram400 = LSTMPredictor(
        model_dir=".",
        beam_width=25,
        max_word_length=10,
        device="cpu",
        alpha=0.2,
        model_name="model.pt",
        dictionary_csv="unigrams400k.csv"
    )

    ngram_model_unigram400 = NGramPredictor(
        model_name="model_3gram",
        csv_path="unigrams400k.csv"
    )
    
    bielik_4_5B = TransformerPredictor(
        model_id="speakleash/Bielik-4.5B-v3",
        # model_id="speakleash/Bielik-4.5B-v3.0-Instruct",
        beam_width=25,
        max_word_length=10,
        alpha=0.2
    )
    
    bielik_1_5B = TransformerPredictor(
        model_id="speakleash/Bielik-1.5B-v3",
        beam_width=25,
        max_word_length=10,
        alpha=0.2
    )

    models = {
        # 'n-gram': ngram_model,
        # 'lstm0': lstm_model_alpha0,
        # 'lstm2': lstm_model_alpha2,
        # 'ft_lstm2': fine_tuned_lstm_model,
        # 'lstm4': lstm_model_alpha4,
        # 'lstm6': lstm_model_alpha6,
        # 'lstm8': lstm_model_alpha8,
        # 'lstm10': lstm_model_alpha10,
        # 'lstm2_beam50': lstm_model_alpha2_beam50,
        # 'lstm2_no_uni': lstm_model_alpha2_no_unigram,
        # 'lstm2_uni400': lstm_model_alpha2_unigram400,
        # 'n-gram_uni400': ngram_model_unigram400,
        # "bielik_45": bielik_4_5B,
        "bielik_15": bielik_1_5B,
    }

    evaluator = ModelEvaluator(models)

    print("Rozpoczynam ewaluację na zbiorze pisanym ...")

    evaluator.evaluate_directory(
        test_indices=test_indices,
        paths_file=paths_file,
        offsets_file=offsets_file,
        output_path="comparison_results_written_bielik_small.json"
    )

    evaluator.print_comparison()
    evaluator.save_results("comparison_results_written_bielik_small.json")

    # print("Rozpoczynam ewaluację na zbiorze mówionym ...")
    # 
    # evaluator.evaluate_directory(
    #     test_indices=conversations_test_indices,
    #     paths_file=conversations_paths_file,
    #     offsets_file=conversations_offsets_file,
    #     output_path="comparison_results_conversations_acc7.json"
    # )
    # 
    # evaluator.print_comparison()
    # evaluator.save_results("comparison_results_conversations_acc7.json")