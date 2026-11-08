import csv
import kenlm
from typing import List, Tuple


class KenLMPredictor:
    def __init__(self, model_path: str, vocab_csv: str):
        """
        Args:
            model_path: Path to the .binary KenLM model file.
            vocab_csv:  Path to the CSV produced by arpa_unigrams_to_csv.py
                        (columns: word, log10_probability).
        """
        self.model = kenlm.Model(model_path)
        self.order = self.model.order
        self.vocab = self._load_vocab_from_csv(vocab_csv)

    # ------------------------------------------------------------------
    # Vocab loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_vocab_from_csv(csv_path: str) -> List[str]:
        """
        Load vocabulary from the pre-built CSV (already sorted, filtered,
        and pruned). Much faster than re-parsing the ARPA file every run.
        """
        vocab = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vocab.append(row["word"])
        return vocab

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_context_and_prefix(text: str) -> Tuple[List[str], str]:
        """
        Returns:
            context  – list of complete tokens to use as LM history
            prefix   – the partially-typed word ("" if the text ends with a space)
        """
        if not text or text.endswith(" "):
            return text.split(), ""
        tokens = text.split()
        return tokens[:-1], tokens[-1]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_next(self, text: str, k: int = 5) -> List[Tuple[str, float]]:
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


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------

if __name__ == "__main__":
    MODEL_PATH = "model_3gram.binary"
    VOCAB_CSV  = "unigrams.csv"

    predictor = KenLMPredictor(MODEL_PATH, VOCAB_CSV)
    print(f"Vocabulary size: {len(predictor.vocab):,}")
    print(f"Sample vocab: {predictor.vocab[:10]}")

    context = "chociaż mam prawie trzydzieści lat cały czas czuję się "
    top_k = 5

    predictions = predictor.predict_next(context, k=top_k)

    print(f'\nKontekst: "{context}"')
    print("Najbardziej prawdopodobne kolejne słowa:\n")
    for word, logp in predictions:
        prob = 10 ** logp
        print(f"{word:20s}  logP={logp:.4f}  P≈{prob:.6f}")