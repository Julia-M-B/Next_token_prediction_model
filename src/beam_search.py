import heapq
import math
from typing import List, Tuple, Dict, Union
from dataclasses import dataclass, field
import unicodedata
import re
import numpy as np

CLEAN_REGEX = re.compile(r"[^a-ząćęłńóśźż0-9\s]")
MULTIPLE_WHITESPACE = re.compile(r"[ \t\n]+")


@dataclass(
    order=True)  # order=True automatycznie tworzy dunder methods pozwalajace porownywac dwa obiekty typu BeamItem
class BeamItem:
    """Represents a partial word being explored in the beam.

    Attributes:
        neg_log_prob: Negative log probability (for max-heap using min-heap)
        tokens: List of token IDs forming this partial word
        text: Human-readable text of the partial word
    """
    # prawdopodobienstwo: [0, 1] -> logarytm z tego: [-inf (token o prawdop. 0), 0 (token o prawdop. 1)]
    # jak teraz zamienimy to na negative number to tokeny majace najmniejsze prawdopodobienstwo pojawienia sie
    # beda mialy najwyzsza wartosc - co jest idealna sytuacja, jesli korzystamy z min-heap (domyslna implementaxja z heapq)
    # bo elementy najmniejsze beda na poczatku, a najwieksze - na koncu
    neg_log_prob_normalised: float
    neg_log_prob: float = field(compare=False)
    tokens: List[int] = field(compare=False)  # tego nie chcemy porownywac
    text: str = field(compare=False)  # tego tez nie chcemy porownywac


@dataclass(order=True)
class CompletedWord:
    """Represents a completed word with its probability.

    Attributes:
        neg_log_prob: Negative log probability
        tokens: List of token IDs forming the complete word
        text: Human-readable text of the word
        probability: Actual probability (exp of negative neg_log_prob)
    """
    # prawdopodobienstwo: [0, 1] -> logarytm z tego: [-inf (token o prawdop. 0), 0 (token o prawdop. 1)]
    # jak teraz zamienimy to na negative number to tokeny majace najmniejsze prawdopodobienstwo pojawienia sie
    # beda mialy najwyzsza wartosc - co jest idealna sytuacja, jesli korzystamy z min-heap (domyslna implementaxja z heapq)
    # bo elementy najmniejsze beda na poczatku, a najwieksze - na koncu
    neg_log_prob_normalised: float
    tokens: List[int] = field(compare=False)
    text: str = field(compare=False)
    probability: float = field(compare=False)


class WordPredictionBeamSearch:
    """Efficient beam search for finding top-K most probable next words."""

    def __init__(self, model, tokenizer, beam_width: int = 50,
                 max_word_length: int = 15, alpha: float = 0.2):
        """
        Args:
            model: LSTM model with predict() method that returns token probabilities
            tokenizer: Tokenizer with decode() and check if token starts word
            beam_width: Maximum number of partial words to keep in beam
            max_word_length: Maximum number of tokens per word (prunes longer words)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.max_word_length = max_word_length
        self.inference_count = 0
        self.start_new_word_char: str = "▁"
        self.alpha = alpha

        # Pre-build piece list indexed by token ID — avoids rebuilding a dict
        # on every _get_top_matching_tokens call (O(1) lookup instead of dict alloc)
        vocab_size = tokenizer.get_piece_size()
        self._id_to_piece: List[str] = [
            tokenizer.id_to_piece(i) for i in range(vocab_size)
        ]

    def starts_new_word(self, token_id: int) -> bool:
        """Check if a token starts a new word (piece starts with '▁' marker)."""
        token_piece = self.tokenizer.id_to_piece(token_id)
        return token_piece.startswith(self.start_new_word_char)

    def contains_letters_only(self, token_id: int) -> bool:
        token_text = self.tokenizer.decode([token_id])
        return token_text.isalpha()

    def get_top_k_words(self, context_text: str, k: int = 5) -> List[
        Tuple[str, float, int]]:
        """
        Find top-K most probable next complete words.

        Args:
            context_text: Input text to predict next word from
            k: Number of top words to return

        Returns:
            List of (word_text, probability, num_tokens) tuples
        """
        # Reset inference counter
        self.inference_count = 0

        # Encode context
        context_text = self._clean_context_text(context_text)
        context_text, unfinished_word = self._extract_unfinished_word(
            context_text)
        context_tokens = self.tokenizer.encode(context_text)

        beam = [BeamItem(neg_log_prob_normalised=0.0, neg_log_prob=0.0, tokens=[], text="")]
        completed_words = []
        completed_words_texts = []

        # Track explored prefixes to avoid cycles (only mark as explored after processing)
        explored_prefixes: set[Tuple[int, ...]] = set()

        iteration = 0
        max_iterations = k * self.beam_width * 10  # Safety limit to prevent infinite loops

        # if unfinished word, get only matching tokens, that starts new word
        if unfinished_word:
            # Pop the most promising partial word (lowest neg_log_prob = highest prob)
            current = heapq.heappop(beam)

            # Mark this prefix as explored (we're about to process it)
            explored_prefixes.add(tuple(current.tokens))

            # Run model inference
            token_probs = self.model.predict(context_tokens + current.tokens)
            self.inference_count += 1

            top_next_tokens = self._get_top_matching_tokens(token_probs,
                                                            self.beam_width,
                                                            current.text,
                                                            unfinished_word,
                                                            beam_init=True)

            # Expand beam with each possible next token
            for token_id, token_prob in top_next_tokens:
                new_item = self._create_new_beam_prefix(current, token_id,
                                                        token_prob)
                if tuple(new_item.tokens) not in explored_prefixes:
                    heapq.heappush(beam, new_item)

            # Prune beam to width (keep only top beam_width items)
            beam = heapq.nsmallest(self.beam_width, beam)

        # Continue until we have k completed words or beam is exhausted
        while beam and len(completed_words) < (k * 2) and iteration < max_iterations:
            iteration += 1

            # Pop the most promising partial word (lowest neg_log_prob = highest prob)
            current = heapq.heappop(beam)

            # Prune: Skip if prefix is too long (unlikely to be a real word)
            if len(current.tokens) > self.max_word_length:
                continue

            if tuple(current.tokens) in explored_prefixes:
                continue

            # Mark this prefix as explored (we're about to process it)
            explored_prefixes.add(tuple(current.tokens))

            # Run model inference
            token_probs = self.model.predict(context_tokens + current.tokens)
            self.inference_count += 1

            # Get top beam_width tokens
            if unfinished_word:
                top_next_tokens = self._get_top_matching_tokens(token_probs, self.beam_width, current.text, unfinished_word, beam_init=False)
            else:
                top_next_tokens = self._get_top_tokens(token_probs, self.beam_width)

            # Expand beam with each possible next token
            for token_id, token_prob in top_next_tokens:
                if not self.contains_letters_only(token_id):
                    continue

                if self.starts_new_word(token_id):
                    # If we have a partial word to complete, complete it first
                    if current.text.strip():  # Only complete if we have a non-empty prefix
                        completed_word = self._create_complete_word(current)
                        if completed_word and completed_word.text not in completed_words_texts:
                            heapq.heappush(completed_words,
                                           completed_word)
                            completed_words_texts.append(
                                completed_word.text)

                    # no prefixes were made yet; we have to create first prefixes
                    else:
                        new_item = self._create_new_beam_prefix(current, token_id, token_prob)
                        if tuple(new_item.tokens) not in explored_prefixes:
                            heapq.heappush(beam, new_item)

                else:
                    # Word continues, add to beam
                    new_item = self._create_new_beam_prefix(current, token_id,
                                                            token_prob)
                    if tuple(new_item.tokens) not in explored_prefixes:
                        heapq.heappush(beam, new_item)

            # Prune beam to width (keep only top beam_width items)
            beam = heapq.nsmallest(self.beam_width, beam)

        # Return top k completed words
        top_words = heapq.nsmallest(k, completed_words)

        results = [
            (word.text, word.probability, len(word.tokens))
            for word in top_words
        ]

        return results

    def _get_top_matching_tokens(self, token_probs: List[float], k: int, current_prefix: str, unfinished_word: str, beam_init: bool = False) -> List[Tuple[int, float]]:
        unfinished_word = unfinished_word.strip()

        if beam_init and not unfinished_word.startswith(
                self.start_new_word_char):
            unfinished_word = self.start_new_word_char + unfinished_word

        candidates = []

        for token_id, prob in enumerate(token_probs):
            piece = self._id_to_piece[token_id]

            new_prefix = current_prefix + piece
            if new_prefix.startswith(
                    unfinished_word) or unfinished_word.startswith(new_prefix):
                candidates.append((token_id, prob))

        top_k = sorted(candidates, key=lambda x: x[1], reverse=True)[:k]
        return top_k

    @staticmethod
    def _get_top_tokens(token_probs: List[float], k: int) -> List[
        Tuple[int, float]]:
        """Get top-k tokens by probability."""
        # Create list of (token_id, probability)
        token_prob_pairs = [(i, prob) for i, prob in enumerate(token_probs)]
        # Sort by probability (descending) and take top k
        top_k = sorted(token_prob_pairs, key=lambda x: x[1], reverse=True)[:k]
        return top_k

    def get_word_probability(self, context_text: str, next_word: str):
        context_text = context_text.strip()
        next_word = " " + next_word.strip()
        context_tokens = self.tokenizer.encode(context_text)
        word_tokens = self.tokenizer.encode(next_word)
        log_prob = 0
        for i in range(len(word_tokens)):
            tokens_probs = self.model.predict(context_tokens + word_tokens[:i])
            prob = tokens_probs[word_tokens[i]]
            log_prob += np.log(prob)
        return math.exp(log_prob / (len(word_tokens) ** self.alpha))

    @staticmethod
    def _clean_context_text(context_text: str) -> str:
        context_text = context_text.lower()
        context_text = unicodedata.normalize("NFC", context_text)
        context_text = CLEAN_REGEX.sub("", context_text)
        context_text = MULTIPLE_WHITESPACE.sub(" ", context_text)
        return context_text

    @staticmethod
    def _extract_unfinished_word(context_text: str) -> [str, str]:
        # if context text is empty or ends with a space, all words are finished
        if not context_text or context_text[-1] == " ":
            return context_text, ""
        words = context_text.split()
        return " ".join(words[:-1]), words[-1]

    @staticmethod
    def _create_complete_word(current_prefix: BeamItem) -> CompletedWord | None:
        word_text = current_prefix.text.strip()
        if word_text:
            word_neg_log_prob_normalised = current_prefix.neg_log_prob_normalised
            word_tokens = current_prefix.tokens
            word_probability = math.exp(-current_prefix.neg_log_prob_normalised)
            return CompletedWord(neg_log_prob_normalised=word_neg_log_prob_normalised,
                                 tokens=word_tokens,
                                 text=word_text,
                                 probability=word_probability)
        return None

    def _create_new_beam_prefix(self, current_prefix: BeamItem, token_id: int,
                                token_prob: float) -> BeamItem | None:
        new_tokens = current_prefix.tokens + [token_id]
        new_text = current_prefix.text + self.tokenizer.decode([token_id])
        new_log_prob = current_prefix.neg_log_prob - math.log(token_prob)
        new_log_prob_normalised = new_log_prob / (len(new_tokens) ** self.alpha)
        return BeamItem(
            neg_log_prob_normalised=new_log_prob_normalised,
            neg_log_prob=new_log_prob,
            tokens=new_tokens,
            text=new_text
        )


def create_beam_searcher(model_dir: str = None, beam_width: int = 30,
                         max_word_length: int = 5, device: str = None,
                         alpha: float = 0.0, seq_len: int = 256,
                         model_name: str = "model.pt"):
    """
    Create a beam searcher with real model and tokenizer.

    Args:
        model_dir: Directory containing model.pt and spm_pl.model. If None, uses predictions directory.
        beam_width: Maximum number of partial words to keep in beam
        max_word_length: Maximum number of tokens per word
        device: Device to run model on ('cpu' or 'cuda'). If None, auto-detect.

    Returns:
        WordPredictionBeamSearch instance
    """
    from src.model_loader import load_model_and_tokenizer
    # from model_loader import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(model_dir=model_dir,
                                                device=device,
                                                seq_len=seq_len,
                                                model_name=model_name)

    return WordPredictionBeamSearch(
        model=model,
        tokenizer=tokenizer,
        beam_width=beam_width,
        max_word_length=max_word_length,
        alpha=alpha
    )

if __name__ == "__main__":

    print("Loading real LSTM model and tokenizer...")
    searcher = create_beam_searcher(model_dir="../", beam_width=50, max_word_length=10)
    print("Model loaded successfully!\n")

    # Find top 5 most probable next words
    context = "chociaż mam prawie trzydzieści lat cały czas czuję się "
    top_words = searcher.get_top_k_words(context, k=5)

    print(f"\n{'=' * 50}")
    print(f"TOP 5 PREDICTED NEXT WORDS after '{context}':")
    print(f"{'=' * 50}")
    for i, (word, prob, num_tokens) in enumerate(top_words, 1):
        print(f"{i}. '{word}' - probability: {prob:.6f} ({num_tokens} tokens)")

    print(f"\nTotal model inferences: {searcher.inference_count}")