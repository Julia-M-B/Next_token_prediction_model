"""
Top-K next-word prediction via beam search for:
  Model  : speakleash/Bielik-4.5B-v3.0-Instruct  (APT4/SentencePiece tokenizer)
  Device : CPU (or CUDA if available)

Adapted from a custom LSTM WordPredictionBeamSearch implementation.

Key differences from the LSTM version
---------------------------------------
1. model.predict() → HuggingFace forward pass
   The LSTM exposed a stateful per-step predict() that returned token
   probabilities for the next position. Bielik is a stateless transformer:
   every call to model.forward() consumes the full sequence and returns
   logits for every position. We use past_key_values (KV cache) to avoid
   recomputing the context prefix on every beam step.

2. KV cache for the shared context prefix
   The context tokens are run through the model once at startup to produce
   a past_key_values cache. All subsequent beam-step forward passes append
   only the new token(s) to this cache, making each step much cheaper —
   O(new_tokens) instead of O(context + new_tokens).

3. Word boundary detection
   The original code used tokenizer.id_to_piece() from a bare SentencePiece
   model. HuggingFace exposes the same information via
   tokenizer.convert_ids_to_tokens(). APT4 uses the same '▁' prefix
   convention, so the starts_new_word() logic is identical.

4. Probability extraction
   model.predict() returned a plain list of probabilities.
   Here we call F.softmax(logits[0, -1, :]) to get the equivalent
   distribution over the full 32k vocabulary.

Usage
-----
    from bielik_beam_search import create_beam_searcher

    searcher = create_beam_searcher(beam_width=20, max_word_length=5)
    results = searcher.get_top_k_words("Chciałbym powiedzieć, że", k=10)
    for word, prob, n_tokens in results:
        print(f"{word:20s}  prob={prob:.6f}  tokens={n_tokens}")

Standalone benchmark
---------------------
    python bielik_beam_search.py \
        --corpus_dir ./my_corpus \
        --num_samples 50 \
        --context_words 20 \
        --top_k 10 \
        --beam_width 20 \
        --output_csv results.csv
"""

import argparse
import csv
import heapq
import logging
import math
import re
import statistics
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text cleaning  (unchanged from original)
# ---------------------------------------------------------------------------

CLEAN_REGEX = re.compile(r"[^a-ząćęłńóśźż0-9\s]")
MULTIPLE_WHITESPACE = re.compile(r"[ \t\n]+")


# ---------------------------------------------------------------------------
# Dataclasses  (unchanged from original)
# ---------------------------------------------------------------------------

@dataclass(order=True)
class BeamItem:
    """Partial word being explored in the beam.

    neg_log_prob_normalised is the comparison key (min-heap → highest prob
    items have the smallest value and surface first).
    """
    neg_log_prob_normalised: float
    neg_log_prob: float = field(compare=False)
    tokens: List[int] = field(compare=False)
    text: str = field(compare=False)


@dataclass(order=True)
class CompletedWord:
    """A completed word with its joint (length-normalised) probability."""
    neg_log_prob_normalised: float
    tokens: List[int] = field(compare=False)
    text: str = field(compare=False)
    probability: float = field(compare=False)


# ---------------------------------------------------------------------------
# Bielik wrapper  (replaces the custom LSTM model interface)
# ---------------------------------------------------------------------------

class BielikPredictor:
    """
    Wraps a HuggingFace causal LM so it exposes the same
    predict(token_ids) -> List[float]  interface as the original LSTM.

    The key optimisation is a two-level KV cache:
      - context_cache : computed once for the shared context prefix
      - step_cache    : extended one token at a time during beam expansion

    Because all beams share the same context prefix, we only pay the
    O(context_length) prefill cost once per get_top_k_words() call.
    Each subsequent per-token forward pass only processes the new token(s)
    appended since the last call, keeping beam steps fast.
    """

    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._context_cache: Optional[Tuple] = None   # past_key_values for context
        self._context_len: int = 0
        self._step_caches: Dict[Tuple[int, ...], Tuple] = {}  # token seq → kv cache

    def prime_context(self, context_token_ids: List[int]) -> None:
        """
        Run the shared context prefix through the model once and store the
        resulting KV cache.  Called once per get_top_k_words() invocation.
        """
        self._step_caches.clear()
        if not context_token_ids:
            self._context_cache = None
            self._context_len = 0
            return

        input_ids = torch.tensor([context_token_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, use_cache=True)
        self._context_cache = out.past_key_values
        self._context_len = len(context_token_ids)

    def predict(self, all_token_ids: List[int]) -> List[float]:
        """
        Given the full token sequence (context + beam tokens so far), return
        a probability distribution over the vocabulary for the next token.

        Uses cached KV values so only the tokens *after* the context prefix
        are computed in each call.  Per-step caches are also stored so that
        sibling beams that share a common prefix don't recompute it.

        Parameters
        ----------
        all_token_ids : context tokens + beam tokens accumulated so far

        Returns
        -------
        list of floats, length = vocab_size, summing to 1.0
        """
        beam_tokens = all_token_ids[self._context_len:]

        if len(beam_tokens) == 0:
            logits = self.model(
                input_ids=torch.tensor(
                    [all_token_ids],
                    dtype=torch.long,
                    device=self.device
                )
            ).logits

            probs = F.softmax(logits[0, -1], dim=-1)
            return probs.tolist()

        beam_key = tuple(beam_tokens)

        # Find the longest already-cached prefix of beam_tokens
        cached_past = self._context_cache
        start_idx = 0
        for length in range(len(beam_tokens), 0, -1):
            prefix_key = beam_key[:length]
            if prefix_key in self._step_caches:
                cached_past = self._step_caches[prefix_key]
                start_idx = length
                logger.debug("KV cache hit at beam depth %d", length)
                break

        # Forward pass: only the uncached suffix
        new_tokens = beam_tokens[start_idx:]
        if not new_tokens and cached_past is not None:
            # Nothing new to process; re-run last token to get logits
            # (this happens when start_idx == len(beam_tokens) due to a full hit)
            new_tokens = beam_tokens[-1:]
            cached_past = self._step_caches.get(
                beam_key[:-1], self._context_cache
            )

        input_ids = torch.tensor([new_tokens], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                past_key_values=cached_past,
                use_cache=True,
            )

        # Cache the result for this beam prefix
        self._step_caches[beam_key] = out.past_key_values

        # Extract next-token probabilities
        logits = out.logits[0, -1, :]          # (vocab_size,)
        probs = F.softmax(logits, dim=-1)
        return probs.tolist()


# ---------------------------------------------------------------------------
# Beam search  (adapted from original WordPredictionBeamSearch)
# ---------------------------------------------------------------------------

class WordPredictionBeamSearch:
    """
    Beam search for top-K most probable next complete words.

    Adapted from the original LSTM-based implementation.  The only
    substantive changes are:
      - self.model is now a BielikPredictor instead of a custom LSTM
      - starts_new_word() uses HuggingFace convert_ids_to_tokens()
      - contains_letters_only() is unchanged in logic
      - All beam/heap/normalisation logic is identical to the original
    """

    START_NEW_WORD_CHAR = "▁"   # SentencePiece word-boundary marker (APT4 uses this too)

    def __init__(
        self,
        predictor: BielikPredictor,
        beam_width: int = 20,
        max_word_length: int = 5,
        alpha: float = 0.2,
    ):
        """
        Parameters
        ----------
        predictor      : BielikPredictor wrapping the HF model
        beam_width     : max partial sequences kept alive at each step
        max_word_length: max tokens per word (longer candidates are pruned)
        alpha          : length-normalisation exponent (0 = no norm, 1 = full)
        """
        self.predictor = predictor
        self.tokenizer = predictor.tokenizer
        self.beam_width = beam_width
        self.max_word_length = max_word_length
        self.alpha = alpha
        self.inference_count = 0

    # ------------------------------------------------------------------
    # Token introspection  (replaces tokenizer.id_to_piece())
    # ------------------------------------------------------------------

    def _token_piece(self, token_id: int) -> str:
        """Return the raw SentencePiece piece string for a token ID."""
        return self.tokenizer.convert_ids_to_tokens(token_id)

    def starts_new_word(self, token_id: int) -> bool:
        """True if the token begins a new word (piece starts with '▁')."""
        return self._token_piece(token_id).startswith(self.START_NEW_WORD_CHAR)

    def contains_letters_only(self, token_id: int) -> bool:
        """True if the decoded token contains only alphabetic characters."""
        text = self.tokenizer.decode([token_id], skip_special_tokens=True)
        return text.isalpha()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_top_k_words(
        self, context_text: str, k: int = 10
    ) -> List[Tuple[str, float, int]]:
        """
        Find the top-K most probable next complete words after context_text.

        Parameters
        ----------
        context_text : input text (will be cleaned and lowercased)
        k            : number of top words to return

        Returns
        -------
        list of (word_text, probability, num_tokens) tuples, best first
        """
        self.inference_count = 0

        # Clean and split off any unfinished word at the end
        context_text = self._clean_context_text(context_text)
        context_text, unfinished_word = self._extract_unfinished_word(context_text)
        context_tokens = self.tokenizer.encode(context_text, add_special_tokens=False)

        # Prime the shared KV cache for the context prefix (one forward pass)
        self.predictor.prime_context(context_tokens)

        beam: List[BeamItem] = [
            BeamItem(neg_log_prob_normalised=0.0, neg_log_prob=0.0, tokens=[], text="")
        ]
        completed_words: List[CompletedWord] = []
        completed_words_texts: List[str] = []
        explored_prefixes: set = set()

        max_iterations = k * self.beam_width * 10
        iteration = 0

        # ------------------------------------------------------------------
        # Special first step when the context ends mid-word
        # ------------------------------------------------------------------
        if unfinished_word:
            current = heapq.heappop(beam)
            explored_prefixes.add(tuple(current.tokens))

            token_probs = self.predictor.predict(context_tokens + current.tokens)
            self.inference_count += 1

            top_tokens = self._get_top_matching_tokens(
                token_probs, self.beam_width, current.text, unfinished_word, beam_init=True
            )
            for token_id, token_prob in top_tokens:
                new_item = self._create_new_beam_prefix(current, token_id, token_prob)
                if tuple(new_item.tokens) not in explored_prefixes:
                    heapq.heappush(beam, new_item)

            beam = heapq.nsmallest(self.beam_width, beam)

        # ------------------------------------------------------------------
        # Main beam search loop
        # ------------------------------------------------------------------
        while beam and len(completed_words) < (k * 2) and iteration < max_iterations:
            iteration += 1

            current = heapq.heappop(beam)

            if len(current.tokens) > self.max_word_length:
                continue
            if tuple(current.tokens) in explored_prefixes:
                continue

            explored_prefixes.add(tuple(current.tokens))

            token_probs = self.predictor.predict(context_tokens + current.tokens)
            self.inference_count += 1

            if unfinished_word:
                top_tokens = self._get_top_matching_tokens(
                    token_probs, self.beam_width, current.text, unfinished_word, beam_init=False
                )
            else:
                top_tokens = self._get_top_tokens(token_probs, self.beam_width)

            for token_id, token_prob in top_tokens:
                if not self.contains_letters_only(token_id):
                    continue

                if self.starts_new_word(token_id):
                    if current.text.strip():
                        # Current prefix is a complete word — finalise it
                        completed = self._create_complete_word(current)
                        if completed and completed.text not in completed_words_texts:
                            heapq.heappush(completed_words, completed)
                            completed_words_texts.append(completed.text)
                    else:
                        # No prefix yet — start a new one
                        new_item = self._create_new_beam_prefix(current, token_id, token_prob)
                        if tuple(new_item.tokens) not in explored_prefixes:
                            heapq.heappush(beam, new_item)
                else:
                    # Word continues — extend the prefix
                    new_item = self._create_new_beam_prefix(current, token_id, token_prob)
                    if tuple(new_item.tokens) not in explored_prefixes:
                        heapq.heappush(beam, new_item)

            beam = heapq.nsmallest(self.beam_width, beam)

        print(
            f"Search done: {iteration} iterations, {self.inference_count} inferences, {len(completed_words)} words found"
        )

        top_words = heapq.nsmallest(k, completed_words)
        return [(w.text, w.probability, len(w.tokens)) for w in top_words]

    # ------------------------------------------------------------------
    # Admissibility check (improvement over original stopping condition)
    # ------------------------------------------------------------------

    def _can_improve_top_k(
        self,
        completed_words: List[CompletedWord],
        beam: List[BeamItem],
        k: int,
    ) -> bool:
        """
        Return False when no beam item can possibly beat the current k-th
        best completed word — i.e. the search has converged.
        """
        if len(completed_words) < k or not beam:
            return True
        kth_score = heapq.nsmallest(k, completed_words)[-1].neg_log_prob_normalised
        best_beam_score = beam[0].neg_log_prob_normalised   # min-heap head
        return best_beam_score < kth_score

    # ------------------------------------------------------------------
    # Token filtering helpers  (unchanged from original)
    # ------------------------------------------------------------------

    def _get_top_matching_tokens(
        self,
        token_probs: List[float],
        k: int,
        current_prefix: str,
        unfinished_word: str,
        beam_init: bool = False,
    ) -> List[Tuple[int, float]]:
        unfinished_word = unfinished_word.strip()
        if beam_init and not unfinished_word.startswith(self.START_NEW_WORD_CHAR):
            unfinished_word = self.START_NEW_WORD_CHAR + unfinished_word

        candidates = []
        for token_id, prob in enumerate(token_probs):
            piece = self._token_piece(token_id)
            new_prefix = current_prefix + piece
            if (
                new_prefix.startswith(unfinished_word)
                or unfinished_word.startswith(new_prefix)
            ):
                candidates.append((token_id, prob))

        return sorted(candidates, key=lambda x: x[1], reverse=True)[:k]

    @staticmethod
    def _get_top_tokens(
        token_probs: List[float], k: int
    ) -> List[Tuple[int, float]]:
        pairs = [(i, p) for i, p in enumerate(token_probs)]
        return sorted(pairs, key=lambda x: x[1], reverse=True)[:k]

    # ------------------------------------------------------------------
    # Beam item construction  (unchanged from original)
    # ------------------------------------------------------------------

    def _create_new_beam_prefix(
        self, current: BeamItem, token_id: int, token_prob: float
    ) -> BeamItem:
        new_tokens = current.tokens + [token_id]
        new_text = current.text + self.tokenizer.decode(
            [token_id], skip_special_tokens=True
        )
        new_log_prob = current.neg_log_prob - math.log(max(token_prob, 1e-12))
        new_log_prob_norm = new_log_prob / (len(new_tokens) ** self.alpha)
        return BeamItem(
            neg_log_prob_normalised=new_log_prob_norm,
            neg_log_prob=new_log_prob,
            tokens=new_tokens,
            text=new_text,
        )

    @staticmethod
    def _create_complete_word(current: BeamItem) -> Optional[CompletedWord]:
        word_text = current.text.strip()
        if not word_text:
            return None
        return CompletedWord(
            neg_log_prob_normalised=current.neg_log_prob_normalised,
            tokens=current.tokens,
            text=word_text,
            probability=math.exp(-current.neg_log_prob_normalised),
        )

    # ------------------------------------------------------------------
    # Text utilities  (unchanged from original)
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_context_text(text: str) -> str:
        text = text.lower()
        text = unicodedata.normalize("NFC", text)
        text = CLEAN_REGEX.sub("", text)
        text = MULTIPLE_WHITESPACE.sub(" ", text)
        return text

    @staticmethod
    def _extract_unfinished_word(text: str) -> Tuple[str, str]:
        if not text or text[-1] == " ":
            return text, ""
        words = text.split()
        return " ".join(words[:-1]), words[-1]

    # ------------------------------------------------------------------
    # Word probability scorer  (unchanged from original, same caveats)
    # Note: returns length-normalised pseudo-probability, not true P(w|ctx)
    # ------------------------------------------------------------------

    def get_word_probability(self, context_text: str, next_word: str) -> float:
        context_text = context_text.strip()
        next_word = " " + next_word.strip()
        context_tokens = self.tokenizer.encode(context_text, add_special_tokens=False)
        word_tokens = self.tokenizer.encode(next_word, add_special_tokens=False)

        self.predictor.prime_context(context_tokens)

        log_prob = 0.0
        for i in range(len(word_tokens)):
            token_probs = self.predictor.predict(context_tokens + word_tokens[:i])
            log_prob += math.log(max(token_probs[word_tokens[i]], 1e-12))

        return math.exp(log_prob / (len(word_tokens) ** self.alpha))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_beam_searcher(
    model_id: str = "speakleash/Bielik-4.5B-v3.0-Instruct",
    beam_width: int = 20,
    max_word_length: int = 5,
    alpha: float = 0.2,
    device: Optional[str] = None,
) -> WordPredictionBeamSearch:
    """
    Load Bielik and return a ready-to-use WordPredictionBeamSearch.

    Parameters
    ----------
    model_id       : HuggingFace model identifier
    beam_width     : max partial sequences in beam (lower = faster, less recall)
    max_word_length: max tokens per word before pruning
    alpha          : length-normalisation exponent
    device         : 'cpu' | 'cuda' | None (auto-detect)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading tokenizer ...  ({model_id})")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading model on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)
    print("Model ready.\n")

    predictor = BielikPredictor(model, tokenizer, device)
    return WordPredictionBeamSearch(
        predictor=predictor,
        beam_width=beam_width,
        max_word_length=max_word_length,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Optional benchmark runner
# ---------------------------------------------------------------------------

def _load_corpus_samples(
    corpus_dir: str, context_words: int, num_samples: int
) -> List[Tuple[str, str]]:
    samples = []
    for path in sorted(Path(corpus_dir).glob("*.txt")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                words = line.split()
                if len(words) <= context_words:
                    continue
                context = " ".join(words[:context_words])
                true_next = words[context_words]
                samples.append((context, true_next))
                if len(samples) >= num_samples:
                    return samples
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Bielik top-K next-word beam search on a Polish corpus."
    )
    parser.add_argument("--corpus_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--context_words", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--beam_width", type=int, default=20)
    parser.add_argument("--max_word_length", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--output_csv", default="results.csv")
    parser.add_argument("--model_id", default="speakleash/Bielik-4.5B-v3.0-Instruct")
    parser.add_argument("--device", default=None)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    searcher = create_beam_searcher(
        model_id=args.model_id,
        beam_width=args.beam_width,
        max_word_length=args.max_word_length,
        alpha=args.alpha,
        device=args.device,
    )

    samples = _load_corpus_samples(args.corpus_dir, args.context_words, args.num_samples)
    print(f"{len(samples)} benchmark samples loaded.")

    if args.warmup and samples:
        print(f"Warming up ({args.warmup} sample(s)) ...")
        for ctx, _ in samples[: args.warmup]:
            searcher.get_top_k_words(ctx, k=args.top_k)
        print("Warmup done.\n")

    rows, latencies = [], []
    for i, (context, true_next) in enumerate(samples, 1):
        t0 = time.perf_counter()
        predictions = searcher.get_top_k_words(context, k=args.top_k)
        latency_ms = (time.perf_counter() - t0) * 1000

        latencies.append(latency_ms)
        predicted_words = [w for w, _, _ in predictions]
        top1 = predicted_words[0] if predicted_words else ""
        top1_correct = int(top1.lower() == true_next.lower())
        topk_hit = int(true_next.lower() in [w.lower() for w in predicted_words])

        rows.append({
            "sample_id"     : i,
            "context"       : context,
            "true_next_word": true_next,
            "top1_prediction": top1,
            "top1_correct"  : top1_correct,
            "topk_hit"      : topk_hit,
            "inferences"    : searcher.inference_count,
            "predictions"   : " | ".join(
                f"{r}.{w}({p:.4f})" for r, (w, p, _) in enumerate(predictions, 1)
            ),
            "latency_ms"    : round(latency_ms, 2),
        })

        if i % 5 == 0 or i == len(samples):
            print(
                f"[{i:>4}/{len(samples)}]  "
                f"last={latency_ms:>8.1f} ms  "
                f"avg={statistics.mean(latencies):>8.1f} ms  "
                f"inferences={searcher.inference_count}  "
                f"top{args.top_k}={predicted_words}"
            )

    sorted_lat = sorted(latencies)
    p95 = sorted_lat[max(0, int(len(latencies) * 0.95) - 1)]
    print("\n" + "=" * 60)
    print(f"LATENCY  mean={statistics.mean(latencies):.1f}  "
          f"median={statistics.median(latencies):.1f}  "
          f"p95={p95:.1f}  "
          f"min={min(latencies):.1f}  max={max(latencies):.1f}  ms")
    top1_acc = sum(r["top1_correct"] for r in rows) / len(rows) * 100
    topk_acc = sum(r["topk_hit"] for r in rows) / len(rows) * 100
    print(f"ACCURACY  top-1={top1_acc:.1f}%  top-{args.top_k}={topk_acc:.1f}%")
    print("=" * 60)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "context", "true_next_word",
                        "top1_prediction", "top1_correct", "topk_hit",
                        "inferences", "predictions", "latency_ms"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
