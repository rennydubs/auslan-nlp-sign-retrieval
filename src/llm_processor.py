"""
LLM fallback processor for Auslan sign retrieval system.

Wraps the Ollama Python package to provide LLM-assisted matching strategies
as the final stage (Stage 5) of the matching pipeline:
  exact -> fuzzy -> synonym -> semantic -> LLM fallback

All methods degrade gracefully when Ollama is not installed or not running.
"""

import json
import logging
import urllib.request
import urllib.error
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    import ollama as _ollama_lib
    _OLLAMA_IMPORTABLE = True
except ImportError:
    _ollama_lib = None  # type: ignore[assignment]
    _OLLAMA_IMPORTABLE = False
    logger.info("ollama package not installed. LLMProcessor will be unavailable.")


class LLMProcessor:
    """LLM-assisted fallback processor using a locally-running Ollama instance.

    Provides four capabilities for the sign-retrieval pipeline:
      - paraphrase      — simplify user input to common vocabulary
      - expand_query    — generate candidate dictionary words for an unmatched token
      - generate_gloss_sequence — reorder matched glosses into Auslan grammar order
      - disambiguate    — choose the best candidate given sentence context

    All public methods return safe fallbacks when the LLM is unavailable.
    """

    def __init__(
        self,
        model: str = "qwen3:8b",
        host: str = "http://localhost:11434",
        timeout: int = 10,
    ) -> None:
        """Initialise the processor and probe Ollama availability.

        Args:
            model:   Ollama model tag to use for inference.
            host:    Base URL of the Ollama HTTP server.
            timeout: Request timeout in seconds.
        """
        self.model = model
        self.host = host
        self.timeout = timeout
        self.available = False

        if not _OLLAMA_IMPORTABLE:
            logger.warning(
                "ollama package is not installed. "
                "Install it with: pip install ollama"
            )
            return

        # Probe the Ollama server by hitting the /api/tags endpoint.
        probe_url = host.rstrip("/") + "/api/tags"
        try:
            req = urllib.request.Request(probe_url)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    self.available = True
                    logger.info(
                        "Ollama server reachable at %s. Model: %s", host, model
                    )
        except (urllib.error.URLError, OSError) as exc:
            logger.warning(
                "Ollama server not reachable at %s: %s. "
                "LLM fallback disabled.",
                host,
                exc,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chat(self, prompt: str) -> Optional[str]:
        """Send a single-turn chat message to Ollama and return the reply text.

        Returns None on any failure (server down, timeout, etc.).
        Strips Qwen3-style ``<think>...</think>`` blocks from the response.
        """
        if not self.available or _ollama_lib is None:
            return None

        try:
            client = _ollama_lib.Client(host=self.host)
            response = client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": 512},
            )
            # ollama-python >= 0.2 returns a dict-like object
            content = response["message"]["content"]
            if not isinstance(content, str):
                return None
            # Strip Qwen3's <think>...</think> reasoning block if present
            content = self._strip_think_block(content)
            return content.strip() if content else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama chat request failed: %s", exc)
            return None

    @staticmethod
    def _strip_think_block(text: str) -> str:
        """Remove Qwen3-style ``<think>...</think>`` blocks from text.

        Qwen3 models emit a reasoning/thinking block before the actual
        answer. We strip it so downstream JSON parsing sees clean output.
        """
        import re
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Extract the first JSON object found in *text*.

        Tries a direct parse first, then scans for a ``{`` ... ``}`` block.
        Returns None if no valid JSON object is found.
        """
        if not text:
            return None

        # Direct parse (model returned clean JSON)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Scan for a JSON object within surrounding prose
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        logger.debug("Could not parse JSON from LLM output: %.120s", text)
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def paraphrase(self, text: str, dictionary_words: List[str]) -> str:
        """Rewrite *text* using simpler vocabulary closer to Auslan dictionary words.

        The LLM is asked to return JSON ``{"simplified": "<rewritten text>"}``.
        Falls back to the original *text* on any failure.

        Args:
            text:             Raw user input string.
            dictionary_words: List of valid gloss/dictionary words for context.

        Returns:
            Simplified text string, or original *text* if the LLM fails.

        Example:
            >>> proc.paraphrase("I require assistance", ["help", "need", "want"])
            "I need help"
        """
        if not self.available:
            return text

        word_list = ", ".join(dictionary_words[:60])  # keep prompt short
        prompt = (
            "You are a vocabulary simplifier for an Auslan sign-language system.\n"
            f"Dictionary words available: {word_list}\n\n"
            "Rewrite the input using simpler, more common words that are likely "
            "to match the dictionary words above. Keep the same meaning.\n"
            f'Input: "{text}"\n\n'
            'Reply with ONLY valid JSON: {"simplified": "<rewritten text>"}'
        )

        raw = self._chat(prompt)
        parsed = self._extract_json(raw or "")
        if parsed and isinstance(parsed.get("simplified"), str):
            simplified = parsed["simplified"].strip()
            if simplified:
                logger.debug("paraphrase: '%s' -> '%s'", text, simplified)
                return simplified

        logger.debug("paraphrase: LLM failed, returning original text")
        return text

    def expand_query(
        self, token: str, dictionary_words: List[str], n: int = 5
    ) -> List[str]:
        """Return up to *n* dictionary words semantically related to *token*.

        The LLM is asked to return JSON ``{"candidates": [...]}``.
        Falls back to an empty list on any failure.

        Args:
            token:            Single unmatched word/token.
            dictionary_words: List of valid gloss/dictionary words.
            n:                Maximum number of candidate words to return.

        Returns:
            List of dictionary words (strings) most related to *token*.

        Example:
            >>> proc.expand_query("squat", ["exercise", "legs", "strong"])
            ["exercise", "legs", "strong"]
        """
        if not self.available:
            return []

        word_list = ", ".join(dictionary_words[:80])
        prompt = (
            "You are a sign-language lookup assistant.\n"
            f"Dictionary words: {word_list}\n\n"
            f"Which {n} dictionary words are most related to '{token}'? "
            "Only choose words that appear in the dictionary list above.\n\n"
            f'Reply with ONLY valid JSON: {{"candidates": ["word1", "word2", ...]}}'
        )

        raw = self._chat(prompt)
        parsed = self._extract_json(raw or "")
        if parsed and isinstance(parsed.get("candidates"), list):
            candidates = [
                w for w in parsed["candidates"]
                if isinstance(w, str) and w in dictionary_words
            ][:n]
            logger.debug("expand_query '%s' -> %s", token, candidates)
            return candidates

        logger.debug("expand_query: LLM failed for token '%s'", token)
        return []

    def generate_gloss_sequence(
        self,
        text: str,
        matched_glosses: List[str],
        all_glosses: List[str],
    ) -> List[str]:
        """Predict the optimal Auslan gloss ordering for *text*.

        Auslan follows topic-comment structure with time markers first.
        The LLM is asked to return JSON ``{"sequence": [...]}``.
        Falls back to *matched_glosses* unchanged on any failure.

        Args:
            text:            Original English input text.
            matched_glosses: Glosses already matched by the pipeline.
            all_glosses:     All available gloss keys in the dictionary.

        Returns:
            Ordered list of gloss strings. Only glosses from *all_glosses*
            are included in the returned sequence.
        """
        if not self.available or not matched_glosses:
            return matched_glosses

        gloss_str = ", ".join(matched_glosses)
        available_str = ", ".join(all_glosses[:80])
        prompt = (
            "You are an Auslan (Australian Sign Language) grammar assistant.\n"
            "Auslan grammar rules: topic first, then comment; time markers go first; "
            "verbs come after objects.\n\n"
            f'English text: "{text}"\n'
            f"Matched glosses (reorder these): {gloss_str}\n"
            f"All available glosses: {available_str}\n\n"
            "Return the matched glosses in the correct Auslan grammar order. "
            "Only use glosses from the matched list — do not add new ones.\n\n"
            'Reply with ONLY valid JSON: {"sequence": ["GLOSS1", "GLOSS2", ...]}'
        )

        raw = self._chat(prompt)
        parsed = self._extract_json(raw or "")
        if parsed and isinstance(parsed.get("sequence"), list):
            sequence = [
                g for g in parsed["sequence"]
                if isinstance(g, str) and g in all_glosses
            ]
            if sequence:
                logger.debug(
                    "generate_gloss_sequence: %s -> %s", matched_glosses, sequence
                )
                return sequence

        logger.debug(
            "generate_gloss_sequence: LLM failed, returning matched_glosses unchanged"
        )
        return matched_glosses

    def disambiguate(
        self, token: str, context: str, candidates: List[str]
    ) -> str:
        """Pick the most contextually appropriate candidate for *token*.

        The LLM is asked to return JSON ``{"best": "<candidate>"}``.
        Falls back to the first candidate on any failure.

        Args:
            token:      The ambiguous word (e.g. "cool").
            context:    The full sentence providing context.
            candidates: List of candidate dictionary words to choose from.

        Returns:
            The best candidate string. Returns ``candidates[0]`` if the LLM
            fails or ``candidates`` is empty.

        Example:
            >>> proc.disambiguate("cool", "cool down after exercise", ["cool", "cold"])
            "cool"
        """
        if not candidates:
            return ""

        if not self.available:
            return candidates[0]

        candidate_str = ", ".join(candidates)
        prompt = (
            "You are a word-sense disambiguation assistant for Auslan sign retrieval.\n"
            f'Sentence: "{context}"\n'
            f'Ambiguous word: "{token}"\n'
            f"Candidates: {candidate_str}\n\n"
            "Which candidate best matches the meaning of the ambiguous word in this "
            "sentence? Choose exactly one candidate from the list.\n\n"
            f'Reply with ONLY valid JSON: {{"best": "<chosen candidate>"}}'
        )

        raw = self._chat(prompt)
        parsed = self._extract_json(raw or "")
        if parsed and isinstance(parsed.get("best"), str):
            best = parsed["best"].strip()
            if best in candidates:
                logger.debug(
                    "disambiguate '%s' in '%s' -> '%s'", token, context, best
                )
                return best

        logger.debug(
            "disambiguate: LLM failed for token '%s', using first candidate", token
        )
        return candidates[0]


# ------------------------------------------------------------------
# Self-test (python src/llm_processor.py)
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logging.basicConfig(level=logging.DEBUG)

    proc = LLMProcessor()
    print(f"LLMProcessor available: {proc.available}")
    print(f"Model: {proc.model}  Host: {proc.host}")

    if proc.available:
        dict_words = ["happy", "sad", "help", "need", "want", "exercise", "run",
                      "walk", "eat", "drink", "home", "work", "good", "bad"]

        print("\n-- paraphrase --")
        result = proc.paraphrase("I require assistance with my workout", dict_words)
        print(f"Result: {result}")

        print("\n-- expand_query --")
        candidates = proc.expand_query("squat", dict_words, n=3)
        print(f"Candidates: {candidates}")

        print("\n-- generate_gloss_sequence --")
        sequence = proc.generate_gloss_sequence(
            "Tomorrow I will eat at home",
            ["eat", "home", "tomorrow"],
            dict_words,
        )
        print(f"Sequence: {sequence}")

        print("\n-- disambiguate --")
        best = proc.disambiguate(
            "cool", "cool down after exercise", ["cool", "cold", "happy"]
        )
        print(f"Best candidate: {best}")
    else:
        print("Skipping live tests (Ollama not available).")
