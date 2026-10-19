# -*- coding: utf-8 -*-
# hailo_whisper/tokenizer.py
"""
tokenizer.py: Lightweight Whisper BPE Tokenizer and Text Decoder.
Provides token decoding, special token filtering, hallucination prevention,
and OpenCC Traditional Chinese normalization for Hailo-10H NPU output.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Set

logger = logging.getLogger("hailo_whisper.tokenizer")

try:
    import opencc
    _OPENCC = opencc.OpenCC('s2twp.json')
except Exception:
    _OPENCC = None


class WhisperTokenizer:
    """
    Lightweight Whisper Tokenizer and Vocabulary Decoder.
    Maps token IDs emitted by Hailo-10H NPU Decoder to UTF-8 text strings.
    """

    # Whisper Standard Special Tokens
    SPECIAL_TOKENS: Dict[str, int] = {
        "<|endoftext|>": 50257,
        "<|startoftranscript|>": 50258,
        "<|en|>": 50259,
        "<|zh|>": 50260,
        "<|de|>": 50261,
        "<|es|>": 50262,
        "<|ru|>": 50263,
        "<|ko|>": 50264,
        "<|fr|>": 50265,
        "<|ja|>": 50266,
        "<|pt|>": 50267,
        "<|tr|>": 50268,
        "<|pl|>": 50269,
        "<|ca|>": 50270,
        "<|nl|>": 50271,
        "<|ar|>": 50272,
        "<|sv|>": 50273,
        "<|it|>": 50274,
        "<|id|>": 50275,
        "<|hi|>": 50276,
        "<|fi|>": 50277,
        "<|vi|>": 50278,
        "<|he|>": 50279,
        "<|uk|>": 50280,
        "<|el|>": 50281,
        "<|ms|>": 50282,
        "<|cs|>": 50283,
        "<|ro|>": 50284,
        "<|da|>": 50285,
        "<|hu|>": 50286,
        "<|ta|>": 50287,
        "<|no|>": 50288,
        "<|th|>": 50289,
        "<|ur|>": 50290,
        "<|hr|>": 50291,
        "<|bg|>": 50292,
        "<|lt|>": 50293,
        "<|la|>": 50294,
        "<|mi|>": 50295,
        "<|ml|>": 50296,
        "<|cy|>": 50297,
        "<|sk|>": 50298,
        "<|te|>": 50299,
        "<|fa|>": 50300,
        "<|lv|>": 50301,
        "<|bn|>": 50302,
        "<|sr|>": 50303,
        "<|az|>": 50304,
        "<|sl|>": 50305,
        "<|kn|>": 50306,
        "<|et|>": 50307,
        "<|mk|>": 50308,
        "<|br|>": 50309,
        "<|eu|>": 50310,
        "<|is|>": 50311,
        "<|hy|>": 50312,
        "<|ne|>": 50313,
        "<|mn|>": 50314,
        "<|bs|>": 50315,
        "<|kk|>": 50316,
        "<|sq|>": 50317,
        "<|sw|>": 50318,
        "<|gl|>": 50319,
        "<|mr|>": 50320,
        "<|pa|>": 50321,
        "<|si|>": 50322,
        "<|km|>": 50323,
        "<|sn|>": 50324,
        "<|yo|>": 50325,
        "<|so|>": 50326,
        "<|af|>": 50327,
        "<|oc|>": 50328,
        "<|ka|>": 50329,
        "<|be|>": 50330,
        "<|tg|>": 50331,
        "<|sd|>": 50332,
        "<|gu|>": 50333,
        "<|am|>": 50334,
        "<|yi|>": 50335,
        "<|lo|>": 50336,
        "<|uz|>": 50337,
        "<|fo|>": 50338,
        "<|ht|>": 50339,
        "<|ps|>": 50340,
        "<|tk|>": 50341,
        "<|nn|>": 50342,
        "<|mt|>": 50343,
        "<|sa|>": 50344,
        "<|lb|>": 50345,
        "<|my|>": 50346,
        "<|bo|>": 50347,
        "<|tl|>": 50348,
        "<|mg|>": 50349,
        "<|as|>": 50350,
        "<|tt|>": 50351,
        "<|haw|>": 50352,
        "<|ln|>": 50353,
        "<|ha|>": 50354,
        "<|ba|>": 50355,
        "<|jw|>": 50356,
        "<|su|>": 50357,
        "<|translate|>": 50358,
        "<|transcribe|>": 50359,
        "<|startoflm|>": 50360,
        "<|startofprev|>": 50361,
        "<|nospeech|>": 50362,
        "<|notimestamps|>": 50363,
    }

    # Common hallucination tokens and phrases to filter out
    FILTER_PHRASES: List[str] = [
        "No Hallucination.",
        "Verbatim speech transcription",
        "Subtitles by",
        "Amara.org",
        "Thank you for watching",
        "謝謝收看",
        "請訂閱",
        "字幕由",
    ]

    def __init__(self, vocab_path: Optional[str] = None):
        self.vocab_path = vocab_path
        self._id_to_token: Dict[int, str] = {}
        self._token_to_id: Dict[str, int] = {}
        self._special_token_ids: Set[int] = set(self.SPECIAL_TOKENS.values())
        self._load_vocab()

    def _load_vocab(self):
        """Initialize or load vocabulary mapping."""
        # Prepopulate with standard special tokens
        for token_str, token_id in self.SPECIAL_TOKENS.items():
            self._id_to_token[token_id] = token_str
            self._token_to_id[token_str] = token_id

        # If external vocabulary JSON exists, load it
        if self.vocab_path and os.path.exists(self.vocab_path):
            try:
                with open(self.vocab_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._token_to_id.update(data)
                    self._id_to_token.update({v: k for k, v in data.items()})
                logger.info(f"[WhisperTokenizer] Loaded {len(self._token_to_id)} tokens from {self.vocab_path}")
            except Exception as e:
                logger.warning(f"[WhisperTokenizer] Failed to load vocab from {self.vocab_path}: {e}")

    def get_initial_tokens(self, language: Optional[str] = None, task: str = "transcribe", timestamps: bool = False) -> List[int]:
        """
        Generate Whisper prompt prefix tokens.
        Default sequence: [<|startoftranscript|>, <|language|>, <|task|>, <|notimestamps|>]
        """
        tokens = [self.SPECIAL_TOKENS["<|startoftranscript|>"]]

        if language:
            lang_token = f"<|{language.lower()}|>"
            if lang_token in self.SPECIAL_TOKENS:
                tokens.append(self.SPECIAL_TOKENS[lang_token])

        task_token = f"<|{task.lower()}|>"
        if task_token in self.SPECIAL_TOKENS:
            tokens.append(self.SPECIAL_TOKENS[task_token])
        else:
            tokens.append(self.SPECIAL_TOKENS["<|transcribe|>"])

        if not timestamps:
            tokens.append(self.SPECIAL_TOKENS["<|notimestamps|>"])

        return tokens

    def decode_tokens(self, token_ids: List[int], convert_traditional: bool = True) -> str:
        """
        Decode a list of Whisper token IDs into human-readable text.
        Strips special tokens, suppresses hallucination prefixes, and aligns Traditional Chinese.
        """
        raw_parts: List[str] = []
        for tid in token_ids:
            if tid in self._special_token_ids or tid >= 50257:
                continue  # Skip special / timestamp tokens
            token_str = self._id_to_token.get(tid, "")
            raw_parts.append(token_str)

        text = "".join(raw_parts).strip()

        # Sanitize known prompt bleed / hallucination triggers
        for phrase in self.FILTER_PHRASES:
            if phrase in text:
                text = text.replace(phrase, "").strip()

        # Convert to Traditional Chinese (Taiwan standard) if required
        if convert_traditional and _OPENCC:
            text = _OPENCC.convert(text)

        return text
