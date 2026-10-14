import re
import unicodedata
from typing import Optional, Tuple


def normalize_text(text: str) -> str:
    # NFC normalization ensures consistent Unicode byte representations
    text = unicodedata.normalize("NFC", text)
    # Replace non-standard whitespace and CRLF line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace null and non-printable control characters (except newline and tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse 3+ consecutive newlines to 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse 3+ consecutive spaces to 1 space
    text = re.sub(r"[ \t]{3,}", " ", text)
    return text.strip()


def is_quality_document(
    text: str,
    min_chars: int = 100,
    max_chars: int = 40000,
    max_repetition_ratio: float = 0.25,
) -> Tuple[bool, Optional[str]]:
    # Filters out malformed, empty, or low-information content
    if len(text) < min_chars:
        return False, f"Too short ({len(text)} < {min_chars} chars)"
    if len(text) > max_chars:
        return False, f"Too long ({len(text)} > {max_chars} chars)"

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return False, "No non-empty lines"

    # Detect degenerate line repetition (e.g. repeated navigation menus or boilerplate)
    total_lines = len(lines)
    if total_lines > 5:
        unique_lines = len(set(lines))
        repeat_ratio = 1.0 - (unique_lines / total_lines)
        if repeat_ratio > max_repetition_ratio:
            return False, f"High line repetition ratio ({repeat_ratio:.2f} > {max_repetition_ratio})"

    # Detect excessive non-alphabetic character dominance (e.g. junk binary dumps or symbol spam)
    alpha_count = sum(c.isalpha() for c in text)
    if alpha_count / len(text) < 0.45:
        return False, "Low alphabetic content ratio"

    return True, None
