import re
import unicodedata
from typing import Optional, Tuple


def normalize_text(text: str, preserve_indentation: bool = False) -> str:
    # NFC normalization ensures consistent Unicode byte representations
    text = unicodedata.normalize("NFC", text)
    # Replace non-standard whitespace and CRLF line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace null and non-printable control characters (except newline and tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse 3+ consecutive newlines to 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    if not preserve_indentation:
        # Collapse 3+ consecutive spaces to 1 space for general text
        text = re.sub(r"[ \t]{3,}", " ", text)
        return text.strip()
    return text.strip("\r\n").rstrip()


def is_quality_code(
    text: str,
    min_chars: int = 50,
    max_chars: int = 40000,
    max_repetition_ratio: float = 0.35,
    min_ascii_ratio: float = 0.80,
) -> Tuple[bool, Optional[str]]:
    """Quality filter tailored for source code (e.g. Python scripts)."""
    if len(text) < min_chars:
        return False, f"Too short ({len(text)} < {min_chars} chars)"
    if len(text) > max_chars:
        return False, f"Too long ({len(text)} > {max_chars} chars)"

    # Code should have high ASCII ratio for identifiers and keywords
    ascii_count = sum(c.isascii() for c in text)
    if ascii_count / len(text) < min_ascii_ratio:
        return False, f"Low ASCII ratio ({ascii_count / len(text):.2f} < {min_ascii_ratio})"

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return False, "No non-empty lines"

    # Reject files with excessively long single lines (e.g. minified code or data blobs)
    max_line_len = max(len(l) for l in lines)
    if max_line_len > 1200:
        return False, f"Line too long ({max_line_len} chars), likely minified or binary blob"

    # Check line repetition ratio
    total_lines = len(lines)
    if total_lines > 5:
        unique_lines = len(set(lines))
        repeat_ratio = 1.0 - (unique_lines / total_lines)
        if repeat_ratio > max_repetition_ratio:
            return False, f"High line repetition ratio ({repeat_ratio:.2f} > {max_repetition_ratio})"

    # Check for presence of code symbols/keywords
    code_chars = sum(c.isalnum() or c in " _():=+-*/[]{}<>,.#'\"\\%$&|~^;@?!" for c in text)
    if code_chars / len(text) < 0.85:
        return False, "Excessive non-standard code characters"

    return True, None


def is_quality_math(
    text: str,
    min_chars: int = 80,
    max_chars: int = 40000,
    max_repetition_ratio: float = 0.25,
) -> Tuple[bool, Optional[str]]:
    """Quality filter for mathematical discourse and LaTeX formulas."""
    if len(text) < min_chars:
        return False, f"Too short ({len(text)} < {min_chars} chars)"
    if len(text) > max_chars:
        return False, f"Too long ({len(text)} > {max_chars} chars)"

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return False, "No non-empty lines"

    total_lines = len(lines)
    if total_lines > 5:
        unique_lines = len(set(lines))
        repeat_ratio = 1.0 - (unique_lines / total_lines)
        if repeat_ratio > max_repetition_ratio:
            return False, f"High line repetition ratio ({repeat_ratio:.2f} > {max_repetition_ratio})"

    # Mathematical documents have symbols, numbers, and formulas; allow lower alpha ratio but require alphanumeric content
    alnum_count = sum(c.isalnum() for c in text)
    if alnum_count / len(text) < 0.35:
        return False, "Low alphanumeric content ratio"

    return True, None


def is_quality_document(
    text: str,
    domain: str = "text",
    min_chars: int = 100,
    max_chars: int = 40000,
    max_repetition_ratio: float = 0.25,
) -> Tuple[bool, Optional[str]]:
    # Domain-aware dispatch
    if domain == "code":
        return is_quality_code(text, min_chars=min_chars if min_chars != 100 else 50, max_chars=max_chars)
    elif domain == "math":
        return is_quality_math(text, min_chars=min_chars if min_chars != 100 else 80, max_chars=max_chars)

    # Standard general text quality filter
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
