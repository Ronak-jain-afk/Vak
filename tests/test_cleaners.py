import pytest
from data_pipeline.clean import is_quality_code, is_quality_document, is_quality_math, normalize_text


def test_normalize_text_preserve_indentation():
    code = "    def factorial(n):\n        if n <= 1:\n            return 1\n        return n * factorial(n - 1)"
    cleaned_code = normalize_text(code, preserve_indentation=True)
    assert "    def factorial(n):" in cleaned_code
    assert "        if n <= 1:" in cleaned_code

    # When preserve_indentation is False (default), 3+ spaces are collapsed
    text = "Hello   world!"
    cleaned_text = normalize_text(text)
    assert cleaned_text == "Hello world!"


def test_is_quality_code_valid_and_invalid():
    valid_python = """
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
"""
    ok, reason = is_quality_code(valid_python)
    assert ok, f"Expected valid python to pass, got: {reason}"

    # Also test via domain dispatch
    ok_dispatch, _ = is_quality_document(valid_python, domain="code")
    assert ok_dispatch

    # Python code would fail standard text filter due to alpha ratio with symbols
    short_code = "x = 1"
    ok, reason = is_quality_code(short_code, min_chars=50)
    assert not ok
    assert "Too short" in reason

    # Minified line check
    minified = "var a = 1;" + "x" * 1300 + "; return a;"
    ok, reason = is_quality_code(minified)
    assert not ok
    assert "Line too long" in reason

    # Low ASCII check (foreign script dump)
    cyrillic_code = "# Привет мир\n" * 20 + "def foo(): pass"
    ok, reason = is_quality_code(cyrillic_code, min_ascii_ratio=0.85)
    assert not ok
    assert "Low ASCII ratio" in reason


def test_is_quality_math_valid_and_invalid():
    valid_math = """
Let $f(x) = \\int_0^x e^{-t^2} dt$ be the error function scaled by a constant factor.
We consider the second-order differential equation:
$$y'' + 2x y' = 0$$
Using the substitution $u = y'$, we obtain the first-order separable system:
$$\\frac{du}{dx} + 2x u = 0 \\implies \\ln|u| = -x^2 + C$$
Therefore, the general solution is $y(x) = C_1 \\int_0^x e^{-t^2} dt + C_2$.
"""
    ok, reason = is_quality_math(valid_math)
    assert ok, f"Expected valid math to pass, got: {reason}"

    # Test via domain dispatch
    ok_dispatch, _ = is_quality_document(valid_math, domain="math")
    assert ok_dispatch

    # Pure symbol noise / spam
    junk = "$$$ \\\\\\ &&& %%% @@@ !!! ??? ~~~ ^^^" * 5
    ok, reason = is_quality_math(junk)
    assert not ok
    assert "Low alphanumeric" in reason


def test_domain_dispatch_honors_explicit_min_chars():
    short = "def foo():\n    return 42\n"
    ok_default, _ = is_quality_document(short, domain="code")
    assert ok_default is False  # below the 50-char code floor
    lines = "".join(f"def func_{i}(x):\n    return x + {i}\n" for i in range(8))
    assert len(lines) > 100
    ok_explicit, reason = is_quality_document(lines, domain="code", min_chars=100)
    assert ok_explicit, f"Explicit floor met but rejected: {reason}"
    ok_higher, reason_higher = is_quality_document(lines, domain="code", min_chars=10000)
    assert not ok_higher and "Too short" in reason_higher
