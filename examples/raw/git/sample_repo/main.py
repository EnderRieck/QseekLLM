def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace for preprocessing examples."""
    return " ".join(text.split())


def add(a: int, b: int) -> int:
    return a + b
