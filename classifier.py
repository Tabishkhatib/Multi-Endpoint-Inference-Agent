"""
classifier.py

Decides which category a query belongs to: "code", "reasoning", or "simple".
This is a deliberately simple keyword heuristic, not a trained model.

Why keyword-based: for 3 broad, lexically distinctive categories (especially
"code", which has very consistent vocabulary), a heuristic is fast to build,
fast to run, and easy to defend/explain. A semantic/embedding-based classifier
was considered and rejected as overkill for this scope — see README for the
full reasoning.

This file only classifies. It does not decide which endpoint to use — that's
router.py's job, using this classification plus live endpoint stats.
"""

CODE_KEYWORDS = [
    "function", "def ", "class ", "debug", "error", "exception",
    "syntax", "code", "python", "javascript", "bug", "compile",
    "variable", "loop", "array", "script", "traceback", "stack trace",
    "{", "}", "()", "import ", "print(",
]

REASONING_SIGNALS = [
    "why", "explain", "how does", "compare", "analyze", "difference between",
    "what causes", "pros and cons", "trade-off", "tradeoff",
]

REASONING_MIN_WORD_COUNT = 25  # long questions tend to need more reasoning


def classify_query(query_text: str) -> str:
    """
    Returns one of: "code", "reasoning", "simple"

    Order matters: code check runs first, since code vocabulary is the most
    distinctive signal. Reasoning is checked next. Anything left over
    defaults to "simple".
    """
    text = query_text.lower()

    if any(keyword in text for keyword in CODE_KEYWORDS):
        return "code"

    word_count = len(query_text.split())
    if any(signal in text for signal in REASONING_SIGNALS) or word_count >= REASONING_MIN_WORD_COUNT:
        return "reasoning"

    return "simple"


if __name__ == "__main__":
    test_queries = [
        "What's the capital of France?",
        "Why does inflation affect interest rates, and how does that ripple through the housing market?",
        "Fix this python function, it's throwing a syntax error",
        "Write a function to reverse a linked list",
        "Explain how neural networks learn",
        "What time is it",
    ]
    for q in test_queries:
        print(f"[{classify_query(q):>9}]  {q}")
