"""
sentence.py
-----------
Turns a sequence of recognized words ("gloss", e.g. ["I", "want", "water"])
into readable English. Sign language grammar doesn't map 1:1 onto English
word order, so this is inherently approximate:

  1. rule_based_sentence(): cheap, offline, no dependencies. Just cleans
     up repeats/fillers and capitalizes/punctuates.
  2. polish_with_claude(): optional -- sends the gloss sequence to the
     Anthropic API to smooth it into a fluent, grammatical English
     sentence. Off by default; the app only calls it if the user supplies
     an API key, since it leaves the machine.
"""

import os


def rule_based_sentence(gloss_sequence):
    words = [g for g in gloss_sequence if g]
    if not words:
        return ""

    # Drop immediate repeats (e.g. a word recognized twice across two
    # adjacent segments because the signer paused mid-sign).
    deduped = [words[0]]
    for w in words[1:]:
        if w != deduped[-1]:
            deduped.append(w)

    text = " ".join(deduped)
    text = text[0].upper() + text[1:]
    if not text.endswith((".", "?", "!")):
        text += "."
    return text


def polish_with_claude(gloss_sequence, api_key=None):
    """
    Sends the raw gloss sequence to Claude and asks it to produce a
    fluent English sentence. Requires the `anthropic` package and an
    API key (passed explicitly or via ANTHROPIC_API_KEY env var).
    Returns None on any failure so callers can fall back to the
    rule-based sentence instead.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None

    words = [g for g in gloss_sequence if g]
    if not words:
        return ""

    try:
        import anthropic
    except ImportError:
        return None

    try:
        client = anthropic.Anthropic(api_key=key)
        gloss_text = " ".join(words)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    "These words were recognized, in order, from a sign "
                    "language video: \"" + gloss_text + "\". "
                    "Rewrite them as a single fluent, grammatical English "
                    "sentence that best represents the likely intended "
                    "meaning. Reply with only the sentence, nothing else."
                ),
            }],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
    except Exception:
        return None
