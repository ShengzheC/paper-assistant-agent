"""Polish a LaTeX paragraph with an LLM without letting it change math, citations or references.

One structured-output request returns the revision and a list of changes. A deterministic
check then compares every protected span (math, \\cite, \\ref, \\label, ...) before and after.
If anything changed, the model is told exactly what and asked again.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass, field

PROTECTED = re.compile(
    r"\$\$.+?\$\$"  # display math $$...$$
    r"|(?<!\\)\$(?:[^$\\]|\\.)+\$"  # inline math $...$ (ignores escaped \$)
    r"|\\\(.+?\\\)|\\\[.+?\\\]"  # \( ... \) and \[ ... \]
    r"|\\begin\{(?P<env>equation|align|gather|multline|eqnarray)\*?\}.*?\\end\{(?P=env)\*?\}"
    r"|\\(?:cite[a-zA-Z]*|ref|eqref|autoref|cref|Cref|label|url|href)\*?(?:\[[^\]]*\])*\{[^}]*\}",
    re.DOTALL,
)

SYSTEM = """You edit paragraphs from machine-learning papers written in LaTeX. Improve clarity, concision and flow while keeping the technical meaning and the author's voice.

Rules:
- Keep every math expression and every \\cite, \\ref, \\eqref, \\label, \\url and \\href argument exactly as written, character for character.
- Keep the LaTeX valid. Do not add claims, results, or citations.
- Return the revised paragraph and a short list of the changes you made."""

SCHEMA = {
    "type": "object",
    "properties": {
        "revised": {"type": "string"},
        "changes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["revised", "changes"],
    "additionalProperties": False,
}


@dataclass
class Polished:
    original: str
    revised: str
    changes: list[str]
    ok: bool  # True if every protected span survived unchanged
    attempts: int
    problems: list[str] = field(default_factory=list)


def protected_spans(tex: str) -> Counter:
    return Counter(match.group(0) for match in PROTECTED.finditer(tex))


def span_problems(before: str, after: str) -> list[str]:
    expected, got = protected_spans(before), protected_spans(after)
    problems = [f"missing or changed: {span}" for span in (expected - got).elements()]
    problems += [f"added: {span}" for span in (got - expected).elements()]
    return problems


def polish(llm, tex: str, instructions: str = "", max_attempts: int = 2) -> Polished:
    request = f"{instructions.strip()}\n\n" if instructions.strip() else ""
    messages: list[dict] = [{"role": "user", "content": f"{request}Paragraph:\n{tex}"}]
    for attempt in range(1, max_attempts + 1):
        data, response = llm.complete_json(system=SYSTEM, messages=messages, schema=SCHEMA)
        problems = span_problems(tex, data["revised"])
        if not problems:
            return Polished(tex, data["revised"], data["changes"], True, attempt)
        feedback = "Your revision changed protected LaTeX:\n- " + "\n- ".join(problems[:10])
        feedback += "\nRevise the original paragraph again and keep those spans exactly."
        messages += [{"role": "assistant", "content": response.content}, {"role": "user", "content": feedback}]
    return Polished(tex, data["revised"], data["changes"], False, max_attempts, problems)


def word_diff(before: str, after: str) -> str:
    """A git-style word diff: [-removed-] {+added+}."""
    a, b = before.split(), after.split()
    parts = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            parts.append(" ".join(a[i1:i2]))
        if op in ("delete", "replace"):
            parts.append("[-" + " ".join(a[i1:i2]) + "-]")
        if op in ("insert", "replace"):
            parts.append("{+" + " ".join(b[j1:j2]) + "+}")
    return " ".join(parts)
