"""Eval prompt sets: loaded from a file, or generated from the skill descriptions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .frontmatter import load_yaml
from .similarity import tokens
from .skills import Skill


@dataclass
class PromptSet:
    should_trigger: list[str] = field(default_factory=list)
    should_not_trigger: list[str] = field(default_factory=list)
    generated: bool = False


class PromptFileError(ValueError):
    pass


def _strings(v, key) -> list[str]:
    if v is None:
        return []
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise PromptFileError(f"'{key}' must be a list of strings")
    return [x.strip() for x in v if x.strip()]


def load_prompt_file(path: Path) -> PromptSet:
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix == ".json" else load_yaml(text)
    if not isinstance(data, dict):
        raise PromptFileError("prompt file must be a mapping with should_trigger / should_not_trigger")
    ps = PromptSet(
        _strings(data.get("should_trigger"), "should_trigger"),
        _strings(data.get("should_not_trigger"), "should_not_trigger"),
    )
    if not ps.should_trigger and not ps.should_not_trigger:
        raise PromptFileError("prompt file has no prompts")
    return ps


QUOTED = re.compile(r"[\"“']([^\"”']{4,80})[\"”']")

GENERIC_NEGATIVES = [
    "What's the capital of Australia?",
    "Rename the variable `tmp` to `total` in this snippet: tmp = a + b; print(tmp)",
    "Explain the difference between a list and a tuple in Python in two sentences.",
    "Write a haiku about autumn rain.",
    "How many seconds are there in a week?",
    'Convert this JSON object to YAML: {"a": 1, "b": [2, 3]}',
    "Write a SQL query that finds duplicate emails in a users table.",
    "Plot a bar chart of monthly revenue from this spreadsheet.",
    "Summarize this meeting transcript in three bullet points.",
    "Write a regex that validates email addresses.",
    "Draft a short reminder email about tomorrow morning's standup.",
]


def _phrases(desc: str) -> list[str]:
    """Pull 'task-like' phrases out of a description: quoted examples first, then clauses."""
    out = [m.strip() for m in QUOTED.findall(desc)]
    m = re.search(r"\b(?:use|load|invoke)[^.]*?\b(?:when|for|to)\b(.*?)(?:\.|$)", desc, re.I | re.S)
    if m:
        for part in re.split(r",|;|\bor\b", m.group(1)):
            part = part.strip(" -—:()")
            if 3 <= len(part.split()) <= 14:
                out.append(part)
    first = re.split(r"(?<=[.!?])\s", desc.strip(), maxsplit=1)[0].rstrip(".")
    if first:
        out.append(first)
    seen, uniq = set(), []
    for p in out:
        k = p.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


FILE_HINT_RE = re.compile(
    r"\b(files?|csv|tsv|pdfs?|docx?|xlsx?|json|yaml|spreadsheets?|documents?|logs?|datasets?)\b", re.I
)


def _clean(phrase: str) -> str:
    """Strip reported-speech prefixes so templates read like a typed request."""
    return re.sub(
        r"^(the user asks |user asks |asks |wants |want |when |for |to |if )+",
        "",
        phrase.strip(),
        flags=re.I,
    ).strip()


def generate(skill: Skill, others: list[Skill], n: int = 5) -> PromptSet:
    """Template prompts. Crude on purpose: write a prompt file for anything you care about."""
    name = skill.name.replace("-", " ")
    templates = [
        "{p}",
        "Can you help me with this: {p}",
        "I need to {p_lower}. Where do I start?",
        "Please {p_lower} for this project.",
        "Quick one: {p_lower}?",
        "{name} pls",
    ]
    # File-shaped requests only make sense for skills that work on files;
    # "Here's data.csv, bake sourdough bread?" would be a nonsense positive.
    if FILE_HINT_RE.search(skill.description):
        templates += [
            "Here's data.csv: {p_lower}?",
            "My goal: {p_lower}. The file is data.csv, can you handle it?",
            "Pasted from data.csv below. {p}. What does it mean?",
            "Can you look at report.csv and {p_lower}?",
        ]
    phrases = [c for p in _phrases(skill.description) if (c := _clean(p))]
    if not phrases:
        kw = " ".join(tokens(skill.description)[:6]) or name
        phrases.append(f"Help me with {kw}.")

    pos, seen_pos = [], set()
    j = 0
    while len(pos) < n and j < n * len(templates):
        phrase = phrases[j % len(phrases)]
        t = templates[j % len(templates)]
        lower = phrase[0].lower() + phrase[1:]
        prompt = t.format(p=phrase, p_lower=lower, name=name)
        if prompt.lower() not in seen_pos:
            seen_pos.add(prompt.lower())
            pos.append(prompt)
        j += 1

    neg = []
    own = set(tokens(skill.description))
    for o in others:
        if o.name == skill.name or not o.description:
            continue
        if len(own & set(tokens(o.description))) <= 2:
            # the other skill's first sentence reads like a real request for a different job
            first = re.split(r"(?<=[.!?])\s", o.description.strip(), maxsplit=1)[0].rstrip(".")
            if first:
                neg.append(first)
        if len(neg) >= n // 2 + 1:
            break
    for g in GENERIC_NEGATIVES:
        if len(neg) >= n:
            break
        neg.append(g)
    return PromptSet(pos[:n], neg[:n], generated=True)
