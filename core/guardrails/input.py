from __future__ import annotations

import re

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(
    r"(?<!\d)(?:(?:\+?\d{1,3}[-.\s]?)?\d{4}[-.\s]?\d{4}|"
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})(?!\d)"
)
ID_RE = re.compile(r"\b[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b")
INJECTION_PATTERNS = [
    re.compile(r"ignore (all |any )?(previous|prior|above) instructions", re.IGNORECASE),
    re.compile(r"disregard (the )?(system|developer) (prompt|instructions)", re.IGNORECASE),
    re.compile(r"you are now (?:a )?(?:the )?(?:system|developer|assistant)?\s*$", re.IGNORECASE),
    re.compile(r"reveal your (system|developer|base) prompt", re.IGNORECASE),
]


def redact_pii(text: str) -> str:
    redacted = EMAIL_RE.sub("[EMAIL_REDACTED]", text)
    redacted = PHONE_RE.sub("[PHONE_REDACTED]", redacted)
    return ID_RE.sub("[ID_REDACTED]", redacted)


def detect_injection(text: str) -> str | None:
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None


def validate_input(text: str, max_chars: int = 4000) -> tuple[bool, str, list[str]]:
    issues: list[str] = []
    sanitized = redact_pii(text)
    if len(text) > max_chars:
        issues.append(f"input exceeds {max_chars} chars")
    injected = detect_injection(text)
    if injected:
        issues.append("prompt injection pattern detected")
        return False, sanitized, issues
    return not issues, sanitized, issues
