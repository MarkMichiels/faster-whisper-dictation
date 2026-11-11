"""Fixed replacement list applied to every recognised chunk.

Whisper keeps mishearing a few names in the same way (a product name as a
similar-sounding word, an acronym as its neighbour on the keyboard). A prompt
word list fixes more of them, but it also changes other words and can repeat
the previous sentence on noise. A replacement list after recognition changes
nothing except the forms it names, so it is the safe first step.

File format (UTF-8, one rule per line)::

    # comment
    wrong form -> right form
    wrng|wrong frm → right form

The left side is a Python regular expression, matched case-insensitively and
only as whole words: it never matches inside a longer word. Use ``|`` for
alternatives and ``\\s?`` or ``-?`` for an optional space or hyphen. The right
side is literal text (no back-references). Rules apply in file order, each on
the result of the rules above it. The separator is ``->`` or ``→`` with
whitespace on both sides.

A rule's id is ``L<line number>``. The journal records, per chunk and per rule
that changed something, only that id and the number of replacements, never the
text. A missing or unreadable file, or a bad line, gives a journal warning and
dictation goes on without that file or line: recognition never stops for it.
"""
from __future__ import annotations

import os
import re
from typing import Callable, Optional

ARROW_RE = re.compile(r'\s+(?:->|→)\s+')
MAX_BYTES = 256 * 1024


class ReplacementList:
    def __init__(self, rules=(), log: Optional[Callable[..., None]] = None):
        # rules: (rule id, compiled pattern, replacement text)
        self.rules = list(rules)
        self.log = log

    def __bool__(self):
        return bool(self.rules)

    def __len__(self):
        return len(self.rules)

    def apply(self, text: str) -> str:
        """Text with every rule applied; one journal line per rule that fired."""
        if not text or not self.rules:
            return text
        for rule_id, pattern, replacement in self.rules:
            changed = 0

            def substitute(match, r=replacement):
                nonlocal changed
                changed += match.group(0) != r
                return r

            text = pattern.sub(substitute, text)
            # Counts real changes only: a rule that also matches the right
            # spelling (case-insensitive) leaves it as is, silently.
            if changed and self.log:
                self.log('replacement', rule=rule_id, count=changed)
        return text


def compile_rule(left: str) -> re.Pattern:
    """Whole-word, case-insensitive pattern for the left side of a rule."""
    return re.compile(r'(?<!\w)(?:%s)(?!\w)' % left, re.IGNORECASE)


def parse(text: str, log: Optional[Callable[..., None]] = None) -> ReplacementList:
    """Rules from the file text; a bad line is skipped with a journal line (line number only)."""
    rules = []
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = ARROW_RE.split(line, maxsplit=1)
        reason = None
        if len(parts) != 2:
            reason = 'no_arrow'
        else:
            left, right = parts[0].strip(), parts[1].strip()
            if not left:
                reason = 'empty_pattern'
            elif not right:
                reason = 'empty_replacement'
            else:
                try:
                    pattern = compile_rule(left)
                except re.error:
                    reason = 'bad_regex'
                else:
                    if pattern.search('') or pattern.search(' '):
                        reason = 'matches_empty'
        if reason:
            if log:
                log('replacements_rule_skipped', line=number, reason=reason)
            continue
        rules.append(('L%d' % number, pattern, right))
    return ReplacementList(rules, log)


def load(path: Optional[str], log: Optional[Callable[..., None]] = None) -> ReplacementList:
    """The list from `path`; an empty list (with one journal warning) when it cannot be used.

    No path means the feature is off: an empty list and no journal line.
    """
    if not path:
        return ReplacementList((), log)
    path = os.path.expanduser(path)
    reason = None
    try:
        with open(path, 'rb') as fh:
            data = fh.read(MAX_BYTES + 1)
    except FileNotFoundError:
        reason = 'missing'
    except IsADirectoryError:
        reason = 'not_a_file'
    except OSError:
        reason = 'cannot_read'
    else:
        if len(data) > MAX_BYTES:
            reason = 'too_large'
        else:
            try:
                text = data.decode('utf-8-sig')
            except UnicodeDecodeError:
                reason = 'not_utf8'
    if reason:
        if log:
            log('replacements_unavailable', reason=reason)
        return ReplacementList((), log)
    replacements = parse(text, log)
    if log:
        log('replacements_loaded', rules=len(replacements))
    return replacements
