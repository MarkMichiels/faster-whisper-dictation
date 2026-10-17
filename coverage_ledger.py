"""Session-local sample accounting. No audio files, model or desktop dependencies."""
import datetime
import json
import re
import time
from typing import Any, Callable, Optional

import numpy as np

RATE = 16000
MAX_SECONDS = 30 * 60


def diagnostic(kind: str, **details: Any) -> None:
    print('DCT-COVERAGE ' + json.dumps(dict(
        timestamp=datetime.datetime.now().astimezone().isoformat(), kind=kind, **details),
        ensure_ascii=False), flush=True)


ANCHOR_WORDS = 4


def _delivered(entry: dict[str, Any]) -> bool:
    """Text whose clipboard write and paste shortcut both succeeded."""
    return bool(entry['text']) and bool((entry['delivery'] or {}).get('ok'))


def _anchor(text: str, tail: bool) -> str:
    words = text.split()
    if len(words) <= ANCHOR_WORDS:
        return ' '.join(words)
    return ('…' + ' '.join(words[-ANCHOR_WORDS:])) if tail else (' '.join(words[:ANCHOR_WORDS]) + '…')


def correction_text(before: str, after: str, text: str) -> str:
    """Plain-language correction that quotes delivered text around the gap."""
    if before and after:
        where = 'tussen "%s" en "%s" is' % (_anchor(before, True), _anchor(after, False))
    elif before:
        where = 'na "%s" is' % _anchor(before, True)
    elif after:
        where = 'vóór "%s" is' % _anchor(after, False)
    else:
        where = 'in deze opname is'
    return '[Correctie: %s waarschijnlijk dit gemist: "%s"]' % (where, text)


# Mark 17/10: for now, always end a session with a short visible check line so
# he can see that the check ran, what it found and what it cost. Set False to
# drop it once the check has proven itself.
STOP_STATUS = True


def stop_status(checked: int, corrected: int, unresolved: int, seconds: float) -> str:
    """Short Dutch end-of-session line, e.g. '[Controle: niets gemist, 0,1 s]'."""
    took = ('%.1f s' % seconds).replace('.', ',')
    if not checked:
        return '[Controle: niets gemist, %s]' % took
    parts = ['%d %s nagekeken' % (checked, 'stuk' if checked == 1 else 'stukken')]
    if corrected:
        parts.append('%d %s' % (corrected, 'correctie' if corrected == 1 else 'correcties'))
    if unresolved:
        parts.append('%d niet terug te vinden' % unresolved)
    return '[Controle: %s, %s]' % (', '.join(parts), took)


class CoverageLedger:
    def __init__(self) -> None:
        self.frames = []
        self.samples = 0
        self.entries = []
        self.started = time.time()
        self.warned = False

    def incident(self, kind: str, **details: Any) -> None:
        # Journal only; no desktop notifications.
        diagnostic(kind, **details)

    def capture(self, data: bytes) -> bytes:
        room = MAX_SECONDS * RATE - self.samples
        data = data[:max(0, room) * 2]
        self.frames.append(data)
        self.samples += len(data) // 2
        if not self.warned and self.samples >= (MAX_SECONDS - 60) * RATE:
            self.warned = True
            self.incident('duration_warning', remaining_seconds=60)
        return data

    def outcome(self, start: int, end: int, text: str = '', raw: str = '',
                error: Optional[str] = None) -> dict[str, Any]:
        # Exclusive ranges must form one partition, regardless of decoder result.
        expected = self.entries[-1]['end'] if self.entries else 0
        if start != expected or end <= start or end > self.samples:
            raise ValueError('non-contiguous or uncaptured sample range')
        kind = 'decoder_error' if error else ('text' if text else ('artifact' if raw else 'empty'))
        entry = dict(start=start, end=end, start_s=start/RATE, end_s=end/RATE,
                     outcome=kind, text=text, raw=raw, error=error,
                     discard_reason='known_silence_artifact' if kind == 'artifact' else None,
                     decoded_at=time.time(),
                     delivery=None if text else dict(path='none', ok=False, reason=kind),
                     recovery=None)
        self.entries.append(entry)
        diagnostic('outcome', **entry)
        if kind == 'decoder_error':
            self.incident(kind, start=start, end=end, error=error)
        elif kind == 'artifact' or ((end-start)/RATE >= 8 and len(text.split()) < (end-start)/RATE/2):
            self.incident('suspect_decode', start=start, end=end, outcome=kind)
        return entry

    def delivered(self, entry: dict[str, Any], result: dict[str, Any]) -> None:
        entry['delivery'] = dict(result, at=time.time())
        diagnostic('delivery', start=entry['start'], end=entry['end'], **entry['delivery'])
        if not result.get('ok') or result.get('path') != 'copyq':
            self.incident('delivery_degraded', start=entry['start'], result=result)

    def _failed_delivery(self, e: dict[str, Any]) -> bool:
        # An xclip fallback that still pasted is degraded (no CopyQ history) but
        # landed; re-pasting it would duplicate the text in Mark's window.
        return bool(e['text']) and not _delivered(e)

    def suspects(self) -> list[dict[str, Any]]:
        result = []
        for e in self.entries:
            duration = (e['end'] - e['start']) / RATE
            if (e['outcome'] == 'decoder_error'
                    or (duration >= 8 and len(e['text'].split()) < duration / 2)
                    or self._failed_delivery(e)):
                result.append(e)
        return result

    def _corrections(self, recovered: list[tuple[int, str]]) -> list[tuple[list[dict[str, Any]], str]]:
        """One correction per gap: adjacent recovered ranges merge, anchors are
        the nearest delivered text before and after the gap."""
        groups = []
        for index, text in recovered:
            if groups and groups[-1][-1][0] == index - 1:
                groups[-1].append((index, text))
            else:
                groups.append([(index, text)])
        result = []
        for group in groups:
            first, last = group[0][0], group[-1][0]
            before = next((e['text'] for e in reversed(self.entries[:first]) if _delivered(e)), '')
            after = next((e['text'] for e in self.entries[last+1:] if _delivered(e)), '')
            text = ' '.join(t for _, t in group)
            result.append(([self.entries[i] for i, _ in group], correction_text(before, after, text)))
        return result

    def finish(self, decode: Callable[[np.ndarray], str],
               deliver: Callable[[str], dict[str, Any]]) -> list[str]:
        """Called after original delivery queue drains.

        Retry each suspect exactly once, in <=10s windows. Corrections are
        pasted through the ordinary chunk path (`deliver`), after the text that
        was already pasted; earlier text is never edited. No audio retained
        after return.
        """
        t0 = time.perf_counter()
        corrections = []
        corrected_entries = []
        targets = []
        try:
            covered = self.entries[-1]['end'] if self.entries else 0
            # A missing queue tail is made explicit, not treated as silence.
            for start in range(covered, self.samples, 10 * RATE):
                self.outcome(start, min(start + 10 * RATE, self.samples), error='queue_gap')
            targets = self.suspects()
            raw = b''.join(self.frames) if targets else b''
            index = {id(e): i for i, e in enumerate(self.entries)}
            recovered_texts = []
            for e in targets:
                self.incident('stop_check', start=e['start'], end=e['end'], reason=e['outcome'])
                recovered = []
                try:
                    if self._failed_delivery(e):
                        recovered.append(e['text'])  # Delivery failure needs no GPU work.
                    else:
                        if e['outcome'] == 'decoder_error':
                            time.sleep(0.2)
                        for start in range(e['start'], e['end'], 10 * RATE):
                            end = min(start + 10 * RATE, e['end'])
                            audio = np.frombuffer(raw[start*2:end*2], dtype=np.int16).astype(np.float32) / 32768
                            recovered.append(decode(audio))
                    text = ' '.join(t.strip() for t in recovered if t.strip())
                    if not re.search(r'\w', text):
                        text = ''
                    # Do not re-deliver unchanged text that was already delivered.
                    if text and text != (e['text'] if _delivered(e) else ''):
                        recovered_texts.append((index[id(e)], text))
                    e['recovery'] = dict(text=text, at=time.time())
                    diagnostic('recovery_outcome', start=e['start'], end=e['end'], **e['recovery'])
                    if not text:
                        self.incident('unresolved', start=e['start'], end=e['end'])
                except Exception as exc:
                    e['recovery'] = dict(error=str(exc), at=time.time())
                    self.incident('recovery_failed', start=e['start'], error=str(exc))
            for entries, correction in self._corrections(recovered_texts):
                try:
                    result = deliver(correction)
                except Exception as exc:
                    result = dict(ok=False, path='exception', error=str(exc))
                corrections.append(correction)
                corrected_entries.extend(entries)
                for e in entries:
                    e['recovery']['delivery'] = dict(result, at=time.time())
                # Full text stays in the journal, also when the paste failed.
                diagnostic('correction_delivery', start=entries[0]['start'], end=entries[-1]['end'],
                           correction=correction, **dict(result, at=time.time()))
            if STOP_STATUS and (corrections or any(_delivered(e) for e in self.entries)):
                status = stop_status(len(targets), len(corrections),
                                     sum(not (e['recovery'] or {}).get('text') for e in targets),
                                     time.perf_counter() - t0)
                try:
                    result = deliver(status)
                except Exception as exc:
                    result = dict(ok=False, path='exception', error=str(exc))
                diagnostic('stop_status_delivery', status=status, **dict(result, at=time.time()))
            diagnostic('stop_summary', samples=self.samples, ranges=len(self.entries),
                       suspects=len(targets), corrections=len(corrections),
                       unresolved_ranges=sum(not (e['recovery'] or {}).get('text') for e in targets),
                       original_delivery_failures=sum(self._failed_delivery(e) for e in self.entries),
                       correction_delivery_failures=sum(not
                           (e['recovery'].get('delivery') or {}).get('ok') for e in corrected_entries),
                       check_seconds=time.perf_counter()-t0,
                       note='clipboard/paste command success is not target application acknowledgement')
        finally:
            self.frames.clear()
            diagnostic('audio_released', samples=self.samples)
        return corrections
