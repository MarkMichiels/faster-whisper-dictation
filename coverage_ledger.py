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


class CoverageLedger:
    def __init__(self, notice: Callable[[str], None] = lambda message: None) -> None:
        self.frames = []
        self.samples = 0
        self.entries = []
        self.notice = notice
        self.started = time.time()
        self.warned = False

    def incident(self, kind: str, **details: Any) -> None:
        diagnostic(kind, **details)
        self.notice('Dictatie: ' + kind)

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

    def suspects(self) -> list[dict[str, Any]]:
        result = []
        for e in self.entries:
            duration = (e['end'] - e['start']) / RATE
            if (e['outcome'] == 'decoder_error'
                    or (duration >= 8 and len(e['text'].split()) < duration / 2)
                    or (e['text'] and (not (e['delivery'] or {}).get('ok')
                                      or (e['delivery'] or {}).get('path') != 'copyq'))):
                result.append(e)
        return result

    def finish(self, decode: Callable[[np.ndarray], str],
               deliver: Callable[[str], dict[str, Any]]) -> list[str]:
        """Called after original delivery queue drains. Never edit/paste corrections.

        Retry each suspect exactly once, in <=10s windows; all corrections are
        one separate, labelled clipboard item. No audio retained after return.
        """
        t0 = time.perf_counter()
        corrections = []
        corrected_entries = []
        try:
            covered = self.entries[-1]['end'] if self.entries else 0
            # A missing queue tail is made explicit, not treated as silence.
            for start in range(covered, self.samples, 10 * RATE):
                self.outcome(start, min(start + 10 * RATE, self.samples), error='queue_gap')
            targets = self.suspects()
            raw = b''.join(self.frames) if targets else b''
            for e in targets:
                self.incident('stop_check', start=e['start'], end=e['end'], reason=e['outcome'])
                recovered = []
                try:
                    if e['text'] and (not (e['delivery'] or {}).get('ok')
                                      or (e['delivery'] or {}).get('path') != 'copyq'):
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
                    if text and text != (e['text'] if (e['delivery'] or {}).get('ok')
                                       and (e['delivery'] or {}).get('path') == 'copyq' else ''):
                        corrected_entries.append(e)
                        corrections.append('[mogelijk gemist %.2f–%.2fs: %s]' %
                                           (e['start_s'], e['end_s'], text))
                    e['recovery'] = dict(text=text, at=time.time())
                    diagnostic('recovery_outcome', start=e['start'], end=e['end'], **e['recovery'])
                    if not text:
                        self.incident('unresolved', start=e['start'], end=e['end'])
                except Exception as exc:
                    e['recovery'] = dict(error=str(exc), at=time.time())
                    self.incident('recovery_failed', start=e['start'], error=str(exc))
            if corrections:
                try:
                    result = deliver('\n'.join(corrections))
                except Exception as exc:
                    result = dict(ok=False, path='exception', error=str(exc))
                for e in corrected_entries:
                    e['recovery']['delivery'] = dict(result, at=time.time(), paste='never_sent')
                    diagnostic('correction_delivery', start=e['start'], end=e['end'],
                               **e['recovery']['delivery'])
                self.incident('corrections_clipboard' if result.get('ok') else 'corrections_delivery_failed',
                              count=len(corrections))
            diagnostic('stop_summary', samples=self.samples, ranges=len(self.entries),
                       suspects=len(targets), corrections=len(corrections),
                       unresolved_ranges=sum(not (e['recovery'] or {}).get('text') for e in targets),
                       original_delivery_failures=sum(bool(e['text']) and not
                           (e['delivery'] or {}).get('ok') for e in self.entries),
                       correction_delivery_failures=sum(not
                           (e['recovery'].get('delivery') or {}).get('ok') for e in corrected_entries),
                       check_seconds=time.perf_counter()-t0,
                       note='clipboard/paste command success is not target application acknowledgement')
        finally:
            self.frames.clear()
            diagnostic('audio_released', samples=self.samples)
        return corrections
