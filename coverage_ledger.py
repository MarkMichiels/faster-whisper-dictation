"""Session-local sample accounting. No audio files, model or desktop dependencies."""
import bisect
import datetime
import difflib
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

# Audio diagnosis per range (RAM only, numbers only). A 32 ms frame counts as
# "voiced" when its RMS exceeds VOICED_DBFS: close-mic speech sits around
# -30..-15 dBFS, a quiet room or an idle wireless receiver well below -50.
# A range with less than SILENT_VOICED_S of such frames cannot hold a single
# spoken word (one short "ja" is ~0.3 s), so any text decoded from it is a
# Whisper hallucination. A clipped click lifts one frame, not five.
FRAME = 512
VOICED_DBFS = -45.0
SILENT_VOICED_S = 0.15
SILENT_MIN_S = 1.0
_FLOOR_DBFS = -120.0


def _dbfs(value: float) -> float:
    return round(max(_FLOOR_DBFS, 20 * np.log10(value)) if value > 0 else _FLOOR_DBFS, 1)


def audio_stats(samples: np.ndarray) -> Optional[dict[str, Any]]:
    """Level and digital-silence figures for one range; no audio is kept."""
    if samples is None or not len(samples):
        return None
    x = samples.astype(np.float64)
    if samples.dtype == np.int16:
        x /= 32768.0
    zero = x == 0
    longest = 0
    if zero.any():
        edges = np.flatnonzero(np.diff(np.concatenate(([0], zero.view(np.int8), [0]))))
        longest = int((edges[1::2] - edges[::2]).max())
    usable = len(x) // FRAME * FRAME
    if usable:
        frame_rms = np.sqrt(np.mean(x[:usable].reshape(-1, FRAME) ** 2, axis=1))
        voiced = int(np.count_nonzero(frame_rms > 10 ** (VOICED_DBFS / 20))) * FRAME
    else:
        voiced = len(x) if np.sqrt(np.mean(x ** 2)) > 10 ** (VOICED_DBFS / 20) else 0
    return dict(seconds=round(len(x) / RATE, 3),
                rms_dbfs=_dbfs(float(np.sqrt(np.mean(x ** 2)))),
                peak_dbfs=_dbfs(float(np.max(np.abs(x)))),
                voiced_s=round(voiced / RATE, 2),
                zero_fraction=round(float(zero.mean()), 3),
                longest_zero_s=round(longest / RATE, 2))


def silent_audio(stats: Optional[dict[str, Any]]) -> bool:
    """True only when a range is long enough to judge and holds no speech energy."""
    return bool(stats) and stats['seconds'] >= SILENT_MIN_S and stats['voiced_s'] < SILENT_VOICED_S


# A correction carries words, never pause marks: '* * *', '. . .', '…', '---'.
_PAUSE_MARKS_RE = re.compile(r'(?:\.\s*){3,}|\u2026+|[*_~#=]+|-{2,}')


def clean_correction(text: str, is_artifact: Optional[Callable[[str], bool]] = None) -> str:
    text = re.sub(r'\s+', ' ', _PAUSE_MARKS_RE.sub(' ', text)).strip()
    text = re.sub(r'\s+([,.;:!?])', r'\1', text).strip(' ,;:')
    if not re.search(r'\w', text) or (is_artifact and is_artifact(text)):
        return ''
    return text


DUPLICATE_SHARE = 0.6


def _words(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


def repeats_delivered(text: str, reference: str) -> bool:
    """True when most recovered words already reached Mark, in the same order.

    Only runs of two or more consecutive words count (a lone 'de' or 'is'
    matches any long neighbour); a one-word recovery needs that word itself.
    """
    words, ref = _words(text), _words(reference)
    if not words or not ref:
        return False
    least = min(2, len(words))
    blocks = difflib.SequenceMatcher(None, words, ref, autojunk=False).get_matching_blocks()
    covered = sum(b.size for b in blocks if b.size >= least)
    return covered >= DUPLICATE_SHARE * len(words)


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


# Mark 18/10: paste the check line only when something deviated (a correction
# or a range whose text could not be found back). A clean session ends with
# nothing extra in the text; the journal still logs stop_summary. Set False to
# never paste it.
STOP_STATUS = True


def stop_status(checked: int, corrected: int, unresolved: int, seconds: float,
                silent: int = 0) -> str:
    """Short Dutch end-of-session line, e.g. '[Controle: niets gemist, 0,1 s]'."""
    took = ('%.1f s' % seconds).replace('.', ',')
    if not checked:
        return '[Controle: niets gemist, %s]' % took
    parts = ['%d %s nagekeken' % (checked, 'stuk' if checked == 1 else 'stukken')]
    if corrected:
        parts.append('%d %s' % (corrected, 'correctie' if corrected == 1 else 'correcties'))
    if unresolved:
        parts.append('%d niet terug te vinden' % unresolved)
    if silent:
        parts.append('%d zonder geluid' % silent)
    return '[Controle: %s, %s]' % (', '.join(parts), took)


class CoverageLedger:
    def __init__(self) -> None:
        self.frames = []
        self.frame_starts = []
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
        self.frame_starts.append(self.samples)
        self.frames.append(data)
        self.samples += len(data) // 2
        if not self.warned and self.samples >= (MAX_SECONDS - 60) * RATE:
            self.warned = True
            self.incident('duration_warning', remaining_seconds=60)
        return data

    def audio(self, start: int, end: int) -> np.ndarray:
        """int16 samples [start, end) from the RAM frames, without joining all."""
        if not self.frames or len(self.frame_starts) != len(self.frames):
            return np.zeros(0, dtype=np.int16)  # Frames already released.
        first = max(0, bisect.bisect_right(self.frame_starts, start) - 1)
        last = bisect.bisect_left(self.frame_starts, end)
        data = b''.join(self.frames[first:last])
        offset = self.frame_starts[first]
        return np.frombuffer(data, dtype=np.int16)[start - offset:end - offset]

    def outcome(self, start: int, end: int, text: str = '', raw: str = '',
                error: Optional[str] = None, audio: Optional[dict[str, Any]] = None,
                discard_reason: Optional[str] = None) -> dict[str, Any]:
        # Exclusive ranges must form one partition, regardless of decoder result.
        expected = self.entries[-1]['end'] if self.entries else 0
        if start != expected or end <= start or end > self.samples:
            raise ValueError('non-contiguous or uncaptured sample range')
        kind = 'decoder_error' if error else ('text' if text else ('artifact' if raw else 'empty'))
        entry = dict(start=start, end=end, start_s=start/RATE, end_s=end/RATE,
                     outcome=kind, text=text, raw=raw, error=error,
                     discard_reason=(discard_reason or 'known_silence_artifact') if kind == 'artifact' else None,
                     decoded_at=time.time(),
                     audio=audio if audio is not None else audio_stats(self.audio(start, end)),
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

    @staticmethod
    def _unresolved(targets: list[dict[str, Any]]) -> int:
        return sum(not (e['recovery'] or {}).get('text') and not (e['recovery'] or {}).get('silent')
                   for e in targets)

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

    def _reference(self, i: int) -> str:
        """Delivered text of range i and its nearest delivered neighbours."""
        before = next((e['text'] for e in reversed(self.entries[:i]) if _delivered(e)), '')
        after = next((e['text'] for e in self.entries[i+1:] if _delivered(e)), '')
        own = self.entries[i]['text'] if _delivered(self.entries[i]) else ''
        return ' '.join(t for t in (before, own, after) if t)

    def finish(self, decode: Callable[[np.ndarray], str],
               deliver: Callable[[str], dict[str, Any]],
               is_artifact: Optional[Callable[[str], bool]] = None) -> list[str]:
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
            index = {id(e): i for i, e in enumerate(self.entries)}
            recovered_texts = []
            for e in targets:
                self.incident('stop_check', start=e['start'], end=e['end'], reason=e['outcome'],
                              audio=e.get('audio'))
                recovered = []
                try:
                    if self._failed_delivery(e):
                        recovered.append(e['text'])  # Delivery failure needs no GPU work.
                    elif silent_audio(e.get('audio')):
                        # No speech energy reached the ledger: decoding it again
                        # only invites a hallucination. Mark sees "zonder geluid".
                        e['recovery'] = dict(text='', silent=True, at=time.time())
                        diagnostic('recovery_outcome', start=e['start'], end=e['end'], **e['recovery'])
                        continue
                    else:
                        if e['outcome'] == 'decoder_error':
                            time.sleep(0.2)
                        for start in range(e['start'], e['end'], 10 * RATE):
                            end = min(start + 10 * RATE, e['end'])
                            audio = self.audio(start, end).astype(np.float32) / 32768
                            recovered.append(clean_correction(decode(audio), is_artifact))
                    raw_text = ' '.join(t.strip() for t in recovered if t.strip())
                    text = clean_correction(raw_text, is_artifact)
                    duplicate = bool(text) and not self._failed_delivery(e) \
                        and repeats_delivered(text, self._reference(index[id(e)]))
                    if text and not duplicate:
                        recovered_texts.append((index[id(e)], text))
                    e['recovery'] = dict(text=text, at=time.time())
                    if duplicate:
                        e['recovery']['duplicate'] = True  # Checked; nothing new to paste.
                    if raw_text != text:
                        e['recovery']['raw'] = raw_text
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
            unresolved = self._unresolved(targets)
            if STOP_STATUS and (corrections or unresolved):
                status = stop_status(len(targets), len(corrections), unresolved,
                                     time.perf_counter() - t0,
                                     sum(bool((e['recovery'] or {}).get('silent')) for e in targets))
                try:
                    result = deliver(status)
                except Exception as exc:
                    result = dict(ok=False, path='exception', error=str(exc))
                diagnostic('stop_status_delivery', status=status, **dict(result, at=time.time()))
            diagnostic('stop_summary', samples=self.samples, ranges=len(self.entries),
                       suspects=len(targets), corrections=len(corrections),
                       unresolved_ranges=self._unresolved(targets),
                       silent_ranges=sum(bool((e['recovery'] or {}).get('silent')) for e in targets),
                       duplicate_ranges=sum(bool((e['recovery'] or {}).get('duplicate')) for e in targets),
                       original_delivery_failures=sum(self._failed_delivery(e) for e in self.entries),
                       correction_delivery_failures=sum(not
                           (e['recovery'].get('delivery') or {}).get('ok') for e in corrected_entries),
                       check_seconds=time.perf_counter()-t0,
                       note='clipboard/paste command success is not target application acknowledgement')
        finally:
            self.frames.clear()
            self.frame_starts.clear()
            diagnostic('audio_released', samples=self.samples)
        return corrections
