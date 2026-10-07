"""Модульные тесты логики, которым не нужны модели: python -m pytest tests"""
import numpy as np
import pytest

from app.diarizer import OnlineSpeakerClusterer, find_turns
from app.llm import clip_transcript
from app.session import refine_cuts, tokens_to_words, words_text
from app.vad import FRAME_SEC, SAMPLE_RATE, WINDOW, StreamingVad, find_pauses


def voice(seed: int, noise: float = 0.15, dim: int = 192) -> np.ndarray:
    """Вектор «голоса»: общее направление диктора + небольшой разброс от реплики к реплике."""
    base = np.random.default_rng(seed).normal(size=dim)
    v = base / np.linalg.norm(base) + np.random.default_rng().normal(scale=noise / np.sqrt(dim), size=dim)
    return (v / np.linalg.norm(v)).astype(np.float32)


# ------------------------------------------------------------ собеседники
def mix(seed: int, purity: float, dim: int = 192) -> np.ndarray:
    """Вектор, близкий к голосу seed ровно на purity (косинус) — имитация короткой, неточной реплики."""
    base = np.random.default_rng(seed).normal(size=dim)
    base /= np.linalg.norm(base)
    other = np.random.default_rng(1000 + seed).normal(size=dim)
    other -= other.dot(base) * base
    other /= np.linalg.norm(other)
    return (purity * base + np.sqrt(1 - purity**2) * other).astype(np.float32)


def test_two_speakers_are_separated_and_stable():
    c = OnlineSpeakerClusterer(threshold=0.45, max_speakers=5)
    seq = [1, 2, 1, 1, 2, 2, 1, 2]
    got = [c.assign(voice(s), 3.0).speaker for s in seq]
    assert got == seq
    assert len(c.speakers) == 2


def test_short_utterance_never_creates_speaker():
    c = OnlineSpeakerClusterer(threshold=0.45)
    assert c.assign(voice(1), 3.0).is_new
    res = c.assign(voice(99), 0.5)            # незнакомый голос, но всего полсекунды
    assert res.speaker == 1 and not res.is_new and not res.confident
    assert len(c.speakers) == 1


def test_short_first_utterance_does_not_leave_ghost_speaker():
    """«Алло» в начале разговора: тот же человек дальше говорит длинно — собеседник остаётся один."""
    c = OnlineSpeakerClusterer(threshold=0.45)
    first = c.assign(mix(1, 0.36), 0.4)       # короткая реплика похожа на свой голос лишь на 0,36
    assert first.is_new and first.speaker == 1
    nxt = c.assign(voice(1, noise=0.0), 4.0)
    assert nxt.speaker == 1 and not nxt.is_new
    assert len(c.speakers) == 1
    # центроид теперь определяется длинной репликой, чужой голос к нему не липнет
    assert c.assign(voice(2), 4.0).speaker == 2


def test_short_first_utterance_of_other_voice_stays_separate():
    c = OnlineSpeakerClusterer(threshold=0.45)
    c.assign(mix(1, 0.36), 0.4)
    res = c.assign(voice(2), 4.0)             # совсем другой голос
    assert res.is_new and res.speaker == 2


def test_threshold_depends_on_how_much_speech_the_speaker_has():
    c = OnlineSpeakerClusterer(threshold=0.45)
    c.assign(voice(1, noise=0.0), 5.0)        # у собеседника накоплено много речи
    weak_match = mix(1, 0.36)
    # длинная реплика с близостью 0,36 — это чужой голос
    assert c.assign(weak_match, 4.0).speaker == 2
    c2 = OnlineSpeakerClusterer(threshold=0.45)
    c2.assign(voice(1, noise=0.0), 5.0)
    # а короткая (0,5 с) с той же близостью — допустимо тот же
    assert c2.assign(weak_match, 0.5).speaker == 1


def test_max_speakers_is_respected():
    c = OnlineSpeakerClusterer(threshold=0.45, max_speakers=2)
    for s in (1, 2, 3, 4):
        c.assign(voice(s), 3.0)
    assert len(c.speakers) == 2


def test_missing_embedding_goes_to_last_speaker():
    c = OnlineSpeakerClusterer()
    c.assign(voice(1), 3.0)
    c.assign(voice(2), 3.0)
    assert c.assign(None, 0.2).speaker == 2


def test_first_utterance_without_embedding_does_not_break_later_ones():
    """Раньше такой «пустой» собеседник ломал сравнение векторов для всех следующих реплик."""
    c = OnlineSpeakerClusterer()
    first = c.assign(None, 0.2)
    assert first.speaker == 1 and first.is_new and not first.confident
    assert c.score(voice(1)) == (None, -1.0)
    second = c.assign(voice(1), 3.0)          # первый настоящий голос занимает пустого собеседника
    assert second.speaker == 1 and len(c.speakers) == 1
    assert c.assign(voice(1), 3.0).speaker == 1
    assert c.assign(voice(2), 3.0).speaker == 2


def test_duplicate_speakers_get_merged_and_number_is_reused():
    c = OnlineSpeakerClusterer(threshold=0.45)
    c.assign(voice(1, noise=0.0), 4.0)
    # искусственно заводим второго собеседника с тем же голосом
    c.speakers.append(type(c.speakers[0])(id=2, total=c.speakers[0].total.copy(), weight=4.0, count=1))
    c.assign(voice(3), 4.0)
    assert sorted(s.id for s in c.speakers) == [1, 2, 3]
    res = c.assign(voice(1, noise=0.0), 4.0)
    assert res.merged == [(2, 1)] and sorted(s.id for s in c.speakers) == [1, 3]
    # освободившийся номер 2 достаётся следующему новому голосу — номера не выходят за 1..5
    assert c.assign(voice(4), 4.0).speaker == 2


def test_speaker_numbers_stay_within_limit():
    c = OnlineSpeakerClusterer(threshold=0.45, max_speakers=5)
    ids = {c.assign(voice(s), 3.0).speaker for s in range(1, 30)}
    assert ids <= {1, 2, 3, 4, 5}


def test_threshold_is_lower_for_short_utterances():
    c = OnlineSpeakerClusterer(threshold=0.5)
    assert c.threshold_for(5.0) == pytest.approx(0.5)
    assert c.threshold_for(0.3) < c.threshold_for(0.6) < c.threshold_for(1.0) < c.threshold_for(2.0) < c.threshold_for(3.0)
    assert OnlineSpeakerClusterer(threshold=0.2).threshold_for(0.1) >= 0.12


def test_find_turns_splits_at_voice_change():
    embs = [voice(1) for _ in range(6)] + [voice(2) for _ in range(6)]
    centers = [0.75 + 0.5 * i for i in range(12)]
    turns = find_turns(embs, centers, duration=7.0, threshold=0.33)
    assert len(turns) == 2
    assert 3.0 < turns[0][1] < 4.2


def test_find_turns_keeps_single_speaker_whole():
    embs = [voice(1) for _ in range(10)]
    centers = [0.75 + 0.5 * i for i in range(10)]
    assert find_turns(embs, centers, 6.0, 0.33) == [(0.0, 6.0)]


def test_find_turns_ignores_single_outlier_window():
    embs = [voice(1) for _ in range(5)] + [voice(7)] + [voice(1) for _ in range(5)]
    centers = [0.75 + 0.5 * i for i in range(11)]
    assert find_turns(embs, centers, 6.5, 0.33) == [(0.0, 6.5)]


# ------------------------------------------------------------------ слова
def test_tokens_to_words_and_text():
    tokens = [" Доб", "рый", " день", ".", " Как", " дела", "?"]
    stamps = [0.1, 0.3, 0.6, 0.8, 1.4, 1.7, 2.0]
    w = tokens_to_words(tokens, stamps, 2.5)
    assert [x["w"] for x in w] == ["Добрый", "день.", "Как", "дела?"]
    assert w[0]["s"] == pytest.approx(0.1) and w[0]["e"] <= w[1]["s"]
    assert w[-1]["e"] <= 2.5
    assert words_text(w) == "Добрый день. Как дела?"


def test_words_text_strips_dialogue_dash_and_capitalises():
    w = tokens_to_words([" —", " какие", " работы", "?"], [0.1, 0.2, 0.5, 0.7], 1.0)
    assert words_text(w) == "Какие работы?"
    w = tokens_to_words([" работы", ".", " —"], [0.1, 0.4, 0.6], 1.0)
    assert words_text(w) == "Работы."
    w = tokens_to_words([" кто", "-", "то", " пришёл"], [0.1, 0.2, 0.3, 0.6], 1.0)
    assert words_text(w) == "Кто-то пришёл"


def test_refine_cuts_prefers_real_pause():
    words = [{"w": "а", "s": 0.2, "e": 1.9}, {"w": "б", "s": 2.6, "e": 4.0}]
    probs = np.ones(int(4.2 / FRAME_SEC), dtype=np.float32)
    probs[int(2.0 / FRAME_SEC): int(2.5 / FRAME_SEC)] = 0.05          # пауза 2.0–2.5 с
    cuts = refine_cuts([2.9], words, probs, 4.2, 0.35)
    assert len(cuts) == 1
    assert cuts[0][0] == pytest.approx(2.0, abs=0.05) and cuts[0][1] == pytest.approx(2.5, abs=0.05)


def test_refine_cuts_falls_back_to_word_gap_without_pause():
    words = [{"w": "а", "s": 0.2, "e": 1.9}, {"w": "б", "s": 2.0, "e": 4.0}]
    probs = np.ones(int(4.2 / FRAME_SEC), dtype=np.float32)
    cuts = refine_cuts([2.2], words, probs, 4.2, 0.35)
    assert cuts == [(1.95, 1.95)]                                   # пауза нулевая — перебивание


def test_find_pauses():
    probs = np.array([1, 1, 0.1, 0.1, 0.1, 1, 0.1, 1], dtype=np.float32)
    assert find_pauses(probs, 0.35) == [(2 * FRAME_SEC, 5 * FRAME_SEC)]


# --------------------------------------------------------- детектор речи
class ScriptedModel:
    """Вместо нейросети: вероятность речи = 1, если в кадре есть ненулевой сигнал."""

    def run(self, _, feed):
        frame = feed["input"][0, 64:]
        return np.array([[1.0 if np.abs(frame).max() > 0.01 else 0.0]], dtype=np.float32), feed["state"]


def tone(sec: float) -> np.ndarray:
    return (0.3 * np.sin(np.arange(int(sec * SAMPLE_RATE)) * 0.05)).astype(np.float32)


def silence(sec: float) -> np.ndarray:
    return np.zeros(int(sec * SAMPLE_RATE), dtype=np.float32)


def feed(vad, audio, chunk=1600):
    out = []
    for i in range(0, len(audio), chunk):
        out.extend(vad.accept(audio[i:i + chunk]))
    return out


def test_vad_cuts_segment_after_pause_with_padding():
    vad = StreamingVad(ScriptedModel(), min_silence=0.45)
    segs = feed(vad, np.concatenate([silence(1.0), tone(2.0), silence(1.0)]))
    assert len(segs) == 1
    s = segs[0]
    assert s.start / SAMPLE_RATE == pytest.approx(1.0 - s.lead / SAMPLE_RATE, abs=0.05)
    voiced = (s.samples.size - s.lead - s.trail) / SAMPLE_RATE
    assert voiced == pytest.approx(2.0, abs=0.07)
    assert 0 < s.lead <= 0.3 * SAMPLE_RATE and 0 < s.trail <= 0.3 * SAMPLE_RATE
    assert len(s.probs) == s.samples.size // WINDOW
    assert not vad.speaking


def test_vad_keeps_short_gap_inside_one_segment():
    vad = StreamingVad(ScriptedModel(), min_silence=0.45)
    segs = feed(vad, np.concatenate([silence(0.5), tone(1.0), silence(0.2), tone(1.0), silence(1.0)]))
    assert len(segs) == 1
    # До начала речи VAD добавляет поле тишины. Оно не является проверяемой
    # паузой между двумя кусками речи и может тоже попасть в find_pauses.
    internal = [(a, b) for a, b in find_pauses(segs[0].probs, 0.35)
                if a >= segs[0].lead / SAMPLE_RATE]
    assert len(internal) == 1
    assert internal[0][1] - internal[0][0] == pytest.approx(0.2, abs=0.07)


def test_vad_splits_two_phrases():
    vad = StreamingVad(ScriptedModel(), min_silence=0.45)
    segs = feed(vad, np.concatenate([silence(0.5), tone(1.0), silence(0.8), tone(1.5), silence(1.0)]))
    assert len(segs) == 2
    assert segs[1].start > segs[0].end - 1


def test_vad_drops_clicks_shorter_than_min_speech():
    vad = StreamingVad(ScriptedModel(), min_speech=0.25)
    assert feed(vad, np.concatenate([silence(0.5), tone(0.1), silence(1.0)])) == []


def test_vad_forces_cut_on_endless_speech_without_losing_audio():
    vad = StreamingVad(ScriptedModel(), max_speech=3.0)
    segs = feed(vad, np.concatenate([silence(0.3), tone(10.0), silence(1.0)]))
    assert len(segs) >= 3
    assert all(s.samples.size <= 3.0 * SAMPLE_RATE + WINDOW for s in segs)
    for a, b in zip(segs, segs[1:]):
        assert b.start == a.end                                   # части идут встык
    total = sum(s.samples.size for s in segs) / SAMPLE_RATE
    assert total == pytest.approx(10.0, abs=0.4)


def test_vad_flush_returns_unfinished_segment():
    vad = StreamingVad(ScriptedModel())
    assert feed(vad, np.concatenate([silence(0.3), tone(1.0)])) == []
    assert vad.speaking and vad.current().size > 0
    segs = vad.flush()
    assert len(segs) == 1 and not vad.speaking


# -------------------------------------------------------------------- LLM
def test_clip_transcript_keeps_head_and_tail():
    text = "\n".join(f"[00:{i:02d}] Собеседник 1: реплика номер {i}" for i in range(60))
    out = clip_transcript(text, 600)
    assert len(out) < 800
    assert out.startswith("[00:00]") and out.rstrip().endswith("реплика номер 59")
    assert "пропущена" in out
    assert clip_transcript("коротко", 600) == "коротко"
