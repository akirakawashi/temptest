"""Разделение собеседников на лету.

Каждая реплика превращается в вектор голоса (эмбеддинг). Векторы сравниваются
с уже известными собеседниками по косинусной близости: похож — относим к нему,
не похож и реплика достаточно длинная — заводим нового собеседника.
Никаких данных о голосах на диск не пишется: всё живёт в памяти сессии.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


@dataclass
class Speaker:
    id: int
    total: Optional[np.ndarray] = None   # сумма векторов голоса, взвешенная длительностью
    weight: float = 0.0                  # сколько секунд речи накоплено в центроиде
    count: int = 0                       # число отнесённых реплик

    @property
    def centroid(self) -> Optional[np.ndarray]:
        return None if self.total is None else _unit(self.total)


@dataclass
class Assignment:
    speaker: int
    similarity: float            # близость к выбранному собеседнику, -1..1
    is_new: bool = False
    confident: bool = True
    merged: List[Tuple[int, int]] = field(default_factory=list)   # (кого, в кого) слили


class OnlineSpeakerClusterer:
    """Инкрементальная кластеризация голосов.

    threshold     — порог близости для длинной реплики (>= 3 с). Чем короче реплика
                    (или чем меньше речи накоплено у собеседника), тем ниже порог:
                    короткий фрагмент даёт менее точный вектор, и тот же голос
                    выглядит «менее похожим». Значения подобраны по замерам на живых голосах.
    max_speakers  — потолок числа собеседников; лишние голоса относятся к ближайшему.

    Номера собеседников берутся из диапазона 1..max_speakers и переиспользуются
    после слияния — поэтому у каждого всегда свой цвет в интерфейсе.
    """

    NEW_MIN_SEC = 0.8    # короче — новый собеседник не создаётся (если хоть один уже есть)
    WEAK_SEC = 1.5       # собеседник «слабый», пока у него накоплено меньше речи
    UPDATE_MIN_SEC = 0.6 # короче — центроид не обновляется

    def __init__(self, threshold: float = 0.45, max_speakers: int = 5):
        self.threshold = threshold
        self.max_speakers = max_speakers
        self.speakers: List[Speaker] = []
        self._last_id: Optional[int] = None

    # ------------------------------------------------------------------ utils
    def threshold_for(self, duration: float) -> float:
        """Порог близости для фрагмента заданной длины (секунды речи)."""
        t = self.threshold
        if duration >= 3.0:
            thr = t
        elif duration >= 1.0:
            thr = t - 0.12 * (3.0 - duration) / 2.0
        elif duration >= 0.4:
            thr = t - 0.12 - 0.11 * (1.0 - duration) / 0.6
        else:
            thr = t - 0.23
        return max(0.12, thr)

    def _scores(self, emb: np.ndarray) -> List[Tuple[Speaker, float]]:
        e = _unit(emb)
        out = []
        for s in list(self.speakers):            # снимок: метод зовут и из другого потока
            c = s.centroid
            if c is not None and c.shape == e.shape:
                out.append((s, float(np.dot(c, e))))
        return out

    def score(self, emb: np.ndarray, duration: float = 3.0) -> Tuple[Optional[int], float]:
        """Подходящий известный собеседник и близость к нему (состояние не меняется).

        Возвращает (None, близость к ближайшему), если никто не проходит порог.
        """
        scores = self._scores(emb)
        if not scores:
            return None, -1.0
        ok = [(s, v) for s, v in scores if v >= self.threshold_for(min(duration, s.weight))]
        if ok:
            s, v = max(ok, key=lambda x: x[1])
            return s.id, v
        return None, max(v for _, v in scores)

    def _get(self, sid: int) -> Speaker:
        return next(s for s in self.speakers if s.id == sid)

    def _free_id(self) -> int:
        used = {s.id for s in self.speakers}
        return next(i for i in range(1, len(used) + 2) if i not in used)

    # ----------------------------------------------------------------- assign
    def assign(self, emb: Optional[np.ndarray], duration: float) -> Assignment:
        """Отнести реплику к собеседнику, при необходимости создав нового.

        duration — длительность самой речи в секундах (без полей тишины).
        """
        if emb is None:
            # фрагмент слишком короткий, вектор не посчитать: берём последнего говорившего
            if self.speakers and self._last_id is not None:
                self._get(self._last_id).count += 1
                return Assignment(self._last_id, 0.0, confident=False)
            return self._new(None, 0.0)

        e = _unit(emb.astype(np.float32))
        scores = self._scores(e)
        ok = [(s, v) for s, v in scores if v >= self.threshold_for(min(duration, s.weight))]
        if ok:
            sp, sim = max(ok, key=lambda x: x[1])
            weak = sp.weight < self.WEAK_SEC
            if duration >= self.UPDATE_MIN_SEC:
                # уверенное совпадение учитываем полностью, пограничное — вполсилы;
                # слабый центроид (мало речи) новая реплика уточняет всегда в полную силу
                w = min(duration, 6.0) * (1.0 if weak or sim >= self.threshold else 0.5)
                self._update(sp, e, w)
            sp.count += 1
            self._last_id = sp.id
            # уверенно: реплика не совсем короткая либо голос совпал с большим запасом
            confident = duration >= self.NEW_MIN_SEC or sim >= self.threshold
            return Assignment(sp.id, sim, confident=confident, merged=self._merge_close())

        # никто не подошёл. Собеседник без вектора (первая реплика была слишком короткой)
        # получает этот голос; иначе заводим нового, если реплика достаточно длинная
        blank = next((s for s in self.speakers if s.total is None), None)
        if blank is not None:
            self._update(blank, e, max(min(duration, 6.0), 0.2))
            blank.count += 1
            self._last_id = blank.id
            return Assignment(blank.id, 1.0, confident=False)

        can_create = len(self.speakers) < self.max_speakers and (not self.speakers or duration >= self.NEW_MIN_SEC)
        if can_create:
            return self._new(e, duration)

        # короткая реплика или достигнут потолок — относим к ближайшему, но помечаем
        sp, sim = max(scores, key=lambda x: x[1]) if scores else (self.speakers[0], 0.0)
        sp.count += 1
        self._last_id = sp.id
        return Assignment(sp.id, sim, confident=False)

    def _new(self, e: Optional[np.ndarray], duration: float) -> Assignment:
        sp = Speaker(id=self._free_id())
        if e is not None:
            self._update(sp, e, max(min(duration, 6.0), 0.2))
        sp.count = 1
        self.speakers.append(sp)
        self._last_id = sp.id
        return Assignment(sp.id, 1.0, is_new=True, confident=e is not None)

    @staticmethod
    def _update(sp: Speaker, e: np.ndarray, w: float) -> None:
        sp.total = e * w if sp.total is None else sp.total + e * w
        sp.weight += w

    def _merge_close(self) -> List[Tuple[int, int]]:
        """Если два собеседника со временем оказались одним голосом — объединяем.

        Порог слияния заметно выше порога отнесения: объединяем только когда оба
        центроида накопили достаточно речи и явно совпадают. Остаётся меньший номер.
        """
        merged: List[Tuple[int, int]] = []
        thr = min(0.92, self.threshold + 0.22)
        changed = True
        while changed:
            changed = False
            order = sorted(self.speakers, key=lambda s: s.id)
            for i, a in enumerate(order):
                for b in order[i + 1:]:
                    if a.weight < 3.0 or b.weight < 3.0:
                        continue
                    if float(np.dot(a.centroid, b.centroid)) >= thr:
                        a.total = a.total + b.total
                        a.weight += b.weight
                        a.count += b.count
                        self.speakers.remove(b)
                        merged.append((b.id, a.id))
                        if self._last_id == b.id:
                            self._last_id = a.id
                        changed = True
                        break
                if changed:
                    break
        return merged


# --------------------------------------------------------------------------
#  Поиск смены говорящего внутри одного непрерывного фрагмента речи
# --------------------------------------------------------------------------
def find_turns(
    window_embs: List[np.ndarray],
    window_centers: List[float],
    duration: float,
    threshold: float,
    min_turn: float = 1.2,
) -> List[Tuple[float, float]]:
    """Вернуть границы реплик [(начало, конец)] внутри фрагмента длиной duration секунд.

    window_embs — эмбеддинги скользящих окон, window_centers — центры окон (секунды).
    Окна объединяются в группы по близости; если получилось больше одной устойчивой
    группы, фрагмент делится в точках смены группы.
    """
    n = len(window_embs)
    if n < 4:
        return [(0.0, duration)]
    X = np.stack([_unit(e) for e in window_embs])

    # жадная последовательная кластеризация окон
    cents: List[np.ndarray] = []
    labels: List[int] = []
    for x in X:
        if cents:
            sims = [float(np.dot(_unit(c), x)) for c in cents]
            k = int(np.argmax(sims))
            if sims[k] >= threshold:
                cents[k] = cents[k] + x
                labels.append(k)
                continue
        cents.append(x.copy())
        labels.append(len(cents) - 1)
    if len(cents) == 1:
        return [(0.0, duration)]

    # одиночные «выбросы» между одинаковыми соседями сглаживаем
    for i in range(1, n - 1):
        if labels[i] != labels[i - 1] and labels[i - 1] == labels[i + 1]:
            labels[i] = labels[i - 1]

    # серии одинаковых меток -> кандидаты в реплики
    runs: List[List[int]] = []          # [метка, первый индекс, последний индекс]
    for i, lab in enumerate(labels):
        if runs and runs[-1][0] == lab:
            runs[-1][2] = i
        else:
            runs.append([lab, i, i])
    # слишком короткие серии (одно окно) приклеиваем к соседу
    stable = [r for r in runs if r[2] - r[1] >= 1]
    if len(stable) < 2:
        return [(0.0, duration)]

    bounds: List[float] = []
    prev = stable[0]
    for r in stable[1:]:
        if r[0] == prev[0]:
            prev = [prev[0], prev[1], r[2]]
            continue
        bounds.append((window_centers[prev[2]] + window_centers[r[1]]) / 2.0)
        prev = r

    turns: List[Tuple[float, float]] = []
    start = 0.0
    for b in bounds:
        if b - start >= min_turn and duration - b >= min_turn:
            turns.append((start, b))
            start = b
    turns.append((start, duration))
    return turns
