// Дашборд: главный экран (плитки собеседников вокруг центра, кольца эмоций, графики
// с вкладками) и подробные разделы ниже. Всё считается в браузере из событий сервера
// и перерисовывается только когда изменились данные.

import * as C from "./charts.js";
import {
  EMOTIONS, EMO_RU, EMO_VAR, cssVar, speakerVar, fmtClock, fmtNum, fmtSec, fmtDur, fmtPct,
  median, quantile, mean, words, countFillers, topWords, plural, h,
} from "./util.js";

const cards = {};
const memo = new Map();
const view = { window: 0, live: "level", mood: "trend" };      // что выбрано во вкладках
const SLOTS = ["l1", "r1", "l2", "r2", "l3"];                   // места плиток собеседников
let requestRender = () => {};

function makeCard(id, title, sub) {
  const root = document.getElementById(id);
  const body = h("div", {});
  const extra = h("div");
  const table = h("div", { hidden: true });
  const toggle = h("button", { class: "link-btn", hidden: true, "aria-pressed": "false" }, "Таблица");
  toggle.addEventListener("click", () => {
    const on = table.hidden;
    table.hidden = !on; body.hidden = on; extra.hidden = on;
    toggle.textContent = on ? "График" : "Таблица";
    toggle.setAttribute("aria-pressed", String(on));
  });
  root.replaceChildren(
    h("div", { class: "panel-head" }, h("span", { class: "chev", "aria-hidden": "true" }), h("h3", {}, title), h("div", { class: "panel-tools" }, toggle)),
    h("div", { class: "panel-body" }, ...[sub ? h("p", { class: "panel-sub" }, sub) : null, body, extra, table].filter(Boolean))
  );
  const card = {
    root, body, extra,
    setTable(headers, rows) { toggle.hidden = false; table.replaceChildren(C.dataTable(headers, rows)); },
  };
  cards[id] = card;
  return card;
}

function bindTabs(id, attr, onPick) {
  const box = document.getElementById(id);
  box.addEventListener("click", (e) => {
    const btn = e.target.closest(".hud-tab");
    if (!btn) return;
    for (const b of box.querySelectorAll(".hud-tab")) b.classList.toggle("is-active", b === btn);
    onPick(btn.dataset[attr]);
    memo.clear();
    requestRender();
  });
}

export function initDashboard(onChange) {
  requestRender = onChange || (() => {});

  // легенда цвета плиток
  document.getElementById("heroLegend").replaceChildren(
    h("b", {}, "Цвет плитки — преобладающая эмоция:"),
    ...EMOTIONS.slice().reverse().map((e) => h("span", {}, h("i", { style: { "--c": `var(${EMO_VAR[e]})` } }), EMO_RU[e]))
  );
  bindTabs("windowTabs", "window", (v) => { view.window = Number(v) || 0; });
  bindTabs("liveTabs", "key", (v) => { view.live = v; });
  bindTabs("moodTabs", "key", (v) => { view.mood = v; });

  // плитки и центральный круг создаются один раз и дальше только обновляются
  const hub = document.getElementById("hub");
  for (const slot of [...SLOTS, "r3"]) {
    hub.append(h("div", { class: `tile slot-${slot} is-empty`, "data-slot": slot },
      h("span", { class: "t-icon", "aria-hidden": "true" }), h("span", { class: "t-name" }), h("span", { class: "t-val" }),
      h("span", { class: "t-note" }), h("span", { class: "t-pct" })));
  }
  C.orb(hub, { label: "Длительность", value: "00:00", note: "" });

  makeCard("c-timeline", "Лента разговора", "Кто и когда говорил. Цвет отрезка — эмоция, определённая по голосу.");
  makeCard("c-speakers", "Показатели по собеседникам", "Средняя и макс. — длина реплики. Паразиты — доля слов-паразитов. Сходство голоса — насколько уверенно реплики отнесены к собеседнику (от 0 до 1).");
  makeCard("c-durations", "Длительность реплик", "Сколько реплик какой длины, секунды.");
  makeCard("c-gaps", "Паузы перед ответом", "Паузы при смене собеседника, секунды. Меньше 0,3 с — перебивание.");
  makeCard("c-latency", "Задержка до текста", "Секунд от конца фразы до появления реплики, включая паузу, по которой определяется конец фразы.");
  makeCard("c-words", "Частые слова", "От четырёх букв, минимум дважды; служебные слова исключены.");
  makeCard("c-stages", "Время обработки одной реплики", "Среднее по этапам, миллисекунды.");
  document.getElementById("c-llm").replaceChildren(h("div", { id: "llmBox" }));
  document.getElementById("c-models").replaceChildren(
    h("div", { class: "panel-head" }, h("span", { class: "chev", "aria-hidden": "true" }), h("h3", {}, "Модели")),
    h("div", { class: "panel-body", id: "modelsBox" }));
  document.getElementById("c-params").replaceChildren(
    h("div", { class: "panel-head" }, h("span", { class: "chev", "aria-hidden": "true" }), h("h3", {}, "Параметры и окружение")),
    h("div", { class: "panel-body" }, h("dl", { class: "params", id: "paramsBox" })));
}

export function invalidateDashboard() {
  memo.clear();
}

function draw(key, data, fn) {
  const sig = JSON.stringify(data);
  if (memo.get(key) === sig) return;
  memo.set(key, sig);
  fn();
}

// ------------------------------------------------------------- агрегаты
/** Показатели разговора за выбранный период (win секунд назад от текущего момента; 0 — весь разговор). */
export function aggregate(state, win = 0) {
  const now = Math.max(state.audioSec, state.utterances.length ? state.utterances[state.utterances.length - 1].end : 0);
  const from = win ? Math.max(0, now - win) : 0;
  const per = new Map();
  const get = (id) => {
    if (!per.has(id)) {
      per.set(id, {
        id, name: state.speakerName(id), color: cssVar(speakerVar(id)),
        talk: 0, count: 0, words: 0, fillers: 0, durations: [], longest: 0,
        interrupts: 0, interrupted: 0, gaps: [], sims: [], dbs: [], confident: 0,
        emo: { angry: 0, sad: 0, neutral: 0, positive: 0 }, emoTime: 0, valence: [],
      });
    }
    return per.get(id);
  };
  const gapsAll = [], list = [];
  let prev = null, totalWords = 0, interruptions = 0;
  for (const u of state.utterances) {
    if (u.end <= from) { prev = u; continue; }
    list.push(u);
    const s = get(u.speaker);
    const full = Math.max(0, u.end - u.start);
    const d = Math.max(0, u.end - Math.max(u.start, from));        // часть реплики внутри периода
    const tok = words(u.text);
    s.talk += d; s.count += 1; s.words += tok.length; s.fillers += countFillers(tok);
    s.durations.push(full); s.longest = Math.max(s.longest, full);
    if (u.confident) s.confident += 1;
    if (u.similarity != null && !u.new_speaker) s.sims.push(u.similarity);
    if (u.db != null) s.dbs.push(u.db);
    totalWords += tok.length;
    if (prev && prev.speaker !== u.speaker && u.gap != null) {
      s.gaps.push(u.gap); gapsAll.push(u.gap);
      if (u.interrupted) { s.interrupts += 1; get(prev.speaker).interrupted += 1; interruptions += 1; }
    }
    if (u.emotion) {
      for (const e of EMOTIONS) s.emo[e] += (u.emotion.probs[e] || 0) * d;
      s.emoTime += d;
      const p = u.emotion.probs;
      s.valence.push({ x: (u.start + u.end) / 2, raw: (p.positive || 0) - (p.angry || 0) - (p.sad || 0), u });
    }
    prev = u;
  }
  const speakers = [...per.values()].filter((s) => s.count > 0).sort((a, b) => a.id - b.id);
  const talkTotal = speakers.reduce((s, x) => s + x.talk, 0);
  const emo = { angry: 0, sad: 0, neutral: 0, positive: 0 };
  let emoTime = 0;
  for (const s of speakers) { for (const e of EMOTIONS) emo[e] += s.emo[e]; emoTime += s.emoTime; }
  return { speakers, talkTotal, totalWords, interruptions, gapsAll, list, now, from, span: Math.max(0, now - from), emo, emoTime };
}

function dominant(emo, emoTime) {
  return emoTime > 0 ? EMOTIONS.reduce((best, e) => (emo[e] > emo[best] ? e : best), "neutral") : null;
}
function histogram(values, edges) {
  const out = new Array(edges.length - 1).fill(0);
  for (const v of values) {
    for (let i = 0; i < edges.length - 1; i++) if (v >= edges[i] && v < edges[i + 1]) { out[i]++; break; }
  }
  return out;
}
function niceTop(v) {
  if (!(v > 0)) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}
function secParts(sec) {
  // для плиток: до 10 минут — секунды, дальше — минуты
  return sec < 600 ? [fmtNum(sec, sec < 10 && sec > 0 ? 1 : 0), "с"] : [fmtNum(sec / 60, 1), "мин"];
}

// --------------------------------------------------------------- рендер
export function renderDashboard(state) {
  const a = aggregate(state, view.window);
  const m = state.metrics[state.metrics.length - 1] || {};
  const emoColor = Object.fromEntries(EMOTIONS.map((e) => [e, cssVar(EMO_VAR[e])]));
  const tech = cssVar("--tech");
  const emoState = state.components.emo ? state.components.emo.state : "pending";

  renderHub(state, a, emoColor);
  renderRings(a, emoColor, emoState);
  renderSystem(state, a, m, tech);
  renderStats(state, a, m);
  renderLive(state, a, tech);
  renderMood(state, a, emoColor, emoState);
  renderDetails(state, a, m, emoColor, tech);
  renderLlm(state);
}

// ---- главный экран: плитки вокруг центра
function setTile(el, { cls, color, name, value, unit, note, pct, title }) {
  el.className = `tile slot-${el.dataset.slot}${cls ? " " + cls : ""}`;
  if (color) el.style.setProperty("--c", color); else el.style.removeProperty("--c");
  el.title = title || "";
  el.querySelector(".t-name").textContent = name;
  el.querySelector(".t-note").textContent = note || "";
  const val = el.querySelector(".t-val");
  val.replaceChildren(value, unit ? h("small", {}, unit) : "");
  el.querySelector(".t-pct").textContent = pct || "";
}

function renderHub(state, a, emoColor) {
  const sig = [Math.floor(a.span), a.speakers.map((s) => [s.id, s.name, Math.round(s.talk), s.count, dominant(s.emo, s.emoTime)]), state.utterances.length];
  draw("hub", sig, () => {
    const hub = document.getElementById("hub");
    SLOTS.forEach((slot, i) => {
      const el = hub.querySelector(`[data-slot="${slot}"]`), s = a.speakers[i];
      if (!s) return setTile(el, { cls: "is-empty", name: "Ожидание голоса", value: "—", note: `собеседник ${i + 1}` });
      const dom = dominant(s.emo, s.emoTime);
      const [v, unit] = secParts(s.talk);
      setTile(el, {
        cls: dom ? "" : "is-quiet", color: dom ? emoColor[dom] : "", name: s.name, value: v, unit,
        note: `${s.count} ${plural(s.count, "реплика", "реплики", "реплик")} · ${dom ? EMO_RU[dom].toLowerCase() : "эмоция не определена"}`,
        pct: a.span > 0 ? fmtPct(s.talk / a.span) : "",
        title: `${s.name}: ${fmtDur(s.talk)} речи`,
      });
    });
    const quiet = Math.max(0, a.span - a.talkTotal);
    const [qv, qu] = secParts(quiet);
    setTile(hub.querySelector('[data-slot="r3"]'), a.span > 0
      ? { cls: "is-quiet", name: "Паузы и тишина", value: qv, unit: qu, note: "без речи", pct: fmtPct(quiet / a.span) }
      : { cls: "is-empty", name: "Паузы и тишина", value: "—", note: "без речи" });
    C.orb(hub, {
      label: view.window ? `Последние ${view.window === 60 ? "60 с" : "5 мин"}` : "Длительность",
      value: fmtClock(a.span),
      note: a.speakers.length
        ? `${a.speakers.length} ${plural(a.speakers.length, "собеседник", "собеседника", "собеседников")}\n${a.list.length} ${plural(a.list.length, "реплика", "реплики", "реплик")}`
        : "запись не начата",
    });
  });
}

// ---- кольца эмоций
function renderRings(a, emoColor, emoState) {
  draw("rings", [emoState, EMOTIONS.map((e) => Math.round(a.emo[e] * 10)), Math.round(a.emoTime)], () => {
    const box = document.getElementById("emoRings");
    if (a.emoTime <= 0) {
      const why = emoState === "ready" ? "Кольца заполнятся после первых реплик."
        : emoState === "off" ? "Определение эмоций отключено в настройках."
        : emoState === "error" ? "Модель эмоций не загрузилась — подробности в разделе «Модели»."
        : "Модель эмоций ещё загружается.";
      box.replaceChildren(h("div", { class: "chart-empty", style: { "grid-column": "1 / -1" } }, why));
      return;
    }
    const order = ["neutral", "positive", "angry", "sad"];
    const cells = order.map(() => h("div", {}));
    box.replaceChildren(...cells);
    order.forEach((e, i) => {
      C.ring(cells[i], {
        pct: a.emo[e] / a.emoTime, color: emoColor[e], name: EMO_RU[e],
        sub: `${fmtNum(a.emo[e])} из ${fmtNum(a.emoTime)} с`,
        tipContent: { title: EMO_RU[e], rows: [{ value: fmtPct(a.emo[e] / a.emoTime, 1), key: "доля времени речи" }, { value: fmtDur(a.emo[e]), key: "время" }] },
      });
    });
  });
}

// ---- малые кольца системы
function renderSystem(state, a, m, tech) {
  const total = (state.config.mem_total_mb || 0) / 1024, rss = (m.rss_mb || 0) / 1024;
  const all = state.utterances.length, conf = state.utterances.filter((u) => u.confident).length;
  const speech = m.audio_sec > 0 ? Math.min(1, (m.speech_sec || 0) / m.audio_sec) : 0;
  draw("sys", [Math.round(m.cpu_app || 0), Math.round(m.cpu_sys || 0), m.rss_mb, Math.round(speech * 100), conf, all, m.queue], () => {
    document.getElementById("sysRings").replaceChildren(
      C.miniRing({ pct: (m.cpu_app || 0) / 100, color: tech, name: "Процессор", sub: m.cpu_sys != null ? `вся система ${fmtNum(m.cpu_sys)}%` : "" }),
      C.miniRing({ pct: total ? rss / total : 0, color: tech, name: "Память", sub: total ? `${fmtNum(rss, 2)} из ${fmtNum(total, 1)} ГБ` : "" }),
      C.miniRing({ pct: speech, color: tech, name: "Речь в записи", sub: m.audio_sec != null ? `${fmtNum(m.speech_sec || 0)} из ${fmtNum(m.audio_sec)} с` : "" }),
      C.miniRing({ pct: all ? conf / all : 0, color: tech, name: "Голос определён уверенно", sub: all ? `${conf} из ${all} ${plural(all, "реплики", "реплик", "реплик")}` : "" })
    );
    document.getElementById("sysMeta").textContent = m.queue != null ? `в очереди: ${m.queue}` : "";
  });
}

// ---- строка показателей
function stat(label, value, unit, note) {
  return h("div", { class: "stat frame" },
    h("div", { class: "stat-label" }, label),
    h("div", { class: "stat-value" }, value, unit ? h("small", {}, unit) : ""),
    h("div", { class: "stat-note" }, note || ""));
}
function renderStats(state, a, m) {
  const lat = a.list.map((u) => u.latency_ms / 1000).filter((x) => isFinite(x));
  const rtf = a.list.map((u) => u.rtf).filter((x) => x > 0);
  const med = median(lat), p95 = quantile(lat, 0.95), rtfMed = median(rtf), gapMed = median(a.gapsAll);
  const wpm = a.talkTotal > 5 ? (a.totalWords / a.talkTotal) * 60 : null;
  draw("stats", [a.list.length, a.totalWords, a.interruptions, Math.round((gapMed || 0) * 10), Math.round((med || 0) * 10), Math.round((p95 || 0) * 10), Math.round((rtfMed || 0) * 100), Math.round(a.talkTotal)], () => {
    document.getElementById("statStrip").replaceChildren(
      stat("Реплик", fmtNum(a.list.length), "", a.list.length ? `в среднем ${fmtSec(a.talkTotal / a.list.length)}` : ""),
      stat("Слов", fmtNum(a.totalWords), "", wpm ? `темп ${fmtNum(wpm)} слов/мин` : ""),
      stat("Перебиваний", fmtNum(a.interruptions), "", "ответ без паузы"),
      stat("Пауза до ответа", gapMed == null ? "—" : fmtNum(gapMed, 1), gapMed == null ? "" : "с", "медиана"),
      stat("Задержка до текста", med == null ? "—" : fmtNum(med, 1), med == null ? "" : "с", p95 == null ? "медиана" : `медиана · 95% — до ${fmtSec(p95)}`),
      stat("Скорость распознавания", rtfMed ? `×${fmtNum(1 / rtfMed, 1)}` : "—", "", "к реальному времени")
    );
  });
}

// ---- график «в реальном времени» с вкладками
function renderLive(state, a, tech) {
  const box = document.getElementById("liveChart"), meta = document.getElementById("liveMeta");
  if (view.live === "level") {
    const now = state.levels.length ? state.levels[state.levels.length - 1].t : 0;
    const pts = state.levels.filter((p) => p.t >= now - 60);
    draw("live", ["level", Math.floor(now * 2), pts.length], () => {
      meta.textContent = "уровень сигнала, последние 60 с";
      if (pts.length < 2) return C.emptyState(box, "График появится после начала записи.");
      const bands = [];
      let start = null;
      for (const p of pts) {
        if (p.speaking && start == null) start = p.t;
        if (!p.speaking && start != null) { bands.push({ a: start, b: p.t, color: tech }); start = null; }
      }
      if (start != null) bands.push({ a: start, b: now, color: tech });
      C.lines(box, {
        series: [{ name: "уровень", color: tech, endDot: false, points: pts.map((p) => ({ x: p.t, y: p.db, tip: { title: fmtClock(p.t), rows: [{ value: `${fmtNum(p.db)} дБ`, key: p.speaking ? "речь" : "тишина" }] } })) }],
        xDomain: [Math.max(0, now - 60), Math.max(now, 60)], yDomain: [-70, 0], xFmt: fmtClock, area: true, height: 190, bands,
        yTicks: [{ v: 0, label: "0" }, { v: -20, label: "−20" }, { v: -40, label: "−40" }, { v: -60, label: "−60" }],
      });
    });
  } else if (view.live === "latency") {
    const pts = state.utterances.filter((u) => isFinite(u.latency_ms)).map((u) => ({ x: u.end, y: u.latency_ms / 1000, u }));
    draw("live", ["latency", pts.length], () => {
      meta.textContent = "от конца фразы до текста, по репликам";
      if (pts.length < 2) return C.emptyState(box, "Нужно хотя бы две реплики.");
      const top = niceTop(Math.max(...pts.map((p) => p.y)));
      C.lines(box, {
        series: [{ name: "задержка", color: tech, points: pts.map((p) => ({ x: p.x, y: p.y, tip: { title: `${state.speakerName(p.u.speaker)} · ${fmtClock(p.u.start)}`, rows: [{ value: fmtSec(p.y, 2), key: "задержка" }] } })) }],
        xDomain: [pts[0].x, Math.max(pts[pts.length - 1].x, pts[0].x + 10)], yDomain: [0, top], xFmt: fmtClock, area: true, height: 190,
        yTicks: [0, 0.5, 1].map((k) => ({ v: top * k, label: fmtNum(top * k, top < 4 ? 1 : 0) })),
      });
    });
  } else {
    const hist = state.metrics.slice(-120);
    const isCpu = view.live === "cpu";
    const vals = hist.map((x) => (isCpu ? x.cpu_app || 0 : (x.rss_mb || 0) / 1024));
    draw("live", [view.live, hist.length, Math.round((vals[vals.length - 1] || 0) * 100)], () => {
      meta.textContent = isCpu ? "доля всех ядер, последние 2 мин" : "память приложения, последние 2 мин";
      if (hist.length < 2) return C.emptyState(box, "Собираем данные…");
      const top = isCpu ? 100 : niceTop(Math.max(...vals));
      const n = vals.length;
      C.lines(box, {
        series: [{ name: isCpu ? "процессор" : "память", color: tech, points: vals.map((v, i) => ({ x: i - (n - 1), y: v, tip: { title: i === n - 1 ? "сейчас" : `${n - 1 - i} с назад`, rows: [{ value: isCpu ? `${fmtNum(v)}%` : `${fmtNum(v, 2)} ГБ` }] } })) }],
        xDomain: [-(Math.max(n, 30) - 1), 0], yDomain: [0, top], xFmt: (v) => (Math.round(v) === 0 ? "сейчас" : `${Math.round(v)} с`), area: true, height: 190,
        yTicks: [0, 0.5, 1].map((k) => ({ v: top * k, label: isCpu ? `${fmtNum(top * k)}` : fmtNum(top * k, 1) })),
      });
    });
  }
}

// ---- «настроение» с вкладками
function renderMood(state, a, emoColor, emoState) {
  const box = document.getElementById("moodChart"), legend = document.getElementById("moodLegend"), meta = document.getElementById("moodMeta");
  const none = () => {
    legend.replaceChildren();
    C.emptyState(box, emoState === "ready" ? "Появится, когда будут определены эмоции."
      : emoState === "off" ? "Определение эмоций отключено в настройках."
      : emoState === "error" ? "Модель эмоций не загрузилась." : "Модель эмоций ещё загружается.");
  };
  if (view.mood === "trend") {
    draw("mood", ["trend", emoState, a.from, a.speakers.map((s) => [s.name, s.valence.map((p) => Math.round(p.raw * 100))])], () => {
      meta.textContent = "выше нуля — позитив, ниже — негатив";
      const series = a.speakers.filter((s) => s.valence.length).map((s) => {
        let ema = null;
        return {
          name: s.name, color: s.color,
          points: s.valence.map((p) => {
            ema = ema == null ? p.raw : ema * 0.5 + p.raw * 0.5;
            return { x: p.x, y: ema, tip: { title: `${s.name} · ${fmtClock(p.u.start)}`, rows: [{ value: fmtNum(ema, 2), key: "настроение", color: s.color }, { value: EMO_RU[p.u.emotion.label], key: "эмоция реплики" }] } };
          }),
        };
      });
      if (!series.length) return none();
      C.lines(box, {
        series, xDomain: [a.from, Math.max(a.now, a.from + 10)], yDomain: [-1, 1], baseline: 0, xFmt: fmtClock, height: 190,
        yTicks: [{ v: 1, label: "позитив" }, { v: 0, label: "0" }, { v: -1, label: "негатив" }],
      });
      legend.replaceChildren(C.legend(series, { line: true }));
    });
  } else {
    draw("mood", ["share", emoState, a.speakers.map((s) => [s.name, EMOTIONS.map((e) => Math.round(s.emo[e] * 10))])], () => {
      meta.textContent = "доля времени речи с каждой эмоцией";
      const have = a.speakers.filter((s) => s.emoTime > 0);
      if (!have.length) return none();
      C.stackedBars(box, have.map((s) => ({ label: s.name, dot: s.color, segments: EMOTIONS.map((e) => ({ name: EMO_RU[e], value: s.emo[e], color: emoColor[e] })) })), { barHeight: 18, rowGap: 18 });
      legend.replaceChildren(C.legend(EMOTIONS.map((e) => ({ name: EMO_RU[e], color: emoColor[e] }))));
    });
  }
}

// ---- подробные разделы
function renderDetails(state, a, m, emoColor, tech) {
  const none = cssVar("--e-none");

  const tl = cards["c-timeline"];
  draw("timeline", [Math.ceil(a.now / 2), a.from, a.list.map((u) => [u.id, u.speaker, u.emotion && u.emotion.label]), a.speakers.map((s) => s.name)], () => {
    if (!a.list.length) { tl.extra.replaceChildren(); return C.emptyState(tl.body, "Лента появится с первой репликой."); }
    C.timeline(tl.body, {
      lanes: a.speakers.map((s) => ({ id: s.id, label: s.name, dot: s.color })),
      items: a.list.map((u) => ({
        lane: u.speaker, start: u.start, end: u.end, color: u.emotion ? emoColor[u.emotion.label] : none,
        tip: {
          title: `${state.speakerName(u.speaker)} · ${fmtClock(u.start)}–${fmtClock(u.end)}`,
          rows: [{ value: u.emotion ? EMO_RU[u.emotion.label] : "эмоция не определена", key: u.interrupted ? "перебил" : "" }],
          text: u.text.length > 140 ? u.text.slice(0, 140) + "…" : u.text,
        },
      })),
      tMin: a.from, tMax: Math.max(a.now, a.from + 10), xFmt: fmtClock,
    });
    tl.extra.replaceChildren(C.legend([...EMOTIONS.map((e) => ({ name: EMO_RU[e], color: emoColor[e] })), { name: "ещё не определена", color: none }]));
  });

  draw("speakers", a.speakers.map((s) => [s.name, s.count, Math.round(s.talk), s.words, s.fillers, s.interrupts, s.interrupted, s.sims.length, Math.round(s.emoTime)]), () => {
    const sp = cards["c-speakers"];
    if (!a.speakers.length) return C.emptyState(sp.body, "Таблица заполнится с первой репликой.");
    const rows = a.speakers.map((s) => {
      const dom = dominant(s.emo, s.emoTime);
      return [
        h("span", {}, h("span", { class: "dot", style: { "--c": s.color } }), s.name),
        fmtDur(s.talk), a.talkTotal ? fmtPct(s.talk / a.talkTotal) : "—", s.count, fmtSec(mean(s.durations)), fmtSec(s.longest),
        s.talk > 5 ? fmtNum((s.words / s.talk) * 60) : "—",
        s.interrupts, s.interrupted, s.gaps.length ? fmtSec(median(s.gaps)) : "—",
        s.words ? fmtNum((s.fillers / s.words) * 100, 1) : "—",
        dom ? EMO_RU[dom] : "—",
        s.sims.length ? fmtNum(mean(s.sims), 2) : "—",
        s.dbs.length ? `${fmtNum(mean(s.dbs))} дБ` : "—",
      ];
    });
    sp.body.replaceChildren(C.dataTable(
      ["Собеседник", "Время", "Доля", "Реплик", "Средняя", "Макс.", "Темп, сл./мин", "Перебил", "Перебили его", "Пауза до ответа", "Паразиты, %", "Преобладает", "Сходство голоса", "Громкость"],
      rows
    ));
  });

  const du = cards["c-durations"];
  const dEdges = [0, 1, 2, 4, 8, 15, Infinity], dBins = ["до 1", "1–2", "2–4", "4–8", "8–15", "15+"];
  draw("durations", a.speakers.map((s) => [s.name, histogram(s.durations, dEdges)]), () => {
    if (!a.speakers.length) { du.extra.replaceChildren(); return C.emptyState(du.body, "Нет данных."); }
    const series = a.speakers.map((s) => ({ name: s.name, color: s.color, values: histogram(s.durations, dEdges) }));
    C.columns(du.body, { bins: dBins, series, unit: " с" });
    du.extra.replaceChildren(series.length > 1 ? C.legend(series) : "");
    du.setTable(["Длина, с", ...series.map((s) => s.name)], dBins.map((b, i) => [b, ...series.map((s) => s.values[i])]));
  });

  const ga = cards["c-gaps"];
  const gEdges = [0, 0.3, 0.6, 1, 2, 5, Infinity], gBins = ["до 0,3", "0,3–0,6", "0,6–1", "1–2", "2–5", "5+"];
  draw("gaps", histogram(a.gapsAll, gEdges), () => {
    if (!a.gapsAll.length) return C.emptyState(ga.body, "Появится после первой смены собеседника.");
    const values = histogram(a.gapsAll, gEdges);
    C.columns(ga.body, { bins: gBins, series: [{ name: "пауз", color: tech, values }], unit: " с" });
    ga.setTable(["Пауза, с", "Сколько раз"], gBins.map((b, i) => [b, values[i]]));
  });

  const lat = a.list.map((u) => u.latency_ms / 1000).filter((x) => isFinite(x));
  const la = cards["c-latency"];
  const lEdges = [0, 0.75, 1, 1.5, 2, 3, 5, Infinity], lBins = ["до 0,75", "0,75–1", "1–1,5", "1,5–2", "2–3", "3–5", "5+"];
  draw("latency", histogram(lat, lEdges), () => {
    if (!lat.length) return C.emptyState(la.body, "Появится после первой реплики.");
    const values = histogram(lat, lEdges);
    C.columns(la.body, { bins: lBins, series: [{ name: "реплик", color: tech, values }], unit: " с" });
    la.setTable(["Задержка, с", "Реплик"], lBins.map((b, i) => [b, values[i]]));
  });

  const wo = cards["c-words"];
  const top = topWords(a.list.map((u) => u.text), 8);
  draw("words", top, () => {
    if (!top.length) return C.emptyState(wo.body, "Пока нет повторяющихся слов.");
    C.hbars(wo.body, top.map(([w, n]) => ({ label: w, value: n, color: tech, display: String(n) })), { barHeight: 10, rowGap: 12 });
  });

  const st = cards["c-stages"];
  const emoMs = a.list.filter((u) => u.emotion && u.emotion.ms != null).map((u) => u.emotion.ms);
  const stages = [
    ["Ожидание в очереди", mean(a.list.map((u) => u.queue_ms || 0))],
    ["Распознавание текста", mean(a.list.map((u) => u.asr_ms || 0))],
    ["Определение голоса", mean(a.list.map((u) => u.diar_ms || 0))],
    ["Эмоции", mean(emoMs)],
  ].filter(([, v]) => v != null);
  draw("stages", stages.map(([n, v]) => [n, Math.round(v)]), () => {
    if (!a.list.length) return C.emptyState(st.body, "Появится после первой реплики.");
    C.hbars(st.body, stages.map(([name, v]) => ({ label: name, value: v, color: tech, display: `${fmtNum(v)} мс` })), { barHeight: 12, rowGap: 16 });
    st.setTable(["Этап", "Среднее, мс"], stages.map(([n, v]) => [n, fmtNum(v)]));
  });

  draw("models", state.components, () => {
    const names = { ready: "готово", loading: "загрузка", pending: "ожидание", error: "ошибка", off: "выключено" };
    document.getElementById("modelsBox").replaceChildren(C.dataTable(
      ["Компонент", "Модель", "Состояние", "Подробности"],
      Object.values(state.components).map((c) => [
        c.title, h("span", { style: { "white-space": "normal" } }, c.model),
        h("span", { class: `st ${c.state}` }, names[c.state] || c.state),
        h("span", { style: { "white-space": "normal" } }, [c.detail, c.load_sec != null ? `загрузка ${fmtSec(c.load_sec)}` : ""].filter(Boolean).join(" · ") || "—"),
      ]),
      { text: true }
    ));
  });

  const cfg = state.config || {}, v = state.versions || {}, mic = state.mic || {};
  draw("params", [cfg, v, mic, state.threshold, state.split, state.wsState, m.bytes_in && Math.round(m.bytes_in / 104858), m.empty_segments], () => {
    const rows = [
      ["Звук для распознавания", "16 кГц, моно, 16 бит, порции по 100 мс"],
      ["Микрофон", mic.label || "—"],
      ["Частота устройства", mic.rate ? `${fmtNum(mic.rate / 1000, 1)} кГц` : "—"],
      ["Обработка звука браузером", mic.dsp ? "включена" : "выключена"],
      ["Принято звука", m.bytes_in != null ? `${fmtNum(m.bytes_in / 1048576, 1)} МБ` : "—"],
      ["Фрагментов без речи", m.empty_segments != null ? fmtNum(m.empty_segments) : "—"],
      ["Порог детектора речи", cfg.vad_threshold != null ? fmtNum(cfg.vad_threshold, 2) : "—"],
      ["Пауза для конца фразы", cfg.vad_min_silence != null ? `${fmtNum(cfg.vad_min_silence, 2)} с` : "—"],
      ["Самая длинная реплика", cfg.vad_max_speech != null ? `${fmtNum(cfg.vad_max_speech)} с` : "—"],
      ["Строгость разделения голосов", state.threshold != null ? fmtNum(state.threshold, 2) : "—"],
      ["Деление реплик без паузы", state.split ? "включено" : "выключено"],
      ["Собеседников не больше", cfg.max_speakers],
      ["Потоков на распознавание", cfg.asr_threads],
      ["Ядер процессора", cfg.cpu_count],
      ["Память контейнера", cfg.mem_total_mb ? `${fmtNum(cfg.mem_total_mb / 1024, 1)} ГБ` : "—"],
      ["Модель для разбора", cfg.llm_model],
      ["Версии", [v["sherpa-onnx"] && `sherpa-onnx ${v["sherpa-onnx"]}`, v.torch && `torch ${v.torch}`, v.ollama && `ollama ${v.ollama}`, v.python && `python ${v.python}`, v.arch].filter(Boolean).join(", ") || "—"],
      ["Соединение с сервером", state.wsState],
      ["Хранение данных", "только в памяти до закрытия вкладки"],
    ];
    document.getElementById("paramsBox").replaceChildren(...rows.map(([k, val]) => h("div", {}, h("dt", {}, k), h("dd", {}, val == null || val === "" ? "—" : String(val)))));
  });
}

// ------------------------------------------------------------ разбор LLM
function list(title, items) {
  if (!Array.isArray(items) || !items.length) return null;
  return h("div", { class: "llm-block" }, h("h4", {}, title), h("ul", {}, items.map((x) => h("li", {}, typeof x === "string" ? x : JSON.stringify(x)))));
}

function renderLlm(state) {
  const llm = state.llm, comp = state.components.llm || { state: "pending", detail: "" };
  draw("llm", [llm, comp, state.utterances.length > 0, state.stale], () => {
    const box = document.getElementById("llmBox");
    const canRun = comp.state === "ready" && state.utterances.length > 0 && llm.state !== "running" && !state.stale;
    const btn = h("button", { class: "btn btn-small btn-primary", disabled: !canRun, onclick: () => state.requestAnalysis() }, llm.state === "done" ? "Разобрать заново" : "Разобрать разговор");
    const status =
      state.stale ? "Связь прерывалась: сервер больше не помнит этот разговор."
      : comp.state === "ready" ? (state.utterances.length ? "" : "Нужна хотя бы одна реплика.")
      : comp.state === "off" ? "Разбор отключён в настройках."
      : comp.state === "error" ? `Модель недоступна: ${comp.detail}`
      : `Модель готовится: ${comp.detail || "ожидание"}`;
    const head = h("div", { class: "panel-head" }, h("span", { class: "chev", "aria-hidden": "true" }), h("h3", {}, "Разбор разговора"),
      h("div", { class: "panel-tools" }, h("span", { class: "muted small" }, status), btn));
    const body = [h("p", { class: "panel-sub" }, `Текст разговора отправляется локальной модели ${state.config.llm_model || ""} в соседнем контейнере. Наружу ничего не уходит.`)];

    if (llm.state === "running") {
      body.push(h("div", { class: "progress" }, h("i")), h("p", { class: "muted small" }, `Модель пишет разбор… ${llm.elapsed ? fmtSec(llm.elapsed, 0) : ""}${llm.chars ? ` · ${fmtNum(llm.chars)} символов` : ""}. На процессоре без видеокарты это занимает от десятков секунд до нескольких минут.`));
    } else if (llm.state === "error") {
      body.push(h("p", {}, `Не получилось: ${llm.message}`));
    } else if (llm.state === "done") {
      const r = llm.result;
      if (!r) {
        body.push(h("p", { class: "muted small" }, "Модель ответила не в ожидаемом формате — показываю ответ как есть."), h("div", { class: "llm-raw" }, llm.raw || ""));
      } else {
        const q = r.quality && typeof r.quality === "object" ? r.quality : null;
        const score = q ? Math.max(0, Math.min(5, Math.round(Number(q.score) || 0))) : 0;
        body.push(h("div", { class: "llm-grid" },
          h("div", { class: "llm-block llm-wide" }, h("h4", {}, r.topic ? `Суть · ${r.topic}` : "Суть"), h("p", { class: "llm-summary" }, r.summary || "—")),
          r.outcome ? h("div", { class: "llm-block" }, h("h4", {}, "Итог"), h("p", {}, r.outcome)) : null,
          q ? h("div", { class: "llm-block" }, h("h4", {}, "Качество разговора"), h("p", {}, h("span", { class: "llm-score", title: `${score} из 5` }, [1, 2, 3, 4, 5].map((i) => h("i", { class: i <= score ? "on" : "" }))), `${score} из 5`), h("p", { class: "muted small" }, q.comment || "")) : null,
          Array.isArray(r.speakers) && r.speakers.length ? h("div", { class: "llm-block" }, h("h4", {}, "Участники"), h("ul", {}, r.speakers.map((s) => h("li", {}, [s.name, s.role, s.tone].filter(Boolean).join(" — "))))) : null,
          list("Договорённости", r.agreements), list("Следующие шаги", r.next_steps), list("Риски и недовольство", r.risks)
        ));
      }
      const meta = [`${fmtSec(llm.elapsed, 0)}`, llm.tokens_out ? `${fmtNum(llm.tokens_out)} токенов ответа` : "", llm.tokens_per_sec ? `${fmtNum(llm.tokens_per_sec, 1)} токенов/с` : "", llm.tokens_in ? `запрос ${fmtNum(llm.tokens_in)} токенов` : "", llm.truncated ? "разговор длинный — в модель ушли начало и конец" : ""].filter(Boolean).join(" · ");
      body.push(h("p", { class: "muted small", style: { "margin-top": "12px" } }, meta));
    } else {
      body.push(h("p", { class: "muted" }, "Резюме, договорённости, риски и оценка разговора появятся здесь после нажатия кнопки."));
    }
    box.replaceChildren(head, h("div", { class: "panel-body" }, ...body.filter(Boolean)));
  });
}
