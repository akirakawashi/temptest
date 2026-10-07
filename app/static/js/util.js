// Общие мелочи: форматирование, слова, статистика.

export const EMOTIONS = ["angry", "sad", "neutral", "positive"];
export const EMO_RU = { angry: "Раздражение", sad: "Грусть", neutral: "Нейтрально", positive: "Позитив" };
export const EMO_VAR = { angry: "--e-angry", sad: "--e-sad", neutral: "--e-neutral", positive: "--e-positive" };

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
export function speakerVar(id) {
  return `--s${((id - 1) % 5) + 1}`;
}

export function fmtClock(sec) {
  sec = Math.max(0, Math.floor(sec || 0));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  const mm = String(m).padStart(2, "0"), ss = String(s).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}
export function fmtNum(x, digits = 0) {
  if (x == null || !isFinite(x)) return "—";
  return x.toLocaleString("ru-RU", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
export function fmtSec(x, digits = 1) {
  return x == null || !isFinite(x) ? "—" : `${fmtNum(x, digits)} с`;
}
export function fmtDur(sec) {
  if (sec == null || !isFinite(sec)) return "—";
  if (sec < 60) return `${fmtNum(sec, sec < 10 ? 1 : 0)} с`;
  const m = Math.floor(sec / 60), s = Math.round(sec % 60);
  return `${m} мин ${s} с`;
}
export function fmtPct(x, digits = 0) {
  return x == null || !isFinite(x) ? "—" : `${fmtNum(x * 100, digits)}%`;
}

export function median(arr) {
  if (!arr.length) return null;
  const a = [...arr].sort((x, y) => x - y);
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}
export function quantile(arr, q) {
  if (!arr.length) return null;
  const a = [...arr].sort((x, y) => x - y);
  const pos = (a.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return a[lo] + (a[hi] - a[lo]) * (pos - lo);
}
export function mean(arr) {
  return arr.length ? arr.reduce((s, x) => s + x, 0) / arr.length : null;
}

// ------------------------------------------------------------------ слова
const WORD_RE = /[a-zа-яё0-9]+(?:-[a-zа-яё0-9]+)*/gi;
export function words(text) {
  return (text || "").toLowerCase().replace(/ё/g, "е").match(WORD_RE) || [];
}

// Слова-паразиты: одиночные слова и устойчивые сочетания.
const FILLER_WORDS = new Set(["ну", "вот", "типа", "короче", "значит", "блин", "э", "ээ", "эм", "эмм", "мм", "ммм", "а-а", "э-э", "собственно", "допустим", "слушай", "слушайте", "понимаешь", "понимаете"]);
const FILLER_PHRASES = [["как", "бы"], ["это", "самое"], ["так", "сказать"], ["в", "общем"], ["в", "принципе"], ["на", "самом", "деле"], ["скажем", "так"], ["по", "сути"]];

export function countFillers(tokens) {
  let n = 0;
  for (let i = 0; i < tokens.length; i++) {
    if (FILLER_WORDS.has(tokens[i])) { n++; continue; }
    for (const ph of FILLER_PHRASES) {
      if (ph.every((w, k) => tokens[i + k] === w)) { n++; i += ph.length - 1; break; }
    }
  }
  return n;
}

const STOP = new Set(("и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об другой хоть после над больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между это эта так вас ваш ваша ваше наш наша наше мои твой очень просто давайте давай пожалуйста спасибо здравствуйте день добрый").split(" "));

export function topWords(texts, limit = 10) {
  const freq = new Map();
  for (const t of texts) {
    for (const w of words(t)) {
      if (w.length < 4 || STOP.has(w) || /^\d+$/.test(w)) continue;
      freq.set(w, (freq.get(w) || 0) + 1);
    }
  }
  return [...freq.entries()].filter(([, n]) => n >= 2).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, limit);
}

export function plural(n, one, few, many) {
  const a = Math.abs(n) % 100, b = a % 10;
  if (a > 10 && a < 20) return many;
  if (b > 1 && b < 5) return few;
  if (b === 1) return one;
  return many;
}

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style" && typeof v === "object") for (const [p, val] of Object.entries(v)) el.style.setProperty(p, val);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    el.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return el;
}
