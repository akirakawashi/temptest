// Графики на чистом SVG — без внешних библиотек, работает офлайн.
// Оформление — «приборная панель»: тёмный фон, светящиеся линии фирменного зелёного,
// сегментные кольца. Подписи, легенды, подсказки и таблицы-двойники сохранены,
// чтобы значения читались не только по цвету.

import { cssVar, h } from "./util.js";

const NS = "http://www.w3.org/2000/svg";
export function svg(tag, attrs = {}, ...children) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null && v !== false) el.setAttribute(k, v);
  for (const c of children.flat()) if (c != null && c !== false) el.append(c.nodeType ? c : document.createTextNode(String(c)));
  return el;
}

const measureCtx = document.createElement("canvas").getContext("2d");
function textWidth(text, px = 12.5, weight = 400) {
  measureCtx.font = `${weight} ${px}px ${weight >= 600 ? "Montserrat" : "Inter"}, "Segoe UI", Arial, sans-serif`;
  return measureCtx.measureText(String(text)).width;
}
function ellipsize(text, maxPx, px = 12.5) {
  text = String(text);
  if (textWidth(text, px) <= maxPx) return text;
  let s = text;
  while (s.length > 1 && textWidth(s + "…", px) > maxPx) s = s.slice(0, -1);
  return s + "…";
}

// ------------------------------------------------------------- подсказка
const tipEl = () => document.getElementById("tooltip");
export const tip = {
  show(x, y, { title, rows = [], text } = {}) {
    const el = tipEl();
    el.replaceChildren();
    if (title) el.append(h("div", { class: "tt-title" }, title));
    for (const r of rows) {
      const key = h("span", { class: "tt-key" });
      if (r.color) key.append(h("i", { style: { "--c": r.color } }));
      key.append(r.key || "");
      el.append(h("div", { class: "tt-row" }, h("b", {}, r.value), key));
    }
    if (text) el.append(h("div", { class: "tt-text" }, text));
    el.hidden = false;
    const w = el.offsetWidth, hgt = el.offsetHeight;
    let left = x + 14, top = y + 14;
    if (left + w > window.innerWidth - 8) left = x - w - 14;
    if (top + hgt > window.innerHeight - 8) top = y - hgt - 14;
    el.style.left = `${Math.max(8, left)}px`;
    el.style.top = `${Math.max(8, top)}px`;
  },
  hide() {
    tipEl().hidden = true;
  },
};
function hover(el, content) {
  el.classList.add("mark");
  el.setAttribute("tabindex", "0");
  el.addEventListener("pointermove", (e) => tip.show(e.clientX, e.clientY, content()));
  el.addEventListener("pointerleave", tip.hide);
  el.addEventListener("focus", () => {
    const r = el.getBoundingClientRect();
    tip.show(r.left + r.width / 2, r.top + r.height / 2, content());
  });
  el.addEventListener("blur", tip.hide);
  return el;
}

// ------------------------------------------------------------ служебное
function niceMax(v) {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}
function ticksFor(max, count = 4) {
  const top = niceMax(max);
  let step = top / count;
  if (Number.isInteger(max) && step < 1) step = 1;
  const out = [];
  for (let v = 0; v <= top + 1e-9; v += step) out.push(Math.round(v * 1000) / 1000);
  return { top, ticks: out };
}
function barPath(x, y, w, hgt, r, side) {
  // прямоугольник со скруглением только на «конце данных»: side = right | top
  r = Math.max(0, Math.min(r, w / 2, hgt / 2));
  if (side === "right") return `M${x},${y}h${w - r}a${r},${r} 0 0 1 ${r},${r}v${hgt - 2 * r}a${r},${r} 0 0 1 ${-r},${r}h${-(w - r)}z`;
  if (side === "top") return `M${x},${y + hgt}v${-(hgt - r)}a${r},${r} 0 0 1 ${r},${-r}h${w - 2 * r}a${r},${r} 0 0 1 ${r},${r}v${hgt - r}z`;
  return `M${x},${y}h${w}v${hgt}h${-w}z`;
}
/** Тёмный или белый текст — что лучше читается на данной заливке. */
export function inkOn(color) {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(color).trim());
  if (!m) return "#fff";
  const n = parseInt(m[1], 16);
  const lin = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
  const L = 0.2126 * lin((n >> 16) & 255) + 0.7152 * lin((n >> 8) & 255) + 0.0722 * lin(n & 255);
  return L > 0.36 ? "#0A0D12" : "#fff";
}
function mount(el, node) {
  el.replaceChildren(node);
}
export function emptyState(el, text) {
  el.replaceChildren(h("div", { class: "chart-empty" }, text));
}
export function legend(items, { line = false } = {}) {
  return h("div", { class: "legend" }, items.map((it) => h("span", {}, h("i", { class: line ? "line" : "", style: { "--c": it.color } }), it.name)));
}

// --------------------------------------------------- горизонтальные столбцы
// rows: [{label, value, color, display, dot, tip}]
export function hbars(el, rows, { max, labelWidth, barHeight = 14, rowGap = 12 } = {}) {
  const W = el.clientWidth || 400;
  const lw = labelWidth || Math.min(170, Math.max(60, ...rows.map((r) => textWidth(r.label) + (r.dot ? 16 : 0))) + 10);
  const vw = Math.max(40, ...rows.map((r) => textWidth(r.display ?? r.value, 12.5, 600))) + 10;
  const plot = Math.max(40, W - lw - vw);
  const top = max || Math.max(1e-9, ...rows.map((r) => r.value));
  const rowH = barHeight + rowGap;
  const H = rows.length * rowH;
  const root = svg("svg", { class: "chart", width: W, height: H, role: "img" });
  rows.forEach((r, i) => {
    const y = i * rowH + rowGap / 2;
    if (r.dot) root.append(svg("circle", { cx: 5, cy: y + barHeight / 2, r: 4.5, fill: r.dot }));
    root.append(svg("text", { class: "lbl", x: r.dot ? 16 : 0, y: y + barHeight / 2, "dominant-baseline": "central" }, ellipsize(r.label, lw - (r.dot ? 22 : 6))));
    const w = Math.max(r.value > 0 ? 2 : 0, (r.value / top) * plot);
    const bar = svg("path", { d: barPath(lw, y, w, barHeight, 4, "right"), fill: r.color, class: "glow", style: `color:${r.color}` });
    hover(bar, () => r.tip || { title: r.label, rows: [{ value: r.display ?? r.value }] });
    root.append(bar);
    root.append(svg("text", { class: "val", x: lw + w + 7, y: y + barHeight / 2, "dominant-baseline": "central" }, r.display ?? r.value));
  });
  mount(el, root);
}

// --------------------------------------------- составные столбцы (доли, 100%)
// rows: [{label, dot, note, segments:[{name, value, color}]}]
export function stackedBars(el, rows, { barHeight = 16, rowGap = 14, fmt = (v) => `${Math.round(v * 100)}%` } = {}) {
  const W = el.clientWidth || 400;
  const lw = Math.min(170, Math.max(60, ...rows.map((r) => textWidth(r.label) + 16)) + 10);
  const nw = Math.max(0, ...rows.map((r) => (r.note ? textWidth(r.note) + 10 : 0)));
  const plot = Math.max(40, W - lw - nw);
  const rowH = barHeight + rowGap;
  const root = svg("svg", { class: "chart", width: W, height: rows.length * rowH, role: "img" });
  rows.forEach((r, i) => {
    const y = i * rowH + rowGap / 2;
    const total = r.segments.reduce((s, g) => s + g.value, 0);
    if (r.dot) root.append(svg("circle", { cx: 5, cy: y + barHeight / 2, r: 4.5, fill: r.dot }));
    root.append(svg("text", { class: "lbl", x: 16, y: y + barHeight / 2, "dominant-baseline": "central" }, ellipsize(r.label, lw - 22)));
    if (r.note) root.append(svg("text", { x: W, y: y + barHeight / 2, "text-anchor": "end", "dominant-baseline": "central" }, r.note));
    if (total <= 0) {
      root.append(svg("rect", { x: lw, y: y + barHeight / 2 - 0.5, width: plot, height: 1, fill: cssVar("--grid") }));
      return;
    }
    const visible = r.segments.filter((g) => g.value > 0);
    let x = lw;
    visible.forEach((g, k) => {
      const share = g.value / total;
      const full = share * plot;
      const last = k === visible.length - 1;
      const w = Math.max(1, full - (last ? 0 : 2));          // зазор 2 px между сегментами
      const seg = svg("path", { d: barPath(x, y, w, barHeight, 4, last ? "right" : "none"), fill: g.color });
      hover(seg, () => ({ title: r.label, rows: [{ value: fmt(share), key: g.name, color: g.color }], text: g.detail }));
      root.append(seg);
      const label = fmt(share);
      if (w > textWidth(label, 11.5, 600) + 12) {
        // подпись внутри сегмента — только если помещается с запасом
        root.append(svg("text", { x: x + w / 2, y: y + barHeight / 2, "text-anchor": "middle", "dominant-baseline": "central", style: `fill:${g.ink || inkOn(g.color)};font-weight:600;font-size:11.5px;pointer-events:none` }, label));
      }
      x += full;
    });
  });
  mount(el, root);
}

// ------------------------------------------------ гистограмма (столбцы вверх)
// bins: ["0–1 с", ...]; series: [{name, color, values:[...]}] — значения складываются в стопку
export function columns(el, { bins, series, height = 190, unit = "", integer = true }) {
  const W = el.clientWidth || 400;
  const totals = bins.map((_, i) => series.reduce((s, sr) => s + (sr.values[i] || 0), 0));
  const { top, ticks } = ticksFor(Math.max(1, ...totals), 4);
  const padL = Math.max(24, textWidth(top, 11.5) + 10), padB = 24, padT = 16;
  const plotW = W - padL - 4, plotH = height - padB - padT;
  const band = plotW / bins.length;
  const bw = Math.min(24, band * 0.62);
  const root = svg("svg", { class: "chart", width: W, height, role: "img" });
  const y = (v) => padT + plotH - (v / top) * plotH;
  for (const t of ticks) {
    if (integer && !Number.isInteger(t)) continue;
    root.append(svg("line", { class: t === 0 ? "axisline" : "gridline", x1: padL, x2: W - 4, y1: y(t), y2: y(t) }));
    root.append(svg("text", { class: "tick", x: padL - 6, y: y(t), "text-anchor": "end", "dominant-baseline": "central" }, t));
  }
  bins.forEach((b, i) => {
    const cx = padL + band * i + band / 2;
    root.append(svg("text", { x: cx, y: height - 6, "text-anchor": "middle" }, b));
    let acc = 0;
    const parts = series.map((sr) => ({ sr, v: sr.values[i] || 0 })).filter((p) => p.v > 0);
    parts.forEach((p, k) => {
      const y0 = y(acc), y1 = y(acc + p.v);
      acc += p.v;
      const last = k === parts.length - 1;
      const hgt = Math.max(1, y0 - y1 - (last ? 0 : 2));
      const rect = svg("path", { d: barPath(cx - bw / 2, y0 - hgt, bw, hgt, 4, last ? "top" : "none"), fill: p.sr.color, class: "glow", style: `color:${p.sr.color}` });
      hover(rect, () => ({ title: `${b}${unit}`, rows: [{ value: p.v, key: p.sr.name, color: series.length > 1 ? p.sr.color : null }] }));
      root.append(rect);
    });
    if (totals[i] > 0) root.append(svg("text", { class: "val", x: cx, y: y(totals[i]) - 6, "text-anchor": "middle", style: "font-size:11.5px" }, totals[i]));
  });
  mount(el, root);
}

// -------------------------------------------------------------- линии
// series: [{name, color, points:[{x, y, tip}]}]; yTicks: [{v, label}]
export function lines(el, { series, xDomain, yDomain, yTicks, xFmt = (v) => v, height = 200, baseline = null, area = false, bands = [] }) {
  const W = el.clientWidth || 400;
  const padL = Math.max(34, ...yTicks.map((t) => textWidth(t.label, 11.5) + 10)), padR = 12, padT = 10, padB = 24;
  const plotW = W - padL - padR, plotH = height - padT - padB;
  const [x0, x1] = xDomain, [y0, y1] = yDomain;
  const X = (v) => padL + ((v - x0) / Math.max(1e-9, x1 - x0)) * plotW;
  const Y = (v) => padT + plotH - ((Math.max(y0, Math.min(y1, v)) - y0) / (y1 - y0)) * plotH;
  const root = svg("svg", { class: "chart", width: W, height, role: "img" });
  for (const t of yTicks) {
    root.append(svg("line", { class: baseline === t.v ? "axisline" : "gridline", x1: padL, x2: W - padR, y1: Y(t.v), y2: Y(t.v) }));
    root.append(svg("text", { class: "tick", x: padL - 6, y: Y(t.v), "text-anchor": "end", "dominant-baseline": "central" }, t.label));
  }
  const nx = Math.max(2, Math.min(6, Math.floor(plotW / 90)));
  for (let i = 0; i <= nx; i++) {
    const v = x0 + ((x1 - x0) * i) / nx;
    root.append(svg("text", { class: "tick", x: X(v), y: height - 6, "text-anchor": i === 0 ? "start" : i === nx ? "end" : "middle" }, xFmt(v)));
  }
  for (const b of bands) {
    const a = Math.max(x0, b.a), z = Math.min(x1, b.b);
    if (z > a) root.append(svg("rect", { x: X(a), y: padT + plotH + 2, width: Math.max(1, X(z) - X(a)), height: 3, rx: 1.5, fill: b.color }));
  }
  const all = [];
  for (const s of series) {
    if (!s.points.length) continue;
    const d = s.points.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join("");
    if (area) {
      const base = Y(y0);
      root.append(svg("path", { d: `${d}L${X(s.points[s.points.length - 1].x).toFixed(1)},${base}L${X(s.points[0].x).toFixed(1)},${base}Z`, fill: s.color, opacity: 0.1 }));
    }
    root.append(svg("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round", class: "glow", style: `color:${s.color}` }));
    if (s.endDot !== false) {
      const p = s.points[s.points.length - 1];
      root.append(svg("circle", { cx: X(p.x), cy: Y(p.y), r: 4, fill: s.color, stroke: cssVar("--surface"), "stroke-width": 2 }));
    }
    for (const p of s.points) all.push({ ...p, s });
  }
  // перекрестие: ищем ближайшую по времени точку, наводиться на саму линию не нужно
  if (all.length) {
    const cross = svg("line", { class: "crosshair", y1: padT, y2: padT + plotH, visibility: "hidden" });
    const dot = svg("circle", { r: 4.5, stroke: cssVar("--surface"), "stroke-width": 2, visibility: "hidden" });
    const hit = svg("rect", { x: padL, y: padT, width: plotW, height: plotH, fill: "transparent" });
    hit.addEventListener("pointermove", (e) => {
      const box = root.getBoundingClientRect();
      const px = e.clientX - box.left, py = e.clientY - box.top;
      let best = null, bd = 1e12;
      for (const p of all) {
        const d = Math.abs(X(p.x) - px) * 3 + Math.abs(Y(p.y) - py);
        if (d < bd) { bd = d; best = p; }
      }
      cross.setAttribute("x1", X(best.x)); cross.setAttribute("x2", X(best.x)); cross.setAttribute("visibility", "visible");
      dot.setAttribute("cx", X(best.x)); dot.setAttribute("cy", Y(best.y)); dot.setAttribute("fill", best.s.color); dot.setAttribute("visibility", "visible");
      tip.show(e.clientX, e.clientY, best.tip || { title: xFmt(best.x), rows: [{ value: best.y, key: best.s.name, color: best.s.color }] });
    });
    hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); tip.hide(); });
    root.append(cross, dot, hit);
  }
  mount(el, root);
}

// ---------------------------------------------------- лента разговора
// lanes: [{id, label, dot}], items: [{lane, start, end, color, tip}]
export function timeline(el, { lanes, items, tMin = 0, tMax, xFmt, laneH = 28 }) {
  const W = el.clientWidth || 600;
  const lw = Math.min(170, Math.max(70, ...lanes.map((l) => textWidth(l.label) + 16)) + 12);
  const padR = 10, padB = 22;
  const plotW = W - lw - padR;
  const H = lanes.length * laneH + padB + 4;
  const span = Math.max(1, tMax - tMin);
  const X = (t) => lw + ((Math.max(tMin, Math.min(tMax, t)) - tMin) / span) * plotW;
  const root = svg("svg", { class: "chart", width: W, height: H, role: "img" });
  const nx = Math.max(2, Math.min(8, Math.floor(plotW / 110)));
  for (let i = 0; i <= nx; i++) {
    const t = tMin + (span * i) / nx;
    root.append(svg("line", { class: "gridline", x1: X(t), x2: X(t), y1: 0, y2: lanes.length * laneH }));
    root.append(svg("text", { class: "tick", x: X(t), y: H - 5, "text-anchor": i === 0 ? "start" : i === nx ? "end" : "middle" }, xFmt(t)));
  }
  const row = new Map();
  lanes.forEach((l, i) => {
    row.set(l.id, i);
    const cy = i * laneH + laneH / 2;
    root.append(svg("circle", { cx: 5, cy, r: 4.5, fill: l.dot }));
    root.append(svg("text", { class: "lbl", x: 16, y: cy, "dominant-baseline": "central" }, ellipsize(l.label, lw - 24)));
  });
  for (const it of items) {
    const i = row.get(it.lane);
    if (i == null) continue;
    const x = X(it.start), w = Math.max(3, X(it.end) - X(it.start) - 2);      // зазор 2 px между соседними
    if (it.end <= tMin) continue;
    const rect = svg("rect", { x, y: i * laneH + (laneH - 14) / 2, width: w, height: 14, rx: 3, fill: it.color, class: "glow", style: `color:${it.color}` });
    hover(rect, () => it.tip);
    root.append(rect);
  }
  mount(el, root);
}

// ------------------------------------------------- сегментное кольцо
function segments(root, { cx, cy, r1, r2, n, filled, color, dim, width }) {
  for (let i = 0; i < n; i++) {
    const on = i < filled;
    root.append(svg("line", {
      x1: cx, y1: cy - r1, x2: cx, y2: cy - r2, transform: `rotate(${(i * 360) / n} ${cx} ${cy})`,
      stroke: on ? color : dim, "stroke-width": width, "stroke-linecap": "butt", opacity: on ? 1 : 0.5,
    }));
  }
}

/** Большое кольцо: доля в процентах сегментами, в центре — число и подпись. */
export function ring(el, { pct, color, name, sub, tipContent }) {
  const p = Math.max(0, Math.min(1, pct || 0));
  const n = 44, filled = Math.round(p * n);
  const root = svg("svg", { viewBox: "0 0 200 200", role: "img", "aria-label": `${name}: ${Math.round(p * 100)}%` });
  root.append(svg("circle", { cx: 100, cy: 100, r: 95, fill: "none", stroke: color, "stroke-width": 2.5, class: "glow", style: `color:${color}` }));
  segments(root, { cx: 100, cy: 100, r1: 68, r2: 87, n, filled, color, dim: cssVar("--e-none"), width: 7 });
  root.append(svg("circle", { cx: 100, cy: 100, r: 58, fill: "none", stroke: color, "stroke-width": 1, opacity: 0.45 }));
  const hit = svg("circle", { cx: 100, cy: 100, r: 96, fill: "transparent" });
  if (tipContent) hover(hit, () => tipContent);
  root.append(hit);
  mount(el, h("div", { class: "ring", style: { "--c": color } },
    root,
    h("div", { class: "ring-text" }, h("div", { class: "ring-pct" }, `${Math.round(p * 100)}%`), h("div", { class: "ring-name" }, name)),
    h("div", { class: "ring-sub" }, sub || "\u00a0")));
}

/** Малое кольцо для системных показателей: кольцо слева, число и подписи справа. */
export function miniRing({ pct, color, name, sub, valueText }) {
  const p = Math.max(0, Math.min(1, pct || 0));
  const n = 28, filled = Math.round(p * n);
  const root = svg("svg", { viewBox: "0 0 100 100", role: "img", "aria-label": `${name}: ${valueText || Math.round(p * 100) + "%"}` });
  root.append(svg("circle", { cx: 50, cy: 50, r: 47, fill: "none", stroke: color, "stroke-width": 2, class: "glow", style: `color:${color}` }));
  segments(root, { cx: 50, cy: 50, r1: 32, r2: 42, n, filled, color, dim: cssVar("--e-none"), width: 5 });
  root.append(svg("circle", { cx: 50, cy: 50, r: 26, fill: "none", stroke: color, "stroke-width": 1, opacity: 0.4 }));
  return h("div", { class: "mini" }, root,
    h("div", {}, h("div", { class: "mini-pct" }, valueText || `${Math.round(p * 100)}%`), h("div", { class: "mini-name" }, name), h("div", { class: "mini-sub" }, sub || "\u00a0")));
}

/** Центральный круг главного экрана: вращающиеся дуги и итог в середине. */
export function orb(el, { label, value, note }) {
  const c = cssVar("--brand"), deep = cssVar("--deep");
  let box = el.querySelector(".orb");
  if (!box) {
    const root = svg("svg", { viewBox: "0 0 300 300", "aria-hidden": "true" });
    const defs = svg("defs", {});
    const grad = svg("radialGradient", { id: "orbFill", cx: "50%", cy: "45%", r: "60%" });
    grad.append(svg("stop", { offset: "0%", "stop-color": c, "stop-opacity": 0.42 }), svg("stop", { offset: "70%", "stop-color": deep, "stop-opacity": 0.9 }), svg("stop", { offset: "100%", "stop-color": deep, "stop-opacity": 0.2 }));
    defs.append(grad);
    root.append(defs);
    root.append(svg("circle", { cx: 150, cy: 150, r: 146, fill: "none", stroke: c, "stroke-width": 1, opacity: 0.35 }));
    root.append(svg("circle", { class: "spin", cx: 150, cy: 150, r: 136, fill: "none", stroke: c, "stroke-width": 3, "stroke-dasharray": "2 9", opacity: 0.8 }));
    root.append(svg("circle", { class: "spin rev glow", style: `color:${c}`, cx: 150, cy: 150, r: 122, fill: "none", stroke: c, "stroke-width": 5, "stroke-dasharray": "150 62 40 62 90 62 20 281", "stroke-linecap": "round" }));
    root.append(svg("circle", { class: "spin", cx: 150, cy: 150, r: 110, fill: "none", stroke: c, "stroke-width": 1.5, "stroke-dasharray": "60 24 12 24", opacity: 0.6 }));
    root.append(svg("circle", { cx: 150, cy: 150, r: 98, fill: "url(#orbFill)", stroke: c, "stroke-width": 2, class: "glow", style: `color:${c}` }));
    box = h("div", { class: "orb" }, root, h("div", { class: "orb-text" }, h("div", { class: "orb-label" }), h("div", { class: "orb-value" }), h("div", { class: "orb-note" })));
    el.append(box);
  }
  box.querySelector(".orb-label").textContent = label;
  box.querySelector(".orb-value").textContent = value;
  box.querySelector(".orb-note").textContent = note || "";
  return box;
}

// ------------------------------------------------ таблица-двойник графика
export function dataTable(headers, rows, { text = false } = {}) {
  const t = h("table", { class: text ? "data is-text" : "data" });
  t.append(h("thead", {}, h("tr", {}, headers.map((x) => h("th", {}, x)))));
  const body = h("tbody");
  for (const r of rows) {
    body.append(h("tr", {}, r.map((c) => (c && c.nodeType ? h("td", {}, c) : h("td", {}, c == null ? "—" : c)))));
  }
  t.append(body);
  return h("div", { class: "table-scroll" }, t);
}
