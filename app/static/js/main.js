// Точка входа: соединение с сервером, запись с микрофона, чат и боковая панель.

import { MicCapture } from "./audio.js";
import { initDashboard, renderDashboard, invalidateDashboard } from "./dashboard.js";
import { EMO_RU, EMO_VAR, cssVar, speakerVar, fmtClock, fmtNum, fmtSec, fmtDur, fmtPct, median, h, plural } from "./util.js";

const $ = (id) => document.getElementById(id);

const state = {
  ws: null, wsState: "подключение…",
  recording: false,
  components: {}, versions: {}, config: {},
  utterances: [], names: new Map(),
  live: { speaking: false, speaker: null },
  levels: [], metrics: [],
  audioSec: 0,
  llm: { state: "idle" },
  mic: {},
  threshold: null, split: true,
  stale: false,          // разговор остался на экране, но сервер его уже не помнит
  view: "chat",
  speakerName(id) { return this.names.get(id) || `Собеседник ${id}`; },
  requestAnalysis() { send({ type: "analyze" }); state.llm = { state: "running", chars: 0 }; schedule(); },
};
const mic = new MicCapture();
let dirty = true, lastDash = 0, reconnectDelay = 1000, userStopped = true;

// ------------------------------------------------------------ соединение
function connect() {
  const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
  const ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";
  state.ws = ws;
  ws.onopen = () => {
    state.wsState = "установлено";
    reconnectDelay = 1000;
    schedule();
  };
  ws.onmessage = (e) => handle(JSON.parse(e.data));
  ws.onclose = () => {
    state.wsState = "потеряно, переподключение…";
    if (state.recording) stopRecording(false);
    if (state.utterances.length) showBanner("Соединение с сервером прервалось, переподключаюсь. Текст на экране сохранён.", true);
    updateControls();
    setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(10000, reconnectDelay * 1.7);
  };
  ws.onerror = () => ws.close();
}
function send(obj) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify(obj));
}

function handle(msg) {
  switch (msg.type) {
    case "hello":
      state.components = msg.components; state.versions = msg.versions; state.config = msg.config;
      state.threshold = msg.config.speaker_threshold; state.split = msg.config.split_turns;
      $("thrRange").value = state.threshold; $("thrValue").textContent = fmtNum(state.threshold, 2);
      $("splitToggle").checked = state.split;
      if (state.utterances.length) {
        // Переподключение: сервер начал с чистого листа, но текст на экране терять незачем.
        state.stale = true;
        state.live = { speaking: false, speaker: null };
        renderTyping();
        showBanner("Связь с сервером восстановлена. Прежний разговор оставлен на экране только для чтения; новая запись начнёт разговор заново.", false);
      } else {
        clearConversation();
      }
      break;
    case "status":
      state.components = msg.components; state.versions = msg.versions || state.versions;
      break;
    case "config":
      state.threshold = msg.speaker_threshold; state.split = msg.split_turns;
      break;
    case "utterance":
      state.utterances.push(msg);
      state.live.speaker = null;
      appendMessage(msg);
      break;
    case "emotion": {
      const u = state.utterances.find((x) => x.id === msg.id);
      if (u) { u.emotion = { probs: msg.probs, label: msg.label, ms: msg.ms }; updateMessageEmotion(u); }
      break;
    }
    case "relabel":
      for (const u of state.utterances) if (u.speaker === msg.from) u.speaker = msg.to;
      state.names.delete(msg.from);       // номер освободился и может достаться другому голосу
      rebuildChat();
      break;
    case "live":
      state.live.speaking = msg.speaking;
      if (!msg.speaking) state.live.speaker = null;
      else if (msg.speaker !== undefined) state.live.speaker = msg.speaker;
      renderTyping();
      break;
    case "level":
      state.audioSec = msg.t;
      state.levels.push(msg);
      if (state.levels.length > 1500) state.levels.splice(0, 300);
      renderMeter(msg);
      break;
    case "metrics":
      state.metrics.push(msg);
      if (state.metrics.length > 600) state.metrics.splice(0, 100);
      if (msg.audio_sec != null && !state.recording && !state.stale) state.audioSec = msg.audio_sec;
      break;
    case "llm":
      state.llm = msg.state === "running" ? { ...state.llm, ...msg } : msg;
      break;
    case "reset":
      clearConversation();
      break;
    case "error":
      showBanner(msg.message, true);
      break;
  }
  updateControls();
  schedule();
}

// --------------------------------------------------------------- запись
async function startRecording() {
  hideBanner();
  if (state.stale) clearConversation();
  try {
    const dsp = $("dspToggle").checked;
    await mic.start((buf) => {
      if (state.ws && state.ws.readyState === WebSocket.OPEN && state.ws.bufferedAmount < 2_000_000) state.ws.send(buf);
    }, { browserDsp: dsp });
    mic.onEnded = () => stopRecording(true);
    state.mic = { label: mic.deviceLabel, rate: mic.deviceRate, dsp };
    state.recording = true;
    userStopped = false;
    drawSpectrum();
  } catch (err) {
    await mic.stop();
    const denied = err && (err.name === "NotAllowedError" || err.name === "SecurityError");
    const missing = err && (err.name === "NotFoundError" || err.name === "OverconstrainedError");
    showBanner(
      denied ? "Браузер не дал доступ к микрофону. Разрешите его для этого сайта в настройках адресной строки и нажмите «Начать запись» ещё раз."
      : missing ? "Микрофон не найден. Проверьте, что он подключён и не занят другой программой."
      : `Не удалось включить микрофон: ${err && err.message ? err.message : err}`, true);
  }
  updateControls();
}
async function stopRecording(notifyServer = true) {
  state.recording = false;
  userStopped = true;
  await mic.stop();
  if (notifyServer) send({ type: "stop" });
  state.live = { speaking: false, speaker: null };
  renderTyping();
  renderMeter(null);
  updateControls();
}

// ------------------------------------------------------------------ чат
const chat = $("chat");
function nearBottom() {
  return chat.scrollHeight - chat.scrollTop - chat.clientHeight < 140;
}
function emotionChip(u) {
  if (!u.emotion) return null;
  const p = u.emotion.probs[u.emotion.label];
  return h("span", { class: "chip", "data-role": "emo", title: `Определено по голосу, уверенность ${fmtPct(p)}` },
    h("span", { class: "dot", style: { "--c": `var(${EMO_VAR[u.emotion.label]})` } }), EMO_RU[u.emotion.label] || u.emotion.label);
}
function messageNode(u, fresh) {
  const color = `var(${speakerVar(u.speaker)})`;
  const head = h("div", { class: "msg-head" },
    h("span", { class: "dot", style: { "--c": color } }),
    h("span", { class: "msg-name" }, state.speakerName(u.speaker)),
    h("span", {}, fmtClock(u.start)),
    emotionChip(u),
    u.interrupted ? h("span", { class: "chip is-flag", title: "Ответ начался меньше чем через 0,3 с после предыдущего собеседника" }, "перебил") : null,
    !u.confident ? h("span", { class: "chip is-flag", title: "Реплика короткая или голос похож сразу на нескольких собеседников" }, "голос неуверенно") : null
  );
  const foot = h("div", { class: "msg-foot" },
    h("span", {}, fmtSec(u.end - u.start)),
    h("span", { title: "От конца фразы до появления текста" }, `задержка ${fmtSec(u.latency_ms / 1000)}`));
  return h("div", { class: `msg${u.speaker % 2 === 0 ? " is-right" : ""}${fresh ? " is-new" : ""}`, "data-id": u.id, style: { "--c": color } },
    head, h("div", { class: "bubble" }, u.text), foot);
}
function appendMessage(u) {
  const stick = nearBottom();
  $("chatEmpty").hidden = true;
  const typing = chat.querySelector(".typing");
  const node = messageNode(u, true);
  if (typing) chat.insertBefore(node, typing); else chat.append(node);
  if (stick) chat.scrollTop = chat.scrollHeight;
  renderTyping();
}
function updateMessageEmotion(u) {
  const node = chat.querySelector(`.msg[data-id="${u.id}"] .msg-head`);
  if (!node) return;
  const old = node.querySelector('[data-role="emo"]');
  const chip = emotionChip(u);
  if (old) old.replaceWith(chip); else node.insertBefore(chip, node.children[3] || null);
}
function rebuildChat() {
  const stick = nearBottom();
  chat.querySelectorAll(".msg").forEach((n) => n.remove());
  const typing = chat.querySelector(".typing");
  for (const u of state.utterances) {
    const node = messageNode(u, false);
    if (typing) chat.insertBefore(node, typing); else chat.append(node);
  }
  if (stick) chat.scrollTop = chat.scrollHeight;
}
function renderTyping() {
  let el = chat.querySelector(".typing");
  if (!state.live.speaking || !state.recording) { if (el) el.remove(); return; }
  const stick = nearBottom();
  const who = state.live.speaker ? state.speakerName(state.live.speaker) : "Кто-то";
  const color = state.live.speaker ? `var(${speakerVar(state.live.speaker)})` : "var(--muted)";
  const fresh = h("div", { class: "typing", style: { "--c": color } }, h("span", { class: "typing-dots" }, h("i"), h("i"), h("i")), `${who} говорит…`);
  if (el) el.replaceWith(fresh); else chat.append(fresh);
  if (stick) chat.scrollTop = chat.scrollHeight;
}
function clearConversation() {
  state.utterances = []; state.names = new Map(); state.levels = []; state.audioSec = 0;
  state.llm = { state: "idle" }; state.live = { speaking: false, speaker: null };
  state.stale = false;
  chat.querySelectorAll(".msg, .typing").forEach((n) => n.remove());
  $("chatEmpty").hidden = false;
  invalidateDashboard();
}

// -------------------------------------------------------- боковая панель
function renderMeter(msg) {
  const fill = $("meterFill"), pill = $("vadPill");
  if (!msg || !state.recording) {
    fill.style.width = "0%"; $("meterValue").textContent = "—";
    pill.textContent = state.recording ? "тишина" : "микрофон выключен"; pill.className = "pill";
    return;
  }
  const pct = Math.max(0, Math.min(100, ((msg.db + 60) / 60) * 100));
  fill.style.width = `${pct}%`;
  fill.classList.toggle("is-hot", msg.peak_db > -1);
  $("meterValue").textContent = `${fmtNum(msg.db)} дБ`;
  pill.textContent = msg.speaking ? "речь" : "тишина";
  pill.className = `pill${msg.speaking ? " is-speech" : ""}`;
}

const spectrumCanvas = $("spectrum");
function drawSpectrum() {
  const ctx = spectrumCanvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const w = spectrumCanvas.clientWidth, hgt = spectrumCanvas.clientHeight;
  if (spectrumCanvas.width !== w * dpr) { spectrumCanvas.width = w * dpr; spectrumCanvas.height = hgt * dpr; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, hgt);
  const bars = 40, data = state.recording ? mic.spectrum(bars) : null;
  const gap = 2, bw = (w - gap * (bars - 1)) / bars;
  ctx.fillStyle = cssVar("--tech");
  for (let i = 0; i < bars; i++) {
    const v = data ? data[i] : 0;
    const bh = Math.max(2, v * (hgt - 2));
    ctx.globalAlpha = data ? 0.35 + 0.65 * v : 0.2;
    ctx.beginPath();
    ctx.roundRect(i * (bw + gap), hgt - bh, bw, bh, [2, 2, 0, 0]);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  if (state.recording) requestAnimationFrame(drawSpectrum);
}

function renderRail() {
  // собеседники
  const per = new Map();
  let total = 0;
  for (const u of state.utterances) {
    const d = u.end - u.start;
    const s = per.get(u.speaker) || { id: u.speaker, talk: 0, count: 0, last: null };
    s.talk += d; s.count += 1; if (u.emotion) s.last = u.emotion.label;
    per.set(u.speaker, s); total += d;
  }
  const list = $("speakersList");
  const speakers = [...per.values()].sort((a, b) => a.id - b.id);
  $("speakersCount").textContent = speakers.length ? String(speakers.length) : "";
  const active = document.activeElement && document.activeElement.classList.contains("spk-name") ? document.activeElement.dataset.id : null;
  if (!speakers.length) {
    list.replaceChildren(h("p", { class: "muted small" }, "Появятся с первой репликой. Имя можно изменить — нажмите на него."));
  } else if (!active) {
    list.replaceChildren(...speakers.map((s) => {
      const input = h("input", { class: "spk-name", value: state.speakerName(s.id), "data-id": s.id, maxlength: 40, "aria-label": `Имя собеседника ${s.id}`, spellcheck: "false" });
      const commit = () => {
        const name = input.value.trim();
        if (name && name !== `Собеседник ${s.id}`) state.names.set(s.id, name); else state.names.delete(s.id);
        send({ type: "rename", speaker: s.id, name: state.names.get(s.id) || "" });
        rebuildChat(); invalidateDashboard(); schedule();
      };
      input.addEventListener("change", commit);
      input.addEventListener("keydown", (e) => { if (e.key === "Enter") input.blur(); });
      return h("div", { class: "spk", style: { "--c": `var(${speakerVar(s.id)})` } },
        h("span", { class: "dot" }), input, h("span", { class: "spk-val" }, fmtPct(total ? s.talk / total : 0)),
        h("div", { class: "spk-bar" }, h("i", { style: { width: `${total ? (s.talk / total) * 100 : 0}%` } })),
        h("div", { class: "spk-meta" }, `${s.count} ${plural(s.count, "реплика", "реплики", "реплик")} · ${fmtDur(s.talk)}`,
          s.last ? h("span", { class: "chip" }, h("span", { class: "dot", style: { "--c": `var(${EMO_VAR[s.last]})` } }), EMO_RU[s.last]) : null));
    }));
  }

  // факты сессии
  const lat = state.utterances.map((u) => u.latency_ms / 1000);
  const m = state.metrics[state.metrics.length - 1] || {};
  const facts = [
    ["Реплик", fmtNum(state.utterances.length)],
    ["Время речи", fmtDur(total)],
    ["Задержка до текста", lat.length ? fmtSec(median(lat)) : "—"],
    ["Очередь", m.queue != null ? fmtNum(m.queue) : "—"],
  ];
  $("railFacts").replaceChildren(...facts.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)]));

  // модели
  const names = { ready: "готово", loading: "загрузка", pending: "ожидание", error: "ошибка", off: "выключено" };
  $("railComponents").replaceChildren(...Object.values(state.components).map((c) =>
    h("li", {}, h("span", { class: `st ${c.state}` }, ""), h("span", {}, c.title),
      c.state !== "ready" ? h("span", { class: "detail" }, `${names[c.state] || c.state}${c.detail ? `: ${c.detail}` : ""}`) : null)));
}

// ------------------------------------------------------------ управление
function coreReady() {
  return ["vad", "asr", "spk"].every((k) => state.components[k] && state.components[k].state === "ready");
}
function updateControls() {
  const btn = $("micBtn"), connected = state.ws && state.ws.readyState === WebSocket.OPEN;
  const ready = connected && coreReady();
  btn.disabled = !ready && !state.recording;
  btn.textContent = state.recording ? "Остановить" : "Начать запись";
  btn.classList.toggle("is-recording", state.recording);
  $("recState").classList.toggle("is-on", state.recording);
  $("recLabel").textContent = state.recording ? "Идёт запись" : !connected ? "Нет связи с сервером" : !ready ? "Модели загружаются…" : "Микрофон выключен";
  $("recTime").textContent = fmtClock(state.audioSec);
  const failed = ["vad", "asr", "spk"].map((k) => state.components[k]).filter((c) => c && c.state === "error");
  $("emptyStatus").textContent = failed.length ? `Не загрузилось: ${failed.map((c) => `${c.title} (${c.detail})`).join("; ")}` : !ready ? "Модели распознавания загружаются, это занимает несколько секунд." : "";
  $("resetBtn").disabled = !connected;
}

function showBanner(text, isError) {
  const b = $("banner");
  b.textContent = text; b.hidden = false; b.classList.toggle("is-error", !!isError);
}
function hideBanner() { $("banner").hidden = true; }

function schedule() { dirty = true; }
function frame() {
  const now = performance.now();
  if (dirty && now - lastDash > 400) {
    dirty = false; lastDash = now;
    $("recTime").textContent = fmtClock(state.audioSec);
    if (state.view === "chat") renderRail(); else renderDashboard(state);
  }
  requestAnimationFrame(frame);
}

function setView(view) {
  state.view = view;
  for (const t of document.querySelectorAll(".hud-tab[data-view]")) {
    const on = t.dataset.view === view;
    t.classList.toggle("is-active", on); t.setAttribute("aria-selected", String(on));
  }
  $("view-chat").classList.toggle("is-active", view === "chat"); $("view-chat").hidden = view !== "chat";
  $("view-dashboard").classList.toggle("is-active", view === "dashboard"); $("view-dashboard").hidden = view !== "dashboard";
  if (view === "dashboard") invalidateDashboard(); else chat.scrollTop = chat.scrollHeight;
  if (location.hash !== `#${view}`) history.replaceState(null, "", `#${view}`);
  lastDash = 0; schedule();
}

function init() {
  initDashboard(() => { lastDash = 0; schedule(); });
  for (const t of document.querySelectorAll(".hud-tab[data-view]")) t.addEventListener("click", () => setView(t.dataset.view));
  $("micBtn").addEventListener("click", () => (state.recording ? stopRecording(true) : startRecording()));
  $("resetBtn").addEventListener("click", () => {
    if (state.utterances.length && !confirm("Очистить разговор? Текст и распознанные голоса будут удалены без возможности восстановления.")) return;
    send({ type: "reset" });
  });
  const menu = $("settingsMenu"), sbtn = $("settingsBtn");
  sbtn.addEventListener("click", (e) => { e.stopPropagation(); menu.hidden = !menu.hidden; sbtn.setAttribute("aria-expanded", String(!menu.hidden)); });
  document.addEventListener("click", (e) => { if (!menu.hidden && !menu.contains(e.target)) { menu.hidden = true; sbtn.setAttribute("aria-expanded", "false"); } });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { menu.hidden = true; sbtn.setAttribute("aria-expanded", "false"); } });
  $("thrRange").addEventListener("input", (e) => { $("thrValue").textContent = fmtNum(Number(e.target.value), 2); });
  $("thrRange").addEventListener("change", (e) => send({ type: "config", speaker_threshold: Number(e.target.value) }));
  $("splitToggle").addEventListener("change", (e) => send({ type: "config", split_turns: e.target.checked }));
  $("dspToggle").addEventListener("change", async () => { if (state.recording) { await stopRecording(false); await startRecording(); } });
  window.addEventListener("resize", () => { invalidateDashboard(); schedule(); });
  window.addEventListener("beforeunload", (e) => { if (state.recording || state.utterances.length) { e.preventDefault(); e.returnValue = ""; } });
  setView(location.hash === "#dashboard" ? "dashboard" : "chat");
  renderMeter(null); drawSpectrum(); updateControls();
  connect();
  requestAnimationFrame(frame);
  window.__appStarted = true;      // для проверки запуска в index.html
}
init();
