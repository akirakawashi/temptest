// Захват микрофона: поток 16 кГц для сервера и анализатор для живого спектра.

export class MicCapture {
  constructor() {
    this.ctx = null;
    this.stream = null;
    this.node = null;
    this.analyser = null;
    this.deviceLabel = "";
    this.deviceRate = 0;
  }

  get active() {
    return !!this.stream;
  }

  /** onChunk получает ArrayBuffer: 100 мс звука, 16 кГц, моно, 16 бит. */
  async start(onChunk, { browserDsp = false } = {}) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error(
        "Браузер не даёт доступ к микрофону. Откройте страницу по адресу http://localhost или через HTTPS."
      );
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: browserDsp,
        noiseSuppression: browserDsp,
        autoGainControl: browserDsp,
      },
    });
    const track = this.stream.getAudioTracks()[0];
    this.deviceLabel = track ? track.label : "";

    const Ctx = window.AudioContext || window.webkitAudioContext;
    this.ctx = new Ctx({ latencyHint: "interactive" });
    this.deviceRate = this.ctx.sampleRate;
    await this.ctx.audioWorklet.addModule(new URL("./pcm-worklet.js", import.meta.url).href);

    const src = this.ctx.createMediaStreamSource(this.stream);
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    this.analyser.smoothingTimeConstant = 0.6;
    src.connect(this.analyser);

    this.node = new AudioWorkletNode(this.ctx, "pcm-capture", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      processorOptions: { targetRate: 16000, chunkSamples: 1600 },
    });
    this.node.port.onmessage = (e) => onChunk(e.data);
    src.connect(this.node);
    // Узел должен быть подключён к выходу, иначе часть браузеров его не обрабатывает.
    // Громкость нулевая — в динамики ничего не попадает.
    const mute = this.ctx.createGain();
    mute.gain.value = 0;
    this.node.connect(mute).connect(this.ctx.destination);

    if (this.ctx.state === "suspended") await this.ctx.resume();
    if (track) track.addEventListener("ended", () => this.onEnded && this.onEnded());
  }

  async stop() {
    try {
      if (this.node) {
        this.node.port.onmessage = null;
        this.node.disconnect();
      }
      if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
      if (this.ctx) await this.ctx.close();
    } catch (_) {
      /* устройство уже отключено */
    }
    this.ctx = this.stream = this.node = this.analyser = null;
  }

  /** Спектр 0–8 кГц, сгруппированный в bars столбцов (значения 0..1). */
  spectrum(bars) {
    if (!this.analyser) return null;
    const bins = new Uint8Array(this.analyser.frequencyBinCount);
    this.analyser.getByteFrequencyData(bins);
    const top = Math.min(bins.length, Math.floor((8000 / (this.deviceRate / 2)) * bins.length));
    const out = new Float32Array(bars);
    for (let b = 0; b < bars; b++) {
      // шкала ближе к логарифмической: нижним частотам — больше столбцов
      const lo = Math.floor(Math.pow(b / bars, 1.7) * top);
      const hi = Math.max(lo + 1, Math.floor(Math.pow((b + 1) / bars, 1.7) * top));
      let acc = 0;
      for (let i = lo; i < hi; i++) acc += bins[i];
      out[b] = acc / (hi - lo) / 255;
    }
    return out;
  }
}
