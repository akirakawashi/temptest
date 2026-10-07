// Аудио-ворклет: принимает звук с частотой устройства (обычно 44,1 или 48 кГц),
// фильтрует, пересчитывает в 16 кГц и отдаёт порциями по 100 мс (16 бит, моно).

class PcmCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = (options && options.processorOptions) || {};
    this.targetRate = o.targetRate || 16000;
    this.chunk = o.chunkSamples || 1600;
    this.ratio = sampleRate / this.targetRate;

    // Фильтр нижних частот (оконный sinc), чтобы при понижении частоты не было наложений.
    const taps = 63;
    const cutoff = Math.min(0.5, (0.45 * this.targetRate) / sampleRate);
    this.fir = new Float32Array(taps);
    let sum = 0;
    for (let i = 0; i < taps; i++) {
      const m = i - (taps - 1) / 2;
      const sinc = m === 0 ? 2 * cutoff : Math.sin(2 * Math.PI * cutoff * m) / (Math.PI * m);
      const win = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (taps - 1));
      this.fir[i] = sinc * win;
      sum += this.fir[i];
    }
    for (let i = 0; i < taps; i++) this.fir[i] /= sum;

    this.hist = new Float32Array(taps);     // кольцевой буфер входа для фильтра
    this.histPos = 0;
    this.prev = 0;                          // предыдущий отфильтрованный отсчёт
    this.phase = 0;                         // дробная позиция следующего выходного отсчёта
    this.out = new Int16Array(this.chunk);
    this.outPos = 0;
  }

  filter(x) {
    const h = this.hist, n = h.length;
    h[this.histPos] = x;
    this.histPos = (this.histPos + 1) % n;
    let acc = 0, j = this.histPos;
    for (let i = 0; i < n; i++) {
      acc += h[j] * this.fir[i];
      j = j + 1 === n ? 0 : j + 1;
    }
    return acc;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const ch0 = input[0], ch1 = input[1];
    for (let i = 0; i < ch0.length; i++) {
      const x = ch1 ? (ch0[i] + ch1[i]) * 0.5 : ch0[i];
      const y = this.filter(x);
      // линейная интерполяция между соседними отфильтрованными отсчётами
      while (this.phase < 1) {
        const v = this.prev + (y - this.prev) * this.phase;
        const s = Math.max(-1, Math.min(1, v));
        this.out[this.outPos++] = s < 0 ? s * 32768 : s * 32767;
        if (this.outPos === this.chunk) {
          const buf = this.out.buffer;
          this.port.postMessage(buf, [buf]);
          this.out = new Int16Array(this.chunk);
          this.outPos = 0;
        }
        this.phase += this.ratio;
      }
      this.phase -= 1;
      this.prev = y;
    }
    return true;
  }
}

registerProcessor("pcm-capture", PcmCapture);
