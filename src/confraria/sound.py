"""Design de som original, sincronizado com a câmera (camera.Timeline).

Arco:  mistério (base em lá menor)  ->  whoosh para dentro do vidro + "ting" de cristal
       ->  swell reverso + golpe grave no preço (a base sobe para Fmaj7)
       ->  clique pesado + a sala escurece (a base perde brilho)  ->  ting grave suave
Tudo é sintetizado aqui (sem samples, sem direitos autorais).

Parametrizado por reel.json:
  timing              cortes e duração (todos os eventos são relativos a c1, c2, c3 e ao fim)
  sound.bed_root      tônica da base (A = lá menor como no Reel 2; as notas são transpostas)
  sound.ting_hz       [ting da tela 2, ting da tela 4]; o brilho do preço é uma oitava acima do 1º
  focus.lamp/sconces  sem eles não há "luzes apagando": sem clique e a base não escurece
"""
import numpy as np
from scipy.signal import fftconvolve, stft, istft
import soundfile as sf

from .camera import Timeline
from .config import NOTES

SR = 48000


def db(x):
    return 10 ** (x / 20)


def ss(e0, e1, x):
    y = np.clip((x - e0) / (e1 - e0), 0, 1)
    return y * y * (3 - 2 * y)


def transpose_ratio(bed_root):
    """razão de frequência em relação ao lá, escolhendo a oitava mais próxima (-6 a +5 semitons)"""
    n = NOTES[bed_root.upper()]
    return 2 ** ((n if n <= 5 else n - 12) / 12)


class Synth:
    def __init__(self, timeline: Timeline, reel, seed=2026):
        self.tl = timeline
        self.reel = reel
        self.DUR = timeline.dur
        self.N = int(self.DUR * SR)
        self.t = np.arange(self.N) / SR
        self.rng = np.random.default_rng(seed)
        self.IR = self.make_ir()          # a ordem das chamadas ao rng é parte do "som" do Reel 2
        self.SPEED = self.camera_speed()

    # ------------------------------------------------------------ reverb (salão estéreo)
    def make_ir(self, rt_low=2.6, rt_mid=2.1, rt_high=1.1, length=3.0):
        n = int(length * SR)
        ti = np.arange(n) / SR
        out = []
        for ch in range(2):
            noise = self.rng.standard_normal(n)
            F = np.fft.rfft(noise)
            f = np.fft.rfftfreq(n, 1 / SR)
            bands = []
            for lo, hi, rt in ((0, 400, rt_low), (400, 3000, rt_mid), (3000, SR / 2, rt_high)):
                m = ((f >= lo) & (f < hi)).astype(float)
                b = np.fft.irfft(F * m, n) * np.exp(-6.91 * ti / rt)
                bands.append(b)
            ir = sum(bands)
            ir[: int(0.012 * SR)] *= np.linspace(0, 1, int(0.012 * SR))  # pré-atraso suave
            out.append(ir)
        ir = np.stack(out, 1)
        return ir / np.sqrt((ir ** 2).sum() / 2)

    def reverb(self, x):
        """x: (N,2) -> molhado (N,2)"""
        return np.stack([fftconvolve(x[:, c], self.IR[:, c])[: len(x)] for c in range(2)], 1)

    # ------------------------------------------------------------ elementos
    def adsr_exp(self, t0, t60, attack=0.004):
        tt = self.t - t0
        e = np.where(tt < 0, 0.0, np.exp(-6.91 * np.maximum(tt, 0) / t60))
        return e * ss(0, attack, tt)

    def glass_ting(self, t0, f0, amp, decay=1.0, bright=1.0):
        """cristal percutido: parciais inarmônicos, cada um dividido em dois (batimento/brilho)"""
        N, t, rng = self.N, self.t, self.rng
        out = np.zeros((N, 2))
        parts = [(1.0, 1.0, 3.2), (2.32, 0.42 * bright, 1.7), (3.87, 0.20 * bright, 0.95), (5.51, 0.10 * bright, 0.5)]
        for ratio, a, t60 in parts:
            f = f0 * ratio
            e = self.adsr_exp(t0, t60 * decay, 0.0015)
            ph = rng.uniform(0, 2 * np.pi, 2)
            l = np.sin(2 * np.pi * (f - 0.9) * (t - t0) + ph[0])
            r = np.sin(2 * np.pi * (f + 0.9) * (t - t0) + ph[1])
            out[:, 0] += a * e * l
            out[:, 1] += a * e * r
        # ataque: rajada curtíssima de ruído brilhante
        tt = t - t0
        burst = rng.standard_normal(N) * np.where(tt >= 0, np.exp(-np.maximum(tt, 0) / 0.004), 0)
        burst = np.diff(burst, prepend=0)  # passa-altas tosco
        out += 0.08 * bright * burst[:, None]
        return amp * out / np.max(np.abs(out))

    def boom(self, t0, amp):
        """golpe grave que ainda aparece em alto-falante de celular (saturação gera 2º/3º harmônicos)"""
        N, t, rng = self.N, self.t, self.rng
        tt = np.maximum(t - t0, 0)
        f = 38 + 44 * np.exp(-tt / 0.12)               # queda de 82 Hz para 38 Hz
        ph = 2 * np.pi * np.cumsum(f) / SR
        e = self.adsr_exp(t0, 1.9, 0.003)
        sub = np.sin(ph) * e
        sub = np.tanh(2.2 * sub) / np.tanh(2.2)
        body = rng.standard_normal(N) * self.adsr_exp(t0, 0.25, 0.002)
        B = np.fft.rfft(body)
        fr = np.fft.rfftfreq(N, 1 / SR)
        body = np.fft.irfft(B * ((fr > 90) & (fr < 420)), N)
        body /= np.max(np.abs(body)) + 1e-9
        mono = sub + 0.35 * body
        return amp * np.stack([mono, mono], 1) / np.max(np.abs(mono))

    def click(self, t0, amp):
        """interruptor pesado: transiente médio curto + pequena batida grave"""
        N, t, rng = self.N, self.t, self.rng
        tt = t - t0
        n = rng.standard_normal(N) * np.where(tt >= 0, np.exp(-np.maximum(tt, 0) / 0.006), 0)
        Nf = np.fft.rfft(n)
        fr = np.fft.rfftfreq(N, 1 / SR)
        n = np.fft.irfft(Nf * ((fr > 700) & (fr < 3500)), N)
        knock = np.sin(2 * np.pi * 120 * np.maximum(tt, 0)) * self.adsr_exp(t0, 0.09, 0.001)
        mono = n / (np.max(np.abs(n)) + 1e-9) + 0.6 * knock
        return amp * np.stack([mono, mono * 0.96], 1) / np.max(np.abs(mono))

    def camera_speed(self):
        """|d log(largura)/dt| + velocidade da inclinação, amostrado por amostra de áudio (suave)"""
        ts = np.arange(0, self.DUR + 0.01, 0.01)
        vals = []
        for x in ts:
            _, cy1, w1 = self.tl.camera(x)
            _, cy2, w2 = self.tl.camera(min(x + 0.01, self.DUR))
            vals.append(abs(np.log(w2 / w1)) / 0.01 + abs(cy2 - cy1) / 0.01 / w1)
        return np.interp(self.t, ts, np.array(vals))

    def whoosh(self, t_lo, t_hi, amp, f_start, f_end):
        """ruído varrido por um passa-banda móvel, volume seguindo a velocidade da câmera"""
        N, t = self.N, self.t
        noise = self.rng.standard_normal((N, 2))
        env = self.SPEED * ((t >= t_lo) & (t <= t_hi))
        env = env / (env.max() + 1e-9)
        env = np.convolve(env, np.hanning(int(0.03 * SR)) / np.hanning(int(0.03 * SR)).sum(), 'same')
        out = np.zeros((N, 2))
        for c in range(2):
            f, tau, Z = stft(noise[:, c], SR, nperseg=2048, noverlap=1536)
            prog = np.clip((tau - t_lo) / (t_hi - t_lo), 0, 1)
            fc = f_start * (f_end / f_start) ** ss(0, 1, prog)
            ff = np.maximum(f, 1)[:, None]
            gain = np.exp(-0.5 * (np.log(ff / fc[None, :]) / 0.55) ** 2)
            _, y = istft(Z * gain, SR, nperseg=2048, noverlap=1536)
            out[:, c] = y[:N]
        out *= env[:, None]
        return amp * out / (np.max(np.abs(out)) + 1e-9)

    def reverse_swell(self, t_end, amp, length=0.9):
        """swell de reverb reverso clássico que pousa exatamente em t_end"""
        N, rng = self.N, self.rng
        n = int(length * SR)
        burst = np.zeros((n, 2))
        k = int(0.05 * SR)
        b = rng.standard_normal((k, 2)) * np.hanning(k)[:, None]
        burst[:k] = b
        wet = np.stack([fftconvolve(burst[:, c], self.IR[: n, c])[:n] for c in range(2)], 1)
        for c in range(2):     # brilhante e arejado
            W_ = np.fft.rfft(wet[:, c])
            fr = np.fft.rfftfreq(n, 1 / SR)
            wet[:, c] = np.fft.irfft(W_ * (fr > 600), n)
        rev = wet[::-1] * np.linspace(0.2, 1, n)[:, None] ** 2
        out = np.zeros((N, 2))
        e = int(t_end * SR)
        out[e - n:e] = rev
        return amp * out / (np.max(np.abs(out)) + 1e-9)

    def dark_curve(self):
        """0 -> 1 quando as luzes se apagam (sempre 0 se o Reel não tem lâmpadas)"""
        if not self.tl.lights_out:
            return np.zeros(self.N)
        return ss(self.tl.c3 - 0.5, self.tl.c3 + 0.5, self.t)

    def pad(self):
        """pad aditivo quente: Am -> Fmaj7 no preço -> Am, perdendo brilho quando a luz apaga"""
        N, t, rng, tl = self.N, self.t, self.rng, self.tl
        r = transpose_ratio(self.reel.bed_root)
        Am = [f * r for f in (110.0, 164.81, 261.63, 329.63)]
        Fmaj7 = [f * r for f in (87.31, 130.81, 220.0, 329.63)]
        c2, c3 = tl.c2, tl.c3
        w_am1 = 1 - ss(c2 + 0.05, c2 + 0.55, t)
        w_f = ss(c2 + 0.05, c2 + 0.55, t) * (1 - ss(c3 - 0.3, c3 + 0.4, t))
        w_am2 = ss(c3 - 0.3, c3 + 0.4, t)
        dark = self.dark_curve()
        out = np.zeros((N, 2))
        for notes, wch in ((Am, w_am1 + w_am2), (Fmaj7, w_f)):
            for i, f0 in enumerate(notes):
                for k in range(1, 7):
                    keep = 1.0 if k == 1 else (0.5 if k == 2 else 0.12)
                    a = (1 / k ** 1.6) * (1 - dark * (1 - keep)) * (0.9 if i == 0 else 0.55)
                    for det, pan in ((-4, 0.2), (0, 0.5), (4, 0.8)):      # coro de 3 vozes
                        f = f0 * k * 2 ** (det / 1200)
                        ph = rng.uniform(0, 2 * np.pi)
                        trem = 1 + 0.12 * np.sin(2 * np.pi * (0.07 + 0.02 * i) * t + ph)
                        s = np.sin(2 * np.pi * f * t + ph) * a * trem * wch
                        out[:, 0] += s * np.cos(pan * np.pi / 2)
                        out[:, 1] += s * np.sin(pan * np.pi / 2)
        # quinta grave por baixo (tônica + quinta), respirando devagar
        sub = (np.sin(2 * np.pi * 55 * r * t) + 0.5 * np.sin(2 * np.pi * 82.41 * r * t)) \
            * (1 + 0.15 * np.sin(2 * np.pi * 0.1 * t))
        out += 0.6 * sub[:, None]
        out *= (1 - 0.35 * dark)[:, None]                         # a sala fica mais quieta no escuro
        out *= ss(0, 0.9, t)[:, None]                              # entrada suave
        return out / np.max(np.abs(out))

    def room_tone(self):
        N, rng = self.N, self.rng
        n = rng.standard_normal((N, 2))
        out = np.zeros_like(n)
        fr = np.fft.rfftfreq(N, 1 / SR)
        shape = ((fr > 150) & (fr < 2500)) / np.sqrt(np.maximum(fr, 20))   # quase rosa
        for c in range(2):
            out[:, c] = np.fft.irfft(np.fft.rfft(n[:, c]) * shape, N)
        out /= np.max(np.abs(out))
        return out * (1 - 0.7 * self.dark_curve())[:, None]

    # ------------------------------------------------------------ mixagem
    def mix(self):
        N, t, tl, reel = self.N, self.t, self.tl, self.reel
        c1, c2, c3, dur = tl.c1, tl.c2, tl.c3, tl.dur
        ting1_hz, ting2_hz = reel.ting_hz
        dry = np.zeros((N, 2))
        send = np.zeros((N, 2))

        bed = self.pad() * db(-27) + self.room_tone() * db(-44)
        dry += bed
        send += bed * 0.35

        w1 = self.whoosh(c1 - 0.4, c1 + 0.4, db(-11), 280, 4200)                 # para dentro do vidro
        thump = self.boom(c1 + 0.30, db(-20))
        ting1 = self.glass_ting(c1 + 0.40, ting1_hz, db(-11), decay=1.0, bright=1.0)   # o vidro
        dry += w1 + thump + ting1
        send += w1 * 0.2 + ting1 * 0.45 + thump * 0.2

        sw = self.reverse_swell(c2 + 0.30, db(-15))
        w2 = self.whoosh(c2 - 0.35, c2 + 0.35, db(-19), 2200, 600)               # ar suave do recuo
        hit = self.boom(c2 + 0.30, db(-5.5))
        shimmer = self.glass_ting(c2 + 0.32, ting1_hz * 2, db(-26), decay=0.45, bright=0.5)  # brilho dourado
        dry += sw + w2 + hit + shimmer
        send += hit * 0.3 + shimmer * 0.6 + w2 * 0.2

        clk = self.click(c3 - 0.48, db(-15)) if tl.lights_out else None      # interruptor pesado
        w3 = self.whoosh(c3 - 0.5, c3 + 0.55, db(-24), 1800, 350)                # ar se afastando
        ting2 = self.glass_ting(c3 + 0.65, ting2_hz, db(-15), decay=1.35, bright=0.55)   # a pergunta
        if clk is not None:
            dry += clk
            send += clk * 0.3
        dry += w3 + ting2
        send += ting2 * 0.55

        mix = dry + self.reverb(send) * db(-9)
        # bordas amigáveis a loop
        mix *= ss(0, 0.05, t)[:, None] * (1 - ss(dur - 0.45, dur, t))[:, None]
        # cola suave do barramento: saturação de joelho macio, deixando espaço para o loudnorm
        peak = np.max(np.abs(mix))
        mix = np.tanh(1.3 * mix / peak) / np.tanh(1.3) * 0.7
        return mix


def synthesize(timeline, reel, path):
    """Sintetiza a trilha (WAV float 48 kHz estéreo) e devolve o caminho."""
    mix = Synth(timeline, reel).mix()
    sf.write(path, mix.astype(np.float32), SR, subtype='FLOAT')
    return path, float(np.max(np.abs(mix)))
