"""Etapa 4: original sound design for Reel 2, synced to the camera timeline in render.py.

Arc:   mystery (A minor bed)  ->  whoosh into the glass + crystal "ting"
       ->  reverse swell + deep hit on the price (bed lifts to F major 7)
       ->  heavy switch + room goes dark (bed loses its brightness)  ->  soft low ting
Everything is synthesised here (no samples), so there are no rights issues.
"""
import numpy as np
from scipy.signal import fftconvolve, stft, istft
import soundfile as sf
import render as R

SR = 48000
DUR = 15.0
N = int(DUR * SR)
t = np.arange(N) / SR
rng = np.random.default_rng(2026)

def db(x):
    return 10 ** (x / 20)

def ss(e0, e1, x):
    y = np.clip((x - e0) / (e1 - e0), 0, 1)
    return y * y * (3 - 2 * y)

# ------------------------------------------------------------------ reverb (stereo hall)
def make_ir(rt_low=2.6, rt_mid=2.1, rt_high=1.1, length=3.0):
    n = int(length * SR)
    ti = np.arange(n) / SR
    out = []
    for ch in range(2):
        noise = rng.standard_normal(n)
        F = np.fft.rfft(noise)
        f = np.fft.rfftfreq(n, 1 / SR)
        bands = []
        for lo, hi, rt in ((0, 400, rt_low), (400, 3000, rt_mid), (3000, SR / 2, rt_high)):
            m = ((f >= lo) & (f < hi)).astype(float)
            b = np.fft.irfft(F * m, n) * np.exp(-6.91 * ti / rt)
            bands.append(b)
        ir = sum(bands)
        ir[: int(0.012 * SR)] *= np.linspace(0, 1, int(0.012 * SR))  # pre-delay softness
        out.append(ir)
    ir = np.stack(out, 1)
    return ir / np.sqrt((ir ** 2).sum() / 2)

IR = make_ir()

def reverb(x, ir=IR):
    """x: (N,2) -> wet (N,2)"""
    return np.stack([fftconvolve(x[:, c], ir[:, c])[: len(x)] for c in range(2)], 1)

# ------------------------------------------------------------------ elements
def adsr_exp(t0, t60, attack=0.004):
    tt = t - t0
    e = np.where(tt < 0, 0.0, np.exp(-6.91 * np.maximum(tt, 0) / t60))
    return e * ss(0, attack, tt)

def glass_ting(t0, f0, amp, decay=1.0, bright=1.0):
    """struck crystal: inharmonic partials, each split in two for shimmer/beating"""
    out = np.zeros((N, 2))
    parts = [(1.0, 1.0, 3.2), (2.32, 0.42 * bright, 1.7), (3.87, 0.20 * bright, 0.95), (5.51, 0.10 * bright, 0.5)]
    for ratio, a, t60 in parts:
        f = f0 * ratio
        e = adsr_exp(t0, t60 * decay, 0.0015)
        ph = rng.uniform(0, 2 * np.pi, 2)
        l = np.sin(2 * np.pi * (f - 0.9) * (t - t0) + ph[0])
        r = np.sin(2 * np.pi * (f + 0.9) * (t - t0) + ph[1])
        out[:, 0] += a * e * l
        out[:, 1] += a * e * r
    # strike: tiny burst of bright noise
    tt = t - t0
    burst = rng.standard_normal(N) * np.where(tt >= 0, np.exp(-np.maximum(tt, 0) / 0.004), 0)
    burst = np.diff(burst, prepend=0)  # crude high-pass
    out += 0.08 * bright * burst[:, None]
    return amp * out / np.max(np.abs(out))

def boom(t0, amp):
    """deep hit that still reads on phone speakers (saturation adds 2nd/3rd harmonics)"""
    tt = np.maximum(t - t0, 0)
    f = 38 + 44 * np.exp(-tt / 0.12)               # 82 Hz -> 38 Hz pitch drop
    ph = 2 * np.pi * np.cumsum(f) / SR
    e = adsr_exp(t0, 1.9, 0.003)
    sub = np.sin(ph) * e
    sub = np.tanh(2.2 * sub) / np.tanh(2.2)          # harmonics for small speakers
    body = rng.standard_normal(N) * adsr_exp(t0, 0.25, 0.002)
    B = np.fft.rfft(body); fr = np.fft.rfftfreq(N, 1 / SR)
    body = np.fft.irfft(B * ((fr > 90) & (fr < 420)), N)
    body /= np.max(np.abs(body)) + 1e-9
    mono = sub + 0.35 * body
    return amp * np.stack([mono, mono], 1) / np.max(np.abs(mono))

def click(t0, amp):
    """heavy switch: short mid transient + small low knock"""
    tt = t - t0
    n = rng.standard_normal(N) * np.where(tt >= 0, np.exp(-np.maximum(tt, 0) / 0.006), 0)
    Nf = np.fft.rfft(n); fr = np.fft.rfftfreq(N, 1 / SR)
    n = np.fft.irfft(Nf * ((fr > 700) & (fr < 3500)), N)
    knock = np.sin(2 * np.pi * 120 * np.maximum(tt, 0)) * adsr_exp(t0, 0.09, 0.001)
    mono = n / (np.max(np.abs(n)) + 1e-9) + 0.6 * knock
    return amp * np.stack([mono, mono * 0.96], 1) / np.max(np.abs(mono))

def camera_speed():
    """|d log(width)/dt| + tilt speed, sampled per audio sample (smooth)"""
    ts = np.arange(0, DUR + 0.01, 0.01)
    vals = []
    for x in ts:
        _, cy1, w1 = R.camera(x)
        _, cy2, w2 = R.camera(min(x + 0.01, DUR))
        vals.append(abs(np.log(w2 / w1)) / 0.01 + abs(cy2 - cy1) / 0.01 / w1)
    return np.interp(t, ts, np.array(vals))

SPEED = camera_speed()

def whoosh(t_lo, t_hi, amp, f_start, f_end):
    """noise swept through a moving band-pass, loudness following the camera speed"""
    noise = rng.standard_normal((N, 2))
    env = SPEED * ((t >= t_lo) & (t <= t_hi))
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

def reverse_swell(t_end, amp, length=0.9):
    """classic reverse-reverb swell that lands exactly on t_end"""
    n = int(length * SR)
    burst = np.zeros((n, 2))
    k = int(0.05 * SR)
    b = rng.standard_normal((k, 2)) * np.hanning(k)[:, None]
    burst[:k] = b
    wet = np.stack([fftconvolve(burst[:, c], IR[: n, c])[:n] for c in range(2)], 1)
    # keep it bright and airy
    for c in range(2):
        W_ = np.fft.rfft(wet[:, c]); fr = np.fft.rfftfreq(n, 1 / SR)
        wet[:, c] = np.fft.irfft(W_ * (fr > 600), n)
    rev = wet[::-1] * np.linspace(0.2, 1, n)[:, None] ** 2
    out = np.zeros((N, 2))
    e = int(t_end * SR)
    out[e - n:e] = rev
    return amp * out / (np.max(np.abs(out)) + 1e-9)

def pad():
    """warm additive pad: Am -> Fmaj7 on the price -> Am, losing brightness at lights-out"""
    Am = [110.0, 164.81, 261.63, 329.63]
    Fmaj7 = [87.31, 130.81, 220.0, 329.63]
    w_am1 = 1 - ss(7.05, 7.55, t)
    w_f = ss(7.05, 7.55, t) * (1 - ss(10.2, 10.9, t))
    w_am2 = ss(10.2, 10.9, t)
    dark = ss(10.0, 11.0, t)
    out = np.zeros((N, 2))
    for notes, wch in ((Am, w_am1 + w_am2), (Fmaj7, w_f)):
        for i, f0 in enumerate(notes):
            for k in range(1, 7):
                keep = 1.0 if k == 1 else (0.5 if k == 2 else 0.12)
                a = (1 / k ** 1.6) * (1 - dark * (1 - keep)) * (0.9 if i == 0 else 0.55)
                for det, pan in ((-4, 0.2), (0, 0.5), (4, 0.8)):      # 3-voice chorus
                    f = f0 * k * 2 ** (det / 1200)
                    ph = rng.uniform(0, 2 * np.pi)
                    trem = 1 + 0.12 * np.sin(2 * np.pi * (0.07 + 0.02 * i) * t + ph)
                    s = np.sin(2 * np.pi * f * t + ph) * a * trem * wch
                    out[:, 0] += s * np.cos(pan * np.pi / 2)
                    out[:, 1] += s * np.sin(pan * np.pi / 2)
    # sub fifth underneath (A1 + E2), breathing slowly
    sub = (np.sin(2 * np.pi * 55 * t) + 0.5 * np.sin(2 * np.pi * 82.41 * t)) * (1 + 0.15 * np.sin(2 * np.pi * 0.1 * t))
    out += 0.6 * sub[:, None]
    out *= (1 - 0.35 * dark)[:, None]                         # room gets quieter in the dark
    out *= ss(0, 0.9, t)[:, None]                              # soft start
    return out / np.max(np.abs(out))

def room_tone():
    n = rng.standard_normal((N, 2))
    out = np.zeros_like(n)
    fr = np.fft.rfftfreq(N, 1 / SR)
    shape = ((fr > 150) & (fr < 2500)) / np.sqrt(np.maximum(fr, 20))   # pink-ish
    for c in range(2):
        out[:, c] = np.fft.irfft(np.fft.rfft(n[:, c]) * shape, N)
    out /= np.max(np.abs(out))
    return out * (1 - 0.7 * ss(10.0, 11.0, t))[:, None]

# ------------------------------------------------------------------ mix
dry = np.zeros((N, 2))
send = np.zeros((N, 2))

bed = pad() * db(-27) + room_tone() * db(-44)
dry += bed
send += bed * 0.35

w1 = whoosh(3.1, 3.9, db(-11), 280, 4200)                       # into the glass
thump = boom(3.80, db(-20))
ting1 = glass_ting(3.90, 1318.5, db(-11), decay=1.0, bright=1.0)   # E6, the glass
dry += w1 + thump + ting1
send += w1 * 0.2 + ting1 * 0.45 + thump * 0.2

sw = reverse_swell(7.30, db(-15))
w2 = whoosh(6.65, 7.35, db(-19), 2200, 600)                      # soft pull-back air
hit = boom(7.30, db(-5.5))
shimmer = glass_ting(7.32, 2637.0, db(-26), decay=0.45, bright=0.5)  # faint gold sparkle
dry += sw + w2 + hit + shimmer
send += hit * 0.3 + shimmer * 0.6 + w2 * 0.2

clk = click(10.02, db(-15))
w3 = whoosh(10.0, 11.05, db(-24), 1800, 350)                     # air falling away
ting2 = glass_ting(11.15, 659.25, db(-15), decay=1.35, bright=0.55)  # E5, the question
dry += clk + w3 + ting2
send += clk * 0.3 + ting2 * 0.55

mix = dry + reverb(send) * db(-9)
# loop-friendly edges
mix *= ss(0, 0.05, t)[:, None] * (1 - ss(14.55, 15.0, t))[:, None]
# gentle bus glue: soft-knee saturation, then leave headroom for loudnorm
peak = np.max(np.abs(mix))
mix = np.tanh(1.3 * mix / peak) / np.tanh(1.3) * 0.7
sf.write('reel2_audio_raw.wav', mix.astype(np.float32), SR, subtype='FLOAT')
print('written', mix.shape, 'peak', round(float(np.max(np.abs(mix))), 3))
