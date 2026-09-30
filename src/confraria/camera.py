"""Câmera virtual: um movimento contínuo sobre o master graduado (vídeo 1080x1920, 30 fps).

A linha do tempo deriva de `timing` (s = timing.screen, L = timing.last):
  cortes   c1 = s, c2 = 2s, c3 = 3s;  duração = 3s + L   (Reel 2: 3,5 / 7,0 / 10,5 / 15,0)
  0        tela 1  a sala (legenda 1 desde o quadro 0)
  c1-0,3 .. c1+0,3   empurrão rápido da sala para o vidro (com motion blur)
  c1 .. c2 tela 2  o vidro
  c2-0,3 .. c2+0,3   recuo até o herói, a luz se fecha sobre ele
  c2 .. c3 tela 3  o preço (capa)
  c3-0,5 .. c3+0,5   recuo enquanto as luzes da sala se apagam
  c3 .. fim tela 4 "Abre ou guarda?" (a mais longa)
Os keyframes caem exatamente nos quatro enquadramentos de `shots.Scene`.
"""
import math

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

from .shots import Scene

W, H, FPS = 1080, 1920, 30


def ss(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


class Timeline:
    def __init__(self, scene: Scene):
        self.scene = scene
        r = scene.reel
        s = r.screen
        self.c1, self.c2, self.c3 = s, 2 * s, 3 * s
        self.dur = r.duration
        self.nfr = int(round(self.dur * FPS))
        self.lights_out = r.lights_out
        sh = scene.shots
        ww, wcx, wcy = sh['t1_sala']
        dw, dcx, dcy = sh['t2_vidro']
        hw, hcx, hcy = sh['t3_preco']
        k = scene.src_w / 768.0            # deslocamentos finos escalados com a resolução de origem
        c1, c2, c3, dur = self.c1, self.c2, self.c3, self.dur
        # (t, largura do recorte em px de origem, cx, cy)
        self.keys = [
            (0.0,       ww,              wcx, wcy),
            (c1 - 0.3,  ww * 730 / 768,  wcx, wcy - 2 * k),
            (c1 + 0.3,  dw,              dcx, dcy),          # = tela 2
            (c2 - 0.3,  dw * 318 / 340,  dcx, dcy - 4 * k),
            (c2 + 0.3,  hw,              hcx, hcy),          # = tela 3 (capa)
            (c3 - 0.5,  hw * 520 / 540,  hcx, hcy - 8 * k),
            (c3 + 0.5,  ww * 700 / 768,  wcx, wcy - 20 * k),
            (dur,       ww,              wcx, wcy),          # = tela 4
        ]
        kt = np.array([q[0] for q in self.keys])
        self.f_logw = PchipInterpolator(kt, np.log([q[1] for q in self.keys]))
        self.f_cx = PchipInterpolator(kt, [q[2] for q in self.keys])
        self.f_cy = PchipInterpolator(kt, [q[3] for q in self.keys])

    def camera(self, t):
        """(cx, cy, largura) em px de origem, contidos na imagem"""
        t = min(max(t, 0.0), self.dur)
        w = float(np.exp(self.f_logw(t)))
        h = w * 16 / 9
        cx = min(max(float(self.f_cx(t)), w / 2), self.scene.src_w - w / 2)
        cy = min(max(float(self.f_cy(t)), h / 2), self.scene.src_h - h / 2)
        return cx, cy, w

    def grade_weights(self, t):
        a = ss(self.c2 - 0.3, self.c2 + 0.3, t)       # base -> t3
        b = ss(self.c3 - 0.5, self.c3 + 0.5, t)       # t3 -> t4 (luzes apagando)
        if not self.lights_out:                        # sem lâmpadas: a tela 4 volta ao look base
            return {'base': 1 - a + a * b, 't3': a * (1 - b)}
        return {'base': 1 - a, 't3': a * (1 - b), 't4': b}

    def caption_state(self, t):
        """(opacidade, escala) por camada de texto"""
        c1, c2, c3 = self.c1, self.c2, self.c3
        return {
            't1_sala':     (1 - ss(c1 - 0.5, c1 - 0.3, t), 1.0),
            't2_vidro':    (ss(c1 + 0.35, c1 + 0.65, t) * (1 - ss(c2 - 0.5, c2 - 0.3, t)), 1.0),
            't3_preco':    (ss(c2 + 0.3, c2 + 0.55, t) * (1 - ss(c3 - 0.7, c3 - 0.5, t)),
                            1.035 - 0.035 * ss(c2 + 0.3, c2 + 0.65, t)),
            't4_pergunta': (ss(c3 + 0.5, c3 + 1.0, t), 1.0),
        }


class Pyramid:
    """Níveis do master graduado (metade a cada nível) para amostrar sem aliasing."""

    def __init__(self, arr, K):
        self.levels = [arr]
        while self.levels[-1].shape[1] > 1400:
            lv = self.levels[-1]
            self.levels.append(cv2.resize(lv, (lv.shape[1] // 2, lv.shape[0] // 2), interpolation=cv2.INTER_AREA))
        self.K = K

    def view(self, cx, cy, w, out_w, out_h, interp):
        """Renderiza o retângulo de origem (centro cx,cy, largura w) em out_w x out_h."""
        h = w * 16 / 9
        L = 0     # menor nível que ainda tem >= out_w pixels na largura da vista
        while L + 1 < len(self.levels) and w * self.K / (2 ** (L + 1)) >= out_w:
            L += 1
        kL = self.K / (2 ** L)
        s = out_w / (w * kL)
        x0 = (cx - w / 2) * kL
        y0 = (cy - h / 2) * kL
        A = np.array([[s, 0, -s * x0 + 0.5 * s - 0.5], [0, s, -s * y0 + 0.5 * s - 0.5]], dtype=np.float64)
        return cv2.warpAffine(self.levels[L], A, (out_w, out_h), flags=interp, borderMode=cv2.BORDER_REFLECT)


def caption_array(im):
    """PIL RGBA -> float32 RGBA pré-multiplicado"""
    rgba = np.asarray(im.convert('RGBA')).astype(np.float32) / 255.0
    rgba[..., :3] *= rgba[..., 3:4]
    return rgba


class Renderer:
    def __init__(self, timeline, grades, captions, master_scale, seed=7):
        self.tl = timeline
        self.pyrs = {k: Pyramid(v, master_scale) for k, v in grades.items()}
        self.caps = {n: caption_array(im) for n, im in captions.items()}
        self.seed = seed

    def background(self, t):
        """Quadro de fundo graduado, com motion blur (float32, 0-255)."""
        tl = self.tl
        dt = 1.0 / FPS
        cx, cy, w = tl.camera(t)
        _, cy2, w2 = tl.camera(t + dt)
        # deslocamento na tela por quadro no canto (zoom + inclinação)
        disp = abs(math.log(w2 / w)) * 1100 + abs(cy2 - cy) * (W / w)
        blur_len = disp * 0.5                      # obturador de 180 graus
        n = int(min(16, max(1, math.ceil(blur_len / 2.0))))
        weights = {k: v for k, v in tl.grade_weights(t).items() if v > 1e-3}
        acc = np.zeros((H, W, 3), np.float32)
        for i in range(n):
            ts = t + ((i + 0.5) / n - 0.5) * 0.5 * dt if n > 1 else t
            cx_s, cy_s, w_s = tl.camera(ts)
            for k, wt in weights.items():
                if n == 1:   # quase parado: cúbica 2x supersampled, depois média de área
                    im = self.pyrs[k].view(cx_s, cy_s, w_s, W * 2, H * 2, cv2.INTER_CUBIC)
                    im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
                else:        # em movimento: o motion blur esconde o aliasing, linear 1x basta
                    im = self.pyrs[k].view(cx_s, cy_s, w_s, W, H, cv2.INTER_LINEAR)
                acc += im.astype(np.float32) * (wt / n)
        return acc, n

    @staticmethod
    def composite_caption(frame, cap, opacity, scale):
        if opacity <= 1e-3:
            return frame
        c = cap
        if abs(scale - 1.0) > 1e-4:   # assenta em torno do centro óptico do preço
            cxp, cyp = W / 2, 425.0
            A = np.array([[scale, 0, (1 - scale) * cxp], [0, scale, (1 - scale) * cyp]])
            c = cv2.warpAffine(cap, A, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        a = c[..., 3:4] * opacity
        return frame * (1 - a) + c[..., :3] * 255.0 * opacity

    def frame(self, fi):
        t = fi / FPS
        bg, n = self.background(t)
        # grão monocromático fino: evita bandas nos degradês escuros após o re-encode do Instagram
        rng = np.random.default_rng(self.seed * 100003 + fi)
        grain = rng.normal(0.0, 1.4, (H, W, 1)).astype(np.float32)
        frame = bg + grain
        for name, (op, sc) in self.tl.caption_state(t).items():
            frame = self.composite_caption(frame, self.caps[name], op, sc)
        return np.clip(frame + 0.5, 0, 255).astype(np.uint8), n

    def frames_at(self, times):
        """[(t, imagem RGB uint8)] para checagem rápida"""
        return [(t, self.frame(int(round(t * FPS)))[0]) for t in times]

    def write_video(self, path, ffmpeg, log=print):
        import subprocess
        cmd = [ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
               '-vf', 'scale=out_color_matrix=bt709:out_range=tv,format=yuv420p',
               '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-maxrate', '16M', '-bufsize', '32M',
               '-profile:v', 'high', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
               '-movflags', '+faststart', path]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for fi in range(self.tl.nfr):
                fr, n = self.frame(fi)
                p.stdin.write(fr.tobytes())
                if fi % 30 == 0:
                    log(f'quadro {fi}/{self.tl.nfr} (t={fi / FPS:.1f}s, amostras de blur {n})')
        finally:
            p.stdin.close()
            code = p.wait()
        if code != 0:
            raise RuntimeError(f'ffmpeg falhou ao codificar o vídeo (código {code})')
