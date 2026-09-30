"""Modo "4 imagens" (deck): um Reel a partir das 4 telas prontas do Gemini.

Entrada: <pasta>/<nome>_1 ... <nome>_4 (.jpg/.png/.webp), uma imagem por tela, com o texto já desenhado
pelo Gemini no terço superior. Saída: <nome>.mp4 (1080x1920, 30 fps, ~15 s, trilha sintetizada) e <nome>_capa.jpg.

Como funciona
  - cada imagem é ampliada 2x (Lanczos + nitidez; Real-ESRGAN opcional) e separada em fundo + texto (textsplit);
  - o fundo tem uma câmera contínua (aproximação lenta ao longo do Reel, com "empurrões" curtos nos cortes);
    entre telas o fundo faz fusão de uma imagem para a outra (o cenário é o mesmo nas 4);
  - o texto fica parado na tela e é animado por linha (entra subindo e ganhando foco, sai desfocando);
  - efeitos: brilho de luz que atravessa as garrafas, poeira dourada flutuando, cintilação quente,
    vinheta que respira, brilho e reflexo no preço (tela 3, a capa), luzes baixando na tela 4, granulação;
  - o som é o mesmo do pipeline original (sound.Synth), sincronizado com a câmera deste modo.
Se o texto não puder ser separado com segurança, aquela tela usa a imagem inteira, com o texto parado
e sem os efeitos de texto (o aviso aparece no log).
"""
import glob
import math
import os
import subprocess
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from . import textsplit

W, H, FPS = 1080, 1920, 30
MASTER = 2                       # fator do master sobre a imagem de origem
EXTS = ('.jpg', '.jpeg', '.png', '.webp')
WARM = np.array([1.0, 0.86, 0.62], np.float32)       # cor da luz (RGB)
GOLD = np.array([0.91, 0.835, 0.64], np.float32)     # #E8D5A3


class DeckError(Exception):
    pass


def ss(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0) if e1 != e0 else float(x >= e1)
    return t * t * (3 - 2 * t)


def find_images(folder, name):
    out = []
    for i in range(1, 5):
        hits = [p for e in EXTS for p in glob.glob(os.path.join(folder, f'{name}_{i}{e}'))]
        hits += [p for e in EXTS for p in glob.glob(os.path.join(folder, f'{name}_{i}{e.upper()}'))]
        if not hits:
            raise DeckError(f'imagem da tela {i} não encontrada: {os.path.join(folder, name)}_{i}.(jpg|png|webp)')
        out.append(sorted(hits)[0])
    return out


@dataclass
class DeckConfig:
    name: str
    images: list
    screen: float = 3.5              # duração das telas 1 a 3 (s)
    last: float = 4.5                # duração da tela 4 (s)
    bed_root: str = 'A'
    ting_hz: tuple = (1318.5, 659.25)
    upscale: str = 'lanczos'         # lanczos | ai
    seed: int = 7

    @property
    def duration(self):
        return 3 * self.screen + self.last


# ---------------------------------------------------------------- preparo das imagens
def load_master(path, method='lanczos', log=print):
    im = Image.open(path).convert('RGB')
    w, h = im.size
    if abs((w / h) - 9 / 16) > 0.03:
        raise DeckError(f'{os.path.basename(path)} não é vertical 9:16 ({w}x{h})')
    if method == 'ai':
        from .upscale import realesrgan
        big = realesrgan(im, log=log)
        big = big.resize((w * MASTER, h * MASTER), Image.LANCZOS)
    else:
        big = im.resize((w * MASTER, h * MASTER), Image.LANCZOS)
    a = np.asarray(big).astype(np.float32) / 255
    if method != 'ai':                       # nitidez leve para compensar o Lanczos
        blur = cv2.GaussianBlur(a, (0, 0), 1.2)
        a = np.clip(a + 0.55 * (a - blur), 0, 1)
    return a


class Screen:
    """Uma tela: fundo (master), texto na resolução de saída, linhas animáveis."""

    def __init__(self, idx, path, cfg, log=print):
        self.idx = idx
        img = load_master(path, cfg.upscale, log)
        self.src_h, self.src_w = img.shape[:2]
        sp = textsplit.split(img)
        self.ok = sp.ok
        if not sp.ok:
            log(f'AVISO tela {idx}: {sp.reason}; usando a imagem inteira (texto parado, sem animação).')
        self.plate = sp.plate if sp.ok else img
        # recorte "de repouso" (zoom 1): largura cheia 9:16 centrada -> resolução de saída
        self.fit_w = min(self.src_w, self.src_h * 9 / 16)
        self.fit_h = self.fit_w * 16 / 9
        self.rest = (self.src_w / 2, self.src_h / 2)
        if sp.ok:
            color = self._to_out(sp.color)
            alpha = self._to_out(sp.alpha)
            gold = self._to_out(sp.gold)
            self.text_rgb = color
            self.text_a = np.clip(alpha, 0, 1)
            self.gold = np.clip(gold, 0, 1)
            sy = H / self.fit_h
            oy = self.rest[1] - self.fit_h / 2
            self.lines = [(int((a - oy) * sy), int(math.ceil((b - oy) * sy))) for a, b in sp.lines]
            self.band = (int((sp.band[0] - oy) * sy), int(math.ceil((sp.band[1] - oy) * sy)))
        else:
            self.text_rgb = None
            self.text_a = None
            self.gold = None
            self.lines = []
            self.band = (0, 0)
        # mapa de brilho do fundo (onde a luz "pega": vidro, rótulos) para o brilho que atravessa
        L = textsplit.luminance(self.plate)
        self.spec = cv2.GaussianBlur(np.clip((L - 0.30) / 0.5, 0, 1), (0, 0), 2).astype(np.float32)

    def _to_out(self, a):
        """recorte de repouso do master -> 1080x1920"""
        cx, cy = self.rest
        x0, y0 = cx - self.fit_w / 2, cy - self.fit_h / 2
        M = np.array([[W / self.fit_w, 0, -x0 * W / self.fit_w], [0, H / self.fit_h, -y0 * H / self.fit_h]],
                     np.float32)
        return cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------- linha do tempo
class DeckTimeline:
    """Cortes, câmera e estados. Expõe a interface que sound.Synth usa (c1..c3, dur, camera, lights_out)."""

    PULSE = 0.045            # empurrão extra de zoom em cada corte
    lights_out = True        # a tela 4 baixa as luzes (clique + base mais escura no som)

    def __init__(self, cfg: DeckConfig, src_w, src_h):
        s = cfg.screen
        self.c1, self.c2, self.c3 = s, 2 * s, 3 * s
        self.dur = cfg.duration
        self.nfr = int(round(self.dur * FPS))
        self.src_w, self.src_h = src_w, src_h
        self.fit_w = min(src_w, src_h * 9 / 16)
        # ponto de fuga da câmera: centro da cena, abaixo da faixa do texto
        self.anchor = (src_w / 2, src_h * 0.62)

    def zoom(self, t):
        base = 1.0 + 0.10 * ss(0, self.dur, t)                   # aproximação lenta e contínua
        p = 0.0
        for c in (self.c1, self.c2, self.c3):
            p += self.PULSE * math.exp(-((t - c) / 0.22) ** 2)     # empurrão curto no corte
        return base + p

    def camera(self, t):
        """(cx, cy, largura) em px do master, contidos na imagem"""
        t = min(max(t, 0.0), self.dur)
        z = self.zoom(t)
        w = self.fit_w / z
        h = w * 16 / 9
        ax, ay = self.anchor
        cx0, cy0 = self.src_w / 2, self.src_h / 2
        cx = ax + (cx0 - ax) / z
        cy = ay + (cy0 - ay) / z + self.src_h * 0.012 * math.sin(2 * math.pi * t / self.dur)   # leve respiro vertical
        cx = min(max(cx, w / 2), self.src_w - w / 2)
        cy = min(max(cy, h / 2), self.src_h - h / 2)
        return cx, cy, w

    def screen_at(self, t):
        return 0 if t < self.c1 else 1 if t < self.c2 else 2 if t < self.c3 else 3

    def cuts(self):
        return (0.0, self.c1, self.c2, self.c3, self.dur)


# ---------------------------------------------------------------- efeitos
class Dust:
    """poeira dourada em profundidade, subindo devagar"""

    def __init__(self, n=70, seed=7):
        rng = np.random.default_rng(seed)
        self.x = rng.uniform(0, W, n)
        self.y = rng.uniform(0, H, n)
        self.z = rng.uniform(0.2, 1.0, n)                 # profundidade (1 = perto: maior, mais desfocado)
        self.vx = rng.normal(0, 6, n)
        self.vy = -rng.uniform(8, 26, n) * self.z
        self.ph = rng.uniform(0, 2 * np.pi, n)
        self.tw = rng.uniform(0.3, 0.9, n)
        self.sprites = {}

    def sprite(self, r):
        r = max(1, int(round(r)))
        if r not in self.sprites:
            k = 4 * r + 1
            yy, xx = np.mgrid[-2 * r:2 * r + 1, -2 * r:2 * r + 1]
            g = np.exp(-(xx ** 2 + yy ** 2) / (2 * (r * 0.7) ** 2)).astype(np.float32)
            self.sprites[r] = g[:k, :k]
        return self.sprites[r]

    def draw(self, frame, t, strength=1.0):
        if strength <= 0:
            return
        for i in range(len(self.x)):
            x = (self.x[i] + self.vx[i] * t + 14 * self.z[i] * math.sin(0.7 * t + self.ph[i])) % W
            y = (self.y[i] + self.vy[i] * t) % H
            r = 1.5 + 6 * self.z[i] ** 2
            a = strength * (0.10 + 0.22 * self.z[i]) * (0.6 + 0.4 * math.sin(self.tw[i] * 2 * math.pi * t + self.ph[i]))
            sp = self.sprite(r)
            k = sp.shape[0] // 2
            xi, yi = int(x), int(y)
            x0, x1, y0, y1 = xi - k, xi + k + 1, yi - k, yi + k + 1
            if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
                continue
            frame[y0:y1, x0:x1] += (sp * a)[..., None] * WARM


def vignette_map():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.55) / (H * 0.62)) ** 2)
    return np.clip(d, 0, 1.4)


def light_sweep(spec, t, t0, t1, angle_deg=-28, width=0.10):
    """faixa de luz diagonal que atravessa a tela entre t0 e t1; devolve o ganho aditivo (H, W)"""
    if not (t0 <= t <= t1):
        return None
    p = ss(t0, t1, t)
    a = math.radians(angle_deg)
    # coordenada ao longo da direção de varredura, normalizada 0..1 na diagonal
    xs = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    ys = np.linspace(0, 1, H, dtype=np.float32)[:, None] * (H / W)
    u = xs * math.cos(a) - ys * math.sin(a)
    umin, umax = -(H / W) * abs(math.sin(a)), math.cos(a)
    u = (u - umin) / (umax - umin)
    pos = -0.2 + 1.4 * p
    band = np.exp(-((u - pos) / width) ** 2)
    return band * spec * math.sin(math.pi * p)


# ---------------------------------------------------------------- renderizador
class DeckRenderer:
    def __init__(self, cfg: DeckConfig, log=print):
        self.cfg = cfg
        self.log = log
        self.screens = []
        for i, p in enumerate(cfg.images, 1):
            log(f'tela {i}: {os.path.basename(p)} (ampliação {cfg.upscale}, separação do texto)...')
            self.screens.append(Screen(i, p, cfg, log))
        s0 = self.screens[0]
        for s in self.screens[1:]:
            if (s.src_w, s.src_h) != (s0.src_w, s0.src_h):
                raise DeckError('as 4 imagens precisam ter o mesmo tamanho '
                                f'({s0.src_w // MASTER}x{s0.src_h // MASTER} vs {s.src_w // MASTER}x{s.src_h // MASTER})')
        self.tl = DeckTimeline(cfg, s0.src_w, s0.src_h)
        self.dust = Dust(seed=cfg.seed)
        self.vig = vignette_map()
        self.rng = np.random.default_rng(cfg.seed)
        self._spec_cache = {}

    # ---- fundo
    def warp(self, img, t):
        cx, cy, w = self.tl.camera(t)
        h = w * 16 / 9
        sx = W / w
        M = np.array([[sx, 0, -(cx - w / 2) * sx], [0, sx, -(cy - h / 2) * sx]], np.float32)
        return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def background(self, t):
        tl = self.tl
        k = tl.screen_at(t)
        # fusão entre fundos nos cortes (±0,22 s)
        weights = {k: 1.0}
        for i, c in enumerate((tl.c1, tl.c2, tl.c3)):
            if abs(t - c) < 0.22:
                a = ss(c - 0.22, c + 0.22, t)
                weights = {i: 1 - a, i + 1: a}
                break
        # desfoque de movimento só perto dos empurrões
        near = min(abs(t - c) for c in (tl.c1, tl.c2, tl.c3))
        subs = [t] if near > 0.3 else [t - 1 / 90, t, t + 1 / 90]
        out = np.zeros((H, W, 3), np.float32)
        spec = np.zeros((H, W), np.float32)
        for i, wgt in weights.items():
            if wgt <= 0:
                continue
            acc = np.zeros((H, W, 3), np.float32)
            for ts in subs:
                acc += self.warp(self.screens[i].plate, ts)
            out += acc * (wgt / len(subs))
            spec += self.warp(self.screens[i].spec, t) * wgt
        return out, spec

    # ---- texto
    def text_state(self, k, t):
        """(opacidade, deslocamento y px, desfoque px, escala) de cada linha da tela k no tempo t"""
        tl = self.tl
        start, end = tl.cuts()[k], tl.cuts()[k + 1]
        n = max(1, len(self.screens[k].lines))
        states = []
        for j in range(n):
            if k == 0:
                t_in0, dur_in = -0.12 + 0.10 * j, 0.40             # o gancho aparece logo (já visível no quadro 0)
            elif k == 3:
                t_in0, dur_in = start + 0.35 + 0.15 * j, 0.80       # a pergunta entra mais devagar
            else:
                t_in0, dur_in = start + 0.12 + 0.12 * j, 0.50
            a_in = ss(t_in0, t_in0 + dur_in, t)
            if k < 3:
                t_out0 = end - 0.30
                a_out = 1 - ss(t_out0, end - 0.02, t)
            else:
                a_out = 1.0
            op = a_in * a_out
            dy = 26 * (1 - a_in) - 18 * (1 - a_out)
            blur = 7 * (1 - a_in) + 9 * (1 - a_out)
            scale = 1.0
            if k == 2:                                              # o preço "pousa"
                scale = 1.07 - 0.07 * ss(t_in0, t_in0 + 0.7, t)
            states.append((op, dy, blur, scale))
        return states

    def draw_text(self, frame, k, t):
        s = self.screens[k]
        if s.text_a is None:
            return
        lines = s.lines or [s.band]
        states = self.text_state(k, t)
        for (y0, y1), (op, dy, blur, scale) in zip(lines, states):
            if op <= 0.003:
                continue
            pad = 40
            a0, a1 = max(0, y0 - pad), min(H, y1 + pad)
            rgb = s.text_rgb[a0:a1]
            al = s.text_a[a0:a1].copy()
            al[:max(0, y0 - a0 - 6)] = 0                       # só a linha j (sem pedaço da vizinha)
            al[min(al.shape[0], y1 - a0 + 6):] = 0
            gold = s.gold[a0:a1]
            if k == 2:                                          # preço: reflexo que atravessa + brilho
                tl = self.tl
                p0 = tl.c2 + 0.95
                if p0 <= t <= p0 + 0.9:
                    p = ss(p0, p0 + 0.9, t)
                    xs = np.linspace(0, 1, W, dtype=np.float32)[None, :]
                    ys = np.linspace(0, 1, al.shape[0], dtype=np.float32)[:, None] * 0.25
                    band = np.exp(-((xs + ys - (-0.2 + 1.5 * p)) / 0.07) ** 2)
                    rgb = np.clip(rgb + (band * 0.55)[..., None] * (1 - rgb * 0.5), 0, 1)
            if scale != 1.0:
                cy = (a1 - a0) / 2
                M = cv2.getRotationMatrix2D((W / 2, cy), 0, scale)
                rgb = cv2.warpAffine(rgb, M, (W, a1 - a0), flags=cv2.INTER_LINEAR)
                al = cv2.warpAffine(al, M, (W, a1 - a0), flags=cv2.INTER_LINEAR)
                gold = cv2.warpAffine(gold, M, (W, a1 - a0), flags=cv2.INTER_LINEAR)
            if blur > 0.4:
                rgb = cv2.GaussianBlur(rgb, (0, 0), blur)
                al = cv2.GaussianBlur(al, (0, 0), blur)
            al = al * op
            # destino deslocado em y
            d0 = int(round(a0 + dy))
            d1 = d0 + (a1 - a0)
            c0, c1 = max(0, d0), min(H, d1)
            if c1 <= c0:
                continue
            sl = slice(c0 - d0, c0 - d0 + (c1 - c0))
            A = al[sl][..., None]
            region = frame[c0:c1]
            # sombra suave por baixo (legibilidade) + glow dourado no preço
            shadow = cv2.GaussianBlur(al[sl], (0, 0), 10)[..., None] * 0.45
            region *= (1 - shadow)
            if k == 2:
                glow = cv2.GaussianBlur(al[sl] * np.maximum(gold[sl], 0.5), (0, 0), 18)[..., None]
                pulse = 0.22 + 0.10 * math.sin(2 * math.pi * 0.6 * t)
                region += glow * GOLD * pulse * op
            region[:] = region * (1 - A) + rgb[sl] * A

    # ---- quadro
    def frame(self, fi):
        t = fi / FPS
        tl = self.tl
        k = tl.screen_at(t)
        img, spec = self.background(t)

        # cintilação quente (luz de adega) e luzes baixando na tela 4
        flick = 1 + 0.018 * math.sin(2 * math.pi * 0.9 * t) + 0.010 * math.sin(2 * math.pi * 2.3 * t + 1.3)
        dim = ss(tl.c3 - 0.40, tl.c3 + 0.60, t)
        img *= flick * (1 - 0.30 * dim)

        # brilho de luz atravessando as garrafas (tela 1 e na chegada do preço)
        for t0, t1, gain in ((0.7, 2.6, 0.55), (tl.c2 + 0.2, tl.c2 + 1.6, 0.45)):
            sw = light_sweep(spec, t, t0, t1)
            if sw is not None:
                img += (sw * gain)[..., None] * WARM

        # clarão quente no corte para o preço (revelação)
        flash = math.exp(-((t - tl.c2) / 0.12) ** 2)
        if flash > 0.01:
            bloom = cv2.GaussianBlur(np.clip(img - 0.45, 0, 1), (0, 0), 25)
            img += bloom * (0.55 * flash) + 0.03 * flash * WARM

        # poeira (menos quando a luz baixa)
        self.dust.draw(img, t, strength=1.0 - 0.6 * dim)

        # vinheta que respira (fecha mais na tela 4)
        v = 0.42 + 0.04 * math.sin(2 * math.pi * t / 5.0) + 0.25 * dim
        img *= (1 - v * self.vig ** 2)[..., None]

        # texto: a tela atual (e a anterior terminando de sair, caso haja sobreposição)
        self.draw_text(img, k, t)

        # granulação + fade final curto (loop limpo)
        img += self.rng.normal(0, 0.010, (H // 2, W // 2, 1)).astype(np.float32).repeat(2, 0).repeat(2, 1)
        img *= 1 - ss(tl.dur - 0.35, tl.dur, t)
        return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)

    def still(self, k):
        """quadro "de repouso" da tela k (texto inteiro, sem transições) para a capa"""
        s = self.screens[k]
        img = s._to_out(s.plate).copy()
        img *= (1 - 0.40 * self.vig ** 2)[..., None]
        if s.text_a is not None:
            A = s.text_a[..., None]
            shadow = cv2.GaussianBlur(s.text_a, (0, 0), 10)[..., None] * 0.45
            img *= 1 - shadow
            if k == 2:
                glow = cv2.GaussianBlur(s.text_a * np.maximum(s.gold, 0.5), (0, 0), 18)[..., None]
                img += glow * GOLD * 0.25
            img = img * (1 - A) + s.text_rgb * A
        return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)

    def write_video(self, path, ffmpeg):
        cmd = [ffmpeg, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
               '-vf', 'scale=out_color_matrix=bt709:out_range=tv,format=yuv420p',
               '-c:v', 'libx264', '-preset', 'slow', '-crf', '23', '-maxrate', '4.5M', '-bufsize', '9M',
               '-profile:v', 'high', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
               '-movflags', '+faststart', path]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for fi in range(self.tl.nfr):
                p.stdin.write(self.frame(fi).tobytes())
                if fi % 60 == 0:
                    self.log(f'quadro {fi}/{self.tl.nfr} (t={fi / FPS:.1f}s)')
        finally:
            p.stdin.close()
            code = p.wait()
        if code != 0:
            raise RuntimeError(f'ffmpeg falhou ao codificar o vídeo (código {code})')


class _SoundReel:
    """o mínimo de `reel` que sound.Synth usa"""

    def __init__(self, cfg):
        self.bed_root = cfg.bed_root
        self.ting_hz = list(cfg.ting_hz)
