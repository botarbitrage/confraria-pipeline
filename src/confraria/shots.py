"""Enquadramentos (SHOTS) e looks (grades) das 4 telas, derivados de `focus` do reel.json.

Todo o enquadramento é definido em coordenadas da imagem de ORIGEM (antes do upscale) e
escalado para o master usado (x4 por IA ou um substituto Lanczos).

Telas e enquadramentos:
  t1_sala      largo: a sala inteira, largura total da imagem, objeto pequeno (luzes acesas)
  t2_vidro     detalhe: `detail_w` de largura centrado em (detail_cx, detail_cy)
  t3_preco     herói / capa: altura do quadro = 2 x hero_h, herói em 53,6% da altura
  t4_pergunta  largo de novo, luzes apagadas (só o herói e o brilho sobrevivem);
               sem `lamp` nem `sconces` não há luzes apagando e a tela 4 mantém o look base
"""
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

OUT_W, OUT_H = 1080, 1920
HERO_Y_FRAC = 515.0 / 960.0     # posição vertical do herói no quadro do preço (810 - 295 de 960)
LOOKS = ('base', 't3', 't4')
SHOT_LOOK = {'base': 't1_sala', 't3': 't3_preco', 't4': 't4_pergunta'}


@dataclass
class Scene:
    """Um Reel + as dimensões da imagem de origem; ponto único de derivação dos enquadramentos."""
    reel: object
    src_w: int
    src_h: int

    # ---- enquadramentos: nome -> (largura do recorte, cx, cy) em px de origem
    @property
    def hero_w(self):
        return self.reel.focus['hero_h'] * 2 * 9 / 16

    @property
    def hero_cy_frame(self):
        """centro vertical do recorte t3 (o herói fica em HERO_Y_FRAC do quadro)"""
        h3 = self.hero_w * 16 / 9
        return self.reel.focus['hero_cy'] - (HERO_Y_FRAC - 0.5) * h3

    @property
    def shots(self):
        f = self.reel.focus
        return {
            't1_sala': (float(self.src_w), self.src_w / 2, self.src_h / 2),
            't2_vidro': (f['detail_w'], f['detail_cx'], f['detail_cy']),
            't3_preco': (self.hero_w, f['hero_cx'], self.hero_cy_frame),
            't4_pergunta': (float(self.src_w), self.src_w / 2, self.src_h / 2),
        }

    def geom(self, name):
        """(x0, top, w, h) do recorte em px de origem, já contido dentro da imagem"""
        w, cx, cy = self.shots[name]
        h = w * 16 / 9
        x0 = min(max(cx - w / 2, 0.0), self.src_w - w)
        top = min(max(cy - h / 2, 0.0), self.src_h - h)
        return x0, top, w, h

    def frame_to_src(self, name, xo, yo):
        """pixel do quadro de saída (1080x1920) -> coordenadas de origem, para o recorte `name`"""
        x0, top, w, h = self.geom(name)
        return x0 + xo / OUT_W * w, top + yo / OUT_H * h


# ---------------------------------------------------------------- utilidades de imagem
def to_np(im):
    return np.asarray(im).astype(np.float32) / 255.0


def to_im(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))


def src_grid(geom, shape):
    """Coordenadas de origem de cada pixel de um recorte (esparsas: X em (1,w), Y em (h,1))."""
    x0, top, w, h = geom
    hh, ww = shape[:2]
    ys = (top + (np.arange(hh, dtype=np.float32) + 0.5) / hh * h)[:, None]
    xs = (x0 + (np.arange(ww, dtype=np.float32) + 0.5) / ww * w)[None, :]
    return xs, ys


def ellipse_mask(X, Y, cx, cy, rx, ry, soft=1.0):
    d = np.sqrt(((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2)
    return np.clip(1.0 - (d - 1.0) / soft, 0, 1) ** 1.5 if soft > 0 else (d <= 1).astype(np.float32)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def red_mask(a):
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    # só o vermelho do vinho: vermelho forte E verde/azul bem abaixo (exclui o brilho laranja das lâmpadas)
    ratio = np.maximum(g, b) / np.maximum(r, 1e-3)
    return smoothstep(0.10, 0.28, r - np.maximum(g, b)) * smoothstep(0.55, 0.38, ratio)


def base_grade(a):
    # look comum e discreto: leve realce nas luzes do vinho, pretos ricos
    lum = a.mean(axis=2, keepdims=True)
    curve = lum + 0.06 * np.sin(np.pi * lum) * (lum - 0.25)  # S suave
    a = a * (curve / np.maximum(lum, 1e-4))
    return np.clip(a, 0, 1)


def _blur(mask, radius):
    return np.asarray(Image.fromarray((mask * 255).astype(np.uint8))
                      .filter(ImageFilter.GaussianBlur(radius))).astype(np.float32) / 255


def grade(scene, name, im, geom):
    """Aplica o look da tela `name` a `im` (recorte com geometria `geom` em px de origem)."""
    f = scene.reel.focus
    hx, hy, hh = f['hero_cx'], f['hero_cy'], f['hero_h']
    lamp, sconces = scene.reel.lamp, scene.reel.sconces
    a = base_grade(to_np(im))
    X, Y = src_grid(geom, a.shape)
    if name == 't3_preco':
        # herói dramático: poça de luz sobre o pêndulo, a sala some
        pool = ellipse_mask(X, Y, hx, hy, 0.25 * hh, 0.6875 * hh, soft=0.9)
        keep = pool
        if lamp:
            keep = np.maximum(pool, ellipse_mask(X, Y, lamp[0], lamp[1], 0.0458 * hh, 0.0458 * hh, soft=1.5))
        a = a * (0.38 + 0.62 * keep[..., None])
        # zona calma para o preço: faixa suave atrás do texto (em coordenadas do quadro do preço),
        # com o pescoço vermelho protegido para não aparecer uma listra dura
        _, y1 = scene.frame_to_src('t3_preco', 0, 290)
        _, y2 = scene.frame_to_src('t3_preco', 0, 350)
        _, y3 = scene.frame_to_src('t3_preco', 0, 480)
        _, y4 = scene.frame_to_src('t3_preco', 0, 560)
        x1, _ = scene.frame_to_src('t3_preco', 316, 0)
        x2, _ = scene.frame_to_src('t3_preco', 372, 0)
        x3, _ = scene.frame_to_src('t3_preco', 708, 0)
        x4, _ = scene.frame_to_src('t3_preco', 764, 0)
        vert = smoothstep(y1, y2, Y) * smoothstep(y4, y3, Y)
        horiz = smoothstep(x1, x2, X) * smoothstep(x4, x3, X)
        protect = 1 - np.clip(red_mask(a) * 1.5, 0, 1)
        band = _blur(vert * horiz * protect, a.shape[1] / 120)
        a = a * (1 - 0.45 * band[..., None])
    if name == 't4_pergunta':
        # luzes apagadas: só o pêndulo, o vermelho do vinho e o brilho (e a lâmpada) sobrevivem
        rm = _blur(red_mask(a), a.shape[1] / 90)
        parts = [np.clip(rm * 1.3, 0, 1), ellipse_mask(X, Y, hx, hy, 0.1458 * hh, 0.5625 * hh, soft=0.8)]
        if lamp:
            parts.append(ellipse_mask(X, Y, lamp[0], lamp[1], 0.0375 * hh, 0.0375 * hh, soft=2.0))
            parts.append(ellipse_mask(X, Y, lamp[0], lamp[1] + 0.2833 * hh, 0.0833 * hh, 0.3125 * hh, soft=0.8) * 0.5)
        keep = np.maximum.reduce(parts)
        a = a * (0.12 + 0.88 * keep[..., None])
        # apliques baixados a brasa quente (mantêm o matiz, sem fantasmas cinza)
        for sx, sy in sconces:
            s_m = ellipse_mask(X, Y, sx, sy, 0.1458 * hh, 0.1667 * hh, soft=1.2)
            a = a * (1 - 0.8 * s_m[..., None])
    return to_im(a)


# ---------------------------------------------------------------- recortes e looks de tela cheia
def master_scale(scene, master):
    return master.width / scene.src_w


def crop(scene, master, name):
    """Recorte do master para a tela `name`; devolve (imagem, geom)."""
    K = master_scale(scene, master)
    x0, top, w, h = scene.geom(name)
    box = tuple(round(v * K) for v in (x0, top, x0 + w, top + h))
    return master.crop(box), (x0, top, w, h)


def still(scene, master, name):
    """Quadro estático 1080x1920 da tela `name` (com look, sem texto)."""
    im, geom = crop(scene, master, name)
    return grade(scene, name, im, geom).resize((OUT_W, OUT_H), Image.LANCZOS)


def looks_needed(scene):
    return LOOKS if scene.reel.lights_out else ('base', 't3')


def prepare_grades(scene, master, workdir=None, log=print):
    """Grada o master inteiro uma vez por look, para a câmera do vídeo se mover livremente.

    Devolve {look: ndarray uint8}. Com `workdir`, guarda/reaproveita grade_<look>.npy
    (a chave de cache é gravada em grade_stamp.txt).
    """
    import hashlib
    import json
    import time
    f = scene.reel.focus
    key = hashlib.sha1(json.dumps([f, scene.reel.lamp, scene.reel.sconces, master.size, scene.src_w,
                                   scene.src_h], sort_keys=True).encode()
                       + master.resize((96, 96), Image.BOX).tobytes()).hexdigest()
    out = {}
    stamp = os.path.join(workdir, 'grade_stamp.txt') if workdir else None
    cached = (stamp and os.path.exists(stamp) and open(stamp).read().strip() == key
              and all(os.path.exists(os.path.join(workdir, f'grade_{k}.npy')) for k in looks_needed(scene)))
    if cached:
        log('grades: reaproveitando cache')
        return {k: np.load(os.path.join(workdir, f'grade_{k}.npy')) for k in looks_needed(scene)}
    geom = (0.0, 0.0, float(scene.src_w), float(scene.src_h))   # quadro inteiro, coords de origem
    for k in looks_needed(scene):
        t0 = time.time()
        out[k] = np.asarray(grade(scene, SHOT_LOOK[k], master, geom))
        log(f'grade {k}: {out[k].shape[1]}x{out[k].shape[0]} em {time.time() - t0:.1f}s')
        if workdir:
            os.makedirs(workdir, exist_ok=True)
            np.save(os.path.join(workdir, f'grade_{k}.npy'), out[k])
    if stamp:
        with open(stamp, 'w') as fh:
            fh.write(key)
    return out
