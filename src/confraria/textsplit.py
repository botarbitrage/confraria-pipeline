"""Separa o texto que o Gemini desenhou na imagem do fundo (modo "4 imagens").

O prompt pede o texto no terço superior, claro sobre uma área escura e limpa. Isso permite:
  1. achar a faixa do texto (linhas claras no topo, antes da cena começar);
  2. estimar o fundo local (mediana larga) e marcar o que é texto (mais claro que o fundo);
  3. reconstruir o fundo por baixo (inpaint) -> "placa" sem texto;
  4. extrair o texto como camada RGBA (cor do traço + alfa suave), para animar separado.

Tudo em float32 RGB 0..1, na resolução em que a imagem chegar (o master 2x).
Se a separação não for confiável (sem faixa clara, texto encostado na cena), devolve
`ok=False` e o render usa a imagem inteira com câmera suave (sem animar o texto).
"""
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Split:
    ok: bool
    plate: np.ndarray                 # fundo sem texto (RGB float)
    color: np.ndarray                 # cor do traço (RGB float), válida onde alpha > 0
    alpha: np.ndarray                 # alfa do texto (float 0..1)
    band: tuple = (0, 0)              # (y0, y1) da faixa do texto
    lines: list = field(default_factory=list)   # [(y0, y1), ...] das linhas, de cima para baixo
    gold: np.ndarray = None           # peso 0..1 de "dourado" no texto (para o brilho do preço)
    reason: str = ''


def luminance(img):
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def _runs(flags):
    """[(início, fim_exclusivo)] de sequências True"""
    out, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(flags)))
    return out


def find_band(L, top_frac=0.45, thr=0.42, max_gap_frac=0.06):
    """Faixa do texto: grupos de linhas claras a partir do topo, parando no primeiro vão grande."""
    h, w = L.shape
    lim = int(h * top_frac)
    counts = (L[:lim] > thr).sum(1)
    rows = counts >= max(2, w // 400)
    runs = [r for r in _runs(rows) if r[1] - r[0] >= max(3, h // 400)]
    if not runs:
        return None
    band = [runs[0][0], runs[0][1]]
    for a, b in runs[1:]:
        if a - band[1] > h * max_gap_frac:
            break
        band[1] = b
    return tuple(band)


def split_lines(alpha, band, min_gap):
    y0, y1 = band
    prof = (alpha[y0:y1] > 0.15).sum(1) > 0
    runs = _runs(prof)
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [(y0 + a, y0 + b) for a, b in merged if b - a > 4]


def split(img):
    """img RGB float 0..1 -> Split"""
    h, w = img.shape[:2]
    L = luminance(img)
    band = find_band(L)
    empty = np.zeros((h, w), np.float32)
    if band is None:
        return Split(False, img, img, empty, reason='não achei texto claro no topo da imagem')
    pad = max(6, h // 120)
    y0, y1 = max(0, band[0] - pad), min(h, band[1] + pad)

    # a cena não pode começar dentro da faixa: o fundo logo abaixo precisa ser escuro
    below = L[y1:min(h, y1 + h // 40)]
    if below.size and float(np.median(below)) > 0.22:
        return Split(False, img, img, empty, band, reason='o texto encosta na cena (fundo claro logo abaixo)')

    reg = img[y0:y1]
    Lr = L[y0:y1]
    L8 = np.clip(Lr * 255, 0, 255).astype(np.uint8)
    k = max(31, (h // 40) | 1)
    bgL = cv2.medianBlur(L8, min(k, 255)).astype(np.float32) / 255
    diff = Lr - bgL
    core = diff > 0.30
    anyt = diff > 0.035
    grow = max(2, h // 700)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    anyt = cv2.dilate(anyt.astype(np.uint8), ker) > 0
    if core.sum() < 50:
        return Split(False, img, img, empty, band, reason='texto muito fraco para separar')

    reg8 = np.clip(reg * 255, 0, 255).astype(np.uint8)
    hole = cv2.dilate(anyt.astype(np.uint8), ker)
    plate_r = cv2.inpaint(reg8, hole, max(3, grow * 2), cv2.INPAINT_TELEA).astype(np.float32) / 255
    # suaviza o remendo (o inpaint deixa manchas): mistura com um desfoque largo dentro do buraco
    soft = cv2.GaussianBlur(cv2.dilate(hole, ker, iterations=3).astype(np.float32), (0, 0), grow * 3)[..., None]
    plate_r = plate_r * (1 - soft) + cv2.GaussianBlur(plate_r, (0, 0), grow * 8) * soft
    # cor do traço: espalha a cor dos pixels "cheios" para a borda antialias
    fill = (hole > 0) & ~core
    color_r = cv2.inpaint(reg8, fill.astype(np.uint8), max(3, grow * 2), cv2.INPAINT_TELEA).astype(np.float32) / 255

    Lp, Lc = luminance(plate_r), luminance(color_r)
    den = np.maximum(Lc - Lp, 0.08)
    a = np.clip((Lr - Lp) / den, 0, 1)
    a[core] = 1.0
    a[hole == 0] = 0.0
    a = a.astype(np.float32)

    plate = img.copy()
    plate[y0:y1] = plate_r
    color = img.copy()
    color[y0:y1] = color_r
    alpha = empty.copy()
    alpha[y0:y1] = a

    # "dourado": b* alto no Lab (amarelado) em relação ao branco quente
    lab = cv2.cvtColor(np.clip(color_r, 0, 1), cv2.COLOR_RGB2Lab)
    gold_r = np.clip((lab[..., 2] - 14) / 12, 0, 1) * (a > 0)
    gold = empty.copy()
    gold[y0:y1] = gold_r

    lines = split_lines(alpha, (y0, y1), min_gap=max(3, h // 500))
    return Split(True, plate, color, alpha, (y0, y1), lines, gold)
