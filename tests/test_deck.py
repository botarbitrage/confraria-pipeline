"""Modo 4 imagens: separação do texto e linha do tempo (sem renderizar vídeo)."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from confraria import deck, textsplit


def synthetic(w=768, h=1376, text=('64 garrafas', 'de 1951 a 2015.')):
    """topo escuro com 2 linhas claras + "cena" clara na metade de baixo"""
    im = Image.new('RGB', (w, h), (18, 16, 14))
    d = ImageDraw.Draw(im)
    d.rectangle([0, int(h * 0.45), w, h], fill=(150, 110, 70))
    font = ImageFont.load_default(size=60)
    for i, line in enumerate(text):
        d.text((120, 220 + i * 90), line, fill=(245, 242, 236), font=font)
    return np.asarray(im).astype(np.float32) / 255


def test_split_finds_two_lines_and_cleans_plate():
    img = synthetic()
    sp = textsplit.split(img)
    assert sp.ok, sp.reason
    assert len(sp.lines) == 2
    y0, y1 = sp.band
    # o fundo reconstruído não tem mais o texto claro
    assert textsplit.luminance(sp.plate[y0:y1]).max() < 0.25
    # fundo + texto recompõe a imagem
    comp = sp.plate * (1 - sp.alpha[..., None]) + sp.color * sp.alpha[..., None]
    assert np.abs(comp - img).mean() < 0.01


def test_split_refuses_when_no_text():
    img = synthetic(text=())
    assert not textsplit.split(img).ok


def test_timeline_cuts_and_camera_inside_image():
    cfg = deck.DeckConfig('x', ['a', 'b', 'c', 'd'], screen=3.5, last=4.5)
    tl = deck.DeckTimeline(cfg, 1536, 2752)
    assert (tl.c1, tl.c2, tl.c3, tl.dur) == (3.5, 7.0, 10.5, 15.0)
    assert tl.nfr == 450
    for t in np.linspace(0, tl.dur, 61):
        cx, cy, w = tl.camera(t)
        h = w * 16 / 9
        assert w / 2 <= cx <= 1536 - w / 2 + 1e-6
        assert h / 2 <= cy <= 2752 - h / 2 + 1e-6
    assert tl.zoom(tl.c2) > tl.zoom(tl.c2 - 0.8)          # empurrão no corte
