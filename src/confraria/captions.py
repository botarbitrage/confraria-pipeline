"""Camadas de texto transparentes das 4 telas (sistema tipográfico da série).

  frases    Playfair Display 400, 76 px, branco quente, linhas em y 285 e 385
            (uma linha só fica em y 335, o centro do bloco de duas)
  números   dourado #E8D5A3 com algarismos alinhados (`lnum`) quando destacados numa frase
  preço     Playfair Display 700, 140 px, dourado, y 335
  pergunta  Playfair Display itálico 400, 112 px, dourado, y 330

Cada camada é desenhada em 2x e reduzida (pré-multiplicada) para bordas limpas, e traz a
própria sombra/halo, para poder ser animada separadamente no vídeo.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import fonts

W, H, SS = 1080, 1920, 2
WHITE = (245, 242, 236)
GOLD = (0xE8, 0xD5, 0xA3)
COLOR = {'w': WHITE, 'gold': GOLD}

SENT_PX, PRICE_PX, ASK_PX = 76, 140, 112
LINE_Y = (285, 385)
SINGLE_LINE_Y = 335
PRICE_Y, ASK_Y = 335, 330

LAYER_NAMES = ('t1_sala', 't2_vidro', 't3_preco', 't4_pergunta')


def _sentence_lines(lines):
    """[[(texto, cor), ...], ...] -> especificação de desenho por linha"""
    ys = LINE_Y if len(lines) == 2 else (SINGLE_LINE_Y,)
    out = []
    for runs, y in zip(lines, ys):
        # números em dourado no meio de uma frase usam algarismos alinhados
        out.append(([(t, COLOR[c], c == 'gold') for t, c in runs], '400-normal', SENT_PX, y))
    return out


def layer_specs(reel):
    """nome da camada -> lista de (trechos[(texto, rgb, lnum)], estilo, px, y)"""
    return {
        't1_sala': _sentence_lines(reel.lines_t1),
        't2_vidro': _sentence_lines(reel.lines_t2),
        't3_preco': [([(reel.price, GOLD, False)], '700-normal', PRICE_PX, PRICE_Y)],
        't4_pergunta': [([(reel.question, GOLD, False)], '400-italic', ASK_PX, ASK_Y)],
    }


def premult_resize(img, size):
    a = np.asarray(img).astype(np.float32) / 255.0
    rgb, al = a[..., :3] * a[..., 3:4], a[..., 3:4]

    def rs(ch):
        return np.asarray(Image.fromarray((ch * 255).astype(np.uint8)).resize(size, Image.LANCZOS)).astype(np.float32) / 255
    rgb_r = np.dstack([rs(rgb[..., i]) for i in range(3)])
    al_r = rs(al[..., 0])[..., None]
    out = np.where(al_r > 1e-4, rgb_r / np.maximum(al_r, 1e-4), 0)
    return Image.fromarray((np.dstack([np.clip(out, 0, 1), al_r]) * 255 + 0.5).astype(np.uint8), 'RGBA')


def render_layer(lines):
    cw, ch = W * SS, H * SS
    text = Image.new('RGBA', (cw, ch), (0, 0, 0, 0))
    mask = Image.new('L', (cw, ch), 0)          # cobertura dos glifos, para as sombras
    dt, dm = ImageDraw.Draw(text), ImageDraw.Draw(mask)
    for runs, style, px, top in lines:
        loaded = [(t, col, fonts.load(style, px * SS, lnum)) for t, col, lnum in runs]
        total = sum(f.getlength(t, features=ft) for t, _, (f, ft) in loaded)
        x = (cw - total) / 2
        y = top * SS
        for t, col, (f, ft) in loaded:
            dt.text((x, y), t, font=f, fill=col + (255,), features=ft)
            dm.text((x, y), t, font=f, fill=255, features=ft)
            x += f.getlength(t, features=ft)
    # halo: escurecimento largo e suave, para o texto segurar sobre madeira iluminada;
    # tight: sombra curta e nítida
    halo = mask.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(38 * SS))
    tight = Image.new('L', (cw, ch), 0)
    tight.paste(mask, (0, 3 * SS))
    tight = tight.filter(ImageFilter.GaussianBlur(6 * SS))
    shadow_a = np.clip(np.asarray(halo).astype(np.float32) * 0.62 * 1.6
                       + np.asarray(tight).astype(np.float32) * 0.85, 0, 255).astype(np.uint8)
    layer = Image.new('RGBA', (cw, ch), (0, 0, 0, 0))
    layer.putalpha(Image.fromarray(shadow_a))
    layer = Image.alpha_composite(layer, text)
    return premult_resize(layer, (W, H))


def render_layers(reel):
    """dict nome -> imagem RGBA 1080x1920"""
    return {name: render_layer(lines) for name, lines in layer_specs(reel).items()}


def save_layers(layers, directory):
    os.makedirs(directory, exist_ok=True)
    for name, im in layers.items():
        im.save(os.path.join(directory, f'{name}_texto.png'))
