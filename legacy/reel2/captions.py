"""Etapa 2: caption layers for Reel 2 (Penfolds Ampoule).

Series type system (same as Reel 1):
  sentences  Playfair Display 400, warm white
  numbers    gold #E8D5A3 (the "12" and the price)
  price      Playfair Display 700, 140 px, same height as Reel 1's cover
  question   Playfair Display 400 italic, gold  -> recurring series sign-off
Every layer is rendered at 2x and downsampled (premultiplied) for clean edges,
and carries its own shadow so it can be animated separately in the video.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, SS = 1080, 1920, 2
WHITE = (245, 242, 236)
GOLD = (0xE8, 0xD5, 0xA3)
F = 'fonts/PlayfairDisplay-{}.ttf'

def font(style, px):
    return ImageFont.truetype(F.format(style), px * SS, layout_engine=ImageFont.Layout.RAQM)

SENT = font('400-normal', 76)
PRICE = font('700-normal', 140)
ASK = font('400-italic', 112)

# each line: list of (text, colour) runs, font, top y (1x px)
CAPTIONS = {
    't1_sala': [
        ([('Só ', WHITE), ('12', GOLD, ['lnum']), (' garrafas no mundo.', WHITE)], SENT, 285),
        ([('E nenhuma tem rolha.', WHITE)], SENT, 385),
    ],
    't2_vidro': [
        ([('Ela é lacrada em vidro', WHITE)], SENT, 285),
        ([('soprado à mão.', WHITE)], SENT, 385),
    ],
    't3_preco': [
        ([('AU$ 168.000', GOLD)], PRICE, 335),
    ],
    't4_pergunta': [
        ([('Abre ou guarda?', GOLD)], ASK, 330),
    ],
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
    mask = Image.new('L', (cw, ch), 0)          # glyph coverage, for the shadows
    dt, dm = ImageDraw.Draw(text), ImageDraw.Draw(mask)
    for runs, f, top in lines:
        runs = [r if len(r) == 3 else (r[0], r[1], None) for r in runs]
        total = sum(f.getlength(t, features=ft) for t, _, ft in runs)
        x = (cw - total) / 2
        y = top * SS
        for t, col, ft in runs:
            dt.text((x, y), t, font=f, fill=col + (255,), features=ft)
            dm.text((x, y), t, font=f, fill=255, features=ft)
            x += f.getlength(t, features=ft)
    # halo: wide, soft darkening so text holds on lit wood; tight: crisp drop shadow
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

os.makedirs('captions', exist_ok=True)
for name, lines in CAPTIONS.items():
    layer = render_layer(lines)
    layer.save(f'captions/{name}_texto.png')
    bg = Image.open(f'shots/{name}_1080.jpg').convert('RGBA')
    Image.alpha_composite(bg, layer).convert('RGB').save(f'captions/{name}_final.jpg', quality=95)
    print(name, 'ok')
