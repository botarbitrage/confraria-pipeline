"""Build the four Reel 2 shots (Penfolds Ampoule) from one master image.

All framing is defined in SOURCE pixel coordinates (768x1376 original) and scaled
to whatever master is used (x4 AI upscale, or a Lanczos stand-in).
Outputs per shot:
  shots/<name>.png        full-quality crop at master resolution (for the video phase)
  shots/<name>_1080.jpg   1080x1920 preview
"""
import sys, os
import numpy as np
from PIL import Image, ImageFilter

SRC_W, SRC_H = 768, 1376
OUT_W, OUT_H = 1080, 1920
M = None   # master image, loaded in main (or by the video renderer)
K = 4.0    # master scale factor

# Key features (source coords, measured): cabinet top ~375, lamp (383,424),
# neck top ~552, glass bell 600-780, cone 830-1000, silver tip ~1065,
# cabinet shelf ~1100, legs ~1220, sconces (61,567) & (702,568).
SHOTS = {
    # name: (crop width in source px, crop top y in source px)
    't1_sala':    (768, 5),    # wide: the room, object small in its case (lit)
    't2_vidro':   (340, 372),  # close: glass bell centred, lamp above the text band
    't3_preco':   (540, 295),  # medium-close hero / cover: pendant fills the case
    't4_pergunta':(768, 5),    # wide again, lights out: only the pendant glows
}

def crop(name):
    w, top = SHOTS[name]
    h = w * 16 / 9
    x0 = (SRC_W - w) / 2
    box = tuple(round(v * K) for v in (x0, top, x0 + w, top + h))
    return M.crop(box), (x0, top, w, h)

def to_np(im):
    return np.asarray(im).astype(np.float32) / 255.0

def to_im(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))

def src_grid(geom, shape):
    """Source-space coordinates for every pixel of a crop."""
    x0, top, w, h = geom
    hh, ww = shape[:2]
    ys = top + (np.arange(hh) + 0.5) / hh * h
    xs = x0 + (np.arange(ww) + 0.5) / ww * w
    return np.meshgrid(xs, ys)

def ellipse_mask(X, Y, cx, cy, rx, ry, soft=1.0):
    d = np.sqrt(((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2)
    return np.clip(1.0 - (d - 1.0) / soft, 0, 1) ** 1.5 if soft > 0 else (d <= 1).astype(np.float32)

def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)

def red_mask(a):
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    # wine red only: strong red AND green/blue well below red (excludes orange lamp glow)
    ratio = np.maximum(g, b) / np.maximum(r, 1e-3)
    return smoothstep(0.10, 0.28, r - np.maximum(g, b)) * smoothstep(0.55, 0.38, ratio)

def base_grade(a):
    # gentle, shared look: tiny lift in highlights of the wine, keep blacks rich
    lum = a.mean(axis=2, keepdims=True)
    curve = lum + 0.06 * np.sin(np.pi * lum) * (lum - 0.25)  # soft S
    a = a * (curve / np.maximum(lum, 1e-4))
    return np.clip(a, 0, 1)

def grade(name, im, geom):
    a = base_grade(to_np(im))
    X, Y = src_grid(geom, a.shape)
    if name == 't3_preco':
        # dramatic hero: pool of light on the case, room falls away
        # pool of light tight on the pendant; the case and room fall away
        pool = ellipse_mask(X, Y, 384, 810, 120, 330, soft=0.9)
        lamp = ellipse_mask(X, Y, 383, 424, 22, 22, soft=1.5)
        keep = np.maximum(pool, lamp)
        a = a * (0.38 + 0.62 * keep[..., None])
        # calm zone for the price: only inside the case (x 285-483), soft top and
        # bottom ramps, and the red neck protected so no hard stripe appears
        vert = smoothstep(440, 470, Y) * smoothstep(575, 535, Y)
        horiz = smoothstep(272, 300, X) * smoothstep(496, 468, X)
        protect = 1 - np.clip(red_mask(a) * 1.5, 0, 1)
        band = vert * horiz * protect
        band = np.asarray(Image.fromarray((band * 255).astype(np.uint8))
                          .filter(ImageFilter.GaussianBlur(a.shape[1] / 120))).astype(np.float32) / 255
        a = a * (1 - 0.45 * band[..., None])
    if name == 't4_pergunta':
        # lights out: only the lamp, the pendant and its glow survive
        rm = np.asarray(Image.fromarray((red_mask(a) * 255).astype(np.uint8))
                        .filter(ImageFilter.GaussianBlur(a.shape[1] / 90))).astype(np.float32) / 255
        glow = ellipse_mask(X, Y, 384, 810, 70, 270, soft=0.8)
        lamp = ellipse_mask(X, Y, 383, 424, 18, 18, soft=2.0)
        beam = ellipse_mask(X, Y, 384, 560, 40, 150, soft=0.8) * 0.5
        keep = np.maximum.reduce([np.clip(rm * 1.3, 0, 1), glow, lamp, beam])
        a = a * (0.12 + 0.88 * keep[..., None])
        # wall lamps switched down to a warm ember (keep their hue, no grey ghosts)
        for sx, sy in ((61, 567), (702, 568)):
            s_m = ellipse_mask(X, Y, sx, sy, 70, 80, soft=1.2)
            a = a * (1 - 0.8 * s_m[..., None])
    return to_im(a)

if __name__ == '__main__':
    master_path = sys.argv[1] if len(sys.argv) > 1 else 'ampoule_x4.png'
    M = Image.open(master_path).convert('RGB')
    K = M.width / SRC_W
    os.makedirs('shots', exist_ok=True)
    for name in SHOTS:
        im, geom = crop(name)
        im = grade(name, im, geom)
        im.save(f'shots/{name}.png')
        prev = im.resize((OUT_W, OUT_H), Image.LANCZOS)
        prev.save(f'shots/{name}_1080.jpg', quality=95)
        print(name, im.size)
