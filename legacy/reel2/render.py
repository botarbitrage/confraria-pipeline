"""Etapa 3: render Reel 2 as one continuous virtual-camera move over the graded master.

Timeline (15 s, 30 fps):
  0.0-3.5   screen 1  the room          (caption 1 on from frame 0)
  3.2-3.8   fast push from the room into the glass (motion-blurred)
  3.5-7.0   screen 2  the glass
  6.7-7.3   pull back to the hero framing, light narrows onto the ampoule
  7.0-10.5  screen 3  the price (cover)
  10.0-11.0 pull back while the room lights go out
  10.5-15.0 screen 4  "Abre ou guarda?" (held longest)
Keyframes land exactly on the four approved stills.
"""
import sys, math, subprocess
import numpy as np
import cv2
from scipy.interpolate import PchipInterpolator

W, H, FPS, DUR = 1080, 1920, 30, 15.0
NFR = int(round(DUR * FPS))
K = 4.0  # master px per source px

# ---------------- camera: (t, crop width in source px, centre y in source px)
KEYS = [
    (0.0, 768, 688),
    (3.2, 730, 686),
    (3.8, 340, 674),   # = still 2
    (6.7, 318, 670),
    (7.3, 540, 775),   # = still 3 (cover)
    (10.0, 520, 767),
    (11.0, 700, 668),
    (15.0, 768, 688),  # = still 4
]
kt = np.array([k[0] for k in KEYS])
f_logw = PchipInterpolator(kt, np.log([k[1] for k in KEYS]))
f_cy = PchipInterpolator(kt, [k[2] for k in KEYS])
CX = 384.0

def camera(t):
    t = min(max(t, 0.0), DUR)
    w = float(np.exp(f_logw(t)))
    h = w * 16 / 9
    cy = float(f_cy(t))
    cy = min(max(cy, h / 2), 1376 - h / 2)
    return CX, cy, w

def ss(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)

def grade_weights(t):
    a = ss(6.7, 7.3, t)          # base -> t3
    b = ss(10.0, 11.0, t)        # t3 -> t4 (lights out)
    return {'base': 1 - a, 't3': a * (1 - b), 't4': b}

def caption_state(t):
    """opacity (and scale for the price) per caption layer"""
    return {
        't1_sala':     (1 - ss(3.0, 3.2, t), 1.0),
        't2_vidro':    (ss(3.85, 4.15, t) * (1 - ss(6.5, 6.7, t)), 1.0),
        't3_preco':    (ss(7.3, 7.55, t) * (1 - ss(9.8, 10.0, t)), 1.035 - 0.035 * ss(7.3, 7.65, t)),
        't4_pergunta': (ss(11.0, 11.5, t), 1.0),
    }

# ---------------- rendering helpers
def load_pyramid(key):
    g = np.load(f'grade_{key}.npy')
    levels = [g]
    while levels[-1].shape[1] > 1400:
        lv = levels[-1]
        levels.append(cv2.resize(lv, (lv.shape[1] // 2, lv.shape[0] // 2), interpolation=cv2.INTER_AREA))
    return levels

def view(pyr, cx, cy, w, out_w, out_h, interp):
    """Render the source-space rect (centre cx,cy, width w) to out_w x out_h."""
    h = w * 16 / 9
    # pick the smallest level that still has >= out_w pixels across the view
    L = 0
    while L + 1 < len(pyr) and w * K / (2 ** (L + 1)) >= out_w:
        L += 1
    kL = K / (2 ** L)
    s = out_w / (w * kL)
    x0 = (cx - w / 2) * kL
    y0 = (cy - h / 2) * kL
    A = np.array([[s, 0, -s * x0 + 0.5 * s - 0.5], [0, s, -s * y0 + 0.5 * s - 0.5]], dtype=np.float64)
    return cv2.warpAffine(pyr[L], A, (out_w, out_h), flags=interp, borderMode=cv2.BORDER_REFLECT)

def background(t, pyrs):
    """Graded, motion-blurred background frame (float32, 0-255)."""
    dt = 1.0 / FPS
    cx, cy, w = camera(t)
    _, cy2, w2 = camera(t + dt)
    # on-screen displacement per frame at the frame corner (zoom + tilt)
    disp = abs(math.log(w2 / w)) * 1100 + abs(cy2 - cy) * (W / w)
    blur_len = disp * 0.5                      # 180-degree shutter
    n = int(min(16, max(1, math.ceil(blur_len / 2.0))))
    weights = {k: v for k, v in grade_weights(t).items() if v > 1e-3}
    acc = np.zeros((H, W, 3), np.float32)
    for i in range(n):
        ts = t + ((i + 0.5) / n - 0.5) * 0.5 * dt if n > 1 else t
        cx_s, cy_s, w_s = camera(ts)
        for k, wt in weights.items():
            if n == 1:   # still-ish: 2x supersampled cubic, then area-downsample
                im = view(pyrs[k], cx_s, cy_s, w_s, W * 2, H * 2, cv2.INTER_CUBIC)
                im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
            else:        # moving: motion blur hides aliasing, 1x linear is enough
                im = view(pyrs[k], cx_s, cy_s, w_s, W, H, cv2.INTER_LINEAR)
            acc += im.astype(np.float32) * (wt / n)
    return acc, n

def load_caption(name):
    im = cv2.imread(f'captions/{name}_texto.png', cv2.IMREAD_UNCHANGED)  # BGRA
    rgba = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA).astype(np.float32) / 255.0
    rgba[..., :3] *= rgba[..., 3:4]   # premultiply
    return rgba

def composite_caption(frame, cap, opacity, scale):
    if opacity <= 1e-3:
        return frame
    c = cap
    if abs(scale - 1.0) > 1e-4:   # settle around the price's optical centre
        cxp, cyp = W / 2, 425.0
        A = np.array([[scale, 0, (1 - scale) * cxp], [0, scale, (1 - scale) * cyp]])
        c = cv2.warpAffine(cap, A, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    a = c[..., 3:4] * opacity
    return frame * (1 - a) + c[..., :3] * 255.0 * opacity

def render_frame(fi, pyrs, caps, rng_seed=7):
    t = fi / FPS
    bg, n = background(t, pyrs)
    # fine monochrome grain: keeps dark gradients from banding after Instagram's re-encode
    rng = np.random.default_rng(rng_seed * 100003 + fi)
    grain = rng.normal(0.0, 1.4, (H, W, 1)).astype(np.float32)
    frame = bg + grain
    for name, (op, sc) in caption_state(t).items():
        frame = composite_caption(frame, caps[name], op, sc)
    return np.clip(frame + 0.5, 0, 255).astype(np.uint8), n

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'video'
    pyrs = {k: load_pyramid(k) for k in ('base', 't3', 't4')}
    caps = {n: load_caption(n) for n in ('t1_sala', 't2_vidro', 't3_preco', 't4_pergunta')}
    if mode == 'frames':      # quick check at given times
        times = [float(x) for x in sys.argv[2:]]
        for t in times:
            fr, n = render_frame(int(round(t * FPS)), pyrs, caps)
            cv2.imwrite(f'check_{t:05.2f}.jpg', cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(t, 'subsamples', n, flush=True)
        return
    out = sys.argv[2] if len(sys.argv) > 2 else 'reel2_video_sem_audio.mp4'
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
           '-vf', 'scale=out_color_matrix=bt709:out_range=tv,format=yuv420p',
           '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-maxrate', '16M', '-bufsize', '32M', '-profile:v', 'high',
           '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
           '-movflags', '+faststart', out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(NFR):
        fr, n = render_frame(fi, pyrs, caps)
        p.stdin.write(fr.tobytes())
        if fi % 30 == 0:
            print(f'frame {fi}/{NFR} (t={fi / FPS:.1f}s, blur samples {n})', flush=True)
    p.stdin.close()
    p.wait()
    print('done', out)

if __name__ == '__main__':
    main()
