"""CLI do pipeline.

  python -m confraria.cli make   reel.json --out out/        MP4 final + capa JPG
  python -m confraria.cli frames reel.json 0 3.5 7.3         quadros de checagem (JPG)
  python -m confraria.cli check  reel.json                   só valida o reel.json e a imagem
  python -m confraria.cli regress reel.json ref.mp4          compara quadros com um vídeo de referência
"""
import argparse
import os
import sys
import time

from PIL import Image

from . import captions as C
from . import config, mux, shots, sound, upscale
from .camera import FPS, Renderer, Timeline


def log(msg):
    print(msg, flush=True)


def load_scene(path):
    """reel.json -> (Reel, Scene). Valida o JSON e os pontos de foco contra a imagem."""
    reel = config.load(path)
    if not os.path.exists(reel.image):
        raise config.ConfigError([f'imagem não encontrada: {reel.image}'], reel.source)
    with Image.open(reel.image) as im:
        w, h = im.size
    config.validate_image(reel, w, h)
    return reel, shots.Scene(reel, w, h)


def build_renderer(reel, scene, work, args):
    """Upscale -> looks -> legendas -> Renderer (tudo com cache em `work`)."""
    master, method = upscale.make_master(reel.image, work, args.upscale, args.gpu, args.tilesize, log)
    if method != 'ai':
        log('AVISO: master Lanczos (teste): o vídeo não é o resultado final de publicação.')
    grades = shots.prepare_grades(scene, master, work, log)
    layers = C.render_layers(reel)
    C.save_layers(layers, os.path.join(work, 'captions'))
    tl = Timeline(scene)
    return Renderer(tl, grades, layers, shots.master_scale(scene, master)), tl, master, layers


def cmd_check(args):
    reel, scene = load_scene(args.reel)
    tl = Timeline(scene)
    log(f'OK: {reel.basename}  {scene.src_w}x{scene.src_h}  {tl.dur:g}s '
        f'(cortes {tl.c1:g}/{tl.c2:g}/{tl.c3:g})  luzes apagando: {"sim" if reel.lights_out else "não"}')
    return 0


def cmd_frames(args):
    reel, scene = load_scene(args.reel)
    out = args.out
    work = args.work or os.path.join(out, 'work', reel.slug)
    r, tl, _, _ = build_renderer(reel, scene, work, args)
    fdir = os.path.join(out, 'frames')
    os.makedirs(fdir, exist_ok=True)
    import cv2
    for t in args.times:
        if not 0 <= t <= tl.dur:
            raise SystemExit(f'tempo {t:g}s fora do vídeo (0 a {tl.dur:g}s)')
        fr, n = r.frame(int(round(t * FPS)))
        path = os.path.join(fdir, f'check_{t:05.2f}.jpg')
        cv2.imwrite(path, cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        log(f'{path}  (t={t:g}s, amostras de blur {n})')
    return 0


REGRESS_TIMES = (0, 3.5, 7.3, 10.5, 14.9)        # cortes das 4 telas + quadros com legenda abaixo
REGRESS_EXTRA = (5.0, 8.5, 13.0)                 # legendas das telas 2, 3 e 4 visíveis


def psnr(a, b):
    import numpy as np
    mse = float(((a.astype('float32') - b.astype('float32')) ** 2).mean())
    return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def cmd_regress(args):
    """Renderiza quadros do reel.json e compara (PSNR) com os do vídeo de referência."""
    import cv2
    reel, scene = load_scene(args.reel)
    work = args.work or os.path.join(args.out, 'work', reel.slug)
    r, tl, _, _ = build_renderer(reel, scene, work, args)
    ok = True
    log(f'{"t (s)":>6}  {"PSNR dB":>8}  {"erro médio":>10}')
    for t in REGRESS_TIMES + REGRESS_EXTRA:
        ref_png = mux.extract_frame(args.reference, t, os.path.join(work, f'ref_{t:05.2f}.png'))
        ref = cv2.cvtColor(cv2.imread(ref_png), cv2.COLOR_BGR2RGB)
        mine, _ = r.frame(int(round(t * FPS)))
        p = psnr(mine, ref)
        mae = float(abs(mine.astype('float32') - ref.astype('float32')).mean())
        flag = '' if p >= args.min_psnr else '  <-- ABAIXO DO LIMITE'
        ok &= p >= args.min_psnr
        log(f'{t:6.1f}  {p:8.1f}  {mae:10.2f}{flag}')
    log(f'limite: {args.min_psnr:g} dB (as bordas do texto diferem ~1 px entre libraqm e o layout básico do Pillow)')
    log('REGRESSÃO OK' if ok else 'REGRESSÃO FALHOU')
    return 0 if ok else 1


def cmd_make(args):
    t0 = time.time()
    reel, scene = load_scene(args.reel)
    out = args.out
    work = args.work or os.path.join(out, 'work', reel.slug)
    os.makedirs(work, exist_ok=True)
    r, tl, master, layers = build_renderer(reel, scene, work, args)

    silent = os.path.join(work, 'video_sem_audio.mp4')
    log(f'renderizando {tl.nfr} quadros ({tl.dur:g}s, {FPS} fps)...')
    r.write_video(silent, mux.ffmpeg_exe(), log)

    log('sintetizando a trilha...')
    raw = os.path.join(work, 'audio_raw.wav')
    _, peak = sound.synthesize(tl, reel, raw)
    log(f'áudio bruto: pico {peak:.3f}; normalizando (loudnorm {mux.TARGET_I:g} LUFS, TP {mux.TARGET_TP:g})...')
    norm = mux.normalize(raw, os.path.join(work, 'audio_norm.wav'))

    mp4 = mux.mux(silent, norm, os.path.join(out, f'{reel.basename}.mp4'))
    still = shots.still(scene, master, 't3_preco').convert('RGBA')
    cover = Image.alpha_composite(still, layers['t3_preco'])
    capa = mux.save_cover(cover, os.path.join(out, f'{reel.basename}_capa.jpg'))
    m = mux.measure(mp4)
    log(f'pronto em {time.time() - t0:.0f}s:\n  {mp4}\n  {capa}\n'
        f'  loudness {m["input_i"]:.1f} LUFS, true peak {m["input_tp"]:.1f} dBTP')
    return 0


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):      # acentos corretos também no Windows
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(prog='confraria', description='Reels "Abre ou guarda?" a partir de uma imagem e um reel.json')
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p):
        p.add_argument('reel', help='caminho do reel.json')
        p.add_argument('--out', default='out', help='pasta de saída (padrão: out/)')
        p.add_argument('--work', help='pasta de trabalho/cache (padrão: <out>/work/<slug>)')
        p.add_argument('--upscale', choices=('auto', 'ai', 'lanczos'), default='auto',
                       help='ai = Real-ESRGAN x4plus (~9,5 min em CPU); lanczos = substituto de teste; '
                            'auto = ai se instalado (padrão)')
        p.add_argument('--gpu', type=int, default=-1, help='gpuid do Real-ESRGAN (-1 = CPU, padrão)')
        p.add_argument('--tilesize', type=int, default=192)

    p = sub.add_parser('make', help='gera MP4 final + capa JPG')
    common(p)
    p.set_defaults(fn=cmd_make)
    p = sub.add_parser('frames', help='gera quadros de checagem nos tempos dados (s)')
    common(p)
    p.add_argument('times', nargs='+', type=float, help='tempos em segundos, ex.: 0 3.5 7.3')
    p.set_defaults(fn=cmd_frames)
    p = sub.add_parser('regress', help='compara quadros com um vídeo de referência (teste de regressão)')
    common(p)
    p.add_argument('reference', help='MP4 de referência')
    p.add_argument('--min-psnr', type=float, default=24.0)
    p.set_defaults(fn=cmd_regress)
    p = sub.add_parser('check', help='valida o reel.json e a imagem, sem renderizar')
    p.add_argument('reel')
    p.set_defaults(fn=cmd_check)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except config.ConfigError as e:
        print(str(e), file=sys.stderr)
        return 2
    except (upscale.UpscaleUnavailable, FileNotFoundError, RuntimeError) as e:
        print(f'erro: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
