"""Upscale x4 da imagem de origem (Real-ESRGAN x4plus, ncnn, só CPU por padrão).

Custo de referência: ~9,5 min para 768x1376 -> 3072x5504 em CPU (`gpuid=-1`, `tilesize=192`).
Precisa de libomp5 no Linux (`apt install libomp5`, ver setup.sh).
Sem o realesrgan-ncnn-py, o modo "auto" cai num substituto Lanczos (rápido, sem detalhe novo;
serve para checar enquadramento/texto/som, não para publicar).
"""
import os
import time

from PIL import Image

SCALE = 4
MODEL_X4PLUS = 4       # índice do modelo realesrgan-x4plus em realesrgan-ncnn-py


class UpscaleUnavailable(RuntimeError):
    pass


def have_realesrgan():
    try:
        import realesrgan_ncnn_py  # noqa: F401
        return True
    except Exception:
        return False


def lanczos(im):
    return im.resize((im.width * SCALE, im.height * SCALE), Image.LANCZOS)


def realesrgan(im, gpuid=-1, tilesize=192, log=print):
    try:
        from realesrgan_ncnn_py import Realesrgan
    except Exception as e:
        raise UpscaleUnavailable(
            f'realesrgan-ncnn-py indisponível ({e}). Rode ./setup.sh (no Linux precisa de libomp5) '
            'ou use --upscale lanczos para um teste rápido.') from e
    log(f'upscale Real-ESRGAN x4plus (gpuid={gpuid}, tilesize={tilesize}); ~9,5 min em CPU para 768x1376...')
    t0 = time.time()
    out = Realesrgan(gpuid=gpuid, tilesize=tilesize, model=MODEL_X4PLUS).process_pil(im)
    log(f'upscale pronto em {time.time() - t0:.0f}s: {out.width}x{out.height}')
    return out


def make_master(image_path, work_dir, mode='auto', gpuid=-1, tilesize=192, log=print):
    """Devolve (imagem master RGB, rótulo do método). Reaproveita <work_dir>/master_<método>.png."""
    src = Image.open(image_path).convert('RGB')
    if mode == 'auto':
        mode = 'ai' if have_realesrgan() else 'lanczos'
        if mode == 'lanczos':
            log('AVISO: realesrgan-ncnn-py não encontrado; usando substituto Lanczos (qualidade de teste).')
    if mode not in ('ai', 'lanczos'):
        raise ValueError(f'modo de upscale desconhecido: {mode}')
    os.makedirs(work_dir, exist_ok=True)
    cache = os.path.join(work_dir, f'master_{mode}.png')
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(image_path):
        log(f'master: reaproveitando {cache}')
        return Image.open(cache).convert('RGB'), mode
    master = realesrgan(src, gpuid, tilesize, log) if mode == 'ai' else lanczos(src)
    master.save(cache)
    return master, mode
