"""ffmpeg: normalização de loudness (~-16 LUFS, TP <= -1.5 dBTP) e mux do MP4 final; capa em JPG."""
import json
import os
import re
import shutil
import subprocess

from PIL import Image

TARGET_I, TARGET_TP, TARGET_LRA = -16.0, -1.5, 11.0


def ffmpeg_exe():
    """ffmpeg do PATH; senão o binário do pacote imageio-ffmpeg (gratuito)."""
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError('ffmpeg não encontrado. Instale (apt install ffmpeg) ou pip install imageio-ffmpeg; '
                           'setup.sh já faz isso.') from e


def _run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if p.returncode != 0:
        raise RuntimeError('ffmpeg falhou:\n' + ' '.join(cmd) + '\n' + p.stderr[-2000:])
    return p.stderr


def measure(path):
    """Medição loudnorm (JSON) de um arquivo de áudio/vídeo."""
    err = _run([ffmpeg_exe(), '-hide_banner', '-nostats', '-i', path, '-vn', '-af',
                f'loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}:print_format=json', '-f', 'null', '-'])
    m = re.search(r'\{[^{}]*"input_i"[^{}]*\}', err, re.S)
    if not m:
        raise RuntimeError('não consegui ler a medição do loudnorm:\n' + err[-800:])
    return {k: float(v) for k, v in json.loads(m.group(0)).items() if k != 'normalization_type'}


def normalize(raw_wav, out_wav):
    """loudnorm em duas passagens (linear) -> WAV normalizado."""
    m = measure(raw_wav)
    af = (f'loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}:'
          f'measured_I={m["input_i"]}:measured_TP={m["input_tp"]}:measured_LRA={m["input_lra"]}:'
          f'measured_thresh={m["input_thresh"]}:offset={m["target_offset"]}:linear=true')
    _run([ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', raw_wav, '-af', af, '-ar', '48000', out_wav])
    return out_wav


def mux(video, audio_wav, out_mp4):
    """Vídeo (copiado) + áudio AAC 256k em MP4 com faststart."""
    _run([ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', video, '-i', audio_wav,
          '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k',
          '-shortest', '-movflags', '+faststart', out_mp4])
    return out_mp4


def extract_frame(video, t, out_png):
    """Quadro de um vídeo em `t` segundos (usado nos testes de regressão)."""
    _run([ffmpeg_exe(), '-y', '-loglevel', 'error', '-ss', str(t), '-i', video, '-frames:v', '1', out_png])
    return out_png


def save_cover(rgb_image: Image.Image, path, quality=95):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rgb_image.convert('RGB').save(path, quality=quality)
    return path
