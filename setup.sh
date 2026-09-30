#!/usr/bin/env bash
# Prepara uma máquina Linux vazia (Debian/Ubuntu) para rodar o pipeline.
# Só ferramentas gratuitas. Uso:  ./setup.sh   (usa sudo quando não for root)
set -euo pipefail
cd "$(dirname "$0")"

SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi

echo "==> 1/5 pacotes do sistema (libomp5 para o Real-ESRGAN, python, node/npm, ffmpeg, libraqm)"
$SUDO apt-get update -y
$SUDO apt-get install -y --no-install-recommends \
  python3 python3-pip python3-venv libomp5 nodejs npm ffmpeg libraqm0 libgl1 libglib2.0-0 git ca-certificates

echo "==> 2/5 ambiente virtual e dependências Python"
python3 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy opencv-python-headless pillow soundfile fonttools realesrgan-ncnn-py imageio-ffmpeg pytest
pip install -e .

echo "==> 3/5 fontes Playfair Display (npm @fontsource/playfair-display -> .woff -> .ttf via fontTools)"
mkdir -p fonts
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
( cd "$TMP" && npm pack @fontsource/playfair-display >/dev/null && tar xzf ./*.tgz )
python - "$TMP/package/files" fonts <<'PY'
import sys
from fontTools.ttLib import TTFont
src, dst = sys.argv[1], sys.argv[2]
for weight in ("400", "700"):
    for style in ("normal", "italic"):
        f = TTFont(f"{src}/playfair-display-latin-{weight}-{style}.woff")
        f.flavor = None                      # woff -> ttf
        f.save(f"{dst}/PlayfairDisplay-{weight}-{style}.ttf")
        print("fonte:", f"{dst}/PlayfairDisplay-{weight}-{style}.ttf")
PY

echo "==> 4/5 pastas de trabalho"
mkdir -p in out

echo "==> 5/5 verificação"
python - <<'PY'
from PIL import features
import shutil, confraria
print("libraqm (kerning/lnum nativos):", features.check("raqm"))
try:
    import realesrgan_ncnn_py
    print("realesrgan-ncnn-py: ok")
except Exception as e:
    print("AVISO realesrgan-ncnn-py:", e)
print("ffmpeg:", shutil.which("ffmpeg") or "usando imageio-ffmpeg")
PY
python -m pytest -q

cat <<'MSG'

Pronto. Para usar:
  . .venv/bin/activate
  python -m confraria.cli check reference/reel2.json
  python -m confraria.cli make  meu_reel.json --out out/
MSG
