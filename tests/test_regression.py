"""Regressão do Reel 2: quadros gerados x vídeo de referência.

Pesado (upscale + grades): só roda com CONFRARIA_REGRESSION=1 e o vídeo em reference/.
Equivalente pela CLI:
  python -m confraria.cli regress reference/reel2.json reference/confraria_reel2_ampoule.mp4
"""
import os

import pytest

from confraria import cli

HERE = os.path.dirname(__file__)
REEL2 = os.path.join(HERE, '..', 'reference', 'reel2.json')
REF = os.path.join(HERE, '..', 'reference', 'confraria_reel2_ampoule.mp4')

pytestmark = pytest.mark.skipif(
    not (os.environ.get('CONFRARIA_REGRESSION') and os.path.exists(REF)),
    reason='defina CONFRARIA_REGRESSION=1 e coloque o vídeo em reference/')


def test_reel2_frames_match_reference(tmp_path):
    upscale = os.environ.get('CONFRARIA_UPSCALE', 'auto')
    assert cli.main(['regress', REEL2, REF, '--out', str(tmp_path), '--upscale', upscale]) == 0
