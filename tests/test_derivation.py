"""SHOTS e KEYS derivados de focus/timing têm de reproduzir os valores fixos do Reel 2."""
import os

import numpy as np
import pytest

from confraria import config, sound
from confraria.camera import Timeline
from confraria.shots import Scene

REEL2 = os.path.join(os.path.dirname(__file__), '..', 'reference', 'reel2.json')


@pytest.fixture
def scene():
    return Scene(config.load(REEL2), 768, 1376)


def test_shots_match_original(scene):
    # original: (largura, topo) = t1 (768, 5), t2 (340, 372), t3 (540, 295), t4 (768, 5)
    for name, (w, top) in {'t1_sala': (768, 5), 't2_vidro': (340, 372), 't3_preco': (540, 295)}.items():
        gx0, gtop, gw, _ = scene.geom(name)
        assert gw == pytest.approx(w)
        assert gtop == pytest.approx(top, abs=0.8)
        assert gx0 == pytest.approx((768 - w) / 2)


def test_keys_match_original(scene):
    tl = Timeline(scene)
    orig = [(0.0, 768, 688), (3.2, 730, 686), (3.8, 340, 674), (6.7, 318, 670),
            (7.3, 540, 775), (10.0, 520, 767), (11.0, 700, 668), (15.0, 768, 688)]
    for (t, w, cy), key in zip(orig, tl.keys):
        assert key[0] == pytest.approx(t)
        assert key[1] == pytest.approx(w, rel=0.005)
        assert key[3] == pytest.approx(cy, abs=1.0)
        assert key[2] == pytest.approx(384)


def test_timing_scales_everything():
    r = config.load(REEL2)
    r.screen, r.last = 3.0, 5.0
    tl = Timeline(Scene(r, 768, 1376))
    assert (tl.c1, tl.c2, tl.c3, tl.dur) == (3.0, 6.0, 9.0, 14.0)
    assert tl.nfr == 420
    op, _ = tl.caption_state(3.0 + 0.7)['t2_vidro']
    assert op == pytest.approx(1.0)


def test_grade_weights_sum_to_one():
    for lamp in (True, False):
        r = config.load(REEL2)
        if not lamp:
            r.lamp, r.sconces = None, []
        tl = Timeline(Scene(r, 768, 1376))
        for t in np.linspace(0, tl.dur, 60):
            assert sum(tl.grade_weights(t).values()) == pytest.approx(1.0)
        assert ('t4' in tl.grade_weights(tl.dur)) == lamp


def test_bed_root_transposition():
    assert sound.transpose_ratio('A') == 1.0
    assert sound.transpose_ratio('D') == pytest.approx(2 ** (5 / 12))
    assert sound.transpose_ratio('E') == pytest.approx(2 ** (-5 / 12))
