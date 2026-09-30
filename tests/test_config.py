import json
import os

import pytest

from confraria import config

REEL2 = os.path.join(os.path.dirname(__file__), '..', 'reference', 'reel2.json')


def raw():
    with open(REEL2, encoding='utf-8') as f:
        return json.load(f)


def parse(d):
    return config.parse(d, base_dir=os.path.dirname(REEL2))


def problems(d):
    with pytest.raises(config.ConfigError) as e:
        parse(d)
    return '\n'.join(e.value.problems)


def test_reel2_ok():
    r = parse(raw())
    assert r.duration == 15.0 and r.lights_out and r.basename == 'confraria_reel2_ampoule'
    assert r.lines_t1[0][1] == ('12', 'gold')


def test_missing_focus_field_named():
    d = raw()
    del d['focus']['detail_w']
    assert 'focus.detail_w ausente' in problems(d)


def test_collects_all_problems():
    d = raw()
    d['timing'] = {'screen': 1.0, 'last': 1.0}
    d['focus']['hero_h'] = -3
    d['screens']['t3'] = {}
    msg = problems(d)
    assert 'timing.screen' in msg and 'timing.last' in msg and 'hero_h' in msg and 'price' in msg


def test_unknown_fields_rejected():
    d = raw()
    d['focus']['lamps'] = [1, 2]
    d['extra'] = 1
    msg = problems(d)
    assert 'focus.lamps' in msg and '"extra"' in msg


def test_bad_run_color_and_lines():
    d = raw()
    d['screens']['t1']['lines'] = [[['x', 'red']], ['b'], ['c']]
    assert 'screens.t1.lines' in problems(d)


def test_lamp_and_sconces_optional():
    d = raw()
    del d['focus']['lamp'], d['focus']['sconces']
    assert not parse(d).lights_out


def test_string_line_shorthand():
    d = raw()
    d['screens']['t2']['lines'] = ['Uma linha só']
    assert parse(d).lines_t2 == [[('Uma linha só', 'w')]]


def test_image_bounds():
    r = parse(raw())
    config.validate_image(r, 768, 1376)
    r.focus['hero_cy'] = 5000
    with pytest.raises(config.ConfigError):
        config.validate_image(r, 768, 1376)


def test_bad_bed_root():
    d = raw()
    d['sound']['bed_root'] = 'H'
    assert 'bed_root' in problems(d)
