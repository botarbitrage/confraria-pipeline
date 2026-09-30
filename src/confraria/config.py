"""Leitura e validação do reel.json.

Toda a parametrização de um Reel (textos, pontos de foco, tempos, som) vive aqui.
Coordenadas de `focus` estão em pixels da imagem de ORIGEM (antes do upscale).
Erros de configuração são reunidos e reportados juntos, em português.
"""
import json
import os
import re
from dataclasses import dataclass, field

SLUG_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')
COLORS = ('w', 'gold')
# semitons acima de A (lá) dentro de uma oitava
NOTES = {'C': 3, 'C#': 4, 'DB': 4, 'D': 5, 'D#': 6, 'EB': 6, 'E': 7, 'F': 8, 'F#': 9, 'GB': 9,
         'G': 10, 'G#': 11, 'AB': 11, 'A': 0, 'A#': 1, 'BB': 1, 'B': 2}

DEFAULT_SERIES = 'abre-ou-guarda'
DEFAULT_TIMING = {'screen': 3.5, 'last': 4.5}
DEFAULT_SOUND = {'bed_root': 'A', 'ting_hz': [1318.5, 659.25]}

REQUIRED_FOCUS = ('hero_cx', 'hero_cy', 'hero_h', 'detail_cx', 'detail_cy', 'detail_w')
OPTIONAL_FOCUS = ('lamp', 'sconces')


class ConfigError(Exception):
    """Configuração inválida; `problems` lista todos os defeitos encontrados."""

    def __init__(self, problems, source=''):
        self.problems = list(problems)
        head = f'reel.json inválido ({source})' if source else 'reel.json inválido'
        super().__init__(head + ':\n' + '\n'.join(f'  - {p}' for p in self.problems))


@dataclass
class Reel:
    slug: str
    reel_number: int
    image: str                       # caminho absoluto da imagem de origem
    series: str
    lines_t1: list                   # [[(texto, cor), ...], ...]
    lines_t2: list
    price: str
    question: str
    focus: dict
    screen: float                    # duração das telas 1 a 3 (s)
    last: float                      # duração da tela 4 (s)
    bed_root: str
    ting_hz: list
    lamp: tuple = None
    sconces: list = field(default_factory=list)
    source: str = ''

    @property
    def duration(self):
        return 3 * self.screen + self.last

    @property
    def lights_out(self):
        """A cena "luzes apagando" só existe se houver lâmpada ou apliques."""
        return self.lamp is not None or bool(self.sconces)

    @property
    def basename(self):
        return f'confraria_reel{self.reel_number}_{self.slug}'


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _point(v, name, problems):
    if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(_is_num(c) for c in v)):
        problems.append(f'{name} deve ser um ponto [x, y] em pixels da imagem de origem (recebido: {v!r})')
        return None
    return (float(v[0]), float(v[1]))


def _parse_lines(screen_key, node, problems):
    where = f'screens.{screen_key}.lines'
    if not isinstance(node, list) or not 1 <= len(node) <= 2:
        problems.append(f'{where} deve ter 1 ou 2 linhas (lista de listas de trechos)')
        return []
    out = []
    for li, line in enumerate(node):
        if isinstance(line, str):
            line = [[line, 'w']]
        if not isinstance(line, list) or not line:
            problems.append(f'{where}[{li}] deve ser uma lista de trechos [texto, cor] (ou um texto simples)')
            continue
        runs = []
        for ri, run in enumerate(line):
            if not (isinstance(run, (list, tuple)) and len(run) == 2 and isinstance(run[0], str)
                    and run[0] != '' and run[1] in COLORS):
                problems.append(f'{where}[{li}][{ri}] deve ser ["texto", "w"|"gold"] (recebido: {run!r})')
                continue
            runs.append((run[0], run[1]))
        out.append(runs)
    return out


def load(path):
    """Lê e valida um reel.json. Levanta ConfigError com TODOS os problemas."""
    path = os.path.abspath(path)
    try:
        with open(path, encoding='utf-8') as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise ConfigError([f'arquivo não encontrado: {path}'])
    except json.JSONDecodeError as e:
        raise ConfigError([f'JSON malformado: {e}'], path)
    return parse(raw, base_dir=os.path.dirname(path), source=path)


def parse(raw, base_dir='.', source=''):
    p = []
    if not isinstance(raw, dict):
        raise ConfigError(['o topo do arquivo deve ser um objeto JSON'], source)
    known = {'slug', 'reel_number', 'image', 'series', 'screens', 'focus', 'timing', 'sound'}
    for k in raw:
        if k not in known and not k.startswith('_'):
            p.append(f'campo desconhecido "{k}" (campos válidos: {", ".join(sorted(known))})')

    slug = raw.get('slug')
    if not isinstance(slug, str) or not SLUG_RE.match(slug):
        p.append('slug ausente ou inválido (use letras, números, "-", "_" ou "."; ex.: "2026-10-01_ampoule")')
    num = raw.get('reel_number')
    if not isinstance(num, int) or isinstance(num, bool) or num < 1:
        p.append('reel_number deve ser um inteiro >= 1')
    img = raw.get('image')
    if not isinstance(img, str) or not img:
        p.append('image ausente (caminho da imagem 9:16, ex.: "in/2026-10-01_ampoule.png")')
    series = raw.get('series', DEFAULT_SERIES)
    if not isinstance(series, str):
        p.append('series deve ser texto')

    # ---- telas
    screens = raw.get('screens')
    lines_t1 = lines_t2 = []
    price = question = ''
    if not isinstance(screens, dict):
        p.append('screens ausente (precisa de t1, t2, t3 e t4)')
    else:
        for k in ('t1', 't2', 't3', 't4'):
            if not isinstance(screens.get(k), dict):
                p.append(f'screens.{k} ausente')
        for k in screens:
            if k not in ('t1', 't2', 't3', 't4'):
                p.append(f'tela desconhecida screens.{k}')
        if isinstance(screens.get('t1'), dict):
            lines_t1 = _parse_lines('t1', screens['t1'].get('lines'), p)
        if isinstance(screens.get('t2'), dict):
            lines_t2 = _parse_lines('t2', screens['t2'].get('lines'), p)
        if isinstance(screens.get('t3'), dict):
            price = screens['t3'].get('price')
            if not isinstance(price, str) or not price.strip():
                p.append('screens.t3.price ausente (ex.: "AU$ 168.000")')
                price = ''
        if isinstance(screens.get('t4'), dict):
            question = screens['t4'].get('question')
            if not isinstance(question, str) or not question.strip():
                p.append('screens.t4.question ausente (ex.: "Abre ou guarda?")')
                question = ''

    # ---- foco
    focus = raw.get('focus')
    lamp, sconces = None, []
    fnum = {}
    if not isinstance(focus, dict):
        p.append('focus ausente (pontos de foco em pixels da imagem de origem: '
                 + ', '.join(REQUIRED_FOCUS) + ')')
    else:
        for k in REQUIRED_FOCUS:
            v = focus.get(k)
            if v is None:
                p.append(f'focus.{k} ausente')
            elif not _is_num(v) or v <= 0:
                p.append(f'focus.{k} deve ser um número positivo (recebido: {v!r})')
            else:
                fnum[k] = float(v)
        for k in focus:
            if k not in REQUIRED_FOCUS and k not in OPTIONAL_FOCUS:
                p.append(f'campo desconhecido focus.{k}')
        if focus.get('lamp') is not None:
            lamp = _point(focus['lamp'], 'focus.lamp', p)
        if focus.get('sconces') is not None:
            sc = focus['sconces']
            if not isinstance(sc, list):
                p.append('focus.sconces deve ser uma lista de pontos [[x, y], ...]')
            else:
                for i, pt in enumerate(sc):
                    q = _point(pt, f'focus.sconces[{i}]', p)
                    if q:
                        sconces.append(q)

    # ---- tempos
    timing = dict(DEFAULT_TIMING)
    if 'timing' in raw:
        t = raw['timing']
        if not isinstance(t, dict):
            p.append('timing deve ser um objeto {"screen": s, "last": s}')
        else:
            for k in t:
                if k not in DEFAULT_TIMING:
                    p.append(f'campo desconhecido timing.{k}')
            timing.update({k: v for k, v in t.items() if k in DEFAULT_TIMING})
    if not _is_num(timing['screen']) or timing['screen'] < 2.0:
        p.append(f'timing.screen deve ser um número >= 2.0 s (recebido: {timing["screen"]!r})')
        timing['screen'] = DEFAULT_TIMING['screen']
    if not _is_num(timing['last']) or timing['last'] < 2.5:
        p.append(f'timing.last deve ser um número >= 2.5 s (recebido: {timing["last"]!r})')
        timing['last'] = DEFAULT_TIMING['last']

    # ---- som
    snd = dict(DEFAULT_SOUND)
    if 'sound' in raw:
        s = raw['sound']
        if not isinstance(s, dict):
            p.append('sound deve ser um objeto {"bed_root": "A", "ting_hz": [f1, f2]}')
        else:
            for k in s:
                if k not in DEFAULT_SOUND:
                    p.append(f'campo desconhecido sound.{k}')
            snd.update({k: v for k, v in s.items() if k in DEFAULT_SOUND})
    if not isinstance(snd['bed_root'], str) or snd['bed_root'].upper() not in NOTES:
        p.append(f'sound.bed_root deve ser uma nota ({", ".join(sorted(NOTES))}); recebido: {snd["bed_root"]!r}')
        snd['bed_root'] = 'A'
    th = snd['ting_hz']
    if not (isinstance(th, list) and len(th) == 2 and all(_is_num(x) and 100 <= x <= 5000 for x in th)):
        p.append(f'sound.ting_hz deve ser [ting da tela 2, ting da tela 4] em Hz (100 a 5000); recebido: {th!r}')
        th = DEFAULT_SOUND['ting_hz']

    if p:
        raise ConfigError(p, source)

    image = img if os.path.isabs(img) else os.path.abspath(os.path.join(base_dir, img))
    if not os.path.exists(image) and not os.path.isabs(img) and os.path.exists(os.path.abspath(img)):
        image = os.path.abspath(img)       # relativo ao diretório atual
    return Reel(slug=slug, reel_number=num, image=image, series=series,
                lines_t1=lines_t1, lines_t2=lines_t2, price=price, question=question,
                focus=fnum, screen=float(timing['screen']), last=float(timing['last']),
                bed_root=snd['bed_root'].upper(), ting_hz=[float(x) for x in th],
                lamp=lamp, sconces=sconces, source=source)


def validate_image(reel, width, height):
    """Confere se os pontos de foco cabem na imagem real. Levanta ConfigError."""
    p = []
    if height * 9 < width * 16 * 0.98:
        p.append(f'a imagem {width}x{height} é mais larga que 9:16')
    f = reel.focus

    def inside(name, x, y):
        if not (0 <= x <= width and 0 <= y <= height):
            p.append(f'{name} ({x:g}, {y:g}) está fora da imagem {width}x{height}')
    inside('focus.hero (hero_cx, hero_cy)', f['hero_cx'], f['hero_cy'])
    inside('focus.detail (detail_cx, detail_cy)', f['detail_cx'], f['detail_cy'])
    if reel.lamp:
        inside('focus.lamp', *reel.lamp)
    for i, s in enumerate(reel.sconces):
        inside(f'focus.sconces[{i}]', *s)
    hero_w = f['hero_h'] * 2 * 9 / 16
    if hero_w > width:
        p.append(f'focus.hero_h={f["hero_h"]:g} gera um enquadramento mais largo que a imagem '
                 f'(máx. {width * 8 / 9:g})')
    if f['detail_w'] > width:
        p.append(f'focus.detail_w={f["detail_w"]:g} é maior que a largura da imagem ({width})')
    if p:
        raise ConfigError(p, reel.source)
