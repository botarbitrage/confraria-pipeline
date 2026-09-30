"""Fontes Playfair Display (não versionadas; ver setup.sh).

`load(style, px, lnum=False)` devolve (fonte, features). Com libraqm o recurso OpenType
`lnum` (algarismos alinhados) é pedido em tempo de desenho. Sem libraqm (típico do Windows)
geramos uma variante da fonte cujo cmap já aponta para os glifos `.lf`, com o mesmo resultado.
"""
import os
from PIL import ImageFont, features

STYLES = ('400-normal', '400-italic', '700-normal', '700-italic')
HAS_RAQM = features.check('raqm')


def fonts_dir():
    env = os.environ.get('CONFRARIA_FONTS')
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(os.getcwd(), 'fonts'), os.path.abspath(os.path.join(here, '..', '..', 'fonts'))):
        if os.path.isdir(cand):
            return cand
    return os.path.join(os.getcwd(), 'fonts')


def font_path(style):
    p = os.path.join(fonts_dir(), f'PlayfairDisplay-{style}.ttf')
    if not os.path.exists(p):
        raise FileNotFoundError(
            f'Fonte não encontrada: {p}\n'
            'Rode ./setup.sh (baixa @fontsource/playfair-display via npm e converte com fontTools) '
            'ou defina CONFRARIA_FONTS apontando para a pasta com PlayfairDisplay-{400,700}-{normal,italic}.ttf.')
    return p


def _lnum_variant(style):
    """Cria (uma vez) fonts/_lnum/PlayfairDisplay-<style>.ttf com os algarismos alinhados no cmap."""
    src = font_path(style)
    dst_dir = os.path.join(os.path.dirname(src), '_lnum')
    dst = os.path.join(dst_dir, os.path.basename(src))
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return dst
    from fontTools.ttLib import TTFont
    f = TTFont(src)
    gsub = f['GSUB'].table
    mapping = {}
    for fr in gsub.FeatureList.FeatureRecord:
        if fr.FeatureTag == 'lnum':
            for li in fr.Feature.LookupListIndex:
                for st in gsub.LookupList.Lookup[li].SubTable:
                    mapping.update(getattr(st, 'mapping', {}))
    for table in f['cmap'].tables:
        for code, glyph in list(table.cmap.items()):
            if glyph in mapping:
                table.cmap[code] = mapping[glyph]
    os.makedirs(dst_dir, exist_ok=True)
    f.save(dst)
    return dst


def load(style, px, lnum=False):
    """(ImageFont, features|None) para o estilo ('400-normal' etc.) em `px` pixels."""
    if HAS_RAQM:
        f = ImageFont.truetype(font_path(style), px, layout_engine=ImageFont.Layout.RAQM)
        return f, (['lnum'] if lnum else None)
    path = _lnum_variant(style) if lnum else font_path(style)
    return ImageFont.truetype(path, px, layout_engine=ImageFont.Layout.BASIC), None
