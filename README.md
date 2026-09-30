# confraria-pipeline

Gera os Reels da série **"Abre ou guarda?"** (Confraria Australis) a partir de **uma imagem 9:16** e de um **`reel.json`**.
Saída: MP4 1080x1920, 30 fps, ~15 s, com trilha sintetizada (sem samples, sem direitos autorais), e a capa em JPG.
Só ferramentas gratuitas e de código aberto.

## Instalação (Linux vazio)

```bash
./setup.sh
. .venv/bin/activate
```

O `setup.sh` instala `libomp5`, Python, ffmpeg, as dependências (numpy, scipy, opencv, pillow, soundfile, fonttools,
realesrgan-ncnn-py) e baixa as fontes Playfair Display do npm (`@fontsource/playfair-display`), convertendo para
`fonts/PlayfairDisplay-{400,700}-{normal,italic}.ttf`. Fontes, imagens, `.npy` e MP4 **não** vão para o git.

## Uso

```bash
python -m confraria.cli check  reel.json                  # valida o JSON e os pontos de foco (rápido)
python -m confraria.cli frames reel.json 0 3.5 7.3        # quadros de checagem em out/frames/check_*.jpg
python -m confraria.cli make   reel.json --out out/       # MP4 final + capa JPG em out/
```

`make` gera `out/confraria_reel<N>_<slug>.mp4` e `out/confraria_reel<N>_<slug>_capa.jpg`. Os intermediários
(master 4x, grades, legendas, áudio) ficam em `out/work/<slug>/` e são reaproveitados nas execuções seguintes.

Opções: `--upscale auto|ai|lanczos` (padrão `auto`), `--gpu N` (padrão `-1` = CPU), `--work DIR`.
O upscale Real-ESRGAN x4plus leva ~9,5 min em CPU para 768x1376 → 3072x5504. Sem `realesrgan-ncnn-py`, o modo `auto`
usa um substituto Lanczos, **bom só para testar enquadramento, texto e som**, não para publicar.

## O `reel.json`

Um por Reel; exemplo completo em [`reference/reel2.json`](reference/reel2.json).

| campo | o que é |
|---|---|
| `slug`, `reel_number`, `image`, `series` | identificação; `image` é relativa ao próprio JSON |
| `screens.t1.lines` | tela 1 (gancho + número): 1 ou 2 linhas; cada linha é uma lista de trechos `["texto", "w"]` ou `["12", "gold"]` (`w` = branco quente; `gold` = dourado, com algarismos alinhados) |
| `screens.t2.lines` | tela 2 (a resposta), mesmo formato |
| `screens.t3.price` | tela 3: o preço sozinho, sempre em AU$ (é a capa) |
| `screens.t4.question` | tela 4: a pergunta, `"Abre ou guarda?"` |
| `focus` | pontos de foco em **pixels da imagem de origem** (ver abaixo) |
| `timing.screen`, `timing.last` | duração das telas 1–3 (s) e da tela 4 (s); duração total = 3·screen + last |
| `sound.bed_root`, `sound.ting_hz` | tônica da base (A = lá menor; as notas são transpostas) e os dois "tings" [tela 2, tela 4] |

`focus` (preenchido por quem olha a imagem):

- `hero_cx`, `hero_cy`, `hero_h`: centro e altura do objeto herói (a garrafa/ampola inteira). Define o enquadramento do preço (altura do quadro = 2 × `hero_h`) e a poça de luz.
- `detail_cx`, `detail_cy`, `detail_w`: o detalhe da tela 2 (centro e largura do recorte).
- `lamp` `[x, y]` e `sconces` `[[x, y], ...]` (**opcionais**): lâmpada de vitrine e apliques de parede. Só com eles existe a cena "luzes apagando" da tela 4 (e o clique do interruptor no som). Sem nenhum dos dois, a tela 4 mantém o look base.

Se faltar algo, a CLI reúne **todos** os problemas e falha com mensagem clara em português (campos ausentes, fora da imagem, tipo errado, campos desconhecidos).

## Fórmula da série (constantes)

4 telas: (1) gancho + número, (2) resposta, (3) preço = capa, (4) pergunta. Tipografia Playfair Display:
frases 400/76 px branco quente em y 285 e 385; preço 700/140 px `#E8D5A3` em y 335; pergunta itálico 400/112 px dourado em y 330.

## Teste de regressão (Reel 2)

`reference/` guarda o Reel 2 (Penfolds Ampoule): `reel2.json` (versionado) e, localmente, a imagem original, o MP4 e a capa
(não versionados). Para conferir que o pipeline ainda reproduz o vídeo:

```bash
python -m confraria.cli regress reference/reel2.json reference/confraria_reel2_ampoule.mp4 --upscale ai
CONFRARIA_REGRESSION=1 python -m pytest tests/test_regression.py      # mesma checagem via pytest
```

Compara os quadros em 0, 3,5, 7,3, 10,5 e 14,9 s (mais 5,0, 8,5 e 13,0 s, onde as legendas aparecem) por PSNR.
Testes rápidos (`python -m pytest`) checam a validação do JSON e que `SHOTS`/`KEYS` derivados de `focus`/`timing` reproduzem os valores do Reel 2.

## Estrutura

```
src/confraria/  config.py (reel.json) · fonts.py · upscale.py · shots.py (enquadramentos e looks)
                captions.py (texto) · camera.py (câmera virtual + vídeo) · sound.py · mux.py (loudnorm/mux/capa) · cli.py
reference/      Reel 2 (regressão)      skills/  rascunhos: criar-conteudo, exportar-prompts, gerar-reels
```

## Notas

- Com `libraqm` (instalado pelo `setup.sh` no Linux) o kerning e o `lnum` são nativos. Sem ele (ex.: Windows) o pipeline
  gera uma variante da fonte com algarismos alinhados e usa o layout básico do Pillow: o resultado é praticamente idêntico, com bordas de glifo até ~1 px diferentes.
- O áudio é normalizado para ~-16 LUFS (true peak ≤ -1,5 dBTP) em duas passagens e codificado em AAC 256k.
