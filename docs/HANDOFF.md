# Handoff: pipeline de Reels "Abre ou guarda?" (Confraria Australis)

Estes scripts produziram o Reel 2 (Penfolds Ampoule) e funcionam, mas têm valores fixos do Reel 2.
O objetivo é transformá-los num repositório `botarbitrage/confraria-pipeline` (Python, privado) que gere
qualquer Reel da série a partir de UMA imagem e um arquivo de configuração.

## Estado atual (funciona, específico do Reel 2)
- `prepare_grades.py`: lê `ampoule_x4.png` (imagem 768x1376 ampliada 4x com Real-ESRGAN) e grava `grade_base/t3/t4.npy`.
- `build_shots.py`: define SHOTS (recortes) e `grade()` (looks: base, t3 com pool de luz, t4 "luzes apagadas").
  Coordenadas em pixels da imagem de ORIGEM 768x1376: pendente/ampola centrada em x=384, lâmpada (383,424), pescoço ~552,
  sino de vidro 600-780, apliques (61,567) e (702,568), pool (384,810), etc.
- `captions.py`: gera camadas de texto transparentes (Playfair Display, halo + sombra, supersample 2x) para 4 telas.
- `render.py`: câmera virtual (KEYS), crossfade dos looks, opacidade das legendas, motion blur, grão; saída 1080x1920, 30 fps, 15 s,
  libx264 crf 18, bt709, via pipe rawvideo. Modos `frames` (checagem) e `video`.
- `sound.py`: trilha sintetizada (pad Am->Fmaj7->Am, whoosh, ting E6, swell reverso + boom no preço, click, ting E5),
  sincronizada com a câmera importando `render`. Grava `reel2_audio_raw.wav`.
- Pós: `ffmpeg loudnorm` para ~-16 LUFS, TP <= -1.5, e mux AAC 256k com o vídeo -> `confraria_reelN_slug.mp4`.
  Capa: o frame da tela 3 (preço) exportado em JPG (`captions/t3_preco_final.jpg`).
- Referências em `reference/`: imagem original do Reel 2, MP4 final e capa.

## Fórmula da série (constantes)
- 4 telas: (1) gancho + número, (2) resposta, (3) preço sozinho = capa, (4) pergunta "Abre ou guarda?".
- Duração ~15 s: telas 1 a 3 com ~3,5 s cada, a última mais longa (~4,5 s ou mais).
- Preço sempre em AU$ (dólares australianos), sempre mesma fonte/posição: Playfair Display 700, 140 px, dourado #E8D5A3, y ~335.
- Frases: Playfair Display 400, branco quente (245,242,236), 76 px, linhas em y 285 e 385. Números destacados em dourado (usar feature OpenType `lnum`).
- Pergunta: Playfair Display italic 400, 112 px, dourado, y 330.
- Fontes: NÃO vão no git. Baixar do npm `@fontsource/playfair-display` (.woff) e converter para .ttf com fontTools
  (pesos 400 e 700, normal e italic). Guardar em `fonts/PlayfairDisplay-{400,700}-{normal,italic}.ttf`.
- Upscale: `realesrgan-ncnn-py` (Real-ESRGAN x4plus), CPU (`gpuid=-1`, `tilesize=192`), ~9,5 min para 768x1376 -> 3072x5504.
  Precisa `apt install libomp5`.

## O que precisa ser parametrizado
Um `reel.json` por Reel. Campos propostos:
```json
{
  "slug": "2026-10-01_ampoule",
  "reel_number": 2,
  "image": "in/2026-10-01_ampoule.png",
  "series": "abre-ou-guarda",
  "screens": {
    "t1": {"lines": [[["Só ", "w"], ["12", "gold"], [" garrafas no mundo.", "w"]], [["E nenhuma tem rolha.", "w"]]]},
    "t2": {"lines": [[["Ela é lacrada em vidro", "w"]], [["soprado à mão.", "w"]]]},
    "t3": {"price": "AU$ 168.000"},
    "t4": {"question": "Abre ou guarda?"}
  },
  "focus": {
    "hero_cx": 384, "hero_cy": 810, "hero_h": 480,
    "detail_cx": 384, "detail_cy": 674, "detail_w": 340,
    "lamp": [383, 424], "sconces": [[61, 567], [702, 568]]
  },
  "timing": {"screen": 3.5, "last": 4.5},
  "sound": {"bed_root": "A", "ting_hz": [1318.5, 659.25]}
}
```
- `focus` substitui as coordenadas fixas de build_shots/render (KEYS e SHOTS devem ser derivadas de `focus` e `timing`).
- Para cada imagem nova, os pontos de `focus` serão preenchidos por quem olha a imagem (um agente com visão). O código deve
  validar os campos e falhar com mensagem clara se faltar algum.
- A duração total, os tempos de cada corte e a sincronia do áudio devem derivar de `timing` (hoje estão fixos em render.py e sound.py).
- Remover os efeitos específicos do Reel 2 quando `focus.lamp` / `focus.sconces` não existirem (a "luz apagando" só faz sentido com lâmpadas).

## Entregáveis esperados
1. `src/confraria/` com módulos: `upscale.py`, `shots.py`, `captions.py`, `camera.py` (render), `sound.py`, `mux.py`, `cli.py`.
2. CLI: `python -m confraria.cli make reel.json --out out/` (gera MP4 final + capa JPG) e `... frames reel.json 0 3.5 7.3` (frames de checagem).
3. `setup.sh` que prepara uma máquina Linux vazia: apt libomp5, pip (numpy scipy opencv-python-headless pillow soundfile realesrgan-ncnn-py fonttools), baixa e converte fontes.
4. `reference/` com o Reel 2 como teste de regressão: rodar `make` com o `reel.json` do Reel 2 deve reproduzir o vídeo atual (mesmas 4 telas, mesmos tempos).
5. `README.md` em português curto explicando o uso.
6. `skills/` (rascunhos em Markdown): `criar-conteudo`, `exportar-prompts`, `gerar-reels`. Esses serão refinados depois, no Claude.

## Restrições
- Só ferramentas grátis e open source. Sem serviços pagos.
- Todo o áudio é sintetizado (sem samples, sem direitos autorais).
- Repositório privado. Não versionar imagens grandes, .npy nem MP4.
- Idioma do texto voltado ao usuário: português.
