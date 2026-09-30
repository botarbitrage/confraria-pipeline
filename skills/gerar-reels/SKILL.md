---
name: gerar-reels
description: Rascunho. A partir de uma imagem 9:16 e dos textos das 4 telas, mede os pontos de foco olhando a imagem, monta o reel.json, valida, confere quadros e gera o MP4 + capa com o pipeline confraria.
---

# gerar-reels (rascunho)

> Rascunho a ser refinado no Claude. Entrada: `in/<slug>.png` (768x1376) + `screens` de `criar-conteudo`. Saída: `out/confraria_reel<N>_<slug>.mp4` e `_capa.jpg`.

## Passos
1. **Olhar a imagem** (visão) e medir, em **pixels da imagem de origem** (768x1376 típico):
   - `hero_cx`, `hero_cy`, `hero_h`: centro e altura do objeto herói inteiro.
   - `detail_cx`, `detail_cy`, `detail_w`: o detalhe mais bonito para a tela 2 (largura do recorte, ~300 a 400 px).
   - `lamp` `[x, y]` e `sconces` `[[x, y], ...]`, **somente se existirem** na imagem (senão omitir os campos).
   Dica: desenhar uma grade de 100 px sobre a imagem para ler coordenadas com precisão.
2. **Montar o `reel.json`** (modelo: `reference/reel2.json`): `slug`, `reel_number`, `image`, `screens`, `focus`, `timing` (`screen` 3.5, `last` 4.5 por padrão), `sound` (`bed_root` "A"; mudar a tônica só se quiser variar a série).
3. **Validar**: `python -m confraria.cli check reel.json`. Corrigir o que a mensagem apontar (todos os problemas vêm juntos).
4. **Checar enquadramento** (rápido, com master Lanczos): `python -m confraria.cli frames reel.json 0 3.5 5 7.3 9 14.9 --upscale lanczos` e olhar `out/frames/check_*.jpg`:
   - t1: objeto pequeno e centrado, texto legível sobre área escura;
   - t2: detalhe centrado, lâmpada/topo fora da faixa de texto;
   - t3: herói cabe, preço não cobre o objeto;
   - t4: luzes apagando só onde existem lâmpadas.
   Ajustar `focus` e repetir.
5. **Gerar o final**: `python -m confraria.cli make reel.json --out out/` (upscale Real-ESRGAN em CPU: ~10 min; o resto é rápido).
6. **QA final**: duração 15 s (3·screen + last), 4 telas na ordem, preço em AU$ na capa, loudness ~-16 LUFS, true peak ≤ -1,5 dBTP (a CLI imprime).
7. Entregar os dois arquivos e avisar o que ficou pendente.

## Regras
- Nunca versionar imagens, `.npy` nem MP4.
- Se a imagem não for 9:16 ou os pontos ficarem fora dela, a CLI falha com mensagem clara: corrigir a imagem ou o `focus`, não contornar.
- Reel novo sem lâmpada/apliques é válido: a tela 4 mantém o look base e o som não tem o "clique".
