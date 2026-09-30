---
name: exportar-prompts
description: Rascunho. Gera o prompt de imagem 9:16 (768x1376) para um Reel da série "Abre ou guarda?" com a composição que o pipeline espera (herói centralizado, espaço calmo no topo para o texto), pronto para colar num gerador de imagens.
---

# exportar-prompts (rascunho)

> Rascunho a ser refinado no Claude. Entrada: o item + o conteúdo de `criar-conteudo`. Saída: prompt(s) de imagem + nome do arquivo esperado em `in/`.

## Por que a composição importa
O pipeline faz UMA imagem virar 4 telas com câmera virtual. Ele espera:
- **Formato 768x1376** (9:16), imagem única, sem texto, sem marcas d'água.
- **Herói centralizado** horizontalmente (`hero_cx` ≈ metade da largura), ocupando ~35% da altura, na metade de baixo do quadro.
- **Topo escuro e calmo** (~15% a 40% da altura) para as legendas da tela 1 (y 285 a 485 no vídeo 1080x1920). Nada de detalhes brilhantes ali.
- **Sala inteira visível** na tela 1 (objeto pequeno, dentro de um nicho, vitrine ou mesa) e **detalhe rico** no herói para o zoom da tela 2.
- Opcional, para a cena "luzes apagando": **uma lâmpada/spot de vitrine** acima do herói e/ou **apliques de parede** simétricos. Sem eles a tela 4 não escurece.
- Luz quente e baixa, fundo escuro (madeira, veludo, pedra), o objeto como única fonte de brilho.

## Passos
1. Descrever o item fielmente (cor, forma, vidro/metal, rótulo ausente ou legível) a partir dos fatos.
2. Montar o prompt no molde:
   `luxury still life, <objeto> centered in a <vitrine/nicho>, dark warm interior, single spotlight above, symmetrical composition, empty dark space in the upper third, cinematic low-key lighting, ultra detailed glass and metal, photorealistic, vertical 9:16, no text, no watermark`
3. Variantes: (A) com lâmpada + apliques, (B) só spot, (C) sem fontes de luz aparentes (a tela 4 não escurece).
4. **Prompt negativo**: `text, letters, logo, watermark, people, hands, cropped object, off-center, bright background`.
5. Informar o nome do arquivo: `in/AAAA-MM-DD_slug.png`, tamanho **768x1376** (gerar em 9:16 e redimensionar se preciso; o upscale x4 é feito pelo pipeline).
6. Depois de gerada, a imagem segue para `gerar-reels` (que mede os pontos de `focus`).

## Regras
- Só geradores de imagem gratuitos ou já disponíveis ao usuário; não assumir serviço pago.
- Não inventar rótulos legíveis nem marcas: o texto vem do vídeo, não da imagem.
- Se o item real tem forma distintiva, priorizar a fidelidade da silhueta sobre o cenário.
