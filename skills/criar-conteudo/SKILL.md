---
name: criar-conteudo
description: Rascunho. Cria o conteúdo de um Reel da série "Abre ou guarda?" (Confraria Australis) - gancho, número, resposta, preço em AU$, pergunta final e legenda do post - no formato do bloco `screens` do reel.json.
---

# criar-conteudo (rascunho)

> Rascunho a ser refinado no Claude. Entrada: o item (garrafa/objeto de luxo). Saída: textos das 4 telas + legenda + metadados.

## Quando usar
O usuário quer um novo Reel da série ("próximo reel", "reel sobre <garrafa>"). Este skill só produz **texto**; imagem é `exportar-prompts`, vídeo é `gerar-reels`.

## Passos
1. **Fatos do item**: nome, produtor, safra, tiragem/raridade, preço de mercado (converter para **AU$**, dólares australianos, e registrar a fonte e a data da cotação). Não inventar números: se não houver fonte, dizer que falta e perguntar.
2. **Escolher o gancho**: o fato mais surpreendente e verificável (ex.: "Só 12 garrafas no mundo. E nenhuma tem rolha.").
3. **Escrever as 4 telas** (fórmula fixa da série):
   - **t1** gancho + número: até 2 linhas, ~26 caracteres por linha; o número destacado em `gold`.
   - **t2** a resposta/explicação: 2 linhas, mesmo limite.
   - **t3** só o preço: `AU$ 168.000` (ponto como separador de milhar, sempre AU$).
   - **t4** a pergunta: `Abre ou guarda?` (fixa).
4. **Legenda do post** (português): 2 a 4 linhas, terminando com a pergunta para os comentários, mais 5 a 8 hashtags.
5. **Entregar** o bloco JSON de `screens` (formato abaixo), a legenda e a lista de fatos com fontes.

## Formato de saída
```json
"screens": {
  "t1": {"lines": [[["Só ", "w"], ["12", "gold"], [" garrafas no mundo.", "w"]], [["E nenhuma tem rolha.", "w"]]]},
  "t2": {"lines": [[["Ela é lacrada em vidro", "w"]], [["soprado à mão.", "w"]]]},
  "t3": {"price": "AU$ 168.000"},
  "t4": {"question": "Abre ou guarda?"}
}
```
Cores: `w` = branco quente, `gold` = dourado (usar só para números).

## Regras
- Português do Brasil, tom sóbrio e elegante; frases curtas, sem exclamações.
- Cada tela lê em ~3 s. Se a frase não cabe em 2 linhas de ~26 caracteres, cortar.
- Nada de promessa de investimento nem recomendação financeira.
- Numerar o Reel (`reel_number`) e sugerir `slug` no formato `AAAA-MM-DD_nome`.
