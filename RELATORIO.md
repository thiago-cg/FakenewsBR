# Score de Confiança FakenewsBR — Relatório da execução de 2026-09-10

Pipeline completo executado entre 02:34 e 06:49 (4 h 15 min), todas as 9 etapas com
sucesso. Modelo treinado em `models/artifacts/bertimbau_finetuned/`.

---

## 1. O número que você quer ver

O BERTimbau ajustado **supera o baseline linear com folga**:

| modelo | acurácia | F1(fake) |
|---|---:|---:|
| Baseline A (ingênuo, TF-IDF) | 0,7811 | 0,8413 |
| Baseline B (higienizado, TF-IDF) | 0,7670 | 0,8292 |
| **BERTimbau fine-tuned** | **0,8655** | **0,9102** |

PR-AUC 0,9615 · ECE 0,0105 · casos limítrofes 0,9653.

E no regime onde a pergunta está bem-posta — só os grupos em que a origem **não**
prediz o rótulo — o modelo final (fine-tuned + cabeça DFR) entrega:

**acurácia 0,8711 · macro-F1 0,8709 · ECE 0,0190**

---

## 2. O resultado mais importante não é esse

O encoder ajustado sozinho **ainda usa o atalho de proveniência**. Olhe o que ele
faz nos dois subsets degenerados:

| subset | encoder puro | + cabeça DFR |
|---|---:|---:|
| `fakes` (100% falso) | 0,9663 | 0,6157 |
| `true` (100% verdadeiro) | **0,1897** | **0,8128** |

O encoder puro acerta 96,6% em `fakes` e erra **81% dos textos verdadeiros** do
subset `true`. Ele não aprendeu veracidade: aprendeu a responder "falso" para
qualquer coisa com cara de alegação de agência de checagem. A cabeça DFR corrige
isso — a acurácia em `true` sobe de 0,19 para 0,81.

É por isso que a acurácia total do DFR (0,7345) é menor que a do ERM (0,8451) e
isso é o objetivo, não o defeito: o ERM está comprando acurácia com um atalho.

---

## 3. Comparação completa

### Encoder congelado (BERTimbau pré-treinado)

| split | variante | pior-grupo | macro-F1 (total) | ECE |
|---|---|---:|---:|---:|
| iid | ERM | 0,6448 | 0,7503 | 0,0107 |
| iid | **DFR** | **0,7226** | 0,5645 | 0,2303 |
| ood:whatsapp | ERM | 0,4553 | 0,4553 | 0,3321 |
| ood:whatsapp | **DFR** | **0,6229** | 0,6229 | 0,2180 |
| ood:portal | **ERM** | **0,6726** | 0,6724 | 0,0420 |
| ood:portal | DFR | 0,6164 | 0,6161 | 0,0889 |

### Encoder fine-tuned

| split | variante | pior-grupo | macro-F1 (total) | ECE | limítrofes |
|---|---|---:|---:|---:|---:|
| iid | ERM | 0,7474 | 0,8136 | 0,0142 | 0,9306 |
| iid | **DFR** | **0,7516** | 0,7172 | 0,1084 | 0,6458 |
| ood:whatsapp | ERM | 0,7259 | 0,7259 | 0,1054 | — |
| ood:whatsapp | **DFR** | **0,7779** | 0,7779 | 0,0667 | — |
| ood:portal | ERM | 0,9721 | 0,9721 | 0,0924 | — |
| ood:portal | DFR | 0,9522 | 0,9522 | 0,0669 | — |

O fine-tuning ajudou mais fora de domínio do que dentro: pior-grupo em WhatsApp
foi de 0,6229 (congelado) para 0,7779 (ajustado), **+0,155**.

### Por grupo, modelo final (fine-tuned + DFR, split IID)

| grupo | n | fake % | macro-F1 | ECE |
|---|---:|---:|---:|---:|
| `Fake.br` | 1074 | 50,0 | 0,9795 | 0,0345 |
| `FakeWhatsApp.BR_2018` | 957 | 47,9 | 0,7877 | 0,0425 |
| `COVID19.BR` | 290 | 42,4 | 0,7516 | 0,1108 |

---

## 4. Três achados que contradizem premissas anteriores do projeto

### 4.1. Features estilísticas não ajudam

| | com estilo | sem estilo |
|---|---:|---:|
| IID — pior-grupo | 0,7226 | **0,7304** |
| OOD WhatsApp | 0,6229 | **0,6266** |

A EDA §6.3 dizia que elas têm "forte poder discriminante complementar". Têm — mas
o que discriminam é **canal**, não veracidade. WhatsApp é curto, com caixa alta e
exclamações, e concentra fake no corpus. Removido o atalho de proveniência, elas
não somam nada. Recomendação: `--no-style` como referência.

### 4.2. Calibrar na validação completa desfaz o DFR

Calibrar por Platt na validação inteira derrubou o pior-grupo de **0,7301 para
0,3751**. A validação é 71,5% fake, o intercepto se ajusta para reproduzir esse
prior, e o corte volta para "fake" em quase tudo.

Mas 71,5% **não é a prevalência de desinformação em lugar nenhum** — é consequência
de 58,4% da base vir de subsets degenerados. O score passou a ser calibrado sob
prior balanceado (~47%), o que o torna essencialmente uma razão de verossimilhança.
Para um fluxo de prevalência conhecida π, aplique correção de prior por fora.

### 4.3. O viés dialetal PT-PT é grande e está medido

Encoder fine-tuned: macro-F1 **0,6248 em PT-PT contra 0,8503 no resto** — 22 pontos.

O `transformer_comparison.md` recomendava XLM-RoBERTa por esse motivo, mas como
argumento teórico. Agora é medição. Essa é a melhoria de maior retorno disponível.

---

## 5. Limitação metodológica que você precisa saber

**Os números OOD da linha fine-tuned não são OOD honestos.**

O encoder foi ajustado no split IID, que contém todos os canais. Quando depois
avaliei com `whatsapp` ou `portal` "retidos", o encoder **já tinha visto aqueles
textos com rótulo** durante o fine-tuning — só a cabeça foi retreinada sem eles.
É por isso que `ood:portal` aparece com 0,9721: não é generalização, é memória.

Os OOD da linha **congelada são válidos** (o encoder nunca viu rótulo nenhum), e
são eles que sustentam a afirmação de que o DFR melhora robustez de canal.

Corrigir isso exige reajustar o encoder uma vez por canal excluído: **~3,8 h por
canal** nesta máquina.

---

## 6. Próximos passos, em ordem de retorno

1. **Trocar o encoder por XLM-RoBERTa** — ataca diretamente os 22 pontos de queda
   em PT-PT. `python -m models.encoder --model xlm-roberta-base`
2. **OOD honesto** — reajustar o encoder por canal excluído (~7,6 h para os dois).
3. **Investigar os casos limítrofes sob DFR** — caíram de 0,9306 (ERM) para 0,6458.
   Parte é o DFR abandonando o atalho, mas 0,6458 nas meias-verdades é fraco e
   merece análise separada.
4. **Revisar `COVID19.BR`** — é o pior grupo confiável (0,7516) e tem o ECE mais
   alto (0,1108).

---

## 7. Como reproduzir

```bash
python -m models.run_overnight
```

Etapas com artefato existente são puladas. Estado em
`models/artifacts/pipeline_state.json`, logs por etapa em `models/artifacts/logs/`.
A folha de dados automática é `RELATORIO_DADOS.md`; este arquivo é a análise curada
e não é sobrescrito.

| etapa | minutos |
|---|---:|
| score congelado + ablação | 1,2 |
| fine-tuning (2 épocas) | 226,3 |
| export + extração ONNX na Vega 8 | 26,7 |
| score sobre fine-tuned | 0,6 |
