# Relatório — Score de Confiança FakenewsBR

Gerado automaticamente em 2026-09-10 06:50.

## Baseline a superar

| modelo | acurácia | F1(fake) |
|---|---|---|
| A (ingenua, TF-IDF) | 0.7811 | 0.8413 |
| B (higienizada, TF-IDF) | 0.7670 | 0.8292 |

> A comparação honesta é contra a **Variante B** (0,7670), que teve os atalhos removidos. A Variante A é o teto contaminado.

## BERTimbau congelado + estilo

| split | variante | macro-F1 (bal.) | pior-grupo | ECE (bal.) | acc (total) | macro-F1 (total) | PR-AUC | limítrofes |
|---|---|---|---|---|---|---|---|---|
| iid | ERM (ingenua) | 0.7646 | 0.6448 | 0.0178 | 0.7936 | 0.7503 | 0.9143 | 0.9097 |
| iid | DFR (honesta) | 0.8117 | 0.7226 | 0.0224 | 0.5704 | 0.5645 | 0.8461 | 0.3611 |
| ood:whatsapp | ERM (ingenua) | 0.4553 | 0.4553 | 0.3321 | 0.5247 | 0.4553 | 0.5263 | — |
| ood:whatsapp | DFR (honesta) | 0.6229 | 0.6229 | 0.2180 | 0.6289 | 0.6229 | 0.6355 | — |
| ood:portal | ERM (ingenua) | 0.6726 | 0.6726 | 0.0424 | 0.6759 | 0.6724 | 0.7310 | — |
| ood:portal | DFR (honesta) | 0.6164 | 0.6164 | 0.0884 | 0.6399 | 0.6161 | 0.7484 | — |

No split IID, a ingênua tem acurácia total 0.7936 contra 0.5704 da honesta — mas o pior-grupo inverte: 0.6448 contra **0.7226**. A queda na acurácia total é o modelo deixando de responder "falso" automaticamente para tudo que tem cara de alegação de agência.

Fora de domínio em `whatsapp`: ingênua macro-F1 0.4553 (ECE 0.3321) contra honesta 0.6229 (ECE 0.2180).

Fora de domínio em `portal`: ingênua macro-F1 0.6724 (ECE 0.0420) contra honesta 0.6161 (ECE 0.0889).

## BERTimbau congelado, sem estilo

| split | variante | macro-F1 (bal.) | pior-grupo | ECE (bal.) | acc (total) | macro-F1 (total) | PR-AUC | limítrofes |
|---|---|---|---|---|---|---|---|---|
| iid | ERM (ingenua) | 0.7646 | 0.6414 | 0.0186 | 0.7904 | 0.7473 | 0.9132 | 0.9028 |
| iid | DFR (honesta) | 0.8129 | 0.7304 | 0.0226 | 0.5720 | 0.5660 | 0.8460 | 0.3611 |
| ood:whatsapp | ERM (ingenua) | 0.4549 | 0.4549 | 0.3290 | 0.5244 | 0.4549 | 0.5243 | — |
| ood:whatsapp | DFR (honesta) | 0.6266 | 0.6266 | 0.2119 | 0.6341 | 0.6266 | 0.6393 | — |

No split IID, a ingênua tem acurácia total 0.7904 contra 0.5720 da honesta — mas o pior-grupo inverte: 0.6414 contra **0.7304**. A queda na acurácia total é o modelo deixando de responder "falso" automaticamente para tudo que tem cara de alegação de agência.

Fora de domínio em `whatsapp`: ingênua macro-F1 0.4549 (ECE 0.3290) contra honesta 0.6266 (ECE 0.2119).

## BERTimbau fine-tuned + estilo

| split | variante | macro-F1 (bal.) | pior-grupo | ECE (bal.) | acc (total) | macro-F1 (total) | PR-AUC | limítrofes |
|---|---|---|---|---|---|---|---|---|
| iid | ERM (ingenua) | 0.8610 | 0.7474 | 0.0115 | 0.8451 | 0.8136 | 0.9532 | 0.9306 |
| iid | DFR (honesta) | 0.8709 | 0.7516 | 0.0190 | 0.7345 | 0.7172 | 0.9326 | 0.6458 |
| ood:whatsapp | ERM (ingenua) | 0.7259 | 0.7259 | 0.1054 | 0.7314 | 0.7259 | 0.8050 | — |
| ood:whatsapp | DFR (honesta) | 0.7779 | 0.7779 | 0.0667 | 0.7779 | 0.7779 | 0.8367 | — |
| ood:portal | ERM (ingenua) | 0.9721 | 0.9721 | 0.0925 | 0.9721 | 0.9721 | 0.9958 | — |
| ood:portal | DFR (honesta) | 0.9522 | 0.9522 | 0.0667 | 0.9522 | 0.9522 | 0.9939 | — |

No split IID, a ingênua tem acurácia total 0.8451 contra 0.7345 da honesta — mas o pior-grupo inverte: 0.7474 contra **0.7516**. A queda na acurácia total é o modelo deixando de responder "falso" automaticamente para tudo que tem cara de alegação de agência.

Fora de domínio em `whatsapp`: ingênua macro-F1 0.7259 (ECE 0.1054) contra honesta 0.7779 (ECE 0.0667).

Fora de domínio em `portal`: ingênua macro-F1 0.9721 (ECE 0.0924) contra honesta 0.9522 (ECE 0.0669).

## Execução

| etapa | status | minutos |
|---|---|---|
| 1_embed_export_base | skipped | — |
| 2_embed_run_base | skipped | — |
| 3_score_frozen | ok | 0.7 |
| 4_score_frozen_nostyle | ok | 0.5 |
| 5_finetune | ok | 226.3 |
| 6_embed_export_ft | ok | 0.7 |
| 7_embed_run_ft | ok | 26.0 |
| 8_score_finetuned | ok | 0.6 |
| 9_relatorio | ok | 0.0 |

## Como ler estes números

- **pior-grupo** é a métrica de manchete, restrita a grupos com massa estatística (n ≥ 100 e classe minoritária ≥ 20). Acurácia média é enganosa porque 58,4% da base tem o rótulo determinado pela origem.
- As colunas **(bal.)** são medidas apenas nos grupos onde a origem não prediz o rótulo (`Fake.br`, `FakeWhatsApp.BR_2018`, `COVID19.BR`, `LLM4BR_300`). São as que valem: é o único regime em que a pergunta "o modelo entende desinformação?" está bem-posta.
- As colunas **(total)** incluem os subsets degenerados. Servem para mostrar o contraste, não para julgar o modelo.
- O score é calibrado sob **prior balanceado** (~47% fake), não sob os 71,5% da base — esse número é artefato de compilação, não prevalência real. Consequência: `ECE (bal.)` é a medida válida de calibração; o ECE sobre a base inteira penaliza uma escolha deliberada. Para um fluxo de prevalência conhecida π, aplique correção de prior por fora.
- Espera-se que **DFR tenha acurácia total menor e pior-grupo maior** que ERM. Isso é a troca aceita pelo projeto, não um defeito.
- `limítrofes` é a acurácia nas meias-verdades (`Enganoso`, `Distorcido`, `Fora de contexto`).

## Limitação conhecida

Os números **fora de domínio da linha fine-tuned não são OOD honestos**. O encoder é ajustado no split IID, que contém todos os canais — quando depois avaliamos com `whatsapp` ou `portal` "retidos", o encoder já viu aqueles textos com rótulo. Só a cabeça foi retida, não o encoder.

Os OOD da linha **congelada são válidos** (o encoder nunca viu rótulo nenhum). Para OOD honesto com fine-tuning seria preciso reajustar o encoder uma vez por canal excluído: ~3,8 h por canal nesta máquina.
