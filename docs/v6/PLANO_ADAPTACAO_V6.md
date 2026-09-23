# Plano de adaptação v6 — pool completo no fine-tuning BERTimbau

- Projeto: FakenewsBR
- Base: `models/v4/PLANO_FT_V4_T4.md` (decisões, controles e orçamento v4)
- Alvo: `neuralmind/bert-base-portuguese-cased` no pool v6 (91.080 linhas rotuladas)
- Hardware alvo: 1x Tesla T4 16 GB (Colab), fp16
- Data: 2026-09-12
- Scripts: `models/v6/{prepare_v6.py, train_bertimbau_v6.py, colab_bertimbau_v6.ipynb}`
- Medições: `models/v6/BUILD_NOTES.md`, `models/v6/processed/prepare_stats.json`

Este documento é o DELTA do plano v4. Tudo o que não é redefinido aqui
continua valendo: protocolo `val_sel`/`val_calib`, Platt em `val_calib` com
máscara `is_balanced_group`, `worst_group_f1` como manchete, teste intocado,
ordem AMP, escada de OOM, retomada por época, semente e cudnn determinístico.

## 0. Mudança de produto

O default deixa de ser treinar apenas nos grupos informativos (36.896) e passa
a ser **treinar no pool completo da v6 (91.080)** com **pesos DFR por célula
grupo×rótulo, `weight_clip=25`, 2 épocas**. Motivo: o pool v6 ganhou massa de
checadores que o default v4 descartava; o DFR é o controle para o atalho de
procedência dos 39.373 rótulo-por-origem. O run informativo continua existindo
como ablação (`--ablate informative`), comparável ao A0 v4.

## 1. Pool v6 medido (tolerância 0)

| item | valor |
|---|---:|
| linhas sanitizadas (com e sem rótulo) | 297.672 |
| pool treinável (`train_label` fake/true) | **91.080** |
| fake / true | 66.772 (73,31%) / 24.308 (26,69%), razão **2,75:1** |
| `label_tier` do pool | checker 44.347, v1 39.466, checker_match 4.166, llm_local 3.092, corroborated 9 |
| grupos | 36 |
| grupos de rótulo constante | **11 grupos, 39.373 linhas** |
| duplicatas exatas normalizadas | **427 grupos / 900 linhas / 473 cópias extras** |
| `U+FFFD` | 4 linhas (rids 11447, 16958, 836990361129089716, 839894016325789543) |

Grupos constantes (linhas): `fakes` 20.347 fake, `FC_BOATOS` 10.772 fake,
`true` 2.710 true, `FC_BOATOS_VIRAL` 2.360 fake, `NEWS_PODER360` 1.338 true,
`NEWS_BRASILDEFATO` 909 true, `NEWS_OECO` 468 true, `NEWS_ECO` 453 true,
`FC_ALETHEIA` 9 fake, `MuMiN-PT_raw` 4 fake, `FC_NEXO` 3 fake.

Informativos (`is_balanced_group`, regra `models/data.py`: n>=200 e minoria>=15%
para `FC_*`/`EXT_*`, mais `BALANCED_GROUPS`): 36.896 linhas, fake 51,98%
(= 8 grupos: FC_POLIGRAFO 10.831, Fake.br 7.160, EXT_LIARBR 6.996,
FakeWhatsApp.BR_2018 6.381, EXT_AVERITECBR 2.925, COVID19.BR 1.931,
COVID19.BR_raw 373, LLM4BR_300 299). Existem 29 grupos com n>=200 (90.822
linhas); a lista de 21 grupos não-constantes do pedido soma 51.465, mas inclui
checadores com minoria <15% (FC_G1 3,5%, FC_ESTADAO 0,1% etc.), que a regra
canônica não considera informativos. `bal_iid` segue `is_balanced_group`
(ver BUILD_NOTES §10.1).

`NEWS_*`: 3.168 linhas `true` com `label_tier` llm_local/checker_match/
corroborated — rótulo fraco. Ablação `llmoff` remove os tiers llm_local e
corroborated do treino.

## 2. Splits (medidos, seed 42)

Regras: estratificação por `grupo|rótulo` (`_safe_strat`); **split no nível do
grupo de texto** (duplicatas exatas normalizadas forçadas ao mesmo lado); nada
removido do pool.

| split | train | val_sel (10%) | val_calib (5%) | test | unused |
|---|---:|---:|---:|---:|---:|
| `full_iid` (default) | 63.752 (73,31% fake) | 9.105 (73,29%) | 4.557 (73,38%) | 13.666 (73,31%) | 0 |
| `ood_wa` | 71.994 (75,23%) | 8.470 (75,23%) | 4.231 (75,21%) | 6.381 (47,88%, todo whatsapp) | 4 |
| `bal_iid` (informativos) | 25.832 (51,95%) | 3.685 (51,89%) | 1.842 (52,23%) | 5.537 (52,09%) | 54.184 |

- `full_iid`: pool completo 70/10/5/15. O teste IID contém os grupos de
  rótulo constante; por isso a manchete é `worst_group` (grupos confiáveis,
  n>=100 e minoria>=20) e o ECE de manchete é `ece_balanced`.
- `ood_wa`: teste = canal `whatsapp` (FakeWhatsApp.BR_2018, 6.381); restante
  85/10/5. As 4 cópias não-whatsapp de textos do teste ficam `unused` (nunca
  entram no treino). Responde "modelo treinado no resto do pool detecta
  WhatsApp?".
- `bal_iid`: 70/10/5/15 nos informativos; referência comparável ao A0 v4
  (`--ablate informative`, treino 25.832 vs 25.827 do v4; a diferença são as
  cópias duplicadas reagrupadas).

Comparação de contagens com o v4: `v4 bal_iid` 25.827/3.689/1.845/5.535;
`v4 full_iid` 67.725/7.968/3.984 (holdout canônico). A v6 não usa holdout no
`full_iid` (o default treina no pool completo); o teste é o test do próprio
split, construído por grupo de texto.

## 3. Duplicatas (473 cópias extras, medido)

Normalização = `investigation/expansion/dedup.py::normalize_text` (NFKD sem
acento, `[a-z0-9 ]`, espaços colapsados), a mesma do quality report da v6.
No pool: **427 grupos de texto, 900 linhas, 473 cópias extras**, das quais 3
grupos (8 linhas) têm rótulos conflitantes fake/true e 12 grupos cruzam canais.

Política: **não remover** (preserva 91.080 e o test de 6.381); o split
(`four_way_grouped`) seleciona um representante por grupo de texto e propaga o
lado para todas as cópias. Verificações no `prepare_stats.json`:
`*.textkey_um_lado = 0`, `*.sem_vazamento_texto = 0`, `full_iid_copias_mesmo_lado
= 0`. Quase-duplicatas (não idênticas) continuam existindo: sonda 5-gram
5k×5k do `full_iid` mede `max_jaccard=0,984` e 25 pares >0,8 — limitação
residual registrada, não bloqueante.

## 4. DFR cell, `weight_clip=25` (medido por split)

Peso por amostra = 1/tamanho da célula grupo×rótulo, normalizado a média 1;
depois `min(w, 25)` e renormalização (idêntico a `build_sample_weights`).

| split | células | prior fake efetivo (sem clip) | após clip: min / p50 / p95 / p99 / max | prior fake efetivo (após clip) |
|---|---:|---:|---|---:|
| `full_iid` | 59 | 0,5254 | 0,100 / 0,532 / 2,533 / 11,807 / 33,06 | **0,6032** |
| `ood_wa` | 59 | 0,5085 | 0,097 / 0,438 / 2,448 / 11,407 / 34,36 | **0,6062** |
| `bal_iid` | 16 | 0,5000 | 0,337 / 0,659 / 2,616 / 7,503 / 25,56 | **0,4888** |

Sem o clip a célula `FC_AFP|true` (1 exemplo no full_iid) teria peso ~1.080 por
amostra; com clip vira 25 e, após renormalização, 33,06. O prior efetivo do
`full_iid` não fica em 0,5 porque 11 grupos são de rótulo constante (6 fake,
5 true) e o clip muda a massa relativa das células pequenas — é o prior que a
CE realmente vê, e vai registrado no run (`loss_info`).

## 5. Comprimento e tokens pagos/época (medido)

| recorte | n | média | p50 | p95 | p99 | >192 |
|---|---:|---:|---:|---:|---:|---:|
| pool | 91.080 | 69,8 | 24 | 293 | 766 | 8,76% |
| informativos | 36.896 | 119,65 | 33 | 453 | 1.046 | 18,82% |

Custo do sampler (`length_grouped_batches`, batch 32, mega 50, seed 42,
comprimentos truncados no cap), 192 tokens:

| split de treino | n | batches @192 | tokens pagos/época | média paga/amostra | batches no cap | razão 256/192 |
|---|---:|---:|---:|---:|---:|---:|
| `full_iid` | 63.752 | 1.993 | **3.183.840 (3,18 M)** | 49,9 | 9,73% | 1,107 |
| `ood_wa` | 71.994 | 2.250 | **3.337.376 (3,34 M)** | 46,4 | 8,36% | 1,098 |
| `bal_iid` | 25.832 | 808 | **1.933.472 (1,93 M)** | 74,9 | 20,05% | 1,145 |

O custo do `bal_iid` reproduz o v4 (1,933 M; razão 1,145), o que é o controle
de sanidade do sampler/tokenizer.

## 6. Orçamento T4 (preliminar, com conta explícita)

Base medida no smoke CPU (batch 8, seq<=128, 6 camadas congeladas):
**213 tokens/s pagos**. A projeção usa a hipótese v4 de speedup T4/CPU em
[10, 30]x, ainda NÃO medida em GPU (o smoke da T4 da seção 5.4 do spec v4
substitui). Conta: `tempo_CPU_época = tokens_pagos / 213`; `tempo_T4 =
tempo_CPU / speedup`.

| run | tokens/época | CPU (min/época) | T4 10x (min/época) | T4 30x (min/época) | 2 épocas (10x–30x) |
|---|---:|---:|---:|---:|---:|
| R0 `full_iid`+DFR | 3,18 M | 249,1 | 24,9 | 8,3 | **50–17 min** |
| R1 `ood_wa`+DFR | 3,34 M | 261,1 | 26,1 | 8,7 | **52–17 min** |
| Abl `informative` | 1,93 M | 151,3 | 15,1 | 5,0 | **30–10 min** |

Overheads por run: tokenização 1–2 min; validação em `val_sel` por época ~1
min (9,1k/8,5k/3,7k); Platt + avaliação final (val_sel+val_calib+teste:
27,3k/19,1k/11,1k linhas) + `--predict-all` (91,1k) 4–9 min (extrapolado do v4);
artefatos e espelho no Drive 1–2 min. Total por run: **R0 ~57–30 min**,
**R1 ~59–30 min**, **abl ~37–19 min**.

- Pacote R0 + R1 + informative: **~2,5–1,3 h de treino** + 8–12 min de setup.
- Margem contra as 12 h do Colab: folgada; 2–3 runs por sessão.

## 7. Riscos específicos da v6 e mitigação

1. **473 duplicatas exatas** (a v4 tinha 0): split por grupo de texto; nada
   removido; sonda de quase-duplicata segue acusando 25 pares >0,8 no
   `full_iid` (residual, documentado).
2. **39.373 linhas de rótulo por origem** (43,2% do pool): o DFR cell
   limita o peso total de cada célula, mas o teste IID contém esses grupos.
   Por isso a manchete é `worst_group` + `ece_balanced`, e o OOD `ood_wa` é a
   régua de generalização entre canais. Ablação `informative` mede o custo de
   largar essa massa.
3. **Prior 2,75:1 e prior efetivo DFR 0,603**: Platt em `val_calib` restrito a
   `is_balanced_group` (prior ~0,52) corrige o deslocamento para implantação;
   `calib_prior_fake` fica em `calibration.json`.
4. **NEWS_* com rótulo fraco** (3.168 linhas `true`, llm_local principalmente):
   ablação `llmoff` remove llm_local/corroborated; medir antes de manter.
5. **Definição de "informativos"** para `bal_iid`: seguimos `is_balanced_group`
   (36.896); a lista n>=200 do coordenador (51.465) inclui checadores com
   minoria <15% e fica registrada no `prepare_stats.json` para auditoria.
6. **3 grupos de texto com rótulos conflitantes fake/true** ficam juntos no
   mesmo lado, como qualquer duplicata; não removidos.

## 8. Ablações e prioridade (herda a ordem de sacrifício do v4)

| prioridade | id / preset | pergunta | custo (T4) |
|---:|---|---|---:|
| 1 | R0 default `full_iid`+DFR | baseline v6 | 30–57 min |
| 2 | R1 `--ablate ood` | generaliza entre canais (portal→WhatsApp)? | 30–59 min |
| 3 | `--ablate informative` | largar os grupos constantes melhora a fronteira? | 19–37 min |
| 4 | `--ablate llmoff` | o `true` fraco de llm_local polui? | 30–57 min |
| 5 | `--ablate maxlen256` | truncamento (8,76% do pool >192) custa pior-grupo? | 33–63 min |
| 6 | `--ablate freeze4` / `freeze0` | capacidade do encoder importa? | 30–57 min |
| 7 | `--ablate classweights` | mede degradação de calibração | 30–57 min |

## 9. Critérios de aceitação do build (executados)

- `prepare_v6.py` -> `all_pass=True`, contagens da seção 1 com tolerância 0,
  splits da seção 2, duplicatas da seção 3, DFR da seção 4.
- `train_bertimbau_v6.py --check-vendor` -> PASSOU (sha256 e sintéticos).
- Smoke CPU exato (`--split-col full_iid --smoke-n 1024 --device cpu --amp off
  --workers 0`) -> exit 0 em <10 min, contrato de 17 artefatos completo.
- Detalhes e números em `models/v6/BUILD_NOTES.md`.
