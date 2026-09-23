# BUILD_NOTES — v6 (builder)

Registro de decisões, medições, desvios e bugs encontrados na construção de
`models/v6/`. Base: pipeline v4 testado (`models/v4/`), NÃO alterado. Plano
delta: `models/v6/PLANO_ADAPTACAO_V6.md`; spec herdada:
`models/v4/spec_treino_v4.md`. Nada fora de `models/v6/` foi criado ou
modificado (exceto leitura de `models/data.py` e `models/evaluate.py`).

## 1. Arquivos criados

| arquivo | papel |
|---|---|
| `models/v6/prepare_v6.py` | CSVs v6 -> `v6_pool.parquet` + `v6_splits.parquet` + `prepare_stats.json` + `token_stats.json` |
| `models/v6/train_bertimbau_v6.py` | single-file Colab-ready (mesmo vendor v4 + treino/avaliação/resume/smoke) |
| `models/v6/colab_bertimbau_v6.ipynb` | driver Colab (19 células: T4, R0 full+DFR, R1 OOD, ablação informative, coleta, download) |
| `models/v6/.gitignore` | mantém `processed/` e `artifacts/` fora do Git |
| `models/v6/processed/*` | saídas do prepare (ignoradas pelo git) |
| `models/v6/artifacts/*` | saídas de smoke (ignoradas pelo git) |

Comando exato do prepare (codigo 0, `all_pass=True`, 108,0 s no total):

```powershell
python models/v6/prepare_v6.py --sanitized FakenewsBR_sanitized_v6.csv `
  --labels FakenewsBR_v6_labels.csv --provenance FakenewsBR_v6_provenance.csv `
  --out-dir models/v6/processed --seed 42 --token-stats
```

## 2. `prepare_v6.py` — medido vs esperado (tolerância 0)

| grandeza | esperado | medido |
|---|---:|---:|
| sanitizado (labels CSV) | 297.672 | 297.672 |
| pool (`train_label` fake/true) | 91.080 | 91.080 |
| fake / true | 66.772 / 24.308 | 66.772 / 24.308 |
| `label_tier` pool | checker 44.347, v1 39.466, checker_match 4.166, llm_local 3.092, corroborated 9 | idênticos |
| `verified_label` não nulo (rotulado ou "unknown") | 297.672 | 297.672 |
| grupos | 36 | 36 |
| grupos constantes / linhas | 11 / 39.373 | 11 / 39.373 |
| informativos (`is_balanced_group`) | 36.896 | 36.896 |
| U+FFFD | 4 | 4 (rids 11447, 16958, 836990361129089716, 839894016325789543) |
| mojibake (Latin Ext-A/B) | flag | 34 |
| `v6_pool.parquet` | <= 25 MB | 17,16 MB |
| `v6_splits.parquet` | - | 823,2 KB |
| sha256 `v6_pool.parquet` | - | `596b09116d8d54fe35cbdb07ec2ee4930982fbb18b0da29f4aebb799cef927cc` |
| sha256 `v6_splits.parquet` | - | `eb10b51e411cfc02ba03c4920e690b91b462880b28617dc504856d82561722bc` |

`verified_label` é `train_label` quando rotulado e `"unknown"` nas 206.592
linhas de procedência; a checagem registra cobertura 297.672/297.672.

Bugs/iterações do build: (i) o `text_key` normalizado precisa excluir
`unused` da checagem "1 lado por texto" (4 cópias não-whatsapp do `ood_wa`
ficam `unused` de propósito); (ii) a checagem de partição inicial acusava
`unused` como lado conflitante — corrigido antes da rodada final. O prepare
rodou 2x (1 sem token-stats para iterar + 1 final com token-stats).

## 3. Duplicatas — números reais (a v4 tinha 0)

Normalização: `investigation/expansion/dedup.py::normalize_text` (NFKD sem
acento, `[a-z0-9 ]`, espaços colapsados) — a mesma do quality report v6
(`duplicatas exatas remanescentes: 473`). O número só aparece com essa
normalização: com lower+whitespace o pool tem 0; o report mede "exato"
ignorando acento/pontuação.

| métrica | valor |
|---|---:|
| grupos de texto duplicados | **427** |
| linhas em grupos duplicados | **900** |
| cópias extras (linhas - grupos) | **473** |
| grupos com conflito fake/true | 3 (8 linhas) |
| grupos que cruzam canais | 12 |
| cópias não-whatsapp de textos do teste OOD (viram `unused`) | 4 |
| duplicatas entre os informativos | 322 cópias extras (608 linhas) |

Política adotada: **não remover** linhas (o pool preserva 91.080 e o teste OOD
6.381); o split é feito no nível do grupo de texto (`four_way_grouped`):
seleciona um representante por `text_key` e propaga o lado para todas as
cópias. Os 3 grupos conflitantes ficam juntos no mesmo lado (conflito de
rótulo é problema de dados, registrado, não resolvido aqui). Validações que
abortam código 2: `*.textkey_um_lado=0` (exceto `unused`),
`*.sem_vazamento_texto=0`, `full_iid_copias_mesmo_lado=0`.

Sonda de quase-duplicata (5-gram, 5k×5k, `full_iid`): `max_jaccard=0,9836`,
**25 pares > 0,8** (v4: 0,9967 e 141 pares). Duplicatas exatas tratadas, quase
duplicatas não — limitação residual registrada (a v4 tinha o mesmo limite).

## 4. Splits — contagens medidas

| split | train | val_sel | val_calib | test | unused |
|---|---:|---:|---:|---:|---:|
| `split_full_iid` | 63.752 (73,31% fake) | 9.105 (73,29%) | 4.557 (73,38%) | 13.666 (73,31%) | 0 |
| `split_ood_wa` | 71.994 (75,23%) | 8.470 (75,23%) | 4.231 (75,21%) | 6.381 (47,88%, whatsapp) | 4 (100% fake) |
| `split_bal_iid` | 25.832 (51,95%) | 3.685 (51,89%) | 1.842 (52,23%) | 5.537 (52,09%) | 54.184 (87,84%) |

Proporções por linha dentro de 1,0 p.p. dos alvos (70/10/5/15; 85/10/5), com o
desvio vindo das 473 cópias reagrupadas nos grupos. `bal_iid` fica a 5 linhas
do canônico v4 (25.827/3.689/1.845/5.535) por causa das duplicatas
reagrupadas. `ood_wa` test = exatamente os 6.381 do canal whatsapp.

## 5. Prior efetivo com DFR cell, `weight_clip=25` (média 1)

Percentis calculados no mesmo pipeline do trainer (`group_balanced_weights` ->
`min(w, 25)` -> renormaliza média 1), sobre o trem de cada split.

| split | antes: p50 / max | sem clip prior fake | depois: min / p25 / p50 / p75 / p95 / p99 / max | prior fake efetivo |
|---|---|---|---|---:|
| `full_iid` | 0,403 / 1.080,54 | 0,5254 | 0,1004 / 0,1895 / 0,5323 / 0,7531 / 2,5332 / 11,807 / **33,06** | **0,6032** |
| `ood_wa` | 0,319 / 1.220,24 | 0,5085 | 0,0970 / 0,1831 / 0,4383 / 0,7906 / 2,448 / 11,407 / **34,36** | **0,6062** |
| `bal_iid` | 0,644 / 38,44 | 0,5000 | 0,3370 / 0,5238 / 0,6586 / 0,7731 / 2,616 / 7,503 / **25,56** | **0,4888** |

média 1,0 em todos. O `max` pós-clip > 25 é efeito da renormalização pós-clip
(o trainer faz `min` e depois volta à média 1). O prior efetivo do `full_iid`
(0,603) difere de 0,5 porque 11 grupos são constante-rótulo (6 fake, 5 true) e
o clip muda a massa das células minúsculas (ex.: `FC_AFP|true`, n=1, peso bruto
1.080,54 -> 25). O smoke loga exatamente esse bloco (`[E5] DFR cell`).

## 6. Tokens pagos por época (tokenizer BERTimbau, batch 32, mega 50, seed 42)

| split de treino | n | 128 | **192** | 256 | média paga/amostra @192 | batches @192 | no cap | razão 256/192 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_iid` | 63.752 | 2,721 M | **3,184 M** | 3,526 M | 49,9 | 1.993 | 9,73% | 1,107 |
| `ood_wa` | 71.994 | 2,881 M | **3,337 M** | 3,664 M | 46,4 | 2.250 | 8,36% | 1,098 |
| `bal_iid` | 25.832 | 1,548 M | **1,933 M** | 2,213 M | 74,9 | 808 | 20,05% | 1,145 |

Distribuições: pool média 69,78 / p95 293,0 / >192 8,76%; informativos média
119,65 / p95 453,0 / >192 18,82% (idêntico ao v4, como esperado — os grupos
informativos não mudaram). `bal_iid` reproduz o custo 1,933 M e a razão 1,145
do v4.

## 7. `--check-vendor` (local)

`python models/v6/train_bertimbau_v6.py --check-vendor` -> **PASSOU**.
sha256 `models/evaluate.py` = `08d1acb0...1557` e `models/data.py` =
`cd7bb663...b9e4` conferem com os blocos embutidos; `apply_platt`, `fit_platt`,
`core_metrics`, `ece` (float64 e float32), `worst_group_f1`, `per_group`,
`reliability_table`, `report` (dict e stdout) e `group_balanced_weights`
maxdiff 0.

## 8. Smoke CPU (critério central) — medido

Comando exato do pedido (exit code **0**; parede **453,2 s**, tempo interno
425,0 s; limite 10 min):

```powershell
python models/v6/train_bertimbau_v6.py `
  --data models/v6/processed/v6_pool.parquet `
  --splits models/v6/processed/v6_splits.parquet `
  --split-col full_iid --out models/v6/artifacts/smoke_cpu `
  --smoke --smoke-n 1024 --device cpu --amp off --workers 0
```

- E2: treino 63.749 (4 U+FFFD removidas: 3 no treino, 1 no teste), val_sel
  9.105, val_calib 4.557, teste 13.665; interseção rid treino×teste = 0;
  duplicatas normalizadas treino/val×teste = 0.
- E5 (DFR ativo no smoke): antes/depois clip iguais no subconjunto de 1.024
  (max 24,38 < 25), prior fake efetivo 0,619.
- Throughput de treino: **5,23 amostras/s | 212,7 tokens/s pagos** (1024
  amostras, batch 8, seq <= 128, 128 steps). A v4 media 5,1 amostras/s e 307,7
  tokens/s; a diferença de tokens/s é composição: o subset `full_iid` tem
  textos mais curtos (média 39,5 tokens/amostra) e por isso paga menos padding
  por batch que o subset informativo do v4.
- 1 época: `train_loss=0,6655`, `val_loss=0,5620`, `val_macro_f1=0,4301`,
  `val_worst_group=0,4086`, `val_ece=0,0469`; Platt em `is_balanced_group`
  (n=444, prior 0,5248) a=1,0538 b=-0,6717; limiar ótimo 0,54.
- Tempos: token stats/tokenização 0,4 s; treino 195,8 s; avaliação final
  147,4 s; total 425,0 s. VRAM 0 (CPU).
- Contrato: **17/17 artefatos** presentes em `models/v6/artifacts/smoke_cpu`
  (22 arquivos no total, incluindo tokenizer).
- Caminhos extras validados com `--smoke-n 128`: `--ablate informative`
  (`split_col=bal_iid`, 25.832 -> 128, vazamentos 0, exit 0) e `--ablate ood`
  (`split_col=ood_wa`, `metrics.json.ood` não nulo, exit 0).

## 9. Projeção T4 (hipótese explícita, substituir pelo smoke T4)

Base: 212,7 tokens/s pagos medidos na CPU no smoke. Hipótese herdada do v4:
speedup T4/CPU em [10, 30]x, NÃO medido. Conta:
`min/época T4 = tokens_pagos / 212,7 / 60 / speedup`.

| run | tokens/época | CPU min/época | T4 10x | T4 30x | 2 épocas |
|---|---:|---:|---:|---:|---:|
| R0 `full_iid`+DFR (default) | 3.183.840 | 249,5 | 24,9 min | 8,3 min | **50–17 min** |
| R1 `ood_wa`+DFR | 3.337.376 | 261,5 | 26,2 min | 8,7 min | **52–17 min** |
| Abl `informative` | 1.933.472 | 151,5 | 15,1 min | 5,0 min | **30–10 min** |

Overheads por run (tokenização, validação por época, Platt + avaliação final
27,3k/19,1k/11,1k + `--predict-all` 91,1k, artefatos): 6–13 min. Pacote
R0+R1+informative: **~2,5–1,3 h**, dentro da sessão Colab.

## 10. Desvios e observações (honestos)

1. **"Informativos" para `bal_iid`**: uso a coluna canônica `is_balanced_group`
   = 36.896 (8 grupos com n>=200 e minoria >= 15%). O pool tem 29 grupos com
   n>=200 (90.822 linhas); os 21 grupos não-constantes listados no pedido somam
   51.465 e incluem checadores com minoria < 15% (FC_G1 3,5%, FC_ESTADAO 0,1%,
   etc.), que a regra de `models/data.py` não considera informativos.
   `prepare_stats.json` registra os dois cortes para auditoria.
2. **U+FFFD mantido no pool** (marcado) e removido defensivamente no treino,
   como no v4: manter 91.080 com tolerância 0 exige não removê-las no prepare.
3. **`--workers`**: default do CLI continua 2 (Colab); o smoke local roda com
   `--workers 0` (Windows/OneDrive), como no comando de aceitação.
4. **`--ablate fullpool` virou alias vazio**: o default v6 já é full_iid + DFR
   cell + 2 épocas; mantido por compatibilidade com os comandos v4.
5. **`--eval-col` cruzado é recusado**: treinar em `full_iid` e avaliar no
   teste de outro split aborta (código 2) porque o teste estaria dentro do
   treino; no v6 cada split tem teste próprio (o v4 tinha holdout canônico).
6. **`val_worst_group=None` em smokes de 128 linhas**: nenhum grupo confiável
   no subset; seleção cai no fallback macro-F1 (comportamento herdado do v4).
7. **3 grupos de texto com rótulo fake/true conflitante** (8 linhas) ficam no
   pool e no mesmo lado do split; não removidos nem resolvidos.
8. **Ablação `informative` reproduz o A0 v4** (treino 25.832 vs 25.827; custo
   1,933 M tokens/época idêntico, diferença = 322 cópias extras reagrupadas).
9. **Quase-duplicatas não tratadas**: sonda acusa 25 pares > 0,8 no `full_iid`
   (v4 acusava 141); clusters não persistidos — mesma limitação do v4.
10. **Números do quality report**: as "473 duplicatas" só aparecem com a
    normalização do `dedup.py` (sem acento/pontuação); com lower+whitespace o
    pool tem 0. Documentado para não haver divergência de interpretação.

## 11. Reprodução

```powershell
python models/v6/prepare_v6.py --sanitized FakenewsBR_sanitized_v6.csv `
  --labels FakenewsBR_v6_labels.csv --provenance FakenewsBR_v6_provenance.csv `
  --out-dir models/v6/processed --seed 42 --token-stats
python models/v6/train_bertimbau_v6.py --data models/v6/processed/v6_pool.parquet `
  --splits models/v6/processed/v6_splits.parquet --split-col full_iid `
  --out models/v6/artifacts/smoke_cpu --smoke --smoke-n 1024 `
  --device cpu --amp off --workers 0
python models/v6/train_bertimbau_v6.py --check-vendor
```
