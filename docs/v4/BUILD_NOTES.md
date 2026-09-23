# BUILD_NOTES — v4 (builder)

Registro de decisoes, medicoes, desvios e bugs encontrados durante a construcao
de `models/v4/{prepare_v4.py, train_bertimbau_v4.py, colab_bertimbau_v4.ipynb}`.
Spec: `models/v4/spec_treino_v4.md`; plano (vence conflitos):
`models/v4/PLANO_FT_V4_T4.md`. Nada fora de `models/v4/` foi alterado; os
mtimes de `models/*.py` mostram que as modificacoes la ja existiam antes do
build.

## 1. Arquivos criados

| arquivo | papel |
|---|---|
| `models/v4/prepare_v4.py` | CSVs -> `v4_pool.parquet` + `v4_splits.parquet` + `prepare_stats.json` + `token_stats.json` |
| `models/v4/train_bertimbau_v4.py` | single-file Colab-ready (vendor + treino + avaliacao + resume + smoke) |
| `models/v4/colab_bertimbau_v4.ipynb` | driver Colab (nbformat 4.5, 9 passos) |
| `models/v4/.gitignore` | mantem `processed/` e `artifacts/` fora do Git |
| `models/v4/processed/*` | saidas do prepare (ignoradas pelo git) |
| `models/v4/artifacts/*` | saidas de smoke (ignoradas pelo git) |

## 2. `prepare_v4.py` — numeros medidos vs esperados

`python models/v4/prepare_v4.py --sanitized ... --labels ... --provenance ...
--out-dir models/v4/processed --seed 42 --token-stats --force` -> `all_pass=True`
(codigo 0).

| grandeza | esperado (PLANO/spec) | medido |
|---|---:|---:|
| pool (load) | 85.212 | 85.212 |
| fake / true | 60.991 / 24.221 | 60.991 / 24.221 |
| informativos | 36.896 | 36.896 |
| grupos constantes | 39.351 | 39.351 |
| `bal_iid` train/val_sel/val_calib/test | 25.827 / 3.689 / 1.845 / 5.535 | identicos |
| `bal_ood_wa` train/val_sel/val_calib/test | 25.937 / 3.052 / 1.526 / 6.381 | identicos |
| `full_iid` train/val_sel/val_calib (+holdout) | 67.725 / 7.968 / 3.984 (+5.535) | identicos |
| interseccao holdout x teste | 0 | 0 |
| tamanho `v4_pool.parquet` | <= 25 MB | 16,72 MB |
| tokens informativos: media / p95 / >192 | 119,6 / 453 / 18,82% | 119,65 / 453,0 / 18,82% |
| custo 128/192/256 M tokens | 1,5 / 1,9 / 2,2 | 1,545 / 1,933 / 2,213 |
| razao 256/192 | 1,145 | 1,1452 |
| media paga/amostra 192 / batches no cap | 74,8 / 20,3% | 74,83 / 19,8% |

Distribuicoes de pool tambem batem (media 73,0; p50 27; p90 180; p95 309;
p99 745; max 8.575; >192 9,36%; >256 6,52%). A simulacao de custo e
`length_grouped_batches(batch=32, mega=50, seed=42)` com comprimentos truncados
no cap e `tokens_pagos = sum(len(batch) * max(len_no_batch))`, igual ao
Apêndice B do PLANO.

### 2.1 Desvio: linhas U+FFFD permanecem no pool (marcadas)

Spec 3.2 pede remover as 4 linhas com U+FFFD; spec 8.1 exige pool 85.212,
informativos 36.896 e splits identicos ao PLANO 2.1. Medicao: das 4 linhas, 2
sao **informativas** (`rid` 11447 e 16958, `FakeWhatsApp.BR_2018`); remove-las
muda o pool para 85.208, os informativos para 36.894 e desloca as contagens de
todos os splits (violando 8.1 com tolerancia 0). Decisao: manter as 4 linhas no
pool com `has_ufffd=True` (a coluna existe no schema do PLANO 8) e registrar
`ufffd_rids` em `prepare_stats.json`; o trainer remove `has_ufffd` de treino e
avaliacao (E2, "defensivo"), de modo que elas nunca entram no treino. Efeito nas
contagens reais vistas pelo trainer: `bal_iid` train 25.825, `full_iid` train
67.722, OOD test 6.379 etc. (diferencas de 1-3 linhas).

### 2.2 Achado: sonda de quase-duplicata acusou

Sonda 5-gram/Jaccard exato em amostra 5k x 5k do treino x teste de `bal_iid`:
`max_jaccard=0,99665`, `n_pares_>0,8 = 141` (pares reais de virais de WhatsApp
e COVID repostados com pequenas edicoes, majoritariamente fake|fake). O PLANO
11.1 diz que "se a sonda acusar, refazer split por cluster" — porem a spec 8.1
fixa as contagens de split com tolerancia 0 e a secao 3.7 pede apenas
**registrar** a sonda. O builder registra e sinaliza; refazer o split e decisao
do planejador (mudaria as contagens canonicas e invalidaria a reprodutibilidade
do PLANO 2.1).

### 2.3 Outras observacoes do prepare

- `mojibake_flag` (Latin Extended-A/B): 32 linhas (o PLANO estimou "~50";
  flag-only, sem efeito em split/treino).
- `label_tier` NaN (linhas `provenance`) virou `"none"` no parquet.
- `label_source` vem do sanitizado (o `load()` do repo so mergeia
  rid/train_label/label_tier/auto_label/confidence/method), nao do CSV de
  labels; conteudo identico na pratica.
- Linha "treino `bal_iid`" do PLANO 3 (media 106,2 / p95 432) nao reproduz:
  medido 120,8 / 453,0. A linha "informativos" (a usada nos criterios 8.1/8.3)
  reproduz exatamente, e a simulacao de custo (74,8 media paga a 192) tambem;
  a discrepancia parece erro de medicao/typo do PLANO naquela linha.

## 3. Vendor e `--check-vendor`

- Bloco `# === VENDOR: models/evaluate.py @ 08d1acb0...1557 ===` com as
  funcoes `fit_platt`, `apply_platt`, `_decision`,
  `expected_calibration_error`, `reliability_table`, `core_metrics`,
  `per_group`, `worst_group_f1`, `report`, `MIN_GROUP_N`, `MIN_MINORITY_N`;
  bloco extra para `models/data.py::group_balanced_weights @ cd7bb663...b9e4`.
- Verificacao por `inspect.getsource` contra `models/evaluate.py` e
  `models/data.py`: **todas VERBATIM** (inclusive `core_metrics`, apos mover os
  imports sklearn para o topo do script).
- `python models/v4/train_bertimbau_v4.py --check-vendor` -> PASSOU:
  sha256 dos dois arquivos OK; `apply_platt`/`fit_platt`/`core_metrics`/`ece`
  float64 maxdiff 0,0 e float32 0,0; `worst_group_f1`, `per_group`,
  `reliability_table`, `report` (dict e stdout) e `group_balanced_weights`
  maxdiff 0. Fora do repo (Colab) o flag falha com "vendor check e local".

## 4. Smoke CPU (criterio central) — medido

Comando exato da spec 8.3 (`--smoke-n 1024 --device cpu --amp off --workers 0`):

- **Wall 437,8 s (~7,3 min)**; tempo interno 421,8 s; **exit code 0**.
  Reexecucao final limpou os artefatos: 423,2 s internos / 439,6 s de parede,
  com as mesmas metricas (determinismo do caminho CPU).
- Throughput de treino: **5,12 amostras/s | 309 tokens/s pagos** (1024
  amostras, batch 8, seq <= 128, 128 steps, 6 camadas congeladas).
- VRAM de pico: 0 (CPU).
- Tempos por etapa: token stats/tokenizacao 3,3 s (36.896 informativos);
  datasets 0,2 s; treino 199,9 s (128 steps); inferencia final (val_sel +
  val_calib + teste, 3x1024) 160,0 s; total 421,8 s.
- Historico da 1a epoca: `train_loss=0,6779`, `val_loss=0,6586`,
  `val_macro_f1=0,6150`, `val_worst_group=0,4091`, `val_ece=0,0465`; Platt em
  `val_calib` (`is_balanced_group`, n=1024): a=1,9558 b=-0,1468; ECE 0,0644 ->
  0,0197; limiar otimo 0,44.
- `token_stats.json`: informativos media 119,65 / p95 453,0 / >192 18,82% /
  razao 1,1457 (tolerancia 2% do PLANO 3: OK).
- Contrato: **17/17 artefatos presentes** em `models/v4/artifacts/smoke_cpu`
  (`best/` com safetensors+config+tokenizer+calibration.json, `last/` com
  model/tokenizer/optimizer/scheduler/scaler/rng/trainer_state, history,
  run_config, metrics, calibration, predictions, per_group, reliability,
  token_stats).
- Projecao T4 impressa e **marcada como HIPOTESE** (speedup 10-30x, nao
  medido): R0 192 = **10,4-3,5 min/epoca**, 3 epocas ~31-10 min.

### 4.1 Resume

- Repetir o comando com `--resume`: `[E7] retomando ... epoca 1/1`, nenhuma
  epoca nova executada, `history.json` continua com 1 epoca e
  `last/trainer_state.json` com `epoch=1` — sem duplicacao. Exit 0.
- `--resume --seed 43`: aborta com codigo 2 e diff
  `seed: 42 -> 43` (validacao de `data_sha256`, `splits_sha256`, `split_col`,
  `max_length`, `freeze_layers`, `model`, `seed`, `dfr_weights`,
  `class_weights`).

### 4.2 Caminhos extras validados localmente (smoke-n 128)

- `--ablate fullpool` sem flags explicitas: `split_col=full_iid`,
  `dfr_weights=cell` (log antes/depois do clip), holdout impresso
  `interseccao treino x teste canonico = 0`, eval no teste canonico, run_id
  `full_iid_ml192_f6_on_seed42_ablate-fullpool_smoke`. Exit 0.
- `--split-col bal_ood_wa --eval-col bal_ood_wa --grad-accum 2`:
  `metrics.json.ood` **nao nulo** (n=128 no smoke; no run cheio sera 6.379
  apos U+FFFD); `steps/epoca=8` com efetivo 16 (128/16), `global_step=8` no
  trainer_state. Exit 0.

### 4.3 Bugs encontrados e corrigidos durante o build

1. `--split-col bal_iid` era tratado como nome de coluna; o parquet tem
   `split_bal_iid`. Corrigido com resolvedor `split_<nome>`.
2. O parquet tem `label` (string), nao `target`; o trainer agora deriva
   `target = (label == "fake")`.
3. `history`/`resume` sem epocas novas quebrava o resumo do smoke
   (`rec` indefinido); corrigido para `tr_rec`.
4. `_max_diff` (check-vendor) quebrava com `np.ndarray`; tratamento adicionado.
5. Autocast com `torch.amp.autocast` so e aberto em CUDA/AMP (nullcontext no
   caminho CPU) para evitar warnings.
6. `--probe-max-length` com default 256 executaria o probe e **sairia sem
   treinar** em toda invocacao CUDA (o fluxo 5.5 nao passa a flag). Corrigido
   para opt-in com `nargs="?"`, `const=256`, `default=None`: sem a flag, treina;
   `--probe-max-length` usa 256; `--probe-max-length N` usa N. Em CPU o probe
   avisa e o treino continua (testado, exit 0). Desvio registrado na secao 5.9.

## 5. Desvios assumidos (todos documentados)

1. **U+FFFD mantido no pool** (secao 2.1) — para cumprir 8.1; treino remove.
2. **`--token-stats` isolado sai apos gravar `token_stats.json`** (nao treina).
   A reproducao da tabela do PLANO 3 (comando do Apêndice B) fica barata e
   deterministica; sem a flag, o treino roda normalmente.
3. **`pin_memory` so em CUDA** e `workers=0` no smoke local (orientacao do
   coordenador); default do CLI continua 2 como na spec.
4. **`scaler.pt` sempre gravado** (vazio `{}` na CPU) para cumprir o contrato
   de `last/` sem GPU.
5. **Smoke forca `--workers 0` implicito? Nao** — o valor vem do CLI; o
   comando de aceitacao passa 0. Com `workers>0`, `persistent_workers=True` e
   `prefetch_factor=2` (testado apenas o caminho workers=0 localmente).
6. **OOD**: `metrics.json.ood` e o bloco do canal retido quando
   `--eval-col bal_ood_wa`; nos demais runs e `null`. `test` e `ood` coincidem
   no R1 (o teste do R1 *e* o canal retido).
7. **`timing.train_s` em retomada** reflete apenas o processo atual (0 se
   nenhuma epoca rodou); os demais tempos sao regravados.
8. **`config.args` em `metrics.json`** inclui as flags efetivas pos-preset/
   pos-smoke (o que foi realmente executado).
9. **`--probe-max-length` e opt-in** (`nargs="?"`, `const=256`): o default
   literal 256 da spec 4.1 faria o R0/R1 sair apos o probe sem treinar. Sem a
   flag, nao roda; com a flag sem valor, usa 256.

## 6. Riscos / pendencias para a T4

- O caminho AMP (fp16 + GradScaler, `unscale_ -> clip -> step -> update`),
  `--probe-max-length` e a checagem de VRAM > 13,5 GB **nao foram exercitados
  localmente** (sem CUDA). Codigo segue a ordem obrigatoria da spec 5.2/PLANO.
- A projecao de tempo da T4 e hipotese (speedup 10-30x sobre a CPU local); o
  smoke da T4 substitui pelo medido.
- A sonda de quase-duplicata acusou (secao 2.2): decidir se o split por cluster
  sera refeito antes dos runs finais; sem isso, `bal_iid.test` contem virais
  quase identicos a itens de treino (mesmo rotulo), inflando as metricas.
- `predict-all` roda sobre 85.208 linhas no run cheio; nao medido na CPU.

## 7. Reproducao

```powershell
python models/v4/prepare_v4.py --sanitized FakenewsBR_sanitized_v4.csv `
  --labels FakenewsBR_v4_labels.csv --provenance FakenewsBR_v4_provenance.csv `
  --out-dir models/v4/processed --seed 42 --token-stats --force
python models/v4/train_bertimbau_v4.py --data models/v4/processed/v4_pool.parquet `
  --splits models/v4/processed/v4_splits.parquet --split-col bal_iid `
  --out models/v4/artifacts/smoke_cpu --smoke --smoke-n 1024 `
  --device cpu --amp off --workers 0
python models/v4/train_bertimbau_v4.py --check-vendor
```
