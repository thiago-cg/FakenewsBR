# Especificação funcional do treino v4 — para o agente builder

- Dono do plano: `models/v4/PLANO_FT_V4_T4.md` (decisões e números)
- Alvo: BERTimbau em 1x Tesla T4 (Colab), fp16
- Este documento define **o que construir**: arquivos, CLI, comportamento por
  etapa, contrato de saída e critérios de aceitação. Não redefine decisões; em
  caso de conflito, o PLANO vence.

## 1. Arquivos a criar

```
models/v4/
  prepare_v4.py               # local: CSVs v4 -> parquet enxuto + splits + stats
  train_bertimbau_v4.py       # single-file: treino, avaliação, retomada, smoke
  colab_bertimbau_v4.ipynb    # driver Colab (células da seção 7)
  artifacts/                  # (ignorado pelo git) saídas locais/smoke
```

Nada existente em `models/` pode ser alterado. `prepare_v4.py` pode importar
`models.data`; `train_bertimbau_v4.py` **não** importa nada do repositório
(precisa rodar no Colab só com o parquet).

## 2. Decisão de dependências: embutir `evaluate.py` (vendoring)

O Colab só terá `v4_pool.parquet`, `v4_splits.parquet` e o script. Decisão:
**embutir** no `train_bertimbau_v4.py`, em bloco delimitado
`# === VENDOR: models/evaluate.py @ <sha256> ===`, as funções copiadas
verbatim de `models/evaluate.py`:

`core_metrics`, `fit_platt`, `apply_platt`, `_decision`,
`expected_calibration_error`, `reliability_table`, `per_group`,
`worst_group_f1`, `report`, `MIN_GROUP_N`, `MIN_MINORITY_N`.

Motivo: sem clone de repo, sem download de rede em sessão, sem divergência de
branch. Para impedir drift, o script embute também
`group_balanced_weights` de `models/data.py` (10 linhas, usada só com
`--dfr-weights cell`) e expõe:

```
python models/v4/train_bertimbau_v4.py --check-vendor
```

que, **somente quando `models.evaluate` for importável** (local), compara
`core_metrics`, `fit_platt/apply_platt`, `ece`, `worst_group_f1` e `report`
entre o bloco embutido e o módulo original sobre um dataset sintético com seed
fixa e exige diferença 0 (float64) / 1e-9 (float32). No Colab esse flag falha
com mensagem "vendor check é local".

## 3. `prepare_v4.py`

CLI:

```
python models/v4/prepare_v4.py \
  --sanitized FakenewsBR_sanitized_v4.csv \
  --labels FakenewsBR_v4_labels.csv \
  --provenance FakenewsBR_v4_provenance.csv \
  --out-dir models/v4/processed \
  --seed 42 \
  --test-frac 0.15 --val-sel-frac 0.10 --val-calib-frac 0.05 \
  --ood-channel whatsapp \
  [--token-stats] [--force]
```

Defaults: paths da raiz do repo, `--out-dir models/v4/processed`, `--seed 42`,
frações acima, `--ood-channel whatsapp`, `--token-stats` desligado, `--force`
desligado (aborta se as saídas existirem).

Comportamento:

1. `df = models.data.load(csv=sanitized, labels_csv=labels,
   provenance_csv=provenance)` (regras idênticas ao PLANO 8).
2. Remove linhas com `U+FFFD` em `text_no_url` (4 esperadas); marca
   `mojibake_flag` por Latin Extended-A/B fora de pt-BR; mantém.
3. Deriva `group`, `channel`, `is_balanced_group`, `rating_class`, `era`,
   `has_ufffd`; constrói as colunas exatas do parquet do PLANO 8.
4. Constrói os três splits com estratificação `grupo|rótulo` (**reusar
   `models.data._safe_strat`** para semântica idêntica):
   - `split_bal_iid`: informativos, 70/10/5/15 (valores
     `train/val_sel/val_calib/test`);
   - `split_bal_ood_wa`: informativos, teste = canal `whatsapp`; restante
     85/10/5 (`train/val_sel/val_calib/test/unused`);
   - `split_full_iid`: pool completo **menos** `split_bal_iid == test`; restante
     85/10/5 (`train/val_sel/val_calib/unused`).
5. Escreve `v4_pool.parquet`, `v4_splits.parquet` (snappy) e
   `prepare_stats.json`; com `--token-stats`, escreve `token_stats.json`.
6. `prepare_stats.json` deve conter: contagens por split e por rótulo, %fake,
   `sha256` dos dois parquets e dos três CSVs de entrada, versões de
   pandas/sklearn/pyarrow, resultado da sonda de quase-duplicata (ver abaixo),
   flags de validação.
7. Sonda de quase-duplicata: 5-gramas de palavra sobre amostra de 5.000 linhas
   de treino × 5.000 de teste (seed 42, min-hash simples por assinatura ou
   Jaccard exato na amostra); registra `max_jaccard` e `n_pares_>0.8`.
8. Validações que abortam com código 2 e mensagem clara:
   - pool 85.212; informativos 36.896; `rid` único; fake 60.991 / true 24.221;
   - 0 interseção `split_full_iid train|val_*` com `split_bal_iid test`;
   - toda linha do pool tem exatamente um valor por coluna de split;
   - nenhuma linha informativa com `unused` em `split_bal_iid`.
9. Imprime ao final a tabela de contagens (comparável à seção 2.1 do PLANO) e
   o caminho dos arquivos.

`--token-stats` mede tokens reais com
`AutoTokenizer.from_pretrained(neuralmind/bert-base-portuguese-cased)` sobre
`text_no_url` (`add_special_tokens=True`, sem truncar) e grava percentis
(p50/p90/p95/p99/max), `% > 128/192/256/512` e a simulação de custo
(`length_grouped_batches`, batch 32, mega 50, seed 42) para 128/192/256, com
`tokens_pagos_por_epoca` e `razao_256_192`.

## 4. `train_bertimbau_v4.py`

### 4.1 CLI completa

```
python models/v4/train_bertimbau_v4.py \
  [--prepare --sanitized S --labels L --provenance P --out-dir DIR] \
  --data v4_pool.parquet \
  --splits v4_splits.parquet \
  --split-col bal_iid \
  --eval-col bal_iid --eval-value test \
  --out DIR \
  [--drive-out DIR] \
  [--run-id NOME] \
  --model neuralmind/bert-base-portuguese-cased \
  --max-length 192 \
  --batch-size 32 --eval-batch-size 128 --grad-accum 1 \
  --epochs 3 --patience 1 --min-delta 0.005 --best-metric worst_group \
  --lr 2e-5 --warmup-frac 0.10 --weight-decay 0.01 --clip 1.0 \
  --freeze-layers 6 --seed 42 \
  --dfr-weights off --weight-clip 25.0 \
  [--class-weights] [--mask-entities] \
  --drop-tiers "" \
  --amp auto --device auto --workers 2 \
  --calib-col val_calib --threshold-primary 0.5 \
  --save-every-epoch both --predict-all \
  [--token-stats] [--resume] [--force-resume] \
  [--smoke --smoke-n 1024] \
  [--ablate none|fullpool|dfr|maxlen256|freeze4|freeze0|classweights|llmoff] \
  [--probe-max-length 256] [--check-vendor]
```

Defaults exatamente como acima. `--split-col` só aceita colunas presentes no
`v4_splits.parquet`. `--run-id` default:
`{split_col}_ml{max_length}_f{freeze}_{on|off}_seed{seed}` (com sufixo
`_ablate-{nome}` quando houver preset). `--out` default
`models/v4/artifacts/<run_id>`.

Regras de `--eval-col`: para `bal_iid` e `full_iid`, avaliar em
`--eval-col bal_iid --eval-value test`; para `bal_ood_wa`, avaliar no próprio
teste (`--eval-col bal_ood_wa`). O script recusa `--split-col full_iid` com
`--eval-col full_iid` (não há teste no full).

`--prepare` delega para `models.v4.prepare_v4` (importável só no repo local);
no Colab, se flag usada, aborta com instrução para gerar o parquet localmente.

Presets `--ablate` (aplicados antes dos argumentos explícitos, que vencem):

| preset | efeito |
|---|---|
| `none` | nada |
| `fullpool` | `--split-col full_iid --dfr-weights cell --epochs 2 --eval-col bal_iid` |
| `dfr` | `--dfr-weights cell --weight-clip 25` |
| `maxlen256` | `--max-length 256` |
| `freeze4` | `--freeze-layers 4` |
| `freeze0` | `--freeze-layers 0 --batch-size 16` (se OOM, seguir escada 5.5 do PLANO) |
| `classweights` | `--class-weights` |
| `llmoff` | `--split-col full_iid --dfr-weights cell --drop-tiers llm_local,corroborated` |

### 4.2 Comportamento por etapa

**E0. Ambiente.** Resolver device (`auto`: cuda se disponível e != CPU). Logar
torch/transformers/cuda/capability. `amp auto`: fp16+GradScaler se
`capability[0] >= 8` usar bf16 sem scaler; senão fp16 com scaler; CPU ou
`--amp off` desliga. Nunca usar `torch.cuda.is_bf16_supported()` como critério
(emulação). Seed completo e cudnn determinístico (PLANO 7).

**E1. Token stats (se `--token-stats` ou `--smoke`).** Carrega o tokenizer,
mede a distribuição de tokens do parquet e a simulação de custo para 128/192/256;
grava `token_stats.json` no `--out`; imprime a projeção de tempo.

**E2. Seleção de dados.** Lê os parquets; valida que todo `rid` do split existe
no pool; seleciona `--split-col`: `train/val_sel/val_calib/test`, ignorando
`unused`; aplica `--drop-tiers` (só no treino) e remove `has_ufffd` (defensivo).
Para avaliação, monta `df_eval` com `--eval-col/--eval-value` e, se for outra
coluna, mantém o teste canônico intocado. Renomeia `is_ptpt_rule` para `is_ptpt`
na cópia de avaliação (compatibilidade com o `report` embutido). Imprime a
tabela de contagens/fake% e falha se houver interseção treino×teste.

**E3. Tokenização e batching.** `TextDS` (tokens sem padding) para treino,
val_sel, val_calib, eval; `DataCollatorWithPadding(tokenizer)`. Sampler de
treino = `LengthGroupedBatchSampler` reproduzindo `length_grouped_batches`
(megabatches de `batch_size*50`, ordenação por comprimento, embaralhamento
entre épocas com gerador semeado em `seed+epoch`), `DataLoader` com
`pin_memory=True`, `num_workers=2`, `persistent_workers=True` se workers>0.
Inferência: sort estável por comprimento, `eval_batch_size`, sem autograd,
`logits.float()`.

**E4. Modelo/otimizador.** `AutoModelForSequenceClassification(num_labels=2)`;
freeze embeddings + N camadas; AdamW em dois grupos (decay 0,01 nos pesos de
matmul; 0,0 em bias/LayerNorm); linear schedule com warmup de 10% dos steps;
steps = `ceil(n_train/efetivo) * epochs`, com `efetivo = batch_size*grad_accum`.

**E5. Loop de treino (por época).**
1. loss CE (ou ponderada: `--class-weights` ou `--dfr-weights cell`);
   - `cell`: `w = group_balanced_weights(train)` (célula grupo×rótulo, média 1),
     `w = minimum(w, weight_clip)`, renormaliza à média 1; logar min/max/prior
     efetivo antes e depois do clip.
2. forward sob autocast; `scaler.scale(loss).backward()`; a cada `grad_accum`:
   `scaler.unscale_(opt)`, `clip_grad_norm_(params treináveis, clip)`,
   `scaler.step(opt)`, `scaler.update()`, `sched.step()`,
   `opt.zero_grad(set_to_none=True)`.
3. Métricas do loop: loss média, tokens/s pagos, amostras/s, lr, escala do
   scaler, `torch.cuda.max_memory_allocated()`.
4. Fim da época: inferência em `val_sel` (logits fp32) -> `p = sigmoid(z)`;
   calcular `macro_f1`, `worst_group_f1` (fallback `macro_f1` se nan), `ece`,
   `val_loss` CE. Selecionar melhor por `--best-metric` (default
   `worst_group`); critério de parada: sem melhora > `--min-delta` em
   `val_sel.worst_group` por `--patience` épocas.
5. Salvar `last/` (pesos, tokenizer, `optimizer.pt`, `scheduler.pt`,
   `scaler.pt`, `rng.pt`, `trainer_state.json`); se melhor, salvar `best/`
   (pesos + tokenizer). Reescrever `history.json` e `run_config.json`. Copiar
   `last/`, `best/` e `history.json` para `--drive-out` quando fornecido
   (tmp + `os.replace`).

**E6. Avaliação final.**
1. Carregar `best/`.
2. Inferência em `val_sel`, `val_calib`, `eval` (teste), e no pool inteiro se
   `--predict-all` (default True; o smoke força False).
3. Platt em `val_calib`: máscara `is_balanced_group` quando existir e tiver
   >= 50 linhas e as duas classes; senão `val_calib` inteira com aviso. Logar
   ECE antes/depois. Limiar ótimo no grid 0,05–0,95 por macro-F1 em
   `val_calib` calibrada.
4. `report` embutido no teste com limiar 0,5 (primário) e novamente com o
   limiar ótimo (secundário); `per_group` por `group`, `channel`, `publisher`,
   `era`, `label_tier`; `reliability_table` no teste calibrado.
5. Diagnóstico dos grupos constantes (se `--predict-all`): métricas por grupo
   apenas descritivas, marcadas `in_training_pool` true/false.
6. Escrever `metrics.json`, `predictions.csv`, `per_group.csv`,
   `reliability.csv`, `calibration.json` no `--out` e copiar para
   `--drive-out`.

**E7. Retomada (`--resume`).** Carregar `last/` + `trainer_state.json`;
validar contra o `run_config.json` atual (`data_sha256`, `splits_sha256`,
`split_col`, `max_length`, `freeze_layers`, `model`, `seed`, `dfr_weights`,
`class_weights`); se divergente, abortar salvo `--force-resume`. Restaurar
otimizador, scheduler, scaler, RNG (python/numpy/torch/cuda) e `epoch`;
continuar do próximo passo/época. Se `last/` não existir, começar do zero com
aviso.

**E8. Smoke (`--smoke`).** `--smoke-n` controla o tamanho; usa subconjunto
estratificado por rótulo de `train/val_sel/val_calib/test`; 1 época;
`max_length = min(128, max_length)`; `batch_size = min(8, batch_size)`;
`eval_batch_size = 32`; `predict-all` desligado; não copia para Drive.
Imprime: throughput medido (amostras/s e tokens/s), VRAM de pico, tempo de
tokenização, validação e avaliação, e uma projeção para o run default com a
hipótese `speedup T4/CPU` na faixa [10, 30]x, **marcada como hipótese**. Ao
final exige que os artefatos do contrato existam.

**E9. Erros/OOM.** Escada do PLANO 5.5; mensagens claras; `--probe-max-length`
roda um batch sintético em CUDA para medir VRAM de pico sem treinar; aborta se
pico > 13,5 GB.

## 5. Fluxo operacional (comandos exatos)

### 5.1 Local (preparação + smoke)

```powershell
python models/v4/prepare_v4.py `
  --sanitized FakenewsBR_sanitized_v4.csv `
  --labels FakenewsBR_v4_labels.csv `
  --provenance FakenewsBR_v4_provenance.csv `
  --out-dir models/v4/processed --seed 42 --token-stats

python models/v4/train_bertimbau_v4.py `
  --data models/v4/processed/v4_pool.parquet `
  --splits models/v4/processed/v4_splits.parquet `
  --split-col bal_iid --out models/v4/artifacts/smoke_cpu `
  --smoke --smoke-n 1024 --device cpu --amp off

python models/v4/train_bertimbau_v4.py --check-vendor
```

### 5.2 Upload para o Drive

Copiar para `/content/drive/MyDrive/FakenewsBR/v4/data/`:
`v4_pool.parquet`, `v4_splits.parquet`, `prepare_stats.json`,
`train_bertimbau_v4.py`.

### 5.3 Colab, célula de setup

```python
!nvidia-smi
import torch
print(torch.__version__, torch.cuda.get_device_name(0),
      torch.cuda.get_device_capability(0))
!pip -q install "transformers==5.17.0" scikit-learn tqdm pyarrow

from google.colab import drive
drive.mount("/content/drive")

D = "/content/drive/MyDrive/FakenewsBR/v4"
!mkdir -p /content/fakenewsbr
!cp $D/data/v4_pool.parquet $D/data/v4_splits.parquet $D/data/prepare_stats.json \
    $D/data/train_bertimbau_v4.py /content/fakenewsbr/
```

### 5.4 Smoke na T4 (obrigatório antes de qualquer run cheio)

```python
!python /content/fakenewsbr/train_bertimbau_v4.py \
  --data /content/fakenewsbr/v4_pool.parquet \
  --splits /content/fakenewsbr/v4_splits.parquet \
  --split-col bal_iid --out /content/smoke --smoke --smoke-n 512 \
  --amp auto --device auto --workers 2
```

### 5.5 Run default (R0)

```python
!python /content/fakenewsbr/train_bertimbau_v4.py \
  --data /content/fakenewsbr/v4_pool.parquet \
  --splits /content/fakenewsbr/v4_splits.parquet \
  --split-col bal_iid --eval-col bal_iid --eval-value test \
  --out /content/runs/bal_iid_ml192_f6_off_seed42 \
  --drive-out $D/runs/bal_iid_ml192_f6_off_seed42 \
  --max-length 192 --batch-size 32 --epochs 3 --patience 1 \
  --lr 2e-5 --freeze-layers 6 --seed 42 --amp auto --workers 2
```

Retomada: repetir o comando com `--resume` (ou `--force-resume` se o config
mudou de propósito).

### 5.6 OOD WhatsApp (R1)

```python
!python /content/fakenewsbr/train_bertimbau_v4.py \
  --data /content/fakenewsbr/v4_pool.parquet \
  --splits /content/fakenewsbr/v4_splits.parquet \
  --split-col bal_ood_wa --eval-col bal_ood_wa --eval-value test \
  --out /content/runs/bal_ood_wa_ml192_f6_off_seed42 \
  --drive-out $D/runs/bal_ood_wa_ml192_f6_off_seed42 \
  --max-length 192 --batch-size 32 --epochs 3 --patience 1 \
  --lr 2e-5 --freeze-layers 6 --seed 42 --amp auto --workers 2
```

### 5.7 Ablações

```python
# A1: pool completo + pesos DFR (2 épocas)
--ablate fullpool --out /content/runs/full_iid_dfr --drive-out $D/runs/full_iid_dfr

# A2: max_length 256
--ablate maxlen256 --out /content/runs/bal_iid_ml256 --drive-out $D/runs/bal_iid_ml256

# A3: DFR nos informativos
--ablate dfr --out /content/runs/bal_iid_dfr --drive-out $D/runs/bal_iid_dfr
```

### 5.8 Download dos artefatos

Baixar `runs/<run_id>/{best,metrics.json,calibration.json,predictions.csv,
per_group.csv,reliability.csv,history.json,run_config.json}` (o `last/` e o
otimizador são descartáveis fora do Colab). Zipar no fim da sessão.

## 6. Contrato de saída

`<out>/` (e espelho em `<drive-out>/`):

```
best/   model.safetensors, config.json, tokenizer.json, tokenizer_config.json,
        special_tokens_map.json, calibration.json
last/   best/* + optimizer.pt, scheduler.pt, scaler.pt, rng.pt, trainer_state.json
history.json
run_config.json
metrics.json
predictions.csv
per_group.csv
reliability.csv
token_stats.json        (quando pedido)
```

### 6.1 `metrics.json`

```json
{
  "run_id": "bal_iid_ml192_f6_off_seed42",
  "created_utc": "2026-09-12T18:00:00Z",
  "config": { "...": "args efetivos + ablate + versões + sha256 do script" },
  "data": {"data_sha256": "...", "splits_sha256": "...", "split_col": "bal_iid",
           "n_train": 25827, "n_val_sel": 3689, "n_val_calib": 1845,
           "n_test": 5535, "train_fake_pct": 51.97},
  "history": [{"epoch": 1, "train_loss": 0.0, "val_loss": 0.0,
               "val_macro_f1": 0.0, "val_f1_fake": 0.0,
               "val_worst_group": 0.0, "val_ece": 0.0,
               "tokens_per_s": 0.0, "samples_per_s": 0.0, "lr_last": 0.0,
               "scaler_scale": 0.0, "vram_peak_gb": 0.0, "seconds": 0.0,
               "best_so_far": true}],
  "best_epoch": 2,
  "calibration": {"platt_a": 1.0, "platt_b": 0.0, "calib_n": 1845,
                   "calib_prior_fake": 0.52, "ece_before": 0.0,
                   "ece_after": 0.0, "threshold": 0.5,
                   "threshold_val_opt": 0.5},
  "test": { "...": "core_metrics + worst_group_macro_f1 + worst_group_balanced +
            acc_balanced + macro_f1_balanced + ece_balanced + hard_acc +
            pure_acc + ptpt_macro_f1, todos com n quando aplicável" },
  "test_at_val_threshold": { "...": "mesmo bloco com threshold_val_opt" },
  "ood": { "...": "mesmo bloco no canal retido, ou null" },
  "cuts": {"channel": {}, "publisher": {}, "era": {}, "label_tier": {},
            "lang_variant_rule": {}, "lang_variant_prov": {}},
  "constant_groups_diagnostic": {"...": "por grupo, descritivo"},
  "timing": {"tokenize_s": 0.0, "train_s": 0.0, "eval_s": 0.0, "total_s": 0.0},
  "hardware": {"device": "Tesla T4", "capability": [7, 5],
                "torch": "...", "transformers": "...", "cuda": "..."}
}
```

### 6.2 `predictions.csv`

Colunas: `rid, split, y_true, logit0, logit1, z, p_raw, p_cal, pred_05,
pred_val_thr, group, channel, publisher, era, lang_variant, is_ptpt_rule,
rating_class, label_tier, has_ufffd, mojibake_flag, in_training_pool`.
`split` identifica `val_sel|val_calib|test|ood|pool_diag`. Com `--predict-all`,
cobre o pool inteiro; senão só os conjuntos de avaliação.

### 6.3 `calibration.json` (compatível com o consumidor atual)

Chaves legadas preservadas: `platt_a`, `platt_b`, `max_length`, `model`,
`mask_entities`. Adicionadas: `threshold`, `threshold_val_opt`, `calib_on`,
`calib_n`, `calib_prior_fake`, `freeze_layers`, `seed`, `split_col`,
`dfr_weights`, `weight_clip`, `class_weights`, `script_sha256`, `created_utc`.
Escrever o mesmo arquivo em `best/calibration.json` e na raiz de `<out>/`.

### 6.4 `per_group.csv` e `reliability.csv`

`per_group.csv`: concatenação do `per_group` embutido para cada recorte, com
coluna extra `cut` = `group|channel|publisher|era|label_tier`. `reliability.csv`
= `reliability_table(p_cal, y)` do teste.

### 6.5 `token_stats.json`

Distribuições e custo de batching (E1), com a mesma estrutura da seção 3 do
PLANO, incluindo `cap_tokens_pagos_por_epoca` e `razao_256_192`.

## 7. Notebook `colab_bertimbau_v4.ipynb`

Células, nesta ordem (markdown curto + código):

1. **Ambiente**: `nvidia-smi`; assert capability `[7,5]` (T4) com aviso se
   outra GPU; imprimir torch/transformers.
2. **Instalação**: `pip -q install "transformers==5.17.0" scikit-learn tqdm
   pyarrow` (nunca reinstalar torch).
3. **Drive + copiar** os 4 arquivos de `data/` para `/content/fakenewsbr/`;
   conferir `sha256` do parquet contra `prepare_stats.json` e abortar se
   divergir.
4. **Smoke T4** (comando 5.4); imprimir tokens/s e VRAM de pico; recalcular o
   ETA dos runs com o valor medido e imprimir a tabela.
5. **Run default R0** (comando 5.5), com célula anterior checando se existe
   `last/trainer_state.json` no Drive e sugerindo `--resume`.
6. **OOD R1** (comando 5.6).
7. **Ablações** A1/A2/A3 comentadas, executáveis uma a uma (cada uma idempotente
   com `--resume`).
8. **Coleta**: montar `resumo` (pandas) lendo os `metrics.json` de
   `$D/runs/*/`; imprimir `acc, macro_f1, worst_group_macro_f1, ece, brier,
   ood.macro_f1` lado a lado; copiar tudo de `/content/runs/` para o Drive.
9. **Download**: zip de `best/` + JSON/CSVs de cada run (sem `last/`).

Idempotência: cada célula de treino pode ser reexecutada; o script detecta
`last/` e retoma. Nenhuma célula depende de estado só em RAM (tudo no `/content`
e no Drive).

## 8. Critérios de aceitação

### 8.1 `prepare_v4.py`

- [ ] Roda na raiz do repo sem rede e produce os 3 arquivos (+token stats).
- [ ] `prepare_stats.json` acusa 85.212 / 36.896 / fake 60.991 / true 24.221 e
      0 interseção holdout×teste.
- [ ] Contagens de split iguais às da seção 2.1 do PLANO (tolerância 0: os
      splits são persistidos; se a contagem divergir, ajustar o código ou
      documentar a mudança no stats).
- [ ] `v4_pool.parquet` <= 25 MB.

### 8.2 `--check-vendor` (local)

- [ ] Sem diferença contra `models/evaluate.py` nos casos sintéticos.

### 8.3 Smoke CPU (`--smoke --smoke-n 1024 --device cpu --amp off`)

- [ ] Termina em < 10 min e sai com código 0.
- [ ] Imprime throughput (amostras/s e tokens/s), tokens de tokenização,
      VRAM (0 na CPU) e projeção T4 explicitamente marcada como hipótese.
- [ ] Gera `best/`, `last/`, `history.json`, `metrics.json`,
      `calibration.json`, `predictions.csv`, `per_group.csv` no diretório de
      teste.
- [ ] `token_stats.json` deve bater com a tabela do PLANO (tolerância 2%):
      informativos média 119,6 tokens, p95 453, >192 = 18,82%,
      razão de custo 256/192 = 1,145.

### 8.4 Smoke T4

- [ ] Termina em < 5 min; VRAM de pico <= 12 GB em 32x192; sem NaN.
- [ ] `--probe-max-length 256` reporta pico e decide se A2 pode rodar 32x256.
- [ ] `--resume` após o smoke continua de onde parou (testar 1 época + resume).

### 8.5 End-to-end

- [ ] R0 completa e gera artefatos com todos os campos de `metrics.json`; o
      `history` tem as épocas realizadas e `best_epoch` coerente.
- [ ] R1 (OOD) reporta `ood` não nulo e o teste canônico permanece intocado no
      treino (`data.n_test == 6381` para R1).
- [ ] `--ablate fullpool` respeita o holdout: nenhum `rid` do teste canônico no
      treino (verificação impressa no log).
- [ ] Reexecutar um run com `--resume` não duplica épocas no `history`.

## 9. Hipóteses que o builder deve medir e substituir

| # | hipótese | onde medir | substitui em |
|---:|---|---|---|
| H1 | throughput T4 2.500–6.000 tokens/s (batch 32, fp16) | smoke T4 | tabela da seção 9 do PLANO e ETA impresso |
| H2 | inferência 3–5x o treino | smoke T4 + avaliação final | orçamento de avaliação |
| H3 | VRAM de pico de 32x192 e 32x256 | smoke/probe T4 | decisão de A2/256 |
| H4 | tokens pagos/época do OOD (~2,1 M) e do full (~3,7 M) | log de tokens/s de cada run | tabela da seção 9 |
| H5 | platô de 2–3 épocas | `history.json` | `--epochs`/`--patience` de runs futuros |
| H6 | custo real do `--predict-all` (85.212 linhas) | avaliação final | orçamento de avaliação |
