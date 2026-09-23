# INFRA_NOTES — v4 (Colab T4 / BERTimbau)

Mapa do que já existe no repo para treino/avaliação do BERTimbau, para o script
Colab-ready de `models/v4/` reaproveitar decisões comprovadas e não reintroduzir
bugs. Leitura feita em 2026-09-12. Nenhum arquivo existente foi modificado.

Hardware/stack local de referência (log `5_finetune.log` linha 1 e
`models/artifacts/bertimbau_finetuned/config.json:31`):

```
device=cpu threads=8 torch=2.14.0+cpu          # 5_finetune.log
"transformers_version": "5.17.0"               # bertimbau_finetuned/config.json:31
```

Run de referência: 2 épocas CPU = 226,3 min (`pipeline_state.json:23-28`).

---

## 1. Assinaturas exatas de `models/evaluate.py`

O módulo só importa `numpy`, `pandas` e `sklearn` — é 100% copiável para Colab.

```python
# models/evaluate.py:121
def core_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
```

Devolve **8 chaves**: `n`, `acc`, `macro_f1`, `f1_fake`, `brier`, `ece`,
`roc_auc`, `pr_auc`. Se `len(np.unique(y)) <= 1` (grupo degenerado):
`macro_f1`, `brier`, `roc_auc`, `pr_auc` viram `float("nan")` (`evaluate.py:128-138`).

```python
# models/evaluate.py:54
def fit_platt(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:

# models/evaluate.py:76
def apply_platt(logits: np.ndarray, a: float, b: float) -> np.ndarray:
# p = sigmoid(clip(a*z + b, -700, 700)), z = logit1 - logit0 (evaluate.py:81-84)

# models/evaluate.py:87
def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:

# models/evaluate.py:149
def per_group(df: pd.DataFrame, y: np.ndarray, p: np.ndarray,
              col: str = "group", min_n: int = 30) -> pd.DataFrame:

# models/evaluate.py:169
def worst_group_f1(df: pd.DataFrame, y: np.ndarray, p: np.ndarray,
                   col: str = "group", min_n: int = 30,
                   only_groups: list | None = None) -> float:

# models/evaluate.py:192
def report(df: pd.DataFrame, y: np.ndarray, p: np.ndarray, title: str,
           threshold: float = 0.5) -> dict:
```

Também existem (não usadas pelo `encoder.py`, mas úteis): `fit_temperature`
(`:19`), `apply_temperature` (`:46`), `reliability_table` (`:104`) e o privado
`_decision` (`:81`). `fit_platt` usa `LogisticRegression(C=1e10, solver="lbfgs",
max_iter=5000)` sobre `_decision(logits)`.

**`worst_group_f1` depende da coluna `df["group"]`** (default `col="group"`).
Essa coluna é criada em `data.load`: `df["group"] = df["dataset_name"].astype(str)`
(`models/data.py:135`). O `encoder.py:227` chama `E.worst_group_f1(split.val, yva, pva)`
sem `only_groups`, então vale para todos os grupos. Ainda:
- `only_groups` filtra por lista (usado no `score.py:70` com `D.BALANCED_GROUPS`);
- só entram no mínimo grupos "confiáveis": `n >= MIN_GROUP_N (100)` **e**
  `minoria_n >= MIN_MINORITY_N (20)` (`evaluate.py:145-146, 160, 186`);
- `per_group` corta `n < min_n (30)` e devolve colunas
  `[col, n, minoria_n, fake_%, acc, macro_f1, f1_fake, roc_auc, pr_auc, ece, confiavel]`
  ordenado por `macro_f1` crescente (`evaluate.py:164-166`);
- grupo sem massa confiável → retorna `nan` (o `encoder.py:230` cai para `macro_f1`).

**O que `report` imprime** (`evaluate.py:192-257`): cabeçalho `=`×70; linha
`GLOBAL n= acc= macro-F1= F1(fake)=` seguida de `PR-AUC ROC-AUC Brier ECE`;
tabela `POR GRUPO` (pior→melhor) + aviso de grupos não confiáveis;
`>>> PIOR GRUPO macro-F1`; se houver `is_balanced_group`, `>>> PIOR GRUPO (so
grupos balanceados)` + `acc/macro-F1/ECE` dos balanceados; se houver
`rating_class`, `CASOS LIMITROFES` e `FALSO PURO`; se houver `is_ptpt`, linha
`DIALETO PT-PT ... | resto ...`.

**Retorno de `report`**: o dict `glob` = `core_metrics` + chaves condicionais
`worst_group_macro_f1`, `worst_group_balanced`, `acc_balanced`,
`macro_f1_balanced`, `ece_balanced`, `hard_acc`, `pure_acc`, `ptpt_macro_f1`
(confirma os JSONs `models/artifacts/score_frozen.json` / `score_finetuned.json`).

---

## 2. `data.load()` × `labels_csv` e import no Colab

```python
# models/data.py:116
def load(csv: str = DEFAULT_CSV, include_length: bool = False,
         provenance_csv: str | None = None,
         labels_csv: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv, low_memory=False)
    if labels_csv:
        lab = pd.read_csv(labels_csv, low_memory=False)
        cols = [c for c in ("rid", "train_label", "label_tier", "auto_label",
                            "confidence", "method") if c in lab.columns]
        df = df.merge(lab[cols], on="rid", how="left")
        df["label_orig"] = df["label"]
        df["label"] = df["train_label"]
    df = df[df["label"].isin(["fake", "true"])].copy()
```

Comportamento exato (`data.py:119-175`):
1. merge `how="left"` por `rid`; adiciona `train_label`, `label_tier`,
   `auto_label`, `confidence`, `method` ao DataFrame.
2. **substitui** `label` por `train_label`; o rótulo original fica em `label_orig`.
3. filtra `label in {fake,true}` → linhas sem `train_label` (provenance) e
   rótulos extras são **descartados em silêncio**. Não há auditoria de conflito
   (existe `conflicting_labels_audit.csv` na raiz, não é usado pelo loader).
4. `target = (label == "fake")`, `group = dataset_name`, `channel = channel_of(group)`,
   `is_balanced_group` (lista fixa + prefixos `FC_/EXT_/NEWS_` informativos),
   `is_ptpt` (regex em `url_review`+`factcheck_url`), `rating_class` e as taxas
   estilísticas. `provenance_csv` é mergeado **depois** disso, sem recalcular
   `is_ptpt` (v4 tem `lang_variant` = pt-BR/pt-PT que poderia ser usado em vez do
   regex; hoje não é).

Verificado nos arquivos v4 (execução real do `D.load`):
- `FakenewsBR_sanitized_v4.csv`: 291.521 linhas, header idêntico ao v1
  (`rid,dataset_name,...,text,text_clean,text_no_url,...`);
- `FakenewsBR_v4_labels.csv`: 291.521 `rid` únicos; `train_label` =
  `fake 60.991`, `true 24.221`, `NaN 206.309`; `label_tier` dominante
  `provenance` (206.309), depois `v1/checker/checker_match/llm_local/corroborated`;
- resultado de `load(csv=sanitized_v4, labels_csv=v4_labels)`: **85.212 linhas**
  (fake = 71,58%), 36.896 em `is_balanced_group`; grupos novos presentes
  (`FC_*`, `EXT_*`, `NEWS_*`);
- `FakenewsBR_v4_provenance.csv`: 252.055 `rid` únicos, `lang_variant` pt-BR 210.689 / pt-PT 41.366;
- `DEFAULT_CSV` é path absoluto Windows (`data.py:23`) → no Colab **sempre**
  passar `--csv` e `--labels-csv` explícitos;
- 4 linhas de `text_no_url` contêm `U+FFFD` real (ex.: rid 11447/16958:
  `70\ufffdos votos`); ruído pré-existente, não bloqueia.

Para o script v4 rodar no Colab sem instalar nada:
- o repo tem `models/__init__.py`, então basta pôr a raiz no path:
  `sys.path.insert(0, "/content/FakenewsBR")` + `from models import data as D`.
  **Atenção**: rodar `python models/v4/train.py` coloca `models/v4` (não a raiz)
  no `sys.path`; use `Path(__file__).resolve().parents[2]` para achar a raiz ou
  execute como módulo (`python -m models.v4.train`) a partir da raiz.
- alternativa mais robusta para Colab: **copiar** `evaluate.py` inteiro (sem
  imports locais) e um loader enxuto de `data.py` para o próprio script/célula.
  `data.py` só depende de numpy/pandas/sklearn; pode ser importado igual.

---

## 3. Mudanças necessárias para T4 (CPU → CUDA/AMP)

Tudo abaixo referencia `models/encoder.py`. Nada disso existe hoje: o device é
fixo em CPU (`:137`) e não há AMP/DataLoader/pin_memory.

1. **Device** — trocar `device = torch.device("cpu")` (`:137`) por
   `torch.device("cuda" if torch.cuda.is_available() else "cpu")`; logar
   `torch.cuda.get_device_name(0)` e `torch.cuda.get_device_capability(0)`
   (T4 = `(7, 5)`).
2. **Dtype do AMP** — T4 não tem bf16: `AMP_DTYPE = torch.bfloat16` só se
   `torch.cuda.get_device_capability(0)[0] >= 8`, senão `torch.float16`.
   Não confiar em `torch.cuda.is_bf16_supported()` sozinho no torch ≥ 2.14
   (default `including_emulation=True` pode devolver True em sm_75).
3. **Autocast + GradScaler** — trocar o forward/backward cru (`:210-219`) por:
   ```python
   scaler = torch.amp.GradScaler("cuda", enabled=(AMP_DTYPE is torch.float16))
   with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
       loss = loss_fn(model(**batch).logits, labels)
   scaler.scale(loss).backward()
   scaler.unscale_(opt)                     # ANTES do clip
   torch.nn.utils.clip_grad_norm_(params, 1.0)
   scaler.step(opt); scaler.update()
   ```
   bf16 dispensa scaler (`enabled=False`). Ordem `unscale_ → clip → step` é
   obrigatória; sem ela o clip não vê os gradientes reais.
4. **Inferência** — `infer()` (`:96-109`) deve rodar sob `torch.amp.autocast`
   (mesmo dtype) e devolver logits em **fp32**: `model(**batch).logits.float().cpu().numpy()`,
   senão `fit_platt`/ECE recebem `float16` (e podem estourar). `@torch.no_grad()`
   pode virar `@torch.inference_mode()`.
5. **Batch/VRAM** — manter 16 é seguro; recomendado 32 em seq 192 com fp16.
   Se preferir manter 16, usar gradient accumulation (efetivo 32):
   `eff = batch_size * grad_accum`; só chamar `scaler.step/update`,
   `opt.zero_grad(set_to_none=True)` e `sched.step()` quando
   `(step % grad_accum == 0) or ultimo_batch`; recalcular
   `steps = ceil(len(ds_tr)/eff) * epochs` (`:198`) — o warmup de 10% depende disso.
6. **DataLoaders com `pin_memory`** — o loop atual monta o batch na mão
   (`:207-211`), então `pin_memory` não se aplica. Refatorar para
   `DataLoader(ds_tr, batch_sampler=LengthGroupedSampler(lengths, batch_size, seed),
   collate_fn=collate, pin_memory=True, num_workers=2, persistent_workers=True,
   prefetch_factor=2)` e usar `.to(device, non_blocking=True)` para **todas** as
   chaves (inclusive `labels`). O `length_grouped_batches` (`:83-93`) vira o
   `BatchSampler` — a lógica de megabatch/sort/shuffle deve ser preservada.
   Colab tem ~2 vCPU: `num_workers=2` basta (o default 8 de `--threads` não se
   aplica; `torch.set_num_threads` em `:134` deve usar `min(4, os.cpu_count())`
   ou ser ignorado no caminho GPU).
7. **Seed/determinismo** — hoje só `torch.manual_seed`/`np.random.seed` e o
   `Generator` CPU (`:135-136, 197`). Adicionar `random.seed`,
   `torch.cuda.manual_seed_all`; opcionalmente
   `torch.backends.cudnn.deterministic=True` / `benchmark=False` (padrão do
   framework externo, `external/.../fine-tuning/utils.py:63-73`). O
   `length_grouped_batches` usa gerador CPU, então continua determinístico.
8. **Checkpoint de melhor época** — `best_state = {k: v.detach().clone() ...}`
   (`:233`) guarda uma cópia na **GPU** (dobra VRAM do modelo). Mudar para
   `v.detach().to("cpu").clone()` e devolver ao device com `model.load_state_dict`
   (fine, o `state_dict` carrega em GPU).
9. **`pin_memory` + AMP no eval** — o `DataCollatorWithPadding` continua igual;
   batch de inferência pode subir para 64/128 em seq 192 (sem autograd).
10. **Opcional p/ OOM** — `model.gradient_checkpointing_enable()` (mais lento;
    útil só se subir batch/seq) ou manter `--freeze-layers 6` (economiza
    ~660 MB de grads+AdamW e poda ativações das camadas congeladas).
11. **Não fazer** — reinstalar torch no Colab (quebra CUDA); usar
    `torch.cuda.amp.*` (deprecado); `torch.compile` (custo de compilação não
    compensa em sessão curta).
12. **Custo esperado** — o CPU gastou 226 min (2 épocas, batch 16, 6 congeladas);
    em T4 fp16 com batch 32 esperar ordem de ~10-20 min/época (estimativa, medir).

---

## 4. Decisões comprovadas a preservar (e por quê)

| Decisão | Por que existe | Onde vive |
|---|---|---|
| Padding dinâmico por batch + agrupamento por comprimento | `padding=True` na lista inteira contra `max_length=256` desperdiçava 3,20x; mediana real 33 tokens (log: `media=71 p50=33 p95=192`, `desperdiçaria 2.72x`); era a causa de 545 min/época | `encoder.py:6-9, 64-80, 83-93, 172, 207` |
| `TEXT_COL = "text_no_url"` (não `text_clean`) | O encoder é *cased*; `text_clean` é minusculizado e sem acento, contradizendo o pré-treino | `encoder.py:10-11, 147-149`; `data.py:24` |
| Sem class weights por padrão | Objetivo é score **calibrado**; ponderar a loss distorce a probabilidade; desbalanceamento é tratado no limiar/métricas | `encoder.py:17-19, 180-187` |
| Melhor checkpoint por pior-grupo F1 da val | Acurácia média esconde o confundimento de proveniência; fallback para `macro_f1` se `nan` | `encoder.py:202, 223-234` |
| Calibração Platt **na validação**, teste intocado | Ajustar no teste vaza; val só é usada para escolher época e calibrar | `encoder.py:239-249` |
| Clip de gradiente 1.0 | Estabiliza fine-tuning de transformer | `encoder.py:215-216` |
| Weight decay 0.01 sem bias/LayerNorm | Bias/LN não devem sofrer decay; só pesos de matmul | `encoder.py:189-195` |
| Warmup 10% + schedule linear | Evita divergência nos primeiros passos com lr 2e-5 | `encoder.py:198-200` |
| Congelar embeddings + 6 camadas | 43,1M treináveis; corte medido de ~148 → ~103 min/época no CPU; também economiza VRAM no T4 | `encoder.py:157-166` |
| Split estratificado por (grupo × rótulo) | Estratificar só por rótulo faz a composição de origem variar entre splits e invalida a métrica por grupo | `data.py:218-233` |
| `infer` ordenado por comprimento | Menos padding e menos variação de shape (era caro no DirectML); inofensivo/útil na GPU | `encoder.py:96-109` |
| `mask_entities` atrás de flag | Pergunta em aberto do plano; deve ser medida, não assumida | `encoder.py:54-61, 126, 148-151` |

No T4 muda só **como** roda (device/AMP/DataLoader), não **o que** é preservado.

---

## 5. Requisitos/armadilhas do Colab T4

**Versões (2026)**
- Runtime Colab 2026.07: Python 3.12.13, numpy 2.0.2, **PyTorch 2.11.0**
  (faq de runtime do Colab). O default atual pode ser mais novo; checar
  `!python -V`, `!pip show torch` na primeira célula.
- PyPI: `transformers` estável em **5.16.1**; este repo gerou os artefatos com
  **5.17.0** (`config.json:31`) e torch 2.14.0+cpu. Fixar
  `!pip install -q "transformers==5.17.0" "scikit-learn" tqdm` se a versão
  existir; se cair em 5.16.x, `from_pretrained` só emite aviso de versão.
  **Não** reinstalar torch.
- `external/automated-fact-checking-in-pt-br/requirements.txt` **não pina**
  torch/transformers (só `tqdm, scikit-learn, requests, jupyter, pandas, numpy,
  matplotlib, seaborn`) — não serve de referência de versão.
- `os.environ["TOKENIZERS_PARALLELISM"]="false"` evita warning de fork
  (padrão no `main.py:23` do framework externo); relevante com `num_workers>0`.

**AMP: API e dtype**
- Usar `torch.amp.autocast("cuda", dtype=...)` e `torch.amp.GradScaler("cuda")`;
  `torch.cuda.amp.autocast/GradScaler` estão deprecados (FutureWarning) no torch
  atual. No `encoder.py` atual não há AMP nenhum.
- T4 = Turing **sm_75**: bf16 exige compute capability ≥ 8.0. Confirmado por
  mensagem típica de frameworks: *"Bfloat16 is only supported on GPUs with
  compute capability of at least 8.0. Your Tesla T4 GPU has compute capability
  7.5."* → **fp16 + GradScaler**.
- `torch.cuda.is_bf16_supported()` no torch ≥ 2.14 tem
  `including_emulation=True` por default; usar
  `torch.cuda.get_device_capability(0)[0] >= 8` como critério real.
- Com fp16: `scaler.unscale_(opt)` **antes** de `clip_grad_norm_`; monitorar
  `scaler.get_scale()` (quedas indicam overflow).

**VRAM de 16 GB (T4 tem ~15,0 GB usáveis)** — BERT-base (110M, hidden 768,
12 camadas; 6 congeladas = 43,1M treináveis). Conta fixa: pesos fp32 ~0,44 GB +
grads+AdamW dos treináveis ~0,52 GB (full) ou ~0,52 GB dos 43,1M
(grad 0,17 + AdamW 0,34). Ativações fp16 dominam:

| config (fp16) | VRAM estimada | veredito |
|---:|---:|---|
| batch 16 × seq 192 | ~4–6 GB | confortável |
| batch 32 × seq 192 | ~7–9 GB | **recomendado** |
| batch 32 × seq 256 | ~10–13 GB | aperta, mas cabe |
| batch 64 × seq 192 | ~12–16 GB | arriscado (OOM com encoder descongelado) |
| batch 64 × seq 256 | > 16 GB | **não cabe** |
| batch 128 × seq 192 | > 16 GB | **não cabe** |

Congelar 6 camadas devolve ~0,6–1,0 GB e poda ativações antes da primeira
camada treinável. `gradient_checkpointing` é a válvula se precisar de batch
maior. Batch de inferência (sem autograd) pode ser 64–128.

**Outras armadilhas**
- CSV v4 tem 210 MB; `low_memory=False` + tokenização em RAM (27,6k treino) é
  tranquilo nos ~12,7 GB do Colab, mas não rode 2 sessões do loader em paralelo.
- `models/embed.py` default é `DmlExecutionProvider` (DirectML/Windows) —
  **não existe no Colab**; usar `CPUExecutionProvider`/`CUDAExecutionProvider`
  (onnxruntime-gpu) ou pular ONNX.
- Estado de artefatos: `models/artifacts/` é gitignored (`models/.gitignore:2`).

---

## 6. Artefatos no Colab (Drive) e ONNX

Estrutura sugerida no Drive:

```
/content/drive/MyDrive/FakenewsBR/
├── data/
│   ├── FakenewsBR_sanitized_v4.csv
│   ├── FakenewsBR_v4_labels.csv
│   └── FakenewsBR_v4_provenance.csv
├── models/v4/
│   └── bertimbau_ft/            # model.safetensors, config.json,
│       calibration.json         # tokenizer.json, tokenizer_config.json
├── onnx/bertimbau_ft.onnx       # opcional
├── results/score_v4.json + RELATORIO_DADOS.md
└── logs/<etapa>.log
```

- Salvar **uma vez no fim** (e opcionalmente por época útil): `model.save_pretrained(out)`,
  `tok.save_pretrained(out)` e `calibration.json` no mesmo diretório
  (`encoder.py:251-258`). Manter o schema atual (`platt_a`, `platt_b`,
  `max_length`, `model`, `mask_entities`) — `models/artifacts/bertimbau_finetuned/calibration.json`
  é lido por ferramentas a jusante. Se o v4 calibrar sob prior balanceado,
  gravar também `calib_prior_fake`/`calib_groups` sem quebrar o schema.
- Drive é I/O lento: treinar/gravar em `/content/` e copiar (`shutil.copy`) ou
  zipar no fim; **não** salvar `state_dict` por época no Drive (risco de
  arquivo parcial/quota). O `best_state` já é mantido em memória
  (`encoder.py:231-237`); só corrigir para CPU (item 3.8).
- `export(model_name_or_dir, out_path, opset=17)` (`models/embed.py:28-61`):
  carrega `AutoModel` (encoder sem cabeça) e exporta só `last_hidden_state` com
  eixos dinâmicos, `dynamo=False` (exportador legado; gera DeprecationWarning no
  torch ≥ 2.9). **Não precisa de GPU**: o modelo nunca é movido para CUDA e os
  tensores de exemplo são CPU (`embed.py:34, 48-49`) — no Colab deixe assim;
  se mover para CUDA, os inputs também precisam ir. Saída medida: 433,5 MB
  (`6_embed_export_ft.log`).
- Manifesto de integridade: replicar `sha256()`/`save_manifest()` de
  `external/fakenews-data/src/fakenews_br_data/utils.py:11-48` para carimbar
  dataset/modelo no Drive.

---

## 7. Bugs/limitações documentados que o v4 não pode repetir

1. **Vulkan é inference-only** — `torch.device("vulkan")` nunca teve kernels de
   backward; o ramo do `bert_finetune.py:48-53` nunca executava
   (`encoder.py:12-13`; `models/README.md:88-90`). No v4: CUDA de verdade ou CPU.
2. **Padding na lista inteira** — o bug original tokenizava tudo com
   `padding=True, max_length=256` (`bert_finetune.py:21`), 3,20x de desperdício;
   era a causa de 545 min/época (`encoder.py:6-9`). Não reintroduzir.
3. **`text_clean` com modelo cased** — original alimentava o BERTimbau *cased*
   com texto minusculizado/sem acento (`encoder.py:10-11`; `data.py:7`).
4. **Class weights degradam a calibração** — ponderar a loss distorce a
   probabilidade; a flag existe para medir, não como default (`encoder.py:17-19,
   183-184`). O framework externo usa `"class_weights": true` nas configs de
   BERTimbau (ex.: `configs/finetune_BERTimbau_cased_PTBR_averitec_LR_1e-3_2kepochs.json`);
   **não copiar essa parte**.
5. **Calibrar na validação inteira desfaz o DFR** — a val é 71,5% fake (artefato
   de compilação, não prevalência). Medido: pior-grupo 0,73 → 0,38. Calibrar sob
   prior balanceado e reportar `ece_balanced` (`score.py:109-135`;
   `RELATORIO.md:98-107`; `make_report.py:125-129`). Obs.: o `encoder.py:240-242`
   hoje calibra na val inteira — aceitável para o encoder puro, mas se o v4
   treinar cabeça DFR, seguir o `score.py`.
6. **ECE medido na base inteira penaliza uma escolha deliberada** — usar
   `ece_balanced` como número válido e `ece` total só como contraste
   (`make_report.py:126-129`).
7. **Pior-grupo precisa de massa mínima** — `MuMiN-PT` (n=34, 3 exemplos da
   minoria) produzia macro-F1 ruído e dominava a manchete;
   `MIN_GROUP_N=100`, `MIN_MINORITY_N=20` (`evaluate.py:142-146`).
8. **OOD da linha fine-tuned não é OOD honesto** — o encoder foi ajustado no
   split IID com todos os canais; `ood:portal = 0,9721` é memória, não
   generalização (`make_report.py:135-143`; `RELATORIO.md:118-131`). Para OOD
   honesto, reajustar um encoder por canal retido (na T4 isso ficou barato).
9. **Atalho de proveniência persiste no encoder puro** — 96,6% em `fakes`
   (100% falso) e 18,97% em `true` (100% verdadeiro); a cabeça DFR corrige
   (`RELATORIO.md:29-43`). Reportar os grupos degenerados, não escondê-los.
10. **Features estilísticas discriminam canal, não veracidade** — com/sem estilo:
    pior-grupo 0,7226 vs **0,7304** (IID) e 0,6229 vs **0,6266** (OOD WhatsApp);
    recomendo `--no-style` como referência (`RELATORIO.md:86-96`).
11. **Viés dialetal PT-PT é grande e medido** — macro-F1 0,6248 PT-PT vs 0,8503
    resto (22 pontos); argumento para XLM-RoBERTa
    (`RELATORIO.md:109-114`; `encoder.py:115-117`).
12. **Portabilidade** — `DEFAULT_CSV` Windows (`data.py:23`), `--threads 8`
    (`encoder.py:124`) e `device="cpu"` fixo (`encoder.py:137`) não valem no
    Colab (2 vCPU). Passar paths/flags explícitos.
13. **`labels_csv` silencioso** — merge sem checagem de duplicidade/conflito;
    rid duplicado multiplica linhas e rid ausente vira drop silencioso
    (`data.py:120-130`). Validar `rid.nunique()` antes de treinar.
14. **Export ONNX legado** — `dynamo=False` força o exportador TorchScript, com
    DeprecationWarning a partir do torch 2.9 (`6_embed_export_ft.log`);
    funciona, mas não é o caminho novo (`torch.export`).
15. **`score.py:211-213`** referencia uma coluna `temperature` que nenhum
    produtor gera (usa Platt) — chave morta no resumo; não replicar.
16. **Dados**: 4 linhas de `text_no_url` têm `U+FFFD` real (rids 11447, 16958,
    836990361129089716, 839894016325789543) — ruído pré-existente no
    sanitizado, vale filtrar/registrar no v4.

---

### TL;DR para o script Colab

- **Importar**: `evaluate.py` (copiável integral) e `data.load` com
  `csv=sanitized_v4`, `labels_csv=v4_labels`, `provenance_csv=v4_provenance`.
- **Reimplementar (não portar)**: device/AMP (fp16+GradScaler), DataLoader com
  `pin_memory`, gradient accumulation, seed CUDA, cópia de checkpoint para CPU,
  paths/Drive e provider do ONNX.
- **Preservar integralmente**: padding dinâmico + length grouping, `text_no_url`,
  sem class weights, melhor checkpoint por pior-grupo, Platt na validação,
  clip 1.0, weight decay sem bias/LN, warmup 10%, freeze 6, split grupo×rótulo.
