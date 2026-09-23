# CODE_REVIEW — pipeline v6 (revisor adversarial)

- Escopo: `models/v6/{prepare_v6.py, train_bertimbau_v6.py, colab_bertimbau_v6.ipynb}`,
  `PLANO_ADAPTACAO_V6.md`, `BUILD_NOTES.md`, `processed/{prepare_stats,token_stats}.json`,
  `artifacts/smoke_cpu/*` (5 smokes: cpu + informative + ood), contratos v4
  (`models/v4/spec_treino_v4.md`, `models/v4/PLANO_FT_V4_T4.md`, `models/evaluate.py`,
  `models/data.py`).
- Método: leitura integral + reprodução local (numpy/pandas/sklearn/torch 2.14 CPU,
  transformers 5.17.0) sobre os parquets reais + execuções de smoke de 16/32 linhas.
- Arquivos de produção **não foram alterados**. Scripts e logs de reprodução ficam em
  `models/v6/review_tmp/` (descartável). `CODE_REVIEW.md` é o único arquivo novo em produção.
- Reprodução geral:
  - `python models/v6/review_tmp/check_splits.py`
  - `python models/v6/review_tmp/check_leak2.py`
  - `python models/v6/review_tmp/check_dfr.py`
  - `python models/v6/review_tmp/check_artifacts.py`
  - `python models/v6/review_tmp/check_neardup_full.py`
  - `python models/v6/review_tmp/check_neardup_coverage.py`
  - `python models/v6/review_tmp/check_strat_table.py`
  - `python models/v6/review_tmp/check_t4.py`
  - `python models/v6/review_tmp/check_ablates.py`
  - Saídas em `models/v6/review_tmp/*.out.txt`; logs `r_*.log`.
- Confirmações globais: `--check-vendor` PASSOU (sha256 e todos os maxdiff 0);
  smoke CPU 17/17 artefatos; `run_config.json`/`metrics.json` com sha256 de
  script/pool/splits conferidos byte a byte contra o disco.

## Sumário de achados

| ID | Severidade | Título |
|---|---|---|
| V1 | ALTO | Resume do notebook não sobrevive a reinício de sessão (`out_dir` local ≠ `drive_out`) |
| V2 | ALTO | Near-duplicatas contaminam o teste do `full_iid` (20% do FakeWhatsApp; 619 pares >0,8) |
| V3 | ALTO | `--workers 2` (default e notebook) quebra com multiprocessing spawn (`train_collate` closure) |
| V4 | MEDIO | `test_at_val_threshold` e `worst_group`/per-group ignoram o limiar; blocos ficam enganosos |
| V5 | MEDIO | `--ablate classweights` empilha DFR + class weights (semântica mudou vs v4) |
| V6 | MEDIO | Projeção T4 [10,30]x não medida e viesada; auditoria por FLOPs não valida a faixa |
| V7 | BAIXO | `n_test` 13.665 no trainer vs 13.666 no PLANO/aceitação (U+FFFD) |
| V8 | BAIXO | Tabela de tokens do PLANO §5 diverge do `token_stats.json` (p50/p99 do pool) |
| V9 | BAIXO | `--probe-max-length` em CPU é ignorado e o treino continua |
| V10 | BAIXO | Lista de compatibilidade do `--resume` omite `drop_tiers`, `batch_size`, `epochs`, `weight_clip`, `eval_col` |
| V11 | BAIXO | `global_step` não é restaurado no `--resume` |
| V12 | BAIXO | `per_group.csv` com colunas união (NaN) em vez de chave comum por `cut` |
| V13 | BAIXO | `jsonable(np.float32('nan'))` emite `NaN` (JSON estrito inválido) |
| V14 | BAIXO | `calibration.json` não registra o prior efetivo do DFR (0,6032) ao lado de `calib_prior_fake` (0,5201) |
| V15 | BAIXO | Smoke: `ece_after` > `ece_before` (Platt piorou o ECE no próprio val_calib) — monitorar |

---

## V1 — ALTO — Resume do notebook não sobrevive a reinício de sessão

**Evidência:** `models/v6/train_bertimbau_v6.py:1272` (`last_dir = out_dir / "last"`);
`models/v6/train_bertimbau_v6.py:1353-1361` (`mirror_to_drive` só copia `out → drive`);
`models/v6/colab_bertimbau_v6.ipynb` §5/§6/§7 (célula de resume usa
`{RUN}/last/trainer_state.json` **no Drive**, mas `--out /content/runs/...`).

**Reprodução (executada):**

```powershell
# "Drive" de mentira com last/ + run_config; out local vazio; --resume
Copy-Item -Recurse models/v6/review_tmp/r_fresh/last models/v6/review_tmp/r_drive/last
Copy-Item models/v6/review_tmp/r_fresh/run_config.json models/v6/review_tmp/r_drive/run_config.json
python models/v6/train_bertimbau_v6.py `
  --data models/v6/processed/v6_pool.parquet --splits models/v6/processed/v6_splits.parquet `
  --split-col full_iid --out models/v6/review_tmp/r_drive_local `
  --drive-out models/v6/review_tmp/r_drive `
  --smoke --smoke-n 16 --device cpu --amp off --workers 0 --resume
```

**Observado:** `[E7] aviso: last/ nao existe; comecando do zero` e treino de 1 época
(`r_drive.log:48-49`). **Esperado:** retomar do `last/` do Drive (o notebook promete
idempotência entre sessões). O script só lê `last/` de `--out`; nunca pré-carrega do
`--drive-out`. Em um reinício real de runtime do Colab, `/content/runs/<run>` não existe:
o run reinicia do zero e depois sobrescreve `last/`/`best/` do Drive. Não corrompe, mas
joga fora horas de treino e invalida o fluxo de retomada do spec v4 §E7.

**Correção mínima:** no notebook, antes de chamar o script, copiar
`{Drive}/last`, `{Drive}/best`, `run_config.json` para `{out}/` quando existirem; ou no
trainer, no bloco `--resume`, pré-carregar de `drive_out` se `out_dir/last` estiver
ausente (`atomic_copy_tree(drive/"last", out_dir/"last")` + `run_config.json`).

---

## V2 — ALTO — Near-duplicatas contaminam o teste do `full_iid`

**Evidência:** `models/v6/processed/prepare_stats.json:846-853` (sonda 5k×5k: 25 pares
>0,8; `max_jaccard=0,983607`); `models/v6/PLANO_ADAPTACAO_V6.md:96-98` e
`BUILD_NOTES.md:82-84` classificam como "limitação residual, não bloqueante";
`models/v6/prepare_v6.py:268-307` (probe amostral), `:162-180` (`four_way_grouped`
só agrupa duplicata **exata** por `text_key`).

**Reprodução (executada):**
`python models/v6/review_tmp/check_neardup_full.py` (5-gramas de palavra exatos, índice
invertido, Jaccard >0,8, textos com >=20 palavras, sem amostragem) e
`python models/v6/review_tmp/check_neardup_coverage.py` (cobertura de linhas de teste).

**Observado vs esperado:**

- Sonda amostral reproduzida na risca: **25 pares >0,8** em 5k×5k, `max_jaccard=0,9836`
  (idêntico ao stats) — a sonda está correta, mas **subestima** o problema.
- Varredura completa do `full_iid` (>=20 palavras): **619 pares treino↔teste >0,8**
  (329 >0,9; 149 >0,95), 613 do mesmo grupo, 23 com rótulo conflitante.
  **231 linhas de teste** têm gêmea >0,8 no treino; por grupo:
  **FakeWhatsApp.BR_2018 192/958 (20,0%)**, COVID19.BR 20/292 (6,8%),
  COVID19.BR_raw 4/55, `fakes` 10/3054, etc.
- `ood_wa` é limpo: 6 pares >0,8 (5 `fakes` fake→WhatsApp fake; 1 `FC_BOATOS_VIRAL`
  fake→WhatsApp true).
- **Esperado:** 0 (a política de split por grupo de texto só cobre duplicata exata).
  Como FakeWhatsApp é um dos 8 grupos informativos que alimentam
  `worst_group_balanced`/`ece_balanced`, a manchete do R0 `full_iid` está inflada por
  memorização; dizer "residual, não bloqueante" é forte demais para 20% do grupo.
  O R1 (`ood_wa`) permanece utilizável como régua.

**Correção mínima (sem dependência nova):** em `prepare_v6.py`, antes do split, agrupar
near-dups por união (union-find) usando a função `shingles()` já existente, com bloqueio
por `group` (e candidatos via índice invertido de 5-gramas, como o próprio probe);
propagar o lado do componente em `four_way_grouped(key_col="near_key")`. Custo restrito
aos grupos afetados (WhatsApp/COVID). Alternativa mais simples: remover do treino as
linhas com gêmea >0,8 no teste (perde ~449 linhas de treino, teste intacto). Se nada for
corrigido, rebaixar a manchete IID a "limite superior" e usar o OOD como régua — decisão
consciente, registrada.

---

## V3 — ALTO — `--workers 2` quebra com spawn; default do CLI e do notebook

**Evidência:** `models/v6/train_bertimbau_v6.py:1144-1148` (`train_collate` é closure
local de `main`) e `:1155-1160` (`DataLoader(..., num_workers=args.workers,
persistent_workers=True)`); notebook §4-§7 sempre com `--workers 2`.

**Reprodução (executada, Windows):**

```powershell
python models/v6/train_bertimbau_v6.py --data models/v6/processed/v6_pool.parquet `
  --splits models/v6/processed/v6_splits.parquet --split-col full_iid `
  --out models/v6/review_tmp/r_workers2 --smoke --smoke-n 32 --device cpu --amp off --workers 2
```

**Observado:** exit 1 em `r_workers2.log:87`:
`AttributeError: Can't get local object 'main.<locals>.train_collate'`
(dataset e collate enviados a workers por pickle; a closure não é picklable).
**Esperado:** rodar com 2 workers. No Colab (Linux, `fork`) o closure é herdado e
provavelmente funciona hoje; porém o default do CLI é 2 e o mesmo caminho quebra em
qualquer contexto spawn/forkserver (Windows, macOS, Python >=3.14 no Linux, ou
`multiprocessing_context` futuro) — é uma bomba de portabilidade para o run cheio.

**Correção mínima:** tornar o collator classe/função de módulo
(`class TrainCollator: __init__(tokenizer); __call__(features)`), sem closure; idem para
`eval_collate` por simetria. Testar `--workers 2` no smoke T4.

---

## V4 — MEDIO — `test_at_val_threshold` ignora o limiar nas métricas de manchete

**Evidência:** `models/v6/train_bertimbau_v6.py:1515-1520` (dois `report(...)`) e bloco
embutido `models/evaluate.py:195-212` — `report` usa `threshold` só em `core_metrics`;
`worst_group_f1(df, y, p)` e `per_group(...)` não recebem limiar (default 0,5).

**Reprodução:** `python models/v6/review_tmp/check_artifacts.py` → no smoke,
`test.worst_group_macro_f1 = test_at_val_threshold.worst_group_macro_f1 = 0,58607`,
`ece` idêntico, etc. Só `acc/macro_f1/f1_fake` mudam entre os blocos.

**Observado vs esperado:** o bloco "limiar val_opt" reporta `worst_group`/`per_group` a
0,5, contrariando o spec v4 §6.1 ("mesmo bloco com `threshold_val_opt`"). Herdado do
evaluate.py da v4, mas a v6 o promove a manchete; conclusões sobre limiar ótimo ficam
enganosas. **Correção mínima:** em E6, calcular explicitamente
`worst_group_f1(eval_df, y_ev, (p_cal_ev >= thr_opt))` e um `per_group` no limiar para o
segundo bloco (ou documentar que só acc/macro-F1 globais obedecem ao limiar).

---

## V5 — MEDIO — `--ablate classweights` empilha DFR + class weights

**Evidência:** `models/v6/train_bertimbau_v6.py:666-682`
(`"classweights": {"class_weights": True}`) com default v6 `--dfr-weights cell`
(`:636`), e `build_sample_weights` multiplicando `cw * wd` (`:568-589`).
Na v4 o default era `dfr_weights off`, logo a ablação media "só class weights"; na v6
media "class weights **em cima do** DFR". **Esperado:** ou declarar a mudança de
semântica no PLANO §8, ou `{"class_weights": True, "dfr_weights": "off"}` se a intenção
era reproduzir a ablação v4.

**Reprodução:** `python models/v6/review_tmp/check_alates.py` (mostra o preset efetivo).

---

## V6 — MEDIO — Projeção T4 [10,30]x não medida e viesada

**Evidência:** `PLANO_ADAPTACAO_V6.md:136-157`; `BUILD_NOTES.md:172-186`;
`train_bertimbau_v6.py:1706-1718` (projeção no smoke). Base: **212,7 tokens/s** medidos em
CPU com `--smoke` (batch 8, cap 128, subconjunto de 1.024 do `full_iid`), speedup
`T4/CPU ∈ [10,30]x` nunca medido.

**Auditoria independente (executada, `check_t4.py`):** BERTimbau base 108,9M params,
43,1M treináveis (freeze 6); FP/Tok = 2·P + 4·P_treino ≈ **390 MFLOPs/token**;
3.183.840 tokens pagos/época (`full_iid`@192); 49,94 tokens pagos/amostra.

| hipótese | tok/s T4 | amostras/s | época full_iid | 2 épocas |
|---|---:|---:|---:|---:|
| 50–70% de 65 TFLOPS (pedido) | 83k–117k | 1.668–2.335 | 0,5–0,6 min | ~1 min |
| 10% MFU (realista BERT-fine-tune) | 16.659 | 334 | 3,2 min | 6,4 min |
| 5% MFU | 8.330 | 167 | 6,4 min | 12,7 min |
| **faixa do plano [10,30]x** | **2.127–6.381** | 43–128 | 8,3–24,9 min | **17–50 min** |
| 3,83% / 1,28% MFU | 6.381 / 2.127 | — | 8,4 / 24,9 min | — |

**Observado vs esperado:** a eficiência de 50–70% dá ~84k–117k tok/s (época de ~30 s),
não crível; a faixa do plano corresponde a **MFU de 1,3–3,8%**, baixo mas possível para
T4/bs32/seq<=192 (memory-bound). Com MFU 5–15% a banda seria **8,3k–25k tok/s**
(2,1–6,4 min/época), ou seja, o plano é plausivelmente **conservador** — mas sem medição.
Vieses: CPU medida em batch 8/cap 128 num subset curto (≈40,7 tokens pagos/amostra) e
aplicada a tokens de produção a 192 com 49,94/amostra; CPU favorece batch pequeno, GPU
ganha com batch 32; um único epoch de 1024 amostras. **Correção mínima:** rodar o smoke
T4 (spec §5.4) em 32×192 e usar o tokens/s medido; manter a faixa explícita apenas como
placeholder (já é o que o texto faz — só não usar para decidir orçamento).

---

## V7 — BAIXO — `n_test` 13.665 (trainer) vs 13.666 (PLANO/aceitação)

`prepare_stats.json` registra `full_iid.test = 13.666`; o trainer remove a linha U+FFFD
do teste (`train_bertimbau_v6.py:1040-1045`), então `metrics.data.n_test = 13.665`
(política documentada em `BUILD_NOTES.md:151-153`). Se o critério de aceitação for
literal "13.666", ele falha por 1. Sugestão: registrar `n_test_bruto` e `n_ufffd_removidas`
no `metrics.json` ou anotar no PLANO.

## V8 — BAIXO — Tabela de tokens do PLANO §5 divergente

`PLANO_ADAPTACAO_V6.md:119-122` diz pool p50=24 e p99=766; `processed/token_stats.json`
mede **p50=27,0** e **p99=726,21**. Demais valores batem. Corrigir a tabela (fonte =
token_stats.json).

## V9 — BAIXO — `--probe-max-length` em CPU é no-op silencioso

`train_bertimbau_v6.py:1237-1239`: sem CUDA imprime "ignorado (VRAM = 0)" e **segue
treinando**. Reproduzido (`r_probe.log:48` + `[FIM]`). No Colab/CUDA ele roda e retorna 0
sem treinar. Esperado: em CPU, ou abortar com erro claro, ou também retornar sem treinar.

## V10 — BAIXO — Compatibilidade do `--resume` incompleta

`train_bertimbau_v6.py:1277-1278` valida só 9 chaves. `drop_tiers` não está na lista:
retomar um run default como `--resume --ablate llmoff` (mesmo seed/split) prossegue
silenciosamente e mistura distribuições de treino. Incluir `drop_tiers` (e, por higiene,
`batch_size`, `epochs`, `weight_clip`, `eval_col`, `mask_entities`, `smoke`).

## V11 — BAIXO — `global_step` não restaurado no resume

`train_bertimbau_v6.py:1364` reinicia `global_step = 0` mesmo no `--resume`;
`trainer_state.json` posterior fica com passo relativo errado (sem efeito no treino).
Restaurar `global_step` do estado, como já se faz com `epoch/history/best_score`.

## V12 — BAIXO — `per_group.csv` com colunas união (NaN)

`train_bertimbau_v6.py:1578-1586` concatena tabelas cuja primeira coluna é o nome do cut
(`group`, `channel`, ...), gerando header com 5 colunas de chave e NaNs. Consumidores
que procuram `group` recebem NaN nas linhas de `channel`. Corrigir renomeando a chave
para `chave` + coluna `cut` (ou emitir um arquivo por cut).

## V13 — BAIXO — `jsonable` emite NaN não-estrito

`train_bertimbau_v6.py:318-321`: `np.float32('nan')` cai no ramo `np.floating` e é
devolvido como `nan`; `json.dumps` escreve `NaN` (inválido no RFC 8259; o `json.load`
do Python aceita). Reprodução: `jsonable({'a': np.float32('nan')})` → `{"a": NaN}`.
Tratar `np.isnan` também para `np.floating` (ou usar `allow_nan=False` para detectar).

## V14 — BAIXO — Prior efetivo do DFR não vai para `calibration.json`

O prior efetivo pós-clip (0,6032 no `full_iid`) está no `loss_info`/log, mas o
`calibration.json` só tem `calib_prior_fake` (0,5201, máscara). A calibração é coerente
(desloca o intercepto do regime 0,603 → 0,52 da implantação), mas a rastreabilidade fica
espalhada. Adicionar `dfr_prior_efetivo_fake` e `weight_clip` no bloco de calibração
(`weight_clip` já existe; falta o prior).

## V15 — BAIXO — Platt piorou o ECE no smoke (0,0653 → 0,1075)

`artifacts/smoke_cpu/metrics.json` (`calibration.ece_after > ece_before`). Com 444
amostras e cabeça subtreinada, Platt pode piorar ECE no próprio conjunto de ajuste;
não é bug, mas o full run deve monitorar (se persistir, investigar máscara/otimizador).

---

## Itens do checklist sem achado (verificados)

- **Splits/leakage:** interseções entre `train/val_sel/val_calib/test/unused` = 0 nos 3
  splits (partição exata de 91.080); `ood_wa.test` = 6.381, 100% whatsapp; `unused` = 4
  linhas `fakes`/agency_claim, cujos `text_key` colidem com 3 textos do teste e **não**
  aparecem no treino OOD; `full_iid_copias_mesmo_lado=0`; `train/val × test` por
  `text_key` = 0 em todos os splits. Duplicatas: **427 grupos / 900 linhas / 473 extras**
  (reproduzido). 3 clusters com rótulo fake/true (8 linhas):
  (a) "a gente ouve falar... wuhan" — `fakes|fake` + `COVID19.BR|true`, ambos no teste do
  `full_iid` (ruído métrico, não vazamento) e ambos no treino OOD;
  (b) "finalmente chegou coletanea dilma..." — 1 true + 3 fake, todos no treino
  `full_iid`/`bal_iid` e **todos no teste OOD** (teste com 3 fake/1 true do mesmo texto);
  (c) "so para lembrar stf..." — `true|true` + `fakes|fake`, ambos no treino.
  Em treino é ruído de rótulo (peso DFR da célula `fakes` é mínimo); não há vazamento
  porque o lado é propagado. Cluster (b) introduz uma pequena inconsistência no teste OOD.
- **Estratificação `full_iid`:** proporções por `grupo|rótulo` mantidas (fake% 73,31 no
  pool; 73,31/73,29/73,38 no train/val_sel/val_calib; test 73,31). Nenhum grupo inteiro
  no teste; **FC_NEXO (3) e MuMiN-PT_raw (4), ambos constantes, ficaram só no treino**
  (não confiáveis, fora da manchete). 8 informativos com `test_fake%` a <=0,5 p.p. do
  treino (ex.: FakeWhatsApp 48,0→47,9; COVID19.BR 42,4→42,8). Tabela completa:
  `review_tmp/check_strat_table.out.txt`.
- **DFR cell clip 25 (reproduzido do zero):** média exatamente 1,0; `full_iid`
  min 0,1004 / p50 0,5323 / p99 11,81 / max 33,06; prior efetivo pós-clip **0,60318**
  (idêntico ao stats; sem clip 0,52542); célula de 1 exemplo (`FC_AFP|true`,
  `FC_ESTADAO|true`) sai de 1.080,54 → 25 → **33,055** (não explode); `ood_wa` 0,60615;
  `bal_iid` 0,48883. Cálculo feito **só no treino** (`train_df`, inclusive após o subset
  do smoke) em `train_bertimbau_v6.py:1210`. Calibração Platt na máscara
  `is_balanced_group`: `full_iid` n=1.842 prior 0,5201 (`ood_wa` 0,5282; `bal_iid`
  0,5223) — coerente com o deslocamento deliberado 0,603 → 0,52.
- **Treino/avaliação:** ordem AMP exata (`autocast → scale → backward → unscale_ →
  clip_grad_norm_ → step → update → sched.step → zero_grad`) em `:1392-1412`; melhor
  checkpoint por worst_group com fallback macro-F1 (`:1427-1431`) e `best/`+`last/` a
  cada época; teste só tocado no E6 (grep: `eval_df` em E2/E3 é contagem, leak-check e
  tokenização; `l_ev` só em `:1485`); Platt e grid de limiar só no `val_calib`
  (`:1501-1505`); `report` usa `is_ptpt` já renomeado (`:1097-1099`);
  `metrics.json` com todas as chaves do spec §6.1, `predictions.csv` com exatamente as
  21 colunas do §6.2, `calibration.json` legado+novo, `reliability.csv` fechando n=test.
- **Smoke:** força `predict_all=False`, `epochs=1`, cap 128, batch 8 (`:1119-1133`);
  `history.json` com **1 época** e args `epochs=1` — o relato de "3 épocas" é refutado;
  se tivesse rodado 3 épocas de 1024 a CPU gastaria ~2×196 s extra e violaria o `<10 min`.
  Contrato 17/17 artefatos; `history.json` == `metrics.history`; resume não duplica
  (`history` continua 1 após `--resume`); config divergente aborta código 2.
- **Resume (execuções):** `--resume` no mesmo out dir → exit 0, `[E7] retomando ...
  epoca 1/1`, history=1; `--resume --seed 43` → exit 2 com "run_config incompativel";
  `--resume` sem `last/` → aviso + treino do zero (exit 0); `--split-col` inexistente →
  exit 2 com mensagem clara.
- **Notebook:** JSON válido (`nbformat.validate`, 19 células); presets e run-ids
  conferem com o código (`check_alates.py`): R0 `full_iid_ml192_f6_on_seed42`,
  R1 `ood_wa_ml192_f6_on_seed42_ablate-ood`,
  abl `bal_iid_ml192_f6_off_seed42_ablate-informative`; sha256 do pool checado contra
  `prepare_stats.json`; coleta lê `test`/`ood` corretamente.
- **`run_config.json`/`metrics.json`:** sha256 do script, do `v6_pool.parquet` e do
  `v6_splits.parquet` batem byte a byte com o disco.

## Veredito

**GO condicional para o Colab.** Não há BLOQUEANTE de execução; o pipeline treina,
avalia e retoma (mesma sessão) corretamente, e o contrato de artefatos fecha. Condições:

1. **V1:** consertar o resume entre sessões (pré-copy `drive_out → out`) ou remover o
   `--resume` automático das células e documentar que retomada só vale na mesma sessão.
2. **V3:** validar `--workers 2` no smoke T4; aplicar o collator de módulo (correção
   trivial que elimina o risco de vez).
3. **V2:** decidir consciente: clustering de near-dups no `prepare` **ou** declarar a
   manchete `full_iid` como limite superior e usar `ood_wa` (limpo: 6 pares) como régua
   de generalização. Sem essa decisão, os números de `worst_group_balanced`/`ece_balanced`
   do R0 não são confiáveis.
4. **V6:** rodar o smoke T4 em 32×192 antes de qualquer run cheio e substituir a
   projeção [10,30]x pelo throughput medido.

## Sólido, não mexer

- Política de split no nível do grupo de texto com propagação de lado e validações que
  abortam (`textkey_um_lado`, `sem_vazamento_texto`, `full_iid_copias_mesmo_lado`) —
  reproduzido do zero, tudo 0.
- `four_way_grouped`/`four_way` reproduz as contagens do v4 (`bal_iid` a 5 linhas de
  distância, explicado pelas 473 cópias).
- DFR cell/clip/renorm idêntico entre `prepare_v6.dfr_weight_stats`, `build_sample_weights`
  e o esperado; média 1; prior logado; célula n=1 clampada sem explosão.
- Bloco vendor de `evaluate.py` e `group_balanced_weights` — `--check-vendor` PASSOU com
  maxdiff 0 (float64/float32) e sha256 conferido.
- Ordem AMP, determinismo (seed + cudnn), melhor-checkpoint com fallback, separação
  val_sel/val_calib/teste, Platt restrito a `val_calib`/`is_balanced_group`.
- Contrato de saída (17 artefatos do smoke, chaves de `metrics.json`/`calibration.json`,
  colunas de `predictions.csv`) e sha256 de `run_config.json`.
- Notebook: presets/ablações/run-ids, cópia de dados e validação de sha256 do pool.

## Relatório final

**Achados por severidade:** ALTO V1 (resume Colab), V2 (near-dups), V3 (workers spawn);
MEDIO V4 (limiar ignorado no report), V5 (classweights empilhado), V6 (projeção T4);
BAIXO V7–V15 (n_test/U+FFFD, tabela do PLANO, probe CPU, compat de resume, global_step,
per_group união, NaN JSON, prior do DFR, ECE do smoke).

**Duplicatas/near-dups:** 427 grupos exatos / 900 linhas / 473 extras, 0 cruzando
train↔test por `text_key`; 3 clusters de rótulo conflitante (1 no teste IID, 1 no teste
OOD, 1 no treino; ruído/erro de teste, não vazamento). Sonda 5k×5k reproduzida
exatamente (25 pares >0,8; max 0,9836). Varredura completa (>=20 palavras):
**619 pares >0,8 treino↔teste no `full_iid`** (329 >0,9; 149 >0,95), **231 linhas de
teste únicas** com gêmea no treino, **192/958 = 20% do FakeWhatsApp.BR_2018** (13,6%
>0,9; 9,1% >0,95); `ood_wa` tem só 6 pares >0,8 (1 com rótulo conflitante).

**Testes de execução:** smoke CPU 1024 (produção) — 1 época, 212,7 tok/s, exit 0;
smoke 32 novo exit 0 (history=1); `--resume` exit 0 sem duplicar época (history=1);
`--resume --seed 43` exit 2 (config divergente); `--resume` sem `last/` aviso + treino;
`--resume` com `last/` só no `drive_out` **reinicia do zero (V1 confirmado)**;
`--workers 2` exit 1 (V3); `--probe-max-length` CPU no-op e treina (V9);
`--split-col` inexistente exit 2; `--check-vendor` PASSOU.

**Auditoria T4:** FLOPs/token ≈ 390 M (fwd 2·P, bwd 4·P_treino, freeze 6);
50–70% de 65 TFLOPS → 83k–117k tok/s (0,5–0,6 min/época — não crível);
a faixa [10,30]x do plano (2.127–6.381 tok/s, 8,3–24,9 min/época) equivale a MFU
1,3–3,8%; com MFU realista 5–15% a banda corrigida é **8,3k–25k tok/s → 2,1–6,4
min/época** (2 épocas: 4–13 min para o `full_iid`), i.e., o plano é provavelmente
conservador, mas não há âncora medida: substituir pelo smoke T4 32×192. Viés: CPU medida
com batch 8/cap 128 num subset curto (~40,7 tokens pagos/amostra) e aplicada a 192 com
49,94/amostra.
