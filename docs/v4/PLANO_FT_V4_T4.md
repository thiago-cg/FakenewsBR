# Plano de fine-tuning BERTimbau v4 na T4 (Colab)

- Projeto: FakenewsBR
- Alvo: `neuralmind/bert-base-portuguese-cased` no pool v4 (85.212 linhas)
- Hardware alvo: 1x Tesla T4 16 GB (Colab gratuito), fp16 (sm_75, sem bf16 nativo)
- Script do builder: `models/v4/spec_treino_v4.md`
- Data: 2026-09-12
- Método: rascunho v1 e revisão v2 documentados no Apêndice A. No corpo, notas
  `> **[v2]**` marcam o que mudou do v1 para o v2.

## 0. Objetivo e critério de sucesso

O objetivo do projeto (`README.md`, `models/README.md`) é um **score contínuo e
calibrado** de P(desinformação) que generalize **entre canais** — não um
classificador que maximiza acurácia média. Métricas de manchete, nesta ordem:

1. macro-F1 do **pior grupo** (grupos estatisticamente confiáveis);
2. **ECE** no regime em que o score foi definido (grupos informativos);
3. desempenho **OOD** leave-one-channel-out;
4. casos limítrofes (`rating_class == "hard"`).

Baseline honesto a superar: 76,70% acc / 82,92% F1(fake) do baseline linear, com
pior-grupo e OOD como réguas de generalização.

## 1. Conjunto de treino (decisão 1)

### 1.1 O que está medido

| item | valor |
|---|---:|
| pool treinável (`train_label` fake/true) | 85.212 |
| fake / true | 60.991 (71,58%) / 24.221 (28,42%), razão 2,52:1 |
| linhas em grupos de rótulo constante | **39.351 (46,18%)**, 10 de 27 grupos |
| linhas em grupos informativos | **36.896**, fake 51,98% (razão 1,08:1), 8 grupos |
| duplicatas exatas / conflitos texto↔rótulo | 0 / 0 |
| linhas com `U+FFFD` | 4 (rids 11447, 16958, 836990361129089716, 839894016325789543) |
| linhas com mojibake (Latin Extended) | ~50 (flag, não descarte) |

Grupos informativos (n>=200, minoria>=15%):

| grupo | n | %fake | canal |
|---|---:|---:|---|
| FC_POLIGRAFO | 10.831 | 64,6 | agency_claim |
| EXT_LIARBR | 6.996 | 35,7 | external_claim |
| EXT_AVERITECBR | 2.925 | 69,2 | external_claim |
| Fake.br | 7.160 | 50,0 | portal |
| FakeWhatsApp.BR_2018 | 6.381 | 47,9 | whatsapp |
| COVID19.BR | 1.931 | 42,5 | covid |
| COVID19.BR_raw | 373 | 15,8 | covid |
| LLM4BR_300 | 299 | 49,5 | llm |
| **total** | **36.896** | **51,98** | 6 canais |

Grupos constantes que dominam o pool: `fakes` 20.347 (100% fake), `FC_BOATOS`
10.774 (100% fake), `FC_BOATOS_VIRAL` 2.360 (100% fake), `true` 2.710 (100%
true), `NEWS_*` (4 grupos, 3.147 linhas, 100% true). Outros 25.5k vêm de
checadores que falham o critério de minoria (FC_EFARSAS, FC_G1, FC_LUPA etc.).

### 1.2 Decisão DEFAULT: (b) somente grupos informativos (36.896)

Treinar o encoder **apenas** nas linhas `is_balanced_group == True`:

- **Sem atalho de procedência estrutural.** 46,18% do pool é rótulo-por-origem;
  com o pool completo o encoder aprende "estilo de alegação de agência => fake" e
  "manchete de portal => true". O score deixa de medir veracidade.
- **Prior já quase balanceado (1,08:1)**, então não é preciso ponderar a loss por
  classe — o que preserva a calibração (objetivo do projeto).
- **Treino 2,3x menor**: 25.827 amostras em vez de 67.725 no split default,
  deixando orçamento de T4 para OOD e ablações.
- **Avaliação bem-posta**: todos os grupos do teste têm as duas classes, então
  `worst_group_f1` e ECE significam algo.

Trade-off aceito: perde-se diversidade de fake dos checadores desbalanceados
(FC_EFARSAS, FC_G1, FC_LUPA etc., ~8,7k linhas). É exatamente essa massa que o
atalho usa como scoragem; a hipótese de que ela ajuda a fronteira fake/true é
medida na ablação A1, não assumida.

> **[v2]** O rascunho v1 usava o pool completo com pesos DFR como default. A
> revisão mostrou que a ponderação limita o *peso total* de cada grupo, mas não
> remove o atalho: o encoder continua vendo milhares de exemplos puros de
> origem-rótulo, e o teste IID sobre o pool completo contém esses mesmos grupos
> (avaliação contaminada). O pool completo virou ablação A1.

### 1.3 Ablações do conjunto de treino

| id | conjunto | pesos | pergunta |
|---|---|---|---|
| A0 (default) | informativos 36.896 | nenhum | referência |
| A1 | pool completo 85.212 | DFR célula×rótulo, `weight_clip=25` | os grupos constantes ajudam ou atrapalham a fronteira? |
| A1b | pool completo sem `llm_local`/`corroborated` (3.101 linhas `true`, check em `label_tier`) | DFR célula, clip 25 | o `true` fraco da camada LLM polui? |
| A3 | informativos 36.896 | DFR célula, clip 25 | equalizar os 8 grupos ajuda o pior-grupo? |

Com o pool completo e pesos DFR, o prior efetivo é ~1:1 (22 células fake x 22
células true). O clip em 25 é obrigatório: sem ele, a célula `FC_FOLHA|true`
(3 exemplos) recebe peso ~645 por amostra, o que desestabiliza CE e calibração.

## 2. Splits (decisão 2)

### 2.1 Protocolo canônico, 4 vias (medido)

A validação é dividida em duas para não superajustar: `val_sel` (checkpoint e
early stop) e `val_calib` (Platt e limiar). O teste nunca é tocado.

| split (coluna no parquet) | train | val_sel (10%) | val_calib (5%) | test |
|---|---:|---:|---:|---:|
| `bal_iid` (default, informativos) | 25.827 | 3.689 | 1.845 | 5.535 |
| `bal_ood_wa` (leave-whatsapp-out) | 25.937 | 3.052 | 1.526 | 6.381 (todo o canal whatsapp) |
| `full_iid` (completo, ablação A1) | 67.725 | 7.968 | 3.984 | holdout: 5.535 do teste canônico |

Regras de construção (executadas **uma vez no prepare local**, nunca no Colab):

- estratificação por `grupo|rótulo` replicando `models/data.py::_safe_strat`;
- `bal_iid`: 70/10/5/15 por `train_test_split` aninhado, seed 42;
- `bal_ood_wa`: teste = todas as linhas informativas de `whatsapp`; o resto
  dividido 85/10/5 com a mesma estratificação;
- `full_iid`: o teste canônico (`bal_iid.test`) é **removido** do conjunto
  (holdout) e o restante dividido 85/10/5. Verificado: 0 interseção;
- o teste canônico `bal_iid.test` (5.535 linhas, 51,98% fake) é o **eval
  compartilhado** de A0 e A1, o que torna as métricas comparáveis.

### 2.2 Qual teste responde "modelo treinado em portal detecta WhatsApp?"

O `bal_ood_wa` (treina em todos os canais informativos **exceto** WhatsApp,
testa nas 6.381 linhas do FakeWhatsApp.BR_2018). Como o canal portal participa
do treino, é o teste direto da pergunta. O `full_iid` de A1 não tem OOD honesto
(se treinar no pool completo, o WhatsApp entra no treino) — A1 é avaliado apenas
no teste canônico.

Controles de atalho no split IID: (i) estratificação por grupo×rótulo garante
as duas classes em cada grupo nos dois lados; (ii) `worst_group_f1` (n>=100,
minoria>=20) é a manchete; (iii) grupos de rótulo constante ficam **fora** do
teste IID; (iv) todo `predictions.csv` é persistido por `rid` para auditoria
post-hoc.

Limite residual: clusters de quase-duplicata não foram persistidos na v4
(o sanitizer eliminou duplicata exata: 0 no pool). O prepare roda uma sonda
barata de contenção 5-gram (amostra de ~5k pares treino×teste, Jaccard>0,8) e
registra no `prepare_stats.json`. Só seria bloqueante se acusasse casos.

> **[v2]** O v1 usava uma única validação (15%) para escolher época, calibrar e
> escolher limiar — três decisões sobre os mesmos 5,5k exemplos. O protocolo 4
> vias resolve. O v1 também avaliava cada ablação no próprio teste; agora todas
> usam o mesmo `eval_bal`.

## 3. Comprimento máximo (decisão 3)

Tokens BERTimbau reais medidos pelo planejador (tokenizer cacheado,
`add_special_tokens=True`, `text_no_url`):

| recorte | média | p50 | p90 | p95 | p99 | máx | >192 | >256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pool | 73,0 | 27 | 180 | 309 | 745 | 8.575 | 9,36% | 6,52% |
| informativos | 119,6 | 33 | 308 | 453 | 1.046 | 8.575 | **18,82%** | 13,02% |
| treino `bal_iid` | 106,2 | - | - | 432 | - | - | 15,34% | - |

Custo real do padding dinâmico + agrupamento por comprimento (medido na
simulação do sampler, batch 32, mega 50, treino `bal_iid`):

| cap | tokens pagos/época | média/amostra | batches no cap | razão de custo |
|---|---:|---:|---:|---:|
| 128 | 1,5 M | 59,9 | 26,5% | 0,79 |
| 192 | 1,9 M | 74,8 | 20,3% | 1,000 |
| 256 | 2,2 M | 85,7 | 14,1% | **1,145** |

Decisão: **default `max_length=192`, batch 32**. A 256 custa apenas +14,5% (não
+33%, porque só a cauda alonga), mas o ganho é de 5,8 p.p. de amostras sem
truncamento (18,82% → 13,02%). Como 192 é a config já comprovada em CPU, tem
mais folga de VRAM e o orçamento de T4 permite medir, a promoção de 256 ocorre
por resultado, não por intuição:

- A2 = repetir A0 com 256 (primeira ablação de custo);
- **regra de promoção**: se A2 superar A0 em `val_sel.worst_group` por >= 0,010,
  o run final adota 256 (se VRAM permitir; ver escada de OOM na seção 5).

O builder **deve** reproduzir a tabela de tokens com
`train_bertimbau_v4.py --token-stats` (escreve `token_stats.json`) e confirmar
na T4 a VRAM de pico do smoke antes de rodar 256.

> **[v2]** O v1 discutia só palavras (p95 = 209) e supunha que 512 palavras
> cobria quase tudo (0,91%). Em tokens, 18,82% do pool informativo passa de 192.
> O custo de 256 foi medido (1,145x), não estimado por razão de comprimento.

## 4. Épocas e early stop (decisão 4)

- `--epochs 3` (teto), `--patience 1`, `--min-delta 0.005`.
- Checkpoint e parada monitoram **`worst_group_f1` em `val_sel`** (via
  `evaluate.worst_group_f1`, n>=100 e minoria>=20; fallback para `macro_f1` se
  `nan`).
- Log por época: `train_loss`, `val_loss` (CE), `val_macro_f1`, `val_f1_fake`,
  `val_ece`, tokens/s, escala do GradScaler, VRAM de pico, segundos.
- Parada manual antecipada: parar se, na época 2, `worst_group` não melhorou
  >= 0,005 sobre a época 1 **e** `val_loss` subiu nos dois registros; ou se o
  `train_loss` cai > 40% enquanto o pior-grupo estagna/desce (sinal de overfit).
- Esperado: 2–3 épocas (mesmo padrão da v1, que parou em 2 na CPU).

> **[v2]** O v1 misturava seleção de época e calibração na mesma validação.
> `val_sel` e `val_calib` agora separam essas decisões.

## 5. AMP e otimização (decisão 5)

| hiperparâmetro | default | origem |
|---|---|---|
| device | CUDA (T4), fallback CPU | `--device auto` |
| AMP | `torch.amp.autocast("cuda", dtype=torch.float16)` + `GradScaler("cuda")` | sm_75 não suporta bf16 |
| batch | 32 | comprovado seguro em 32x192 |
| grad accumulation | 1 (efetivo 32) | só no fallback de OOM |
| lr | 2e-5 | comprovado |
| warmup | 10% dos steps (242 de 2.424) | comprovado |
| schedule | linear | comprovado |
| clip | 1,0 (`unscale_` antes do clip) | comprovado + ordem AMP |
| weight decay | 0,01, exceto bias/LayerNorm | comprovado |
| freeze | embeddings + 6 camadas (~43,1 M treináveis) | economiza VRAM/tempo |
| class weights | **desligado** | não distorcer a calibração |
| loss | CE pura (com peso DFR opcional em A1/A3) | calibração |

Ordem AMP obrigatória: `loss = criterion(model(**batch).logits, y)` sob
autocast; `scaler.scale(loss).backward()`; `scaler.unscale_(opt)`;
`clip_grad_norm_(params_treináveis, 1.0)`; `scaler.step(opt)`;
`scaler.update()`. Inferência sempre sob autocast e devolvendo **fp32**:
`model(**batch).logits.float().cpu().numpy()`. Logar `scaler.get_scale()`; queda
persistente (< 64) indica overflow e exige investigar (lr ou dados).

Freeze não muda com GPU/T4: congelar 6 é decisão de capacidade/VRAM, não de
hardware. A4 (freeze 4) e A4b (freeze 0) são ablações de capacidade.

Escada de OOM (aplicar em ordem, registrar qual foi usada):
1. batch 32 x 192 (default);
2. batch 32 x 256 (A2): se OOM, cair para o degrau 3;
3. batch 16 x cap + `--grad-accum 2` (efetivo 32);
4. batch 8 x cap + `--grad-accum 4` (último recurso; 4-8x mais lento);
5. `model.gradient_checkpointing_enable()` apenas se realmente necessário
   (troca compute por VRAM, ~+30% tempo).
O script loga `torch.cuda.max_memory_allocated()` por época e aborta com
mensagem clara se a VRAM de pico passar de 13,5 GB antes do primeiro backward.

> **[v2]** O v1 tinha 256 como default sem escada de fallback e salvava
> `best_state` na GPU (dobrava VRAM do modelo). Agora o checkpoint vai para CPU
> e a escada está explícita.

## 6. Avaliação e calibração (decisão 6)

### 6.1 Métricas obrigatórias no teste (via `evaluate.py`)

- `n, acc, macro_f1, f1_fake, brier, ece, roc_auc, pr_auc` (`core_metrics`);
- `worst_group_f1` (manchete; grupos confiáveis);
- bloco de grupos informativos: `worst_group_balanced`, `acc_balanced`,
  `macro_f1_balanced`, `ece_balanced` (prior ~50% é o regime do score);
- recortes: `per_group` por **canal**, **publisher**, **era**, **grupo**, e
  dialeto; cada recorte com `n`, `minoria_n` e flag `confiavel`;
- casos limítrofes: `hard_acc` e `pure_acc` (se houver >= 20 casos);
- OOD: `core_metrics` + `worst_group` no canal retido (só o run R1).

### 6.2 Limiar

- **Primário: 0,5** — comparabilidade com os runs anteriores;
- **Secundário: `threshold_val_opt`** — grid 0,05–0,95 maximizando macro-F1 em
  `val_calib` (calibrado), reportado junto;
- proibido escolher limiar no teste; `predictions.csv` guarda os dois.

### 6.3 Calibração

- Platt (`evaluate.fit_platt`) ajustado em `val_calib`, restrito a
  `is_balanced_group` quando o split default já é informativo (no `full_iid`,
  `val_calib` é 73% fake — calibrar nele desfaria o DFR; por isso a máscara
  `is_balanced_group` é aplicada sempre que disponível e a prior resultante vai
  em `calib_prior_fake`);
- `calibration.json` mantém o esquema legado (`platt_a`, `platt_b`,
  `max_length`, `model`, `mask_entities`) e adiciona `threshold`,
  `threshold_val_opt`, `calib_on`, `calib_n`, `calib_prior_fake`,
  `freeze_layers`, `seed`, `split_col`, `dfr_weights`, `script_sha256`.
- ECE de manchete = `ece_balanced`; `ece` global entra como contraste.
- Implantação: o score é uma razão de verossimilhança sob prior ~50%; para um
  fluxo com prevalência `pi` conhecida aplicar
  `p = sigmoid(logit(p_cal) + log(pi/(1-pi)) - log(pi0/(1-pi0)))`, com `pi0 =
  calib_prior_fake`. Não embutir no modelo a prevalência artificial do corpus.

### 6.4 PT-PT

Dois critérios discordam (provenance `lang_variant`: 449 linhas; regra URL do
`data.py`: 18.420), e das 449 pt-PT no pool, 445 são `true` (0,89% fake) — o
domínio não é confiável para avaliação. Decisão: reportar os dois cortes como
**descritivos**, com a cobertura, e manter a manchete em pt-BR/canal/era. O
FC_POLIGRAFO (PT-PT) **continua no treino** (é um dos 3 grupos mais
informativos). O `report()` usa a coluna `is_ptpt`; o script renomeia
`is_ptpt_rule` para isso só na avaliação.

> **[v2]** O v1 tratava o recorte PT-PT como métrica válida e deixava o `report`
> calibrar na val inteira (71,5% fake). Corrigido: máscara balanceada na
> calibração, ECE balanced como manchete e PT-PT descritivo.

## 7. Robustez do Colab (decisão 7)

Layout no Drive:

```
/content/drive/MyDrive/FakenewsBR/v4/
  data/    v4_pool.parquet, v4_splits.parquet, prepare_stats.json,
           train_bertimbau_v4.py
  runs/<run_id>/
           best/            (model.safetensors, config.json, tokenizer*, calibration.json)
           last/            (model + tokenizer + optimizer.pt, scheduler.pt, scaler.pt,
                             trainer_state.json, rng.pt)
           history.json, run_config.json, metrics.json
           predictions.csv, per_group.csv, reliability.csv, token_stats.json
```

- Treino roda em `/content/` (rápido) e o `last/` é copiado para o Drive **ao
  fim de cada época** (tmp no Drive + `os.replace`, para não deixar arquivo
  parcial). `history.json` é reescrito a cada época (arquivo pequeno).
- Interrupção perde no máximo 1 época: `--resume` carrega `last/` (pesos,
  otimizador, scheduler, scaler, RNG e número da época) e continua; valida que
  `run_config.json` é compatível (senão exige `--force-resume`).
- `best/` é reescrito quando o pior-grupo melhora; `calibration.json` e
  `metrics.json` finais também vão para o Drive.
- Seeds: `random`, `numpy`, `torch`, `torch.cuda.manual_seed_all`;
  `cudnn.deterministic=True`, `cudnn.benchmark=False`. Aviso registrado:
  reduções de CUDA não são bit-exact; a reprodutibilidade forte vale para
  splits e ordem de dados (geradores semeados), não para a 4ª decimal.
- Sessão máxima planejada: 6h (o Colab grátis desconecta por inatividade/12h).
  Cada run é independente; se cair, basta remontar o Drive e chamar `--resume`.
- Sem `--save-every-epoch` de pesos completos por época além de `best`/`last`
  (cada checkpoint ~1,1 GB; 4 runs = ~4,4 GB no Drive, dentro dos 15 GB, mas
  limpar runs velhos antes).

> **[v2]** O v1 salvava só no fim. Agora o pacote de retomada vai ao Drive a cada
> época, com validação de compatibilidade no `--resume`.

## 8. Preparação de dados (decisão 8)

Script local `models/v4/prepare_v4.py` (roda uma vez, fora do Colab):

- carrega os três CSVs via `models.data.load(csv, labels_csv, provenance_csv)`
  (mesmas regras de `train_label`, grupo, canal, `is_balanced_group`, `is_ptpt`);
- filtra as 4 linhas com `U+FFFD` no texto; mantém mojibake apenas marcado;
- deriva `era` (`<=2017`, `2018-2022`, `>=2023`, `sem_data`) e `rating_class`;
- escreve **`v4_pool.parquet`** enxuto com as colunas exatas:

| coluna | tipo | origem |
|---|---|---|
| `rid` | int64 | sanitized |
| `text_no_url` | string | sanitized (TEXT_COL) |
| `group` | string | `dataset_name` |
| `channel` | string | `data.channel_of` |
| `label` | string | `train_label` (fake/true) |
| `label_tier` | string | labels |
| `label_source` | string | labels |
| `is_balanced_group` | bool | regra `data.py` |
| `is_ptpt_rule` | bool | regex `data.py` |
| `lang_variant` | string | provenance |
| `publisher` | string | provenance |
| `rating_class` | string | regra `data.py` |
| `era` | string | derivada de `date_iso` |
| `word_len`, `num_exclamations`, `num_questions`, `num_ellipsis` | int32 | sanitized |
| `uppercase_word_ratio` | float32 | sanitized |
| `has_ufffd`, `mojibake_flag` | bool | detecção no prepare |

  (~15–20 MB em parquet snappy, contra 210 MB de CSV).
- escreve **`v4_splits.parquet`** com `rid` + 3 colunas
  (`split_bal_iid`, `split_bal_ood_wa`, `split_full_iid`);
- escreve `prepare_stats.json` com contagens, cobertura, sha256 dos dois
  parquets, versões (pandas/sklearn) e o resultado da sonda de quase-duplicata;
- falha com código 2 se qualquer invariante quebrar (85.212 linhas, 36.896
  informativos, 0 interseção holdout×teste, etc.).

Como os splits são persistidos com o `rid`, **local e Colab usam exatamente os
mesmos índices**; o Colab nunca chama sklearn para dividir.

> **[v2]** O v1 ia ler os três CSVs (315 MB) direto no Colab e recomputava os
> splits. Agora o Colab lê um parquet de ~15–20 MB e índices prontos.

## 9. Orçamento de tempo na T4 (decisão 9)

Hipóteses marcadas **[H]** devem ser substituídas por medição do builder no
smoke da T4 (`--smoke` imprime tokens/s e VRAM).

| símbolo | valor | status |
|---|---|---|
| tokens pagos/época, `bal_iid` a 192 | 1,9 M | **medido** (sampler simulado) |
| tokens pagos/época, `bal_iid` a 256 | 2,2 M | **medido** |
| tokens pagos/época, `bal_ood_wa` | ~2,1 M | estimado pela média de 123,4 tokens/amostra [H] |
| tokens pagos/época, `full_iid` | ~3,7 M | estimado pela média do pool [H] |
| throughput T4 fp16, batch 32 | 2.500–6.000 tokens/s (pessimista–otimista) | **[H]**; âncora: INFRA estima 10–20 min/época para 27,6k amostras em CPU→T4 |
| throughput de inferência | 3–5x o de treino | **[H]** |

Steps: 808 batches/época (25.827/32); 2.424 steps em 3 épocas; warmup 242.

| run | n treino | tokens/época | épocas | min/época (pess–otim) | total (pess–otim) |
|---|---:|---:|---:|---:|---:|
| R0 default `bal_iid`, 192 | 25.827 | 1,9 M | 3 | 12,7–5,3 | 38–16 min |
| R1 OOD `bal_ood_wa`, 192 | 25.937 | ~2,1 M | 3 | 14,0–5,8 | 42–17 min |
| A1 `full_iid` + DFR, 192 | 67.725 | ~3,7 M | 2 | 24,7–10,3 | 49–21 min |
| A2 `bal_iid` a 256 | 25.827 | 2,2 M | 3 | 14,7–6,1 | 44–18 min |
| A3 `bal_iid` + DFR | 25.827 | 1,9 M | 3 | 12,7–5,3 | 38–16 min |
| A4 freeze 4/0 | 25.827 | 1,9 M | 3 | 12,7–5,3 (+5%) | 40–17 min |

Overheads por run: tokenização 1–3 min; validação 3 épocas ~1–2 min; Platt,
limiar e avaliação completa (teste 5.535 + predições do pool 85.212) 3–8 min;
escrita de artefatos 1–2 min. Setup de sessão (pip + Drive + cópia): 8–12 min.

- Mínimo viável (setup + R0 + R1 + A1 + A2): **2,5–4,5 h**
- Pacote completo (acrescenta A3 + A4): **3,5–6 h**
- Margem contra as 12 h do Colab: folgada. Recomendação: 2 runs + 1–2 ablações
  por sessão, sempre com `--drive-out` ligado.

## 10. Ablações e prioridade (decisão 10)

| prioridade | id | pergunta | custo |
|---:|---|---|---|
| 1 | R1 `bal_ood_wa` | generaliza entre canais? (portal→WhatsApp) | 17–42 min |
| 2 | A1 `full_iid`+DFR | grupos constantes ajudam ou criam atalho? | 21–49 min |
| 3 | A2 256 | truncamento (18,82% a 192) custa pior-grupo? | 18–44 min |
| 4 | A3 DFR no informativo | equalizar grupo pequeno melhora o pior-grupo? | 16–38 min |
| 5 | A4 freeze 4 / freeze 0 | capacidade do encoder importa? | 17–40 min |
| 6 | A5 class weights / A1b sem tiers LLM | mede degradação de calibração / `true` fraco | 16–38 min |

Regra de corte: se o tempo acabar depois do run 2 (R1), o entregável mínimo é
**A0 + R1** (default + OOD), que responde à pergunta do projeto. A ordem acima
é a ordem de sacrifício inversa. Cada ablação é um run independente com o mesmo
`eval_bal`; nenhuma depende de outra.

## 11. Riscos residuais e limites

1. **Quase-duplicata não persistida** na v4: sonda barata no prepare; OOD e
   pior-grupo controlam o resto. Se a sonda acusar, refazer split por cluster.
2. **PT-PT não confiável**: cortes descritivos, não manchete (seção 6.4).
3. **Throughput da T4 é hipótese**: substituir pelo smoke antes de prometer
   prazos; o ETA do script usa a medição.
4. **Colab grátis**: sessões longas podem cair; o desenho é retomável e
   idempotente.
5. **Artefatos em git**: `models/.gitignore` ignora `artifacts/` em qualquer
   nível; os parquets/splits ficam fora do Git (Drive + scripts reprodutíveis).
6. **llm_local/checker_match/corroborated** (`true` fraco): fora do default;
   composição logada em A1/A1b.
7. **CUDA não bit-exact**: seed controla splits/ordem; pequenas variações de
   loss entre execuções são esperadas.

---

## Apêndice A — Revisão iterativa v1 → v2

### A.1 O rascunho v1 (rodada 1)

Decisões do rascunho, antes da autocrítica:

1. treinar no pool completo (85.212) com pesos DFR por célula;
2. split IID 85/15 (val/teste) sobre o pool completo;
3. `max_length=256` para preservar os textos longos;
4. 2 épocas fixas, sem paciência, melhor checkpoint por macro-F1;
5. AMP fp16, batch 32, freeze 6, lr 2e-5, clip 1,0, warmup 10%;
6. Platt na validação inteira, limiar 0,5;
7. salvar artefatos só no fim da sessão;
8. Colab lendo os três CSVs originais e recomputando o split;
9. tempo de T4 extrapolado linearmente da CPU (sem conta explícita);
10. ablações: pool completo × informativos, com 256 e freeze 0.

### A.2 Críticas (12) e correções aplicadas

| # | problema no v1 | correção no v2 | onde |
|---:|---|---|---|
| C1 | **Vazamento de procedência**: 46,18% do pool é rótulo-por-origem; treinar nele e testar nele mede memorização de fonte | default passa a ser só informativos (36.896); pool completo vira A1, avaliado no teste canônico sem os grupos constantes | 1.2, 1.3, 2.2 |
| C2 | **Calibração vs balanceamento**: class weights/DFR no default distorcem a probabilidade, que é o produto do projeto | sem pesos no default; desbalanceamento resolvido pelo pool (1,08:1); DFR vira ablação com clip | 1.2, 5 |
| C3 | **Comparabilidade**: cada ablação avaliada no próprio teste, com A1 avaliando grupos que viu no treino | `eval_bal` canônico (5.535) + `eval_ood_whatsapp` (6.381); `full_iid` obrigado a excluir o teste canônico do treino | 2.1, 2.2 |
| C4 | **Custo de T4**: "~20 min/época" sem conta | conta explícita com tokens pagos medidos (1,9 M/época), faixa de throughput e tabela por run; T4 smoke mede e substitui | 9 |
| C5 | **OOM**: 256 default com batch 32 sem fallback e `best_state` na GPU | default 192; escada de OOM em 5 degraus; checkpoint do melhor em CPU; VRAM de pico logada | 3, 5 |
| C6 | **Reprodutibilidade**: só torch/numpy seed; splits recomputados no Colab | seed completo + cudnn determinístico; splits persistidos com rid e sha256; `run_config.json` e hash do script | 7, 8 |
| C7 | **Overfit val-val-test**: uma única val para seleção, calibração e limiar | 4 vias: `val_sel` (checkpoint/early stop), `val_calib` (Platt/limiar), teste intocado | 2.1, 4, 6 |
| C8 | **Queda do Colab**: artefatos só no fim; perda total da sessão | `last/` no Drive a cada época com otimizador/scheduler/scaler/RNG; `--resume` validado | 7 |
| C9 | **Duplicata/ruído**: supunha que duplicata exata zero bastava; U+FFFD ignorado | sonda de contenção 5-gram no prepare; 4 linhas U+FFFD removidas; mojibake marcado | 2.2, 8 |
| C10 | **PT-PT**: recorte tratado como métrica válida | PT-PT descritivo (critérios discordam; 445/449 true); manchete em pt-BR/canal/era | 6.4 |
| C11 | **Limiar no teste** (implícito) | 0,5 primário; otimizado só em `val_calib`; ambos persistidos | 6.2 |
| C12 | **`true` fraco da camada LLM** entrava silenciosamente no pool completo | default já o exclui (não é informativo); A1 loga `label_tier`; A1b o remove explicitamente | 1.3, 11 |

### A.3 O que mudou de forma consolidada

- Default de dados: pool completo → **informativos**; a decisão central do v2.
- Default de comprimento: 256 → **192**, com 256 promovido por medição.
- Protocolo de validação: 1 val → **val_sel + val_calib**.
- Avaliação: testes separados → **eval canônico compartilhado**.
- Salvamento: fim da sessão → **por época + retomável**.
- Tempo: extrapolação → **conta com tokens medidos + smoke obrigatório**.

### A.4 O que NÃO mudou (decisões comprovadas da v1/CPU)

`text_no_url`; padding dinâmico + agrupamento por comprimento; sem class weights
por default; melhor checkpoint por pior-grupo com fallback; Platt na validação;
clip 1,0; weight decay sem bias/LayerNorm; warmup 10%; freeze 6; AdamW lr 2e-5;
split estratificado por grupo|rótulo.

---

## Apêndice B — Números medidos e como reproduzir

Medições feitas pelo planejador em 2026-09-12, com o repositório local:

- `models.data.load(csv=FakenewsBR_sanitized_v4.csv,
  labels_csv=FakenewsBR_v4_labels.csv,
  provenance_csv=FakenewsBR_v4_provenance.csv)` → 85.212 linhas, 36.896
  informativos, 39.351 em grupos constantes;
- splits 4 vias implementados com `_safe_strat` de `models/data.py`, seed 42
  (contagens da seção 2.1, 0 interseções);
- tokens: `AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased",
  local_files_only=True)` sobre `text_no_url` (tabela da seção 3);
- custo de batching: simulação de `length_grouped_batches` (batch 32, mega 50,
  seed 42) com os comprimentos truncados; tokens pagos = soma de
  `batch_size x max_len_do_batch`; razão 256/192 = 1,145.

Reprodução pelo builder: `prepare_v4.py` (contagens e splits) e
`train_bertimbau_v4.py --token-stats` (tokens e custo). Os valores esperados
estão neste documento para conferência; divergência > 2% indica tokenizer ou
regra de dados diferente.
