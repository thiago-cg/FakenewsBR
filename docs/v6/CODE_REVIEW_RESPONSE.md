# Resposta ao code review v6 (`CODE_REVIEW.md`)

- Data da verificação: 2026-09-12 (pós-correções)
- Método: execução dos comandos de validação no repo + inspeção de código
- Estado: pipeline pronto para o smoke na T4; pendência única é o teste
  funcional do fallback de resume no Drive (só possível em sessão Colab)

## V1 (ALTO) — resume a partir do Drive

- Implementado: ao retomar, o script procura `last/` em `--drive-out` quando
  não existe em `--out`, escolhe o mais recente por
  `(epoch, global_step)` e restaura antes de continuar
  (`train_bertimbau_v6.py:1317-1327`, restauração de `global_step` em
  `:1409`).
- Validado: resume na mesma sessão (smoke reexecutado não duplica época;
  seed divergente aborta com exit 2).
- Pendente: teste do fallback do Drive fim-a-fim (exige um run não-smoke; será
  confirmado na primeira sessão Colab com `--resume`).

## V2 (ALTO) — near-duplicatas cruzando splits

- Implementado: shingles 5-gramas, MinHash K=64, LSH 16x4, verificação exata
  de Jaccard e union-find, unido aos clusters de texto exato; splits
  atribuídos por cluster com greedy estratificado por rótulo e grupo
  (`prepare_v6.py:95-305`, `:869`, `:961-1000`).
- Executado `prepare_v6.py --force`: 85.592 clusters, 4.454 multi
  (máx. 22 linhas), 473 cópias extras agrupadas, 3 conflitos de rótulo
  isolados por cluster.
- Varredura completa pós-split: **0 pares Jaccard>=0,8 cruzando
  treino x teste e treino x val** nos três splits (máximos observados:
  0,798 full_iid, 0,774 ood_wa, 0,798 bal_iid).
- Splits resultantes (train/val_sel/val_calib/test):
  - `full_iid`: 63.756 / 9.108 / 4.554 / 13.662
  - `ood_wa`: 71.986 / 8.468 / 4.235 / 6.381 (+10 unused)
  - `bal_iid`: 25.827 / 3.689 / 1.845 / 5.535

## V3 (ALTO) — workers no Windows

- Implementado: `TrainCollator`/`EvalCollator` em nível de módulo
  (`train_bertimbau_v6.py:429`, `:440`) — pickláveis.
- Validado: smoke com `--workers 2` no Windows, exit 0.

## V4 (MEDIO) — limiar no bloco secundário

- Implementado: `test_at_val_threshold` no `metrics.json` (`:1786`); no smoke,
  os blocos de limiar 0,5 e de limiar ótimo imprimem métricas distintas.

## V5 (MEDIO) — preset classweights

- Implementado: `"classweights": {"class_weights": True, "dfr_weights": "off"}`
  (`:712`).

## Baixos

- V7 (n_test): o split registra 13.662; o trainer remove 1 linha `has_ufffd`
  do teste -> 13.661 (mesmo tratamento do treino, que remove 3). Documentado.
- V8 (tabela de tokens): `token_stats.json` atual — pool média 69,78 /
  p50 27 / p95 293; informativos 119,65 / 33 / 453, >192 = 18,82%;
  custo `full_iid` 192 = 3,231M tokens/época; razão 256/192 = 1,1119.
- V9 (probe em CPU): guarda explícita com aviso e saída, sem treinar
  (`:1277-1281`).
- V10 (resume x drop_tiers): `drop_tiers` incluído na validação de
  compatibilidade (`:1368`).
- V11 (global_step): restaurado no resume (`:1409`, `:1440`).
- V13 (JSON com NaN/Inf): `jsonable` converte para `None`; smoke sem NaN.
- V14 (priors): `train_prior_fake` e `effective_prior_fake` no
  `metrics.json` (`:1706`, `:1780`) — no `full_iid`+DFR: prior efetivo 0,603.
- V15 (ECE pós-Platt no smoke): esperado em subconjunto de 1.024 linhas; nos
  runs cheios a calibração é medida em `val_calib` e reportada antes/depois.

## Evidência de execução (pós-correções)

- `prepare_v6.py --force --token-stats`: `all_pass=True`, 0 falhas.
- `--check-vendor`: PASSOU (sha256 idênticos; diferenças 0 em float64 e
  <=1e-9 em float32; `report` stdout idêntico).
- Smoke CPU `full_iid` (1.024 amostras, 1 época): exit 0 em 371 s,
  17/17 artefatos, 242 tokens/s pagos.
- Smoke CPU `--workers 2`: exit 0.

## O que permanece como hipótese (não verificável localmente)

- Throughput/VRAM reais na T4 (smoke T4 substitui a projeção [10,30]x).
- Fallback de resume via Drive (V1) e limites de VRAM de 32x192/32x256.
