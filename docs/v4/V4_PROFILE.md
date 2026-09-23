# Perfil do dataset FakenewsBR v4

Gerado por `models/v4/explore_v4.py` em 2026-09-12T15:19:33.318109+00:00 (tempo total: 80.6s). Somente leitura; nenhum arquivo do projeto foi alterado.

## Resumo executivo

- Pool treinavel (regra `models/data.py` com labels v4): **85,212 linhas** — fake 60,991 / true 24,221 (71.58% fake; razao fake:true 2.52).
- `sanitized_v4`: 291,521 linhas x 23 colunas; `rid` unico (duplicados: 0).
- `labels_v4`: 291,521 linhas; `rid` unico; train_label NaN em 206,309 (camada `provenance`).
- Grupos informativos (prefixo FC_/EXT_/NEWS_, n>=200, minoria>=15%): **3**; grupos v1 balanceados presentes: 5.
- `NEWS_*` (canal press_true) fora do treino: **206,309 de 209,456** linhas (98.5%).
- Duplicatas normalizadas no pool: 0 linhas em grupos duplicados (0.0%); textos com fake E true: **0** (0 linhas).
- PT-PT no pool: 449 linhas com lang_variant pt-PT; a regra de URL `is_ptpt` do data.py marca 18,420 (criterios discordam; ver secao 9).
- Exemplos acima de 512 palavras (truncamento BERT): **776** (0.91%).

## 1. Arquivos e colunas

| arquivo | MB | linhas fisicas | linhas CSV (parse) | leitura (s) |
|---|---|---|---|---|
| sanitized | 210.6 | 526277 | 291521 | 64.2 |
| labels | 56.0 | 409596 | 291521 | 3.2 |
| provenance | 49.0 | 252056 | 252055 | 2.1 |

Linhas fisicas > linhas parseadas indicam `\n` embutido em campos de texto (esperado nos 3 arquivos).

Colunas de texto presentes: `text` (291,521 nao nulos, 0 vazios), `text_clean` (291,521 nao nulos, 0 vazios), `text_no_url` (291,521 nao nulos, 0 vazios).

| coluna | dtype | nulos | nulos % |
|---|---|---|---|
| rid | int64 | 0 | 0.0 |
| dataset_name | str | 0 | 0.0 |
| source_type | str | 0 | 0.0 |
| source_description | str | 0 | 0.0 |
| label | str | 0 | 0.0 |
| date_iso | str | 20,171 | 6.92 |
| url_review | str | 23,284 | 7.99 |
| text | str | 0 | 0.0 |
| text_clean | str | 0 | 0.0 |
| text_no_url | str | 0 | 0.0 |
| extracted_urls | str | 0 | 0.0 |
| is_duplicated | int64 | 0 | 0.0 |
| is_null | int64 | 0 | 0.0 |
| too_short | int64 | 0 | 0.0 |
| factcheck_rating | str | 241,086 | 82.7 |
| factcheck_claimant | str | 271,097 | 92.99 |
| factcheck_url | str | 38,626 | 13.25 |
| char_len | int64 | 0 | 0.0 |
| word_len | int64 | 0 | 0.0 |
| num_exclamations | int64 | 0 | 0.0 |
| num_questions | int64 | 0 | 0.0 |
| num_ellipsis | int64 | 0 | 0.0 |
| uppercase_word_ratio | float64 | 0 | 0.0 |

## 2. Merge sanitized x labels

| metrica | valor |
|---|---|
| linhas sanitized | 291,521 |
| rids unicos sanitized | 291,521 |
| linhas labels | 291,521 |
| rids labels ausentes no sanitized | 0 |
| rids sanitized sem linha em labels | 0 |
| train_label fake | 60,991 |
| train_label true | 24,221 |
| train_label NaN | 206,309 |
| train_label 'True' (maiusculo) | 0 |
| linhas com train_label fake/true | 85,212 |
|   descartadas por texto vazio/nulo | 0 |
| pool final | 85,212 |

Quebra por `label_tier` (linhas do sanitized apos merge):

| label_tier | n | fake | true | train_label NaN |
|---|---|---|---|---|
| provenance | 206,309 | 0 | 0 | 206,309 |
| v1 | 39,466 | 28,236 | 11,230 | 0 |
| checker | 38,526 | 28,717 | 9,809 | 0 |
| checker_match | 4,119 | 4,038 | 81 | 0 |
| llm_local | 3,092 | 0 | 3,092 | 0 |
| corroborated | 9 | 0 | 9 | 0 |

Divergencias rotulo original x train_label (casefold, dentro do pool): **0** (0.0% do pool).

## 3. Pool treinavel: grupos e canais

Top 30 grupos por n (o pool tem 27 grupos; lista completa no JSON):

| dataset_name | canal | n | fake | true | %fake | minoria | informativo |
|---|---|---|---|---|---|---|---|
| fakes | agency_claim | 20,347 | 20,347 | 0 | 100.0 | 0.0 | - |
| FC_POLIGRAFO | agency_claim | 10,831 | 6,997 | 3,834 | 64.6 | 0.354 | sim |
| FC_BOATOS | agency_claim | 10,774 | 10,774 | 0 | 100.0 | 0.0 | - |
| Fake.br | portal | 7,160 | 3,580 | 3,580 | 50.0 | 0.5 | v1 |
| EXT_LIARBR | external_claim | 6,996 | 2,495 | 4,501 | 35.66 | 0.3566 | sim |
| FakeWhatsApp.BR_2018 | whatsapp | 6,381 | 3,055 | 3,326 | 47.88 | 0.4788 | v1 |
| FC_EFARSAS | agency_claim | 3,208 | 2,777 | 431 | 86.56 | 0.1344 | - |
| FC_G1 | agency_claim | 3,187 | 3,075 | 112 | 96.49 | 0.0351 | - |
| EXT_AVERITECBR | external_claim | 2,925 | 2,023 | 902 | 69.16 | 0.3084 | sim |
| true | press_true | 2,710 | 0 | 2,710 | 0.0 | 0.0 | - |
| FC_BOATOS_VIRAL | social | 2,360 | 2,360 | 0 | 100.0 | 0.0 | - |
| COVID19.BR | covid | 1,931 | 820 | 1,111 | 42.47 | 0.4247 | v1 |
| FC_LUPA | agency_claim | 1,437 | 1,418 | 19 | 98.68 | 0.0132 | - |
| NEWS_PODER360 | press_true | 1,334 | 0 | 1,334 | 0.0 | 0.0 | - |
| NEWS_BRASILDEFATO | press_true | 900 | 0 | 900 | 0.0 | 0.0 | - |
| NEWS_OECO | press_true | 468 | 0 | 468 | 0.0 | 0.0 | - |
| NEWS_ECO | press_true | 445 | 0 | 445 | 0.0 | 0.0 | - |
| COVID19.BR_raw | covid | 373 | 59 | 314 | 15.82 | 0.1582 | v1 |
| LLM4BR_300 | llm | 299 | 148 | 151 | 49.5 | 0.495 | v1 |
| FC_COMPROVA | agency_claim | 249 | 242 | 7 | 97.19 | 0.0281 | - |
| MuMiN-PT | social | 222 | 203 | 19 | 91.44 | 0.0856 | - |
| FC_FOLHA | agency_claim | 211 | 208 | 3 | 98.58 | 0.0142 | - |
| FC_SBT | agency_claim | 210 | 204 | 6 | 97.14 | 0.0286 | - |
| FC_BEREIA | agency_claim | 202 | 173 | 29 | 85.64 | 0.1436 | - |
| Fake.br_raw | portal | 39 | 20 | 19 | 51.28 | 0.4872 | - |
| FC_NEXO | agency_claim | 9 | 9 | 0 | 100.0 | 0.0 | - |
| MuMiN-PT_raw | social | 4 | 4 | 0 | 100.0 | 0.0 | - |

Distribuicao por canal:

| canal | n | fake | true | %fake |
|---|---|---|---|---|
| agency_claim | 50,665 | 46,224 | 4,441 | 91.23 |
| external_claim | 9,921 | 4,518 | 5,403 | 45.54 |
| portal | 7,199 | 3,600 | 3,599 | 50.01 |
| whatsapp | 6,381 | 3,055 | 3,326 | 47.88 |
| press_true | 5,857 | 0 | 5,857 | 0.0 |
| social | 2,586 | 2,567 | 19 | 99.27 |
| covid | 2,304 | 879 | 1,425 | 38.15 |
| llm | 299 | 148 | 151 | 49.5 |

Grupos com prefixo FC_/EXT_/NEWS_ (todos):

| grupo | canal | n | %fake | minoria | passa criterio |
|---|---|---|---|---|---|
| FC_POLIGRAFO | agency_claim | 10,831 | 64.6 | 0.354 | SIM |
| FC_BOATOS | agency_claim | 10,774 | 100.0 | 0.0 | nao |
| EXT_LIARBR | external_claim | 6,996 | 35.66 | 0.3566 | SIM |
| FC_EFARSAS | agency_claim | 3,208 | 86.56 | 0.1344 | nao |
| FC_G1 | agency_claim | 3,187 | 96.49 | 0.0351 | nao |
| EXT_AVERITECBR | external_claim | 2,925 | 69.16 | 0.3084 | SIM |
| FC_BOATOS_VIRAL | social | 2,360 | 100.0 | 0.0 | nao |
| FC_LUPA | agency_claim | 1,437 | 98.68 | 0.0132 | nao |
| NEWS_PODER360 | press_true | 1,334 | 0.0 | 0.0 | nao |
| NEWS_BRASILDEFATO | press_true | 900 | 0.0 | 0.0 | nao |
| NEWS_OECO | press_true | 468 | 0.0 | 0.0 | nao |
| NEWS_ECO | press_true | 445 | 0.0 | 0.0 | nao |
| FC_COMPROVA | agency_claim | 249 | 97.19 | 0.0281 | nao |
| FC_FOLHA | agency_claim | 211 | 98.58 | 0.0142 | nao |
| FC_SBT | agency_claim | 210 | 97.14 | 0.0286 | nao |
| FC_BEREIA | agency_claim | 202 | 85.64 | 0.1436 | nao |
| FC_NEXO | agency_claim | 9 | 100.0 | 0.0 | nao |

Grupos informativos que entram no treino (3): `FC_POLIGRAFO` (10,831, 64.6% fake), `EXT_LIARBR` (6,996, 35.66% fake), `EXT_AVERITECBR` (2,925, 69.16% fake).

## 4. Comprimento (palavras em text_no_url)

| classe | n | media | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| fake | 60,991 | 42.7 | 17.0 | 93.0 | 190.0 | 471.0 | 4859 |
| true | 24,221 | 58.8 | 18.0 | 162.0 | 244.0 | 530.0 | 5938 |
| pool | 85,212 | 47.3 | 17.0 | 121.9 | 209.0 | 490.0 | 5938 |

Acima de 512 palavras: **776** (fake 508, true 268); acima de 510: 784; acima de 400: 1,287.

Conferencia da coluna `word_len`: 0 nulos, 2,634 linhas divergentes do texto (>2: 524), corr=0.9999.

## 5. Duplicatas e conflitos

| metrica | valor |
|---|---|
| textos unicos (norm.) | 85,212 |
| linhas duplicadas (total) | 0 (0.0%) |
| duplicatas exatas (cru) | 0 |
| textos com fake E true | 0 |
| linhas nesses textos conflitantes | 0 (0.0%) |

No sanitized completo: 291,521 textos normalizados unicos, 0 linhas duplicadas extras (0.0%).

## 6. Proveniencia no pool (lang, publisher, era)

Cobertura de `provenance` no pool: 45,746 / 85,212 (53.68%).

Idioma (`lang_variant`):

| lang_variant | n | fake | true | %fake |
|---|---|---|---|---|
| pt-BR | 45,297 | 32,751 | 12,546 | 72.3 |
| <sem_prov> | 39,466 | 28,236 | 11,230 | 71.55 |
| pt-PT | 449 | 4 | 445 | 0.89 |

Publisher (top 15):

| publisher | n | fake | true | %fake |
|---|---|---|---|---|
| <sem_prov> | 39,466 | 28,236 | 11,230 | 71.55 |
| boatos | 11,499 | 11,499 | 0 | 100.0 |
| poligrafo | 10,831 | 6,997 | 3,834 | 64.6 |
| ext_liarbr | 6,996 | 2,495 | 4,501 | 35.66 |
| efarsas | 3,208 | 2,777 | 431 | 86.56 |
| g1 | 3,187 | 3,075 | 112 | 96.49 |
| ext_averitecbr | 2,925 | 2,023 | 902 | 69.16 |
| boatos.org | 1,635 | 1,635 | 0 | 100.0 |
| lupa | 1,437 | 1,418 | 19 | 98.68 |
| poder360 | 1,334 | 0 | 1,334 | 0.0 |
| brasildefato | 900 | 0 | 900 | 0.0 |
| oeco | 468 | 0 | 468 | 0.0 |
| eco | 445 | 0 | 445 | 0.0 |
| comprova | 249 | 242 | 7 | 97.19 |
| folha | 211 | 208 | 3 | 98.58 |

Regra URL PT-PT do `data.py` marca 18,420 linhas do pool; dessas, a provenance diz pt-PT em 449, pt-BR em 10,827 e nao cobre 7,144. Os dois criterios discordam.

Eras temporais (`date_iso` do sanitized):

| era | n | fake | true | %fake |
|---|---|---|---|---|
| <=2017 | 7,553 | 4,843 | 2,710 | 64.12 |
| 2018-2022 | 42,106 | 32,827 | 9,279 | 77.96 |
| >=2023 | 15,382 | 12,903 | 2,479 | 83.88 |
| <sem_data> | 20,171 | 10,418 | 9,753 | 51.65 |

Por ano: 1920: 1; 1921: 1; 1922: 4; 2004: 4; 2005: 2; 2006: 2; 2007: 2; 2008: 8; 2009: 15; 2010: 72; 2011: 114; 2012: 184; 2013: 342; 2014: 757; 2015: 700; 2016: 2,160; 2017: 3,185; 2018: 10,591; 2019: 6,669; 2020: 10,102; 2021: 7,566; 2022: 7,178; 2023: 4,345; 2024: 4,295; 2025: 3,890; 2026: 2,852.

## 7. Vazamento por procedencia (rotulo constante)

- Grupos (dataset_name): **10 de 27** (37.04%) tem rotulo constante; essas linhas sao **39,351** (46.18% do pool).
- Canais com rotulo constante: press_true.

Maiores grupos constantes:

| grupo | n | rotulo unico | canal |
|---|---|---|---|
| fakes | 20,347 | fake | agency_claim |
| FC_BOATOS | 10,774 | fake | agency_claim |
| true | 2,710 | true | press_true |
| FC_BOATOS_VIRAL | 2,360 | fake | social |
| NEWS_PODER360 | 1,334 | true | press_true |
| NEWS_BRASILDEFATO | 900 | true | press_true |
| NEWS_OECO | 468 | true | press_true |
| NEWS_ECO | 445 | true | press_true |
| FC_NEXO | 9 | fake | agency_claim |
| MuMiN-PT_raw | 4 | fake | social |

## 8. Metadados uteis para avaliacao/calibracao

| coluna | fonte | cardinalidade | linhas no pool |
|---|---|---|---|
| publisher | provenance | 17 | 45,746 |
| lang_variant | provenance | 2 | 45,746 |
| era (de date_iso) | sanitized | 4 | 65,041 |
| channel (channel_of) | derived | 8 | 85,212 |
| dataset_name | sanitized | 27 | 85,212 |
| label_tier | labels | 6 | 85,212 |
| label_source (labels) | labels | 3 | 85,212 |

## 9. Achados criticos / surpresas

1. **Encoding corrompido em parte do texto**: `text_no_url` tem 4 linhas com U+FFFD e 50 com caracteres Latin Extended-A/B fora do portugues (mojibake); ex.: 'Paraná Pesquisas: Bolsonaro chega a quase 70�os votos válidos em SP -' / 'Paraná Pesquisas: Bolsonaro chega a quase 70�dos votos válidos em SP -'.
2. **Vazamento de procedencia**: 46.18% do pool esta em grupos com rotulo constante (dataset_name). Amostragem aleatoria IID mede memorizacao de origem, nao detecao.
3. **Duplicatas**: 0 linhas duplicadas (0.0%) e 0 textos com fake E true (0 linhas) — o sanitizer ja deduplicou tudo (flag `is_duplicated`=0 nas 291.521 linhas), entao nao ha vazamento treino/teste por texto repetido.
4. **Desbalanceamento**: pool 71.58% fake (razao 2.52:1) e 776 linhas acima de 512 palavras serao truncadas pelo BERTimbau.
5. **NEWS_* fora do treino**: 206,309 de 209,456 linhas do canal press_true nao tem train_label e ficam fora do pool.
6. **`is_ptpt` x `lang_variant` discordam**: a regra de URL do `data.py` marca 18,420 linhas do pool como PT-PT, mas a provenance registra so 449 pt-PT (e 10,827 pt-BR nessas mesmas linhas). Avaliar o dominio de PT-PT fica ambiguo.
7. **NEWS_* parcialmente no treino**: 3,147 manchetes `NEWS_*` tem train_label e entram no pool (NEWS_PODER360 1,334; NEWS_BRASILDEFATO 900; NEWS_OECO 468; NEWS_ECO 445), todas da classe true — grupos constantes que nao ensinam a fronteira fake/true.

## 10. Reprodutibilidade

```
python models/v4/explore_v4.py
```

- Entrada: `FakenewsBR_sanitized_v4.csv`, `FakenewsBR_v4_labels.csv`, `FakenewsBR_v4_provenance.csv`.
- Saida: `models/v4/v4_profile.json`, `models/v4/V4_PROFILE.md`.
- Nao importa `models.data`; regras replicadas no proprio script.
- Tempos: passada completa 64.2s, total 80.6s.
