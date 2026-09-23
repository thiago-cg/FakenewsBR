# Melhorias e Expansão do FakenewsBR — versão corrigida (2026-09-10)

> Substitui integralmente a versão anterior deste documento, que continha
> erros factuais (atribuição de métrica, tese de dialeto, números de cobertura
> e projeções de volume). Todos os números aqui foram **medidos** no repositório,
> exceto onde marcado como *estimativa*.

## TL;DR

1. O **0,8655 de acurácia não é** média/seleção de ERM/DFR: é o **classificador
   FT** (cabeça `classifier` treinada junto do encoder BERTimbau, com Platt
   ajustado na validação cheia). ERM = 0,8451; DFR = 0,7345.
2. O FT **também cai na armadilha do prior**: 71,6 % do teste é `fake`. Recalibrar
   o Platt só na validação balanceada melhora o subset `true` (0,19 → 0,64) e o
   ECE balanceado (0,049 → 0,028) **sem retreino**, ao custo de acurácia global
   (0,8655 → 0,8331).
3. Há **vazamento por quase-duplicata** no canal WhatsApp: 41,4 % do teste de
   `FakeWhatsApp.BR_2018` tem quase-duplicata no treino. Com quase-dup: FT
   0,869 / AUC 0,947; sem: **0,661 / AUC 0,752**.
4. A tese de "BERTimbau perde 22 pontos por dialeto" **não se sustenta** com a
   análise corrigida (ver §3). PT-PT está quase todo dentro dos subsets
   degenerados e, controlando o checador, a AUC é comparável.
5. O dataset v1 tem **58,4 % de linhas em grupos degenerados** (uma única
   classe) e **210 linhas com rating contradizendo o rótulo** (39 no teste).
6. A expansão v2 mantém a v1 **intacta** e adiciona linhas de checadores e de
   procedência por manchete, com dedup exata + quase-duplicata. Composição
   medida em `investigation/expansion/audit_v2.md`.

---

## 1. O que o `FakenewsBR_sanitized.csv` (v1) é

- 39.466 linhas, 23 colunas (ordem fixa), `rid` int64 0…51.204.
- `label ∈ {fake, true}`; `dataset_name` identifica 11 origens.
- Em **58,4 %** da base o grupo determina o rótulo (grupos `fakes`, `true`,
  `MuMiN-PT`): são grupos **degenerados**, inúteis para treinar a cabeça de
  decisão e úteis apenas como avaliação OOD.
- A fração `fake` global é **71,5 %** (esse é o número que antes estava
  atribuído erroneamente a "grupos degenerados").
- Ratings "hard" (Enganoso, Distorcido, Sem contexto, Exagerado…) são tratados
  como `fake` (`models/data.py::_rating_class`). Ratings no formato
  `"FALSO: explicação…"` precisam ter o token antes de `:`/`-` extraído.

### 1.1 Vazamentos e atalhos conhecidos

- **Proveniência**: em quase 6 de 10 linhas, saber a origem basta para acertar o
  rótulo. É o maior vazamento do projeto — daí o split estratificado por
  (grupo × rótulo) e o DFR.
- **Comprimento**: no WhatsApp, a acurácia do FT sobe monotonicamente com o
  tamanho (≤10 palavras 0,37 / AUC 0,28, n=35; 11–20 0,65; >160 0,90). Mensagens
  curtas são o buraco real do canal. Por isso a expansão usa **manchete como
  texto** para linhas de procedência (tamanho ≈ alegação de checador).
- **Quase-duplicata** (§4).

---

## 2. Análise reversa dos modelos (split IID seed 42, teste n=5.920)

Três leitores do mesmo encoder fine-tuned:

| leitor | o que é | acc | macro-F1 | AUC |
|---|---|---:|---:|---:|
| FT  | cabeça treinada com o encoder; Platt na val val completa | **0,8655** | 0,8214 | 0,9162 |
| ERM | logística sobre `embeddings_ft.npy` (mean-pool) | 0,8451 | 0,8136 | 0,9069 |
| DFR | logística só em grupos balanceados, ponderada | 0,7345 | 0,7172 | 0,8608 |

A reprodução bate exatamente com os logs, confirmando splits idênticos.
Artefatos: `investigation/reverse_analysis/output/reverse_analysis.json`,
`erros_confiantes.csv`, `models/artifacts/preds_ft_classifier.csv`.

### 2.1 Calibração do FT (melhoria gratuita, sem retreino)

| Platt ajustado em | acc subset `true` | pior-grupo | ECE balanceado | acc global |
|---|---:|---:|---:|---:|
| val cheia | 0,190 | 0,7276 | 0,0494 | 0,8655 |
| **val balanceada** (a=0,648, b=−0,764) | **0,638** | **0,7325** | **0,0279** | 0,8331 |

Recomendação: usar a calibração balanceada no score, documentando a troca de
acurácia global por robustez no subset minoritário.

---

## 3. Dialeto PT-PT — conclusão corrigida

A leitura anterior ("BERTimbau perde 22 pontos por dialeto") era artefato de
composição: **7.128 das 7.144 linhas PT-PT estão nos grupos degenerados**
`fakes`/`true` (teste: `fakes` 835 + `true` 242). Comparações válidas:

- Dentro de `fakes`: PT-BR 0,9756 × PT-PT 0,9413 (queda de 3,4 pp).
- Dentro de `true`: PT-BR 0,061 × PT-PT 0,2769 (FT vai **melhor** em PT-PT).
- **Controlado por checador (AUC)** — a comparação correta:

| checador | dialeto | FT acc | FT AUC | n |
|---|---|---:|---:|---:|
| poligrafo.sapo.pt | PT-PT | 0,7048 | 0,708 | 752 |
| lupa.uol.com.br | PT-BR | 0,8134 | 0,683 | 743 |
| politica.estadao.com.br | PT-BR | 0,9821 | 0,531 | 336 |
| observador.pt | PT-PT | 0,9930 | — | 286 |
| checamos.afp.com | PT-BR | 0,9914 | 0,774 | 232 |

**Conclusão:** não há evidência de penalidade de dialeto uma vez controlada a
proveniência. O baixo desempenho em Polígrafo reflete o estilo das alegações
(perguntas curtas, política portuguesa) e o desbalanceamento do checador, não o
tokenizer. Trocar para XLM-R **não** é "o maior retorno" com base nesses dados.

---

## 4. Vazamento por quase-duplicata (TF-IDF char 4–5, cosseno ≥ 0,90)

Percentual do **teste** com quase-duplicata no treino:

| grupo | % quase-dup |
|---|---:|
| FakeWhatsApp.BR_2018 | **41,4** |
| COVID19.BR | 11,4 |
| MuMiN-PT | 5,9 |
| fakes | 4,2 |
| Fake.br | 0,3 (o 0,986 do Fake.br **não** é vazamento) |

Impacto em `FakeWhatsApp.BR_2018`: FT 0,869 / AUC 0,947 **com** quase-dup ×
0,661 / AUC 0,752 **sem**. A métrica de WhatsApp está inflada; o split IID
precisa agrupar quase-duplicatas (**GroupKFold por cluster**) numa próxima
avaliação. A expansão v2 já implementa esse dedup para as linhas novas.

---

## 5. Ruído de rótulo

- **210 linhas** da v1 têm `factcheck_rating` contradizendo o `label`
  (por grupo: `fakes` 129, `true` 63, WhatsApp 6, COVID19.BR_raw 5, COVID19.BR 4,
  MuMiN-PT 3); **39 no teste**. Piso de erro de ~1 pp nas métricas.
- Ratings "Boato", "Pimenta na Língua", "Manipulado", "Descontextualizado" não
  existem na v1 e podem ser adicionados aos conjuntos de `data.py` sem alterar
  métricas da v1.

---

## 6. Expansão v2 — desenho implementado

Pipeline em `investigation/expansion/`:

```
schema.py         23 colunas + proveniência; ratings -> {false_pure,hard,true_rating,other};
                  outras classes de rating (Explica, Contextualizando, Indeterminado…) são descartadas;
                  derivações/métricas idênticas à v1 (paridade testada); rid=sha1(url|role|i)
http.py           PoliteSession: robots.txt, opt-out de IA, 1 req/s/host, backoff 429/5xx
sources.py        registro das fontes + whitelist do feed
collect_wp.py     WP REST (checadores e portais), estratificado por ano
collect_sitemap.py G1 (/fato-ou-fake/) e Polígrafo (fact_check-sitemap*)
collect_feed.py   feed ClaimReview (Data Commons) -> PT por whitelist
collect_gfc.py    Google Fact Check Tools `claims:search` (opcional; exige chave)
extract.py        regras por fonte -> registros canônicos + proveniência
dedup.py          exata + MinHash LSH; conflitos em conflicts_v2.csv
merge_and_audit.py v2 = v1 intacta + novas (dedup, cota NEWS por era) + auditoria
llm_batch.py      OpenRouter Batch (`openai/gpt-5-nano:batch`), dry-run/custo/resume
langchain_agent.py planejador (`--planner rules|llm`)
pipeline.py       CLI: plan | collect | extract | merge | llm-prepare|submit|collect
tests/            unittest offline (27 testes)
```

Política ética: **não** se faz scraping de conteúdo de sites cujo robots.txt
bloqueia robôs de treino de IA (GPTBot/ClaudeBot/CCBot/Google-Extended). Esses
publishers entram apenas via metadado ClaimReview do feed (Lupa, Comprova, Aos
Fatos, Estadão Verifica, Observador, CartaCapital, SAPO…). AFP e UOL devolvem 403.

### 6.1 Fatos que corrigem a versão anterior

- `pages` (o endpoint do link original) é **CRUD do ClaimReview da própria
  organização** (OAuth `factchecktools`) e **não lê** checagens de terceiros.
  A leitura correta é `claims:search` (com `reviewPublisherSiteFilter`).
- O feed ClaimReview PT tem **8.706 itens** (não "50–80 mil").
- RSS só entrega os itens mais recentes; histórico vem de WP REST/sitemaps.
- Wayback "earliest snapshot" **não** é data de publicação — desnecessário, pois
  WP REST e sitemaps trazem a data real.
- O LLM passou de gemini-2.5-flash-lite/Google AI Studio para
  **`openai/gpt-5-nano:batch`** no OpenRouter (decisão do usuário).

### 6.2 Composição medida da v2

Resultado completo e atualizado em `investigation/expansion/audit_v2.md`
(gerado por `merge_and_audit`). A v1 é verificada por hash de conteúdo a cada
merge; a v2 nunca reintroduz linhas de procedência como se fossem veredito.

DoD revisado (o antigo "fake/true 1,2:1–1,8:1" é inviável com checadores, que
são ~93 % "falso" no feed, e irrelevante):
1. `FakenewsBR_sanitized_v2.csv` com v1 intacta e ≥ 200.000 linhas.
2. Cobertura por era: ≥ 30k ≤2017, ≥ 80k 2018–nov/2022, ≥ 80k dez/2022+.
3. PT-PT ≥ 10 % do total (medido: ~40 %, dominado por ECO — ver auditoria).
4. Balanceamento **por grupo e por era**, não global.
5. Re-treino (se pedido) com split que agrupa quase-duplicatas; comparar
   pior-grupo do DFR e AUC por checador.

## 7. LLM (rotulagem/extração)

- Modelo: `openai/gpt-5-nano:batch` (OpenRouter Batch, ~50 % do preço;
  0,025/0,20 US$ por 1M tokens). Sem `temperature`; `reasoning.effort=minimal`;
  `response_format` com `json_schema` estrito.
- Uso: `llm_batch.py` monta requests (verdict_claim para artigos sem taxonomia;
  correction para frases de correção verbatim), com `custom_id` estável,
  `--dry-run`/custo e estado em `processed/llm_batches.json`.
- Submissão exige `OPENROUTER_API_KEY` definida pelo usuário no ambiente
  (nunca no chat).

---

## 8. Erros da versão anterior deste documento (corrigidos)

- Atribuição do 0,8655 ao ERM/DFR → é o FT.
- "71,5 % da base é degenerada" → 58,4 % (71,5 % é a fração fake).
- Tese de dialeto/tokenizer e recomendação de XLM-R → sem suporte (§3).
- "val macro-F1 estagnado" → subiu entre as épocas 1 e 2 (0,8149 → 0,8296).
- Números de cobertura do feed, Fase 1 via `pages`, RSS 60–180 mil/ano,
  DoD falsos, Wayback como data.
- "curadoria do Mavis", caracteres chineses e outras afirmações não verificáveis
  foram removidos.
