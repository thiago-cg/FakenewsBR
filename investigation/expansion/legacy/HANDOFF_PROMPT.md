# Prompt de continuação — goal "melhorias + expansão do FakenewsBR"

> Cole tudo abaixo da linha em uma nova sessão do Claude Code aberta em
> `C:\Users\tito\OneDrive\Documentos\Projetos\FakenewsBR`. Escrito em 2026-09-10
> ao fim da sessão anterior. Todos os números aqui foram medidos (não estimados),
> exceto onde está escrito "estimativa".

---

## 0. Objetivo (texto original do usuário)

> encontre melhorias possíveis no dataset trabalhado. faça uma análise reversa dos
> resultados dos modelos treinados (foco no finetuning do BERTimbau) RELATORIO.md
> RELATORIO_DADOS.md FakenewsBR_sanitized.csv
>
> eu tinha pensado em aumentar o dataset drasticamente com um script de web scraping
> + agente de IA em langchain para fazer uma varredura maior da internet em tipos de
> notícias e datas sub-representadas, com a api de fact check da Google
> (https://developers.google.com/fact-check/tools/api/reference/rest/v1alpha1/pages?hl=pt-br)
> para rotular a notícia como fato ou fake, que seja compatível com o formato do
> dataset original.
>
> Adicionar em torno de 200k a 300k de linhas, com bom volume antes e pós IA
> generativa (lançamento do gpt 3.5 em diante).
>
> muita coisa já foi iniciada em outra sessão, conclua o que foi executado deste goal

Responda sempre em PT-BR com acentuação correta. Não faça commit/push sem pedido.

## 1. Decisões já tomadas pelo usuário (não perguntar de novo)

1. **LLM do rotulador**: `openai/gpt-5-nano:batch` via **OpenRouter Batch API**
   (o usuário escolheu explicitamente, depois de sugerir gemini-2.5-flash-lite:batch).
2. **Download autorizado e concluído** do feed ClaimReview do Google/Data Commons
   → `investigation/expansion/raw/datacommons_claimreview.json` (199,6 MB,
   Last-Modified 2026-09-10, sha256 prefixo `a0df9c17cf0acad0`).
3. Nenhuma chave está no ambiente: `OPENROUTER_API_KEY`, `GOOGLE_FACTCHECK_API_KEY`,
   `GOOGLE_API_KEY` = unset (usuário e sistema). Peça ao usuário para **definir a
   variável de ambiente ele mesmo**; nunca peça para colar chave no chat.
4. Dois vereditos externos foram colados pelo usuário: um PASS e um **FAIL**. O FAIL
   está certo: o entregável central (200–300 mil linhas) não existe. Não existe
   `FakenewsBR_sanitized_v2.csv`, nem `raw/`/`processed/` com dados de expansão
   (só o feed baixado).

## 2. Ambiente

- Windows 11, Python 3.13, CPU-only para treino (ver memória `hardware-constraints`).
- Instalado nesta sessão: `langchain 1.4.0`, `langchain-openai 1.6.2`,
  `langchain-core 1.6.2` (`from langchain.agents import create_agent` funciona).
- Disponíveis: `requests 2.34`, `httpx`, `pandas`, `numpy`, `sklearn 1.9`,
  `onnxruntime 1.24`, `torch`/`transformers`.
- **Ausentes**: bs4, lxml, trafilatura, feedparser, pytest, sentence-transformers,
  faiss → usar regex/`html.parser`/`urllib.robotparser` e `unittest` (ou instalar).
- No Bash tool, use `PYTHONIOENCODING=utf-8` para imprimir acentos (sem isso o
  console mostra `�` — **não é corrupção do CSV**; medido: só 2 linhas com U+FFFD).
- **WebFetch e WebSearch estavam quebrados** (erro de modelo interno). Verifique
  fatos externos com `requests` direto nas fontes primárias.
- Cópia local da doc completa do OpenRouter (3,9 MB), se ainda existir:
  `C:\Users\tito\AppData\Local\Temp\claude\C--Users-tito-OneDrive-Documentos-Projetos-FakenewsBR\0df0fb1c-d6e3-42eb-ba54-825dccfcbbaf\scratchpad\openrouter_llms_full.txt`
  (seção "# Batch API Quickstart" ~linha 13746). Senão: `https://openrouter.ai/docs/llms-full.txt`.

## 3. Arquivos relevantes

| caminho | estado |
|---|---|
| `FakenewsBR_sanitized.csv` | v1, 39.466 linhas, 23 colunas. **Nunca modificar.** |
| `FakenewsBR_factchecked.csv` | bruto (rastreado no git) |
| `sanitize_dataset.py` | gerou a v1 (métricas de estilo; ver §6) |
| `models/` | pipeline do score (data, encoder, embed, score, evaluate) — rastreado |
| `models/artifacts/` | gitignored; modelo FT, embeddings, logs |
| `models/artifacts/preds_ft_classifier.csv` | **novo**: logits+p calibrado do classificador FT em val+teste (por rid) |
| `investigation/reverse_analysis/predict_ft.py` | **novo**, rodou OK, reproduziu acc 0,8655 |
| `investigation/reverse_analysis/analyze.py` | **novo**; bug de alinhamento **já corrigido** no `by()` — **precisa rodar de novo** |
| `investigation/reverse_analysis/output/` | `reverse_analysis.json` e `erros_confiantes.csv` da execução com bug (seções de dialeto/AUC/interrogação inválidas) |
| `MELHORIAS_DATASET.md` | documento da sessão anterior — **cheio de erros (§7)**; reescrever |
| `investigation/expansion/*.py` | scripts da sessão anterior — **quebrados (§8)**; reescrever |
| `investigation/expansion/README.md` | desatualizado (fala de Gemini/Google AI Studio) |
| `investigation/reversefit_audit.py` | auditoria simples da sessão anterior (ok) |
| `RELATORIO.md`, `RELATORIO_DADOS.md` | relatórios do pipeline (corretos para o que medem) |

Memórias do projeto em `C:\Users\tito\.claude\projects\C--Users-tito-OneDrive-Documentos-Projetos-FakenewsBR\memory\` (confundimento subset↔rótulo, prior trap, hardware etc.) — leia o `MEMORY.md`.

## 4. Fatos externos verificados (fonte primária)

### 4.1 OpenRouter
- Catálogo `GET https://openrouter.ai/api/v1/models` (sem chave): 436 modelos, 77 variantes `:batch`.
- Preços por 1M tokens (entrada/saída):
  `openai/gpt-5-nano:batch` 0,025/0,20 (ctx 400k, modelo de raciocínio, sem `temperature`;
  suporta `reasoning`, `reasoning_effort`, `response_format`, `structured_outputs`, `seed`, `max_tokens`) ·
  `google/gemini-2.5-flash-lite:batch` 0,05/0,20 (ctx 1M) ·
  `openai/gpt-5.4-nano:batch` 0,10/0,625 · `openai/gpt-4.1-nano:batch` 0,05/0,20.
- **Batch API**: `POST https://openrouter.ai/api/beta/batches`, corpo JSON com
  `endpoint` (`/v1/chat/completions`), `model`, `requests` = `[{custom_id, body}]`.
  **Serializar `endpoint` e `model` ANTES de `requests`** (senão 400). `custom_id` único
  por batch. Resposta 202 `status: validating`. Poll `GET /api/beta/batches/:id`:
  `validating → in_progress → finalizing → completed` (terminais: completed, failed,
  expired, cancelled). **Resultados vêm inline** no GET quando `completed`
  (`results[].custom_id`, `response.body.choices[0].message.content` ou `error`);
  não há download de arquivo. Janela 24h. Só texto. ~50% do preço. Lista:
  `GET /api/beta/batches?limit=&status=&after=`. Não há limite de tamanho documentado
  (corpo é stream-parsed) → fatiar em ~2.000 requests por batch mesmo assim.
- O exemplo da doc usa slug sem sufixo (`openai/gpt-4o`); variantes `:batch` aparecem
  no catálogo e a doc diz que ":batch endpoints are not usable interactively".
  Implementar: enviar o slug escolhido; se 400 citar o modelo, tentar sem `:batch` e logar.
- `reasoning: {"effort": "minimal"}` é valor aceito na tabela de effort da doc.
  `response_format: {"type":"json_schema","json_schema":{name, strict, schema}}` suportado.
- Para o **agente LangChain** (interativo/síncrono) usar `openai/gpt-5-nano` sem `:batch`,
  `ChatOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])`.

### 4.2 Google Fact Check Tools API (discovery doc v1alpha1)
- `claims:search` params: `query` (**obrigatório só se não houver `reviewPublisherSiteFilter`**),
  `reviewPublisherSiteFilter`, `maxAgeDays`, `languageCode`, `pageSize` (default 10),
  `pageToken`, `offset`. Também existe `claims:imageSearch`.
- **`pages` (o link do usuário) NÃO serve para ler checagens de terceiros**: é o CRUD
  de ClaimReview markup **da própria organização**, OAuth scope
  `https://www.googleapis.com/auth/factchecktools` ("Read, create, update, and delete
  your ClaimReview data"). Explique isso ao usuário; a leitura é via `claims:search`
  ou via o feed (§4.3).

### 4.3 Feed ClaimReview (Data Commons) — já baixado
- `dataFeedElement`: 93.436 elementos; 99.769 itens ClaimReview (alguns elementos têm
  `item: null` — tratar). Chaves: `author`, `url`, `claimReviewed` (99.061),
  `reviewRating.alternateName`, `itemReviewed.author.name`, `datePublished` (88.928).
- **PT é pequeno: 8.706 itens** por domínio `.br/.pt`/checadores: boatos.org 4.268,
  lupa.uol.com.br 3.369, projetocomprova.com.br 354, www1.folha.uol.com.br 249,
  sbtnews.sbt.com.br 224 (+ sbtnews.com.br 47), jornalnh 45, nexojornal 32,
  jc.ne10.uol 24, correiobraziliense 17, … poligrafo **4**, g1 **1**; o resto é ruído
  (blogs/lojas usando ClaimReview indevidamente) → **usar whitelist de publishers**.
- Anos PT: 2019 368, 2020 1.308, 2021 886, 2022 2.255, 2023 1.804, 2024 1.960, 2025 19.
- Ratings PT: falso 6.814, verdadeiro 504, enganoso 221, exagerado 186,
  "verdadeiro, mas..." 130, contextualizando 47, subestimado 37, explica 23,
  insustentável 19, sátira 14, contraditório 12; várias no formato "FALSO: explicação…"
  (pegar o token antes de `:`/`-`).

## 5. Fontes sondadas (robots.txt respeitado, UA `FakenewsBR-research/2.0`, 1 req/s/host)

**Política ética definida**: não fazer scraping de conteúdo de sites cujo robots.txt
bloqueia robôs de treino de IA (GPTBot/ClaudeBot/CCBot/Google-Extended…). Esses
publishers só entram via metadado ClaimReview do feed. AFP e UOL devolvem 403 → fora.

| fonte | acesso | volume / datas | onde está o veredito | opt-out IA |
|---|---|---|---|---|
| Boatos.org | WP REST `/wp-json/wp/v2/posts` (per_page=100, 0,74 MB/página) | 13.633 posts, 2013-06 → hoje | tudo é boato (fake). `content` começa com **"Boato – <alegação>"**; 1º `<blockquote>` = texto viral ("Versão 1: …"). Categorias a excluir: English (451), Español (443), Opinião, Lista | não |
| E-farsas | WP REST | 4.376, **2002-04** → hoje | **categorias**: Falso 2.927, Verdadeiro 513, Fora de Contexto 201, Impreciso 37, Indeterminado 36 (Montagens 323 é tema). Título = pergunta "É verdade que …?"; fim do texto "Conclusão …"; blockquotes são embeds sociais ruidosos | não |
| Coletivo Bereia | WP REST | 1.210, 2019-10 → | categoria "Checamos" (resolver id pelo nome; 662 é a contagem), tags Enganoso 115, Falso 65, Verdadeiro 32, impreciso 28; títulos às vezes contêm o veredito | não |
| Polígrafo (PT-PT) | Yoast sitemaps: `fact_check-sitemap.xml` … `fact_check-sitemap13.xml` (~12,1 mil URLs). `/wp-json/` bloqueado no robots | 2017 → hoje | HTML: `<div class="fact-check-result"><span>Falso</span></div>` (valores: Falso, Verdadeiro, Verdadeiro mas, Pimenta na Língua, Impreciso, Descontextualizado, Manipulado). `og:title` termina com " - Polígrafo"; ld+json `@graph` WebPage (data a confirmar lá) | não |
| G1 Fato ou Fake | índice `https://g1.globo.com/sitemap/g1/sitemap.xml` = 9.086 sitemaps diários (2003→2026), filtrar URLs com `/fato-ou-fake/` (~2/dia) | 2018-07 → hoje | **h1**: "É #FAKE que …" / "É #FATO …"; data na URL `/noticia/AAAA/MM/DD/`; artigos "veja o que é fato e o que é fake no debate" são multi-alegação → LLM | não |
| Lupa | WP REST aberto (8.855; tags Falso 5.485, Verdadeiro 816, Exagerado 613) | 2015 → | tags | **sim** → só feed (3.369) |
| Aos Fatos | sitemap-noticias 6.864 | | | **sim** → fora (0 no feed) |
| Estadão Verifica | sitemap por dia; tem ClaimReview JSON-LD | | | **sim** → fora |
| Comprova | sitemaps por posttype/mês | | | **sim** → só feed (354) |
| Observador | wp-sitemap (367 sub-sitemaps) | | | **sim** → fora |
| AFP Checamos, UOL Confere | 403 | | | fora |
| Agência Pública (Truco) | WP REST 4.992 (maioria reportagem; tags Truco 161/227) | | no texto | não (opcional, LLM) |

Portais para linhas **`true` por procedência** (usar **só a manchete** como texto —
tamanho ≈ alegação de checador, evita recriar o atalho de comprimento):

| portal | WP REST | volume | notas |
|---|---|---|---|
| Poder360 | ok | 217.156 desde 2014-11; **2016 só 428** | categorias Brasil, Governo, Economia, Internacional, Justiça, Congresso, Eleições, Coronavírus, Saúde… excluir vídeos/Infográficos/Sports MKT |
| CNN Brasil | ok | 473.311 | **filtro `after/before` pareceu ignorado** (2016 devolveu o total) → validar; lista de categorias ordenou errado; muito entretenimento/esporte → filtrar |
| Brasil de Fato | ok | 127.720 desde 2016-03; 2016: 3.296 | excluir Opinião, English, Español, Brazil, Politics, Rádio (EN), Editorial; veículo com linha editorial → manter flag |
| O Eco | ok | 36.123 desde **2004**; 2016: 510 | usar Notícias/Reportagens; excluir Colunas, Análises, English |
| ECO (PT-PT) | posts ok (191.918 desde 2015), `/categories` 404 | | sem filtro de categoria |
| sem opt-out mas descartados | NiT (lifestyle) | | |
| com opt-out (fora) | CartaCapital, ZAP, SAPO, SOL, Observador | | |

**Gargalo real**: volume pré-2018 de portais é pequeno nas fontes acima. Sondar mais
sites WP PT-BR/PT-PT sem opt-out com cobertura 2010–2017.

**Estimativa de volume** (estimativa, não medido): linhas com veredito ≈ 55–65 mil
brutas (Boatos ~12,7 mil artigos × até 2 linhas, E-farsas ~3,7 mil, Polígrafo ~12 mil,
G1 ~6–12 mil, feed Lupa+Comprova+Folha+SBT ~4,3 mil, Bereia ~600), **menos a sobreposição
com a v1** (v1 já tem lupa 5.942, poligrafo 5.029, boatos 2.508, e-farsas 123 URLs).
Para 200–300 mil, o restante vem de manchetes `NEWS_*` (procedência). Deixar isso
explícito ao usuário: é o preço do volume; mitigação em §9.

## 6. Fatos da v1 necessários para compatibilidade

- 23 colunas na ordem: `rid, dataset_name, source_type, source_description, label,
  date_iso, url_review, text, text_clean, text_no_url, extracted_urls, is_duplicated,
  is_null, too_short, factcheck_rating, factcheck_claimant, factcheck_url, char_len,
  word_len, num_exclamations, num_questions, num_ellipsis, uppercase_word_ratio`.
- `rid` int64, 0…51.204. `date_iso` `AAAA-MM-DD`. `is_duplicated`, `is_null`,
  `too_short` são **todos 0**; `word_len` mínimo 3.
- `text_no_url == re.sub(r"https?://\S+|www\.\S+", "", text).strip()` em **99,35%**.
- `text_clean`: minúsculas + sem acento + sem não-ASCII + espaços colapsados bate só
  ~80% (derivação exata veio do bruto; aceitar aproximação e documentar).
- Métricas (de `sanitize_dataset.py`): `char_len`/`word_len` sobre **`text_clean`**;
  `num_exclamations` = contagem de `!` em `text`; `num_questions` = `?` em `text`;
  `num_ellipsis` = `len(re.findall(r'\.{3,}|\u2026', text))`;
  `uppercase_word_ratio` = palavras `re.findall(r'\b[A-Za-zÀ-ÖØ-öø-ÿ]+\b', text)` com
  `len>=3 and isupper()` / total. Testar paridade contra linhas reais da v1.
- Convenção de rótulo da v1: ratings "hard" (Enganoso, Distorcido, Sem contexto…)
  são `label=fake` (990 fake × 30 true). `models/data.py::_rating_class` classifica
  pelo texto do rating → gravar o rating original do checador em `factcheck_rating`.
- Ratings no v1 por checador: boatos NA 2.232/falso 122/Falso 91/Errado 33;
  poligrafo NA 4.700/Errado 206/Falso 35/Enganador 34/Certo 7. "Boato",
  "Pimenta na Língua", "Manipulado", "Descontextualizado" **não aparecem** na v1 →
  podem ser adicionados aos conjuntos de `data.py` sem mudar métricas da v1.
- `data.py::is_ptpt` = regex `poligrafo|observador` em url_review+factcheck_url;
  para PT-PT novo (ECO) acrescentar `eco\.sapo\.pt` (não existe na v1).
- **Ruído de rótulo na v1**: 210 linhas com rating contradizendo o rótulo
  (`fakes` 129, `true` 63, WhatsApp 6, COVID raw 5, COVID 4, MuMiN 3); 39 no teste.

## 7. Análise reversa — resultados medidos (válidos)

Três leitores do mesmo encoder fine-tuned, split IID seed 42, teste n=5.920:

| leitor | o que é | acc | macro-F1 |
|---|---|---:|---:|
| FT | cabeça `classifier` treinada com o encoder, Platt na **val cheia** (`5_finetune.log`) | 0,8655 | 0,8214 |
| ERM | logística sobre `embeddings_ft.npy` (mean-pool) em toda a base | 0,8451 | 0,8136 |
| DFR | logística só em grupos balanceados, ponderada, Platt sob prior balanceado | 0,7345 | 0,7172 |

Reprodução exata dos logs → splits idênticos.

1. **O "0,8655" não é média/seleção de ERM/DFR** (erro do `MELHORIAS_DATASET.md`): é
   o classificador FT, que também cai na **armadilha do prior**. Recalibrando o Platt
   do FT só na val balanceada (a=0,648, b=−0,764): acc no subset `true` **0,19 → 0,64**,
   pior-grupo 0,7276 → 0,7325, ECE balanceado 0,049 → 0,028, acc global 0,8655 → 0,8331.
   Melhoria gratuita (sem retreino).
2. **Vazamento por quase-duplicata** (cosseno ≥ 0,90, TF-IDF char 4–5, teste→treino):
   `FakeWhatsApp.BR_2018` **41,4%** do teste tem quase-duplicata no treino; FT acc
   **0,869 (AUC 0,947) com dup × 0,661 (AUC 0,752) sem dup**. COVID19.BR 11,4%;
   Fake.br 0,3% (o 0,986 do Fake.br não é vazamento). → A métrica de WhatsApp está
   inflada; split precisa agrupar quase-duplicatas (group k-fold por cluster).
3. **PT-PT está 100% confundido com os subsets degenerados**: 7.128 das 7.144 linhas
   PT-PT estão em `fakes`/`true`; no teste PT-PT = `fakes` 835 + `true` 242. A leitura
   "BERTimbau perde 22 pontos por dialeto" **não é sustentada** por esse dado.
   A comparação dentro do subset e a AUC dentro do checador (Polígrafo × Lupa) **saíram
   com bug de alinhamento** — já corrigido; **rodar de novo** e só então concluir.
4. **Comprimento em WhatsApp**: acc FT sobe monotonicamente com o tamanho
   (≤10 palavras 0,37 / AUC 0,28, n=35; 11–20 0,65; >160 0,90). Mensagens curtas são o
   buraco real do canal. Subset `true`: FT ≤ 0,23 em todas as faixas.
5. **Ano** (FT/DFR acc): ≤2017 0,860/0,491 (n=114); 2018 0,761/0,732; 2019 0,844/0,610;
   2020 0,882/0,579; 2021 0,896/0,697; 2022+ 0,851/0,703; sem data 0,925/0,875.
   DFR piora muito fora de 2018 e em datas antigas → cobertura temporal importa.
6. **Rating** (FT/DFR acc): false_pure 0,982/0,747 (n=934); hard 0,965/0,646 (144);
   true_rating 0,936/0,419 (**n=31** — pouquíssimo `true` em estilo alegação).
7. **Treino interrompido antes de convergir**: val macro-F1 0,8149 → 0,8296 e
   pior-grupo 0,7181 → 0,7645 entre as épocas 1 e 2; melhor checkpoint = última época.
   (O documento anterior dizia "estagnado".)

Rodar de novo: `python -m investigation.reverse_analysis.analyze` (~5 min; o
`predict_ft` não precisa rodar de novo). Depois ler `output/erros_confiantes.csv`
para análise qualitativa.

## 8. Erros a corrigir em `MELHORIAS_DATASET.md` (reescrever o documento)

- Atribuição do 0,8655 (TL;DR e §1.1) e a citação de "encoder.py linhas 230-249".
- "71,5% da base é degenerate" → **58,4%** (71,5% é a fração fake).
- Tese de dialeto/tokenizer (§1.3, §4 item 1): confundida (§7.3). Recomendar XLM-R
  como "maior retorno" não se sustenta até a reanálise; README diz "BERTimbau mantido" —
  reconciliar.
- "val macro-F1 estagnado" (§4 item 4) → errado (§7.7).
- Caracteres chineses "早期", "curadoria do Mavis", "acerra".
- §2.6: LLM4BR_300 "n<100" (tem 299); COVID19.BR_raw tem 373.
- §3.2.1 "50–80 mil claims PT" → feed tem 8.706.
- Fase 1 "busca o fact-check completo via `pages`" → impossível (§4.2).
- RSS "60–180 mil/ano" → RSS só dá os últimos itens; histórico vem de WP REST/sitemaps.
- Decisão Gemini/Google AI Studio (§3.2.6) → substituída por `openai/gpt-5-nano:batch`
  no OpenRouter; "avaliações internas do Google AI Studio" é inverificável.
- DoD "fake/true entre 1,2:1 e 1,8:1" é inviável com checadores (~93% "falso" no feed)
  e irrelevante; trocar por balanceamento **por grupo** e por era.
- Wayback: "earliest snapshot" não é data de publicação; cutoff 2010 × 2018 contraditório;
  desnecessário (WP REST/sitemaps trazem data real).
- Incluir os achados novos de §7 (vazamento WhatsApp, prior trap no FT, ruído de rótulo,
  comprimento, ano) e a política de opt-out de IA.

## 9. Bugs dos scripts atuais de `investigation/expansion/` (por isso reescrever)

- `collect_gfc.py` grava a linha do registro e depois uma linha de proveniência sem a
  chave `_provenance`; `langchain_agent.process_file` descarta linhas sem `_provenance`
  → **100% das linhas do GFC são descartadas**. `DEFAULT_QUERY` usa `OR` (não é sintaxe
  da API); o certo é iterar `reviewPublisherSiteFilter` sem query.
- `langchain_agent.py` batch (Google genai): `req-{i:06d}` reinicia a cada chunk →
  **colisão de chaves e rótulo atribuído à linha errada**; `job.output_file` não existe
  (resultados nunca baixados); um job de 50 requests por vez com poll de até 24h em série.
  Validador recebe `factcheck_claimant` como se fosse a alegação e pede ao LLM para
  julgar **verdade** (vies de conhecimento) em vez de consistência de extração.
  `PROMPT_ROUTER` com chaves simples (quebraria no `.format`, está sem uso).
  Deduplicador O(n×5000) em Python e **não compara com a v1** (vazamento).
- `merge_and_audit.py`: `df[c].dtype` em coluna inexistente → `KeyError`; `sanitize`
  refiltra e deduplica **as linhas da v1** (altera a v1); `iterrows`; rid str × int.
- `schema.compute_metrics` difere de `sanitize_dataset.py` (§6).
- `wayback_backfill.py` parsing em pares frágil; `rss_scraper.py` sem histórico;
  `_smoke_test.py` não cobre nada disso.
- Arquivos são untracked e foram criados pela sessão anterior deste goal: mover os
  obsoletos para `investigation/expansion/legacy/` (reversível) em vez de apagar.

## 10. Arquitetura a implementar (desenho fechado nesta sessão)

```
investigation/expansion/
  schema.py            COLUMNS (23) + PROVENANCE_COLUMNS; mapa de ratings PT→{false_pure,hard,true_rating,other};
                       label: false_pure/hard→fake, true_rating→true, other→descarta (inclui "explica",
                       "contextualizando", "ainda é cedo", "indeterminado", "de olho", "contraditório");
                       rating "FALSO: …" → token antes de ':'/'-'; derivações e métricas idênticas à v1 (§6);
                       rid = int(sha1(f"{url}|{text_role}|{i}")[:15], 16)  (int64, sem colisão com v1)
  http.py              PoliteSession: robots por host (urllib.robotparser), detecção de opt-out de IA,
                       1 req/s por host, retry/backoff 429/5xx com Retry-After, timeout, thread-safe
  sources.py           registro das fontes de §5 (método, dialeto, label_source, dataset_name,
                       filtros de categoria por nome normalizado, opt-out) + whitelist de publishers do feed
  collect_wp.py        WP REST paginado (per_page=100, _fields), mapa de taxonomias por nome, resume por
                       (fonte, ano, página); portais: estratificado por ano via after/before e páginas
                       espaçadas (linspace) dentro do ano; conteúdo só para checadores
  collect_sitemap.py   G1 (índice diário → filtro /fato-ou-fake/ → página: h1, data da URL) e
                       Polígrafo (fact_check-sitemap{,2..13} → página: og:title, fact-check-result, data)
  collect_feed.py      feed Data Commons → PT por whitelist → registro bruto
  collect_gfc.py       API corrigida (iterar reviewPublisherSiteFilter, maxAgeDays, 1 registro/linha); só com chave
  extract.py           regras por fonte → registros + proveniência:
                         Boatos: claim = texto após "Boato –"; viral = 1º blockquote (sem "Versão N:")
                         E-farsas: veredito por categoria; claim = título sem "É verdade que"/"Será verdade?"
                         G1: veredito e claim do h1 ("É #FAKE que X" → X); pular multi-alegação
                         Polígrafo: veredito do span; claim = og:title sem " - Polígrafo"
                         Feed: claimReviewed + rating; claimant = itemReviewed.author.name
                         NEWS: texto = manchete (6–40 palavras), label true, label_source=provenance
                       Filtro de vazamento: descartar textos escritos pelo checador (claim) ou manchetes
                       que contenham fake|falso|boato|mentira|farsa|verdadeir|engan|checagem|fact-check|#fato;
                       não aplicar a textos virais. Flag mentions_ai (IA/deepfake/ChatGPT) na proveniência.
  dedup.py             chave normalizada exata + MinHash LSH (numpy; shingles de 3 palavras, 5-gram de
                       caractere para textos curtos; hash estável blake2b); precedência v1 > veredito >
                       procedência > data mais antiga; conflito de rótulo entre novas → descarta e loga
                       (conflicts_v2.csv); nova duplicando v1 com rótulo diferente → descarta e loga
  llm_batch.py         OpenRouter Batch (§4.1): tarefas (a) veredito/alegação em artigos sem taxonomia,
                       (b) frase de CORREÇÃO verbatim (vira linha true, text_role=correction,
                       dataset FC_<PUB>_CORRECAO); json_schema strict; reasoning effort minimal; custom_id
                       = hash estável; estado em processed/llm_batches.json; resume; --dry-run grava os
                       requests e estima custo com preço do catálogo; validar que spans são substrings
                       normalizadas do artigo, senão descartar
  langchain_agent.py   create_agent (langchain 1.4) com tools: coverage_gaps (déficits era×rótulo×dialeto×
                       canal vs metas), list_sources (volume/anos/opt-out), propose_quota(fonte, anos,
                       max, motivo) validado, gfc_search (se houver chave), estimate_llm_cost;
                       saída processed/collection_plan.json; --planner rules faz o mesmo sem LLM
  pipeline.py          CLI: plan | collect | extract | llm-prepare | llm-submit | llm-collect | merge
  merge_and_audit.py   v2 = v1 intacta (assert por rid + hash das colunas de texto) + novas após dedup e
                       cotas (--news-max-ratio por era); grava FakenewsBR_sanitized_v2.csv (23 colunas) +
                       FakenewsBR_v2_provenance.csv (rid, label_source, text_role, publisher, lang_variant,
                       collector, source_url, collected_at, rating_norm, mentions_ai) + audit_v2.md/json
  tests/               unittest offline com fixtures (post WP, sitemap, item do feed, resposta de batch
                       mockada, dedup, paridade de métricas com linhas reais da v1, merge não altera v1)
  raw/, processed/     gitignored (criar investigation/expansion/.gitignore)
```

Convenções de `dataset_name`: `FC_BOATOS`, `FC_BOATOS_VIRAL`, `FC_EFARSAS`, `FC_POLIGRAFO`,
`FC_G1`, `FC_LUPA`, `FC_COMPROVA`, `FC_FOLHA`, `FC_SBT`, `FC_BEREIA`, `FC_<PUB>_CORRECAO`,
`NEWS_PODER360`, `NEWS_CNNBRASIL`, `NEWS_BRASILDEFATO`, `NEWS_OECO`, `NEWS_ECO`.
`source_type`: "news" (alegações), "social media post" (virais), "news headline" (NEWS).

Mudanças em `models/data.py` (sem alterar resultado da v1 — testar hash do DataFrame
antes/depois): canal por prefixo (`FC_*_VIRAL`→social, `FC_*`→agency_claim,
`NEWS_*`→press_true); `is_balanced_group` dinâmico só para grupos novos
(minoria ≥15% e n ≥200); `NEWS_*` sempre degenerado (nunca treina a cabeça DFR);
regex PT-PT + `eco\.sapo\.pt`; join opcional do CSV de proveniência.

## 11. Ordem de execução sugerida

1. Rodar `analyze.py` de novo; fechar a conclusão sobre dialeto (§7.3).
2. Implementar `schema/http/sources/collect_*/extract/dedup/merge` + testes; rodar testes.
3. Coletar em background o que não precisa de chave (feed → WP checadores → Polígrafo →
   G1 → portais por ano), monitorando progresso e respeitando robots/opt-out.
4. `extract` → `dedup` → `merge` → `audit_v2.md`. Reportar composição real
   (veredito × procedência, por era: ≤2017, 2018–nov/2022, dez/2022+; PT-PT %; rótulo por grupo).
5. `llm-prepare --dry-run` com `openai/gpt-5-nano:batch` e custo estimado; submeter só
   quando o usuário definir `OPENROUTER_API_KEY`.
6. Reescrever `MELHORIAS_DATASET.md` (§7, §8) e `investigation/expansion/README.md`.
7. Atualizar memória (novas decisões e achados).
8. Retreino v2 (12–36h CPU) só se o usuário pedir; comparar com split que agrupa
   quase-duplicatas.

## 12. Tarefas pendentes (checklist)

- [ ] Reexecutar análise reversa corrigida e registrar conclusão de dialeto
- [ ] Pipeline novo implementado + testes offline passando
- [ ] Obsoletos movidos para `legacy/`
- [ ] Coleta real executada (sem chave) com logs
- [ ] `FakenewsBR_sanitized_v2.csv` + proveniência + auditoria gerados, v1 intacta
- [ ] Requests do LLM preparados (dry-run) com custo; submissão pendente de chave
- [ ] `MELHORIAS_DATASET.md` e README reescritos com números medidos
- [ ] `models/data.py` estendido sem mudar a v1
- [ ] Memória atualizada
