# Análise Exploratória de Dados (EDA) — FakenewsBR

> Consolidação dos achados da análise exploratória, enriquecida com as medições
> posteriores do projeto. **Documento base**: [`FakenewsBR_Exploratory_Report.html`](FakenewsBR_Exploratory_Report.html)
> (síntese das 8 trilhas de investigação). Números aqui são medidos, com a
> fonte indicada; onde a EDA inicial foi corrigida depois, está explicitado
> (§5).
>
> **Escopo temporal**: a EDA foi executada sobre a base **v1**
> (`data/FakenewsBR_sanitized.csv`, 39.466 linhas). O dataset atual é a **v6**
> (297.672 linhas) — ver [§7](#7-da-eda-v1-ao-dataset-v6) e
> [`DATASET_CARD.md`](DATASET_CARD.md).

---

## Sumário executivo

1. **Atalho de proveniência é o achado central**: em 58,4% da base o grupo de
   origem determina o rótulo. Só o prior da fonte "acerta" 0,8055
   (`diagnose_source_leak`), com pior-grupo 0,2717. Qualquer ERM ingênuo
   aprende *de onde veio*, não *se é verdade*.
2. **Duas desinformações distintas**: corrente de WhatsApp (curta, urgente,
   CTA 14,95% vs 1,04% na imprensa) e portais fake (longos, imitam léxico
   jornalístico). Um único modelo mal desenha as duas.
3. **Meias-verdades são outro problema**: os 847 casos limítrofes não usam as
   assinaturas estilísticas das fakes puras — são mais longos, menos
   "gritados", com vocabulário próximo ao jornalismo real.
4. **Vazamentos medidos**: datelines ("feira", "nesta"), comprimento, termos
   PT-PT e quase-duplicatas (41,4% do teste WhatsApp) contaminam métricas
   ingênuas.
5. **A conversão binária esconde gradação**: 9,2% das classificações são
   enganoso/distorcido/manipulado — classes que léxico sozinho não resolve.

---

## 0. Artefatos e reprodução

| artefato | caminho | observação |
|---|---|---|
| Síntese mestre (HTML autocontido, figuras em base64) | [`FakenewsBR_Exploratory_Report.html`](FakenewsBR_Exploratory_Report.html) | fonte canônica dos achados das 8 trilhas |
| 8 scripts de investigação | `investigation/path1_topics.py` … `path8_spurious_shortcuts.py` | leem `data/FakenewsBR_sanitized.csv` |
| EDA estilística / n-gramas / sentimento | `scripts/eda_analysis.py` | gera as 5 figuras de `plots/` |
| Figuras da EDA | `plots/*.png` (5 arquivos, via Git LFS) | comprimento, bigramas, estilo/sentimento |
| Relatórios por trilha (`path*_report.md`) | *fora do repositório* | gravados em `ARTIFACT_DIR` absoluto de outra máquina; **não versionados** — a síntese HTML substitui |
| Conversão MD → HTML | `investigation/generate_html_report.py` | aponta para caminho Windows; ajustar `ARTIFACT_DIR`/`MD_PATH` antes de usar |

**Reprodução**: os CSVs grandes ficam no Release (ver README §Dados) e devem
ser copiados para `data/`. Os scripts das trilhas têm `ARTIFACT_DIR`
hardcoded — ajuste-o (ou rode com o caminho relativo do repo) para regenerar
relatórios e figuras. A EDA estilística:

```bash
python scripts/eda_analysis.py   # requer data/FakenewsBR_sanitized.csv
```

---

## 1. Base analisada (v1)

| métrica | valor |
|---|---:|
| linhas (desduplicadas e auditadas) | 39.466 |
| colunas | 23 (contrato fixo) |
| fração `fake` | 71,5% |
| linhas em **grupos degenerados** (origem ⇒ rótulo) | **58,4%** |
| `date_iso` válida (2014–2024) | 26.383 |
| grupos (`dataset_name`) | 11 origens |

Subconjuntos usados nos contrastes:

| subconjunto | linhas | papel |
|---|---:|---|
| `FakeWhatsApp.BR_2018` | 6.381 | correntes virais (peer-to-peer) |
| `Fake.br` | 3.600 | portais fake "profissionais" |
| imprensa / manchetes reais | — | classe `true` por procedência |
| fakes rótulo `false_pure` | 4.513 | mentiras flagrantes |
| casos limítrofes (`Enganoso`/`Distorcido`/`Fora de contexto`) | 847 | meias-verdades |

---

## 2. Achados por trilha

### Trilha 1 — Tópicos e narrativas latentes (`path1_topics.py`)

Método: NMF (8 tópicos globais + separados por classe) sobre TF-IDF de
1–2 gramas com stopwords jornalísticas removidas.

- **"Isca visual" é marcador quase exclusivo de fake**: tópico
  *video mostra / foto mostra / redes sociais* — **6,9% em fake vs 2,8% em
  true** (razão ~2,5×). Textos falsos funcionam como legenda fraudulenta
  atrelada a mídia real fora de contexto.
- **Dicotomia sanitária (COVID-19)**: fake orbita *ivermectina, cloroquina,
  cura milagrosa, efeitos colaterais da pfizer*; true reflete a vigilância
  epidemiológica (boletim da secretaria, leitos de UTI, fase amarela/vermelha).
- **Mimesis política**: o tópico generalista de política (governo, presidente,
  pt, reais) absorve **>50% dos documentos de ambas as classes** — a
  desinformação em portais espelha o léxico da imprensa tradicional.

### Trilha 2 — Temporal e crises (`path2_temporal.py`)

A desinformação é **cíclica**, puxada por choques exógenos:

| período | pico |
|---|---|
| out/2018 (eleições) | **1.407** publicações falsas checadas no mês — maior da série (atentado a faca, costumes, alegação de fraude em urnas) |
| mar–jun/2020 | média **>600** boatos/mês (curas falsas, origem do vírus, descrédito do isolamento) |
| mar–jun/2021 | 2ª onda de saúde: anti-Coronavac e anti-laboratórios |
| ago–out/2022 | **>1.200** checagens no trimestre eleitoral (Lula × Bolsonaro, questionamento ao TSE) |

**Topic drift em três eras**: Lava Jato (2015–2018, corrupção/delações) →
Pandêmica (2020–2021, saúde chegou a **65%** de todo o fluxo) →
Institucional/Eleitoral (2021–2023, ataques a STF e ao voto eletrônico).

### Trilha 3 — Anatomia do WhatsApp vs. portais vs. imprensa (`path3_whatsapp.py`)

Testes de Mann-Whitney (WhatsApp × real):

| métrica | WhatsApp | portais fake | imprensa | teste |
|---|---:|---:|---:|---|
| mediana de palavras | 38 | 157 | 71 | — |
| % palavras CAIXA ALTA | **6,78%** | 1,20% | 2,56% | p = 5,4×10⁻⁵⁰ |
| % chamadas à ação (CTA) | **14,95%** | 0,83% | 1,04% | p = 6,0×10⁻²²³ |
| % urgência (bomba/atenção) | 7,15% | 1,08% | 4,05% | gatilho de pânico |
| % apelo direto (você/vejam/olhe) | 9,03% | 7,39% | 5,07% | 2ª pessoa |
| % mensagens com URL | **22,68%** | 0,00% | 0,65% | vetor de redirecionamento |

**Implicação**: modelo treinado só em texto jornalístico formal falha em
corrente peer-to-peer. A desinformação de WhatsApp não depende de
sofisticação gramatical — depende de urgência afetiva e replicação explícita
("repassem antes que censurem").

### Trilha 4 — Alvos, checadores e o achado PT-PT (`path4_factcheckers.py`)

**Agências brasileiras**: Aos Fatos 2.549 · Boatos.org 2.508 · Estadão
Verifica 2.213 · AFP Checamos 1.649 · Projeto Comprova 415.

**Achado crítico — Portugal**: Polígrafo (`poligrafo.sapo.pt`) **5.029**
checagens e Observador **1.851**, além de figuras lusas (António Costa, 48
checagens). O corpus **não é 100% PT-BR**: há um subconjunto relevante de
PT-PT ("fato" × "facto"), que depois aparece como artefato de classe
(ver §3.4 e §5).

**Taxonomia granular escondida pela binarização**:

| veredito | casos | share |
|---|---:|---:|
| Falso (fabricação completa) | 4.449 | 60% |
| Enganoso / Distorcido / Manipulado | 675 | 9,2% |
| Fora de contexto / Sem contexto | 70 | — |
| Exagerado / Impreciso | 87 | — |

> ⚠️ "Fora de contexto" e "Distorcido" usam dados/imagens **verdadeiros** com
> narrativa desviada — é onde vocabulário sozinho falha e por que o projeto
> migrou para avaliação dos **casos limítrofes** e módulos de verificação
> externa.

**Alvos (em 28.236 fake news)**: Lula/PT **16,7%** (4.706) · Bolsonaro/família
**14,7%** (4.151) · mídia tradicional (Globo/Folha) 6,1% · vacinas/Anvisa
5,8% · Congresso 5,4% · STF/ministros 4,1%.

### Trilha 5 — NER e redes de coocorrência (`path5_ner_networks.py`)

Log-odds de 32 entidades centrais:

- **Polo fake**: Ivermectina **+2,10** · Cloroquina **+0,94** · Venezuela
  **+0,78** · China **+0,39** · (Dra. Nise Yamaguchi e Alexandre de Moraes
  como protagonistas de alegações infundadas).
- **Polo true**: Ministério Público **−1,12** · Michel Temer **−1,09** ·
  Congresso/Senado **−0,90** · Ciro Gomes **−0,78** · STF **−0,56** — a
  cobertura da burocracia política não reverbera nos circuitos de boataria.

**Topologia (fake news)**: atores encadeados em tríades de acusação —
(1) Bolsonaro & Lula (**504 coocorrências**); (2) Bolsonaro & STF/Moraes
(narrativa de censura); (3) China & Coronavac & Doria (narrativa antivacina).

### Trilha 6 — Casos limítrofes (`path6_hard_cases.py`)

847 meias-verdades × 4.513 falsos puros:

- **Mimetismo factual**: menos CAIXA ALTA (**1,92% vs 2,67%**) e bem menos
  exclamações — não usam as "assinaturas gritantes".
- **Mais longos**: mediana **80 vs 64 palavras** (mais esforço argumentativo
  para desvirtuar fatos reais).
- **Léxico próximo do jornalismo**: os termos que os separam do falso puro
  são âncoras reais (*vacinas, ivermectina, pessoas, crianças*) vs
  (*Lula, STF, fraude*).

### Trilha 7 — Falácias e gatilhos emocionais (`path7_framing_emotions.py`)

- **Assinatura exclusiva**: "repassem", "médico do hospital revelou" são
  marcadores quase 100% da classe fake; o jornalismo true não os usa.
- **Combustível do WhatsApp**: indignação (*absurdo, vergonha, revolta*)
  concentrada nas correntes.
- **Diferença de plataforma**: portais fake usam **menos** apelo emocional
  barato que o WhatsApp — tentam parecer profissionais.

### Trilha 8 — Auditoria de atalhos espúrios (`path8_spurious_shortcuts.py`)

Efeito Clever Hans — correlações de superfície que vazam o rótulo:

| atalho | evidência | direção |
|---|---|---|
| dias da semana (datelines) | feira **r = −0,29**, nesta −0,25, quarta −0,16 | proxy de `true` ("Nesta quarta-feira…") vs boato atemporal |
| comprimento | r = −0,13 (curto ↔ falso) | modelo penaliza textos curtos |
| artefatos PT-PT | *ecrã, equipa, facto* | vazam correlação com `true` |

---

## 3. Atalhos estruturais (a parte que virou o projeto)

### 3.1 Proveniência — o maior vazamento

- **58,4%** da base tem rótulo determinado pelo grupo (v1);
  fração `fake` global 71,5% (não é prevalência real de desinformação).
- **`models/v6/diagnose_source_leak.py`**: prever o grupo pelo texto
  (acc 0,642) e aplicar o prior do grupo dá **acc 0,8055 / macro-F1 0,7474**,
  com **pior-grupo 0,2717**. É o teto do trapaceiro e o piso honesto de
  comparação.
- Sintoma em modelo real (`docs/RELATORIO.md` §2): o encoder puro acerta
  **96,6%** no subset `fakes` e erra **81%** do subset `true` — aprendeu a
  responder "falso" para qualquer coisa com cara de alegação de agência.

### 3.2 Quase-duplicata

- **41,4%** do teste de `FakeWhatsApp.BR_2018` tem quase-duplicata no treino
  em split IID. Impacto: FT **0,869 / AUC 0,947** com quase-dup ×
  **0,661 / AUC 0,752** sem (`MELHORIAS_DATASET.md` §4).
- Por isso o split exige **agrupamento por cluster de quase-duplicata**
  (MinHash; no v6, 0 pares cruzando splits).

### 3.3 Comprimento por canal

No WhatsApp, a acurácia sobe monotonicamente com o tamanho: ≤10 palavras
**0,37** (AUC 0,28, n=35) → 11–20: 0,65 → >160: 0,90. Mensagens curtas são o
buraco real do canal.

### 3.4 Ruído de rótulo

**210 linhas** da v1 têm `factcheck_rating` contradizendo o `label` (39 no
teste) → piso de erro de ~1 pp. Ratings "hard" viram `fake` por convenção
(`models/data.py::_rating_class`).

### 3.5 O baseline linear é forte

TF-IDF + Regressão Logística + Platt: **0,8032 / 0,7404 / pior-grupo 0,5070**
(`models/v6/compare/baseline_tfidf/`). O transformer só se paga se passar
disso **no pior grupo**, não na média.

---

## 4. Síntese: vulnerabilidade → mitigação

| # | desafio | mecanismo de falha | mitigação | status |
|---|---|---|---|---|
| 1 | vazamento de datelines/jornalismo | palavras neutras (*feira, nesta, segundo*) como proxy de `true` | remoção/normalização de marcações temporais; testes com máscara | auditoria feita (trilha 8); mascarar entidades em avaliação |
| 2 | viés de canal/comprimento | nº de palavras e caixa alta como atalho | modelo por canal ou normalizar comprimento; avaliar por grupo | métrica **pior-grupo** + DFR no repo |
| 3 | casos limítrofes | super-otimização em "assinaturas gritantes" | avaliação multi-classe / NLI, não só binário | `limítrofes` é métrica obrigatória nos runs |
| 4 | vazamento de entidades | *ivermectina, cloroquina, moraes* ≡ falso | **entity masking** (`[PESSOA]`, `[REMEDIO]`) | em teste (`mask` no autoresearch; run completo pendente) |
| 5 | mistura PT-BR/PT-PT | artefatos *ecrã, equipa, facto* | isolar PT-PT na avaliação ou normalizar ortografia | `lang_variant` + corte `pt-PT` reportado (0,6992 no R0) |
| 6 | prior de 71,5% | Platt na validação cheia derruba o DFR (pior-grupo 0,7301 → 0,3751) | calibrar sob **prior balanceado** (~47%); correção de prior por fora | implementado (Platt em `val_calib`) |
| 7 | quase-duplicata | 41,4% do teste WhatsApp no treino | split por cluster (GroupKFold/MinHash) | v6: 0 pares near-dup cruzando splits |

---

## 5. Revisões posteriores — o que a EDA disse e foi corrigido

A EDA é o mapa, não a verdade final. Três correções registradas nos relatórios
seguintes:

1. **Features estilísticas "não ajudam"** — a EDA (trilha 7 / §6.3) afirmava
   "forte poder discriminante complementar". A ablação
   (`RELATORIO.md` §4.1) mostrou que o que elas discriminam é **canal**, não
   veracidade: sem estilo o pior-grupo IID sobe 0,7226 → **0,7304** e o OOD
   WhatsApp 0,6229 → **0,6266**. Recomendação: `--no-style` como referência.
2. **Dialeto PT-PT** — `RELATORIO.md` §4.3 mede 22 pp de queda (0,6248 vs
   0,8503) e recomenda XLM-R; `MELHORIAS_DATASET.md` §3 mostra que **7.128 das
   7.144 linhas PT-PT estão em grupos degenerados** e que controlando o
   checador a AUC é comparável (Polígrafo 0,71 × Lupa 0,68) — sem evidência de
   penalidade de tokenizer. **Leitura corrigida: composição, não dialeto.**
3. **O prior 71,5% não é prevalência** — artefato de 58,4% de subsets
   degenerados; calibrar sobre ele compra acurácia global e destrói o
   pior-grupo (`RELATORIO.md` §4.2).

---

## 6. Transição para modelagem

Restrição de EDA encerrada (relatório mestre, §10). Caminho adotado:

1. **Baseline estrutural explicável** — TF-IDF + LR (+ Platt): feito
   (0,8032 / 0,7404).
2. **Transformers com máscara de datelines/entidades**, avaliados contra os
   casos limítrofes — BERTimbau R0 treinado (acc 0,8163 · macro-F1 0,7782 ·
   pior-grupo 0,5749 · ECE 0,0188); `informative+mask` pendente.
3. **Score de confiança**, não rótulo duro — Platt sob prior balanceado +
   avaliação por grupo (pior-grupo como métrica manchete); DFR como régua
   honesta.

Referências de resultado: `README.md` (R0, autoresearch), `docs/RELATORIO.md`
e `docs/RELATORIO_DADOS.md` (pipeline legado), `docs/MELHORIAS_DATASET.md`
(correções).

---

## 7. Da EDA v1 ao dataset v6

| item | EDA (v1) | agora (v6) |
|---|---|---|
| linhas | 39.466 | 297.672 (pool verificado 91.080) |
| rótulo | `label` binário único | camadas (`label_tier`/`verified_label`/`train_label`); 206.592 `unknown` fora do treino |
| grupos degenerados | 58,4% da base | isolados; 11 constantes (39.373 linhas) documentadas |
| quase-duplicata | vazamento medido (41,4% WhatsApp) | MinHash: **0 pares** cruzando splits |
| split | IID ingênuo | por cluster, seed 42 (`full_iid`) |
| métrica | acurácia | **pior-grupo** + macro-F1 balanceado + ECE |
| PT-PT | artefato de classe | corte `pt-PT` reportado (0,6992 no R0) |

Nenhum achado da Eda foi "desfeito" pelo v6 — os vazamentos que ela mediu são
exatamente os controles que a v6 implementa (DFR, splits por cluster, camadas
de rótulo, métrica por grupo).

---

## Referências

- Síntese das 8 trilhas: [`FakenewsBR_Exploratory_Report.html`](FakenewsBR_Exploratory_Report.html)
- Scripts: `investigation/path1_topics.py` … `path8_spurious_shortcuts.py` · `scripts/eda_analysis.py`
- Figuras: `plots/` (LFS)
- Correções e medições: [`MELHORIAS_DATASET.md`](MELHORIAS_DATASET.md) · [`RELATORIO.md`](RELATORIO.md) · [`RELATORIO_DADOS.md`](RELATORIO_DADOS.md)
- Dataset: [`DATASET_CARD.md`](DATASET_CARD.md) · [`SOURCES_AND_LICENSES.md`](SOURCES_AND_LICENSES.md)
- Perfil v4 (automático): `docs/v4/V4_PROFILE.md`
- Diagnóstico de proveniência: `models/v6/diagnose_source_leak.py`
