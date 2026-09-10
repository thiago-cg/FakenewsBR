# Graph Report - project-author-link-f95b17  (2026-09-09)

## Corpus Check
- Corpus is ~450 words - fits in a single context window. You may not need a graph.

## Summary
- 10 nodes · 13 edges · 3 communities (2 shown, 1 thin omitted)
- Extraction: 77% EXTRACTED · 23% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.78)
- Token cost: 79,000 input · 2,200 output

## Community Hubs (Navigation)
- Brazilian Misinformation Research
- Dataset Structure & Labels
- Classification Task & Text Fields

## God Nodes (most connected - your core abstractions)
1. `FakenewsBR_factchecked.csv` - 6 edges
2. `FakenewsBR` - 5 edges
3. `Binary text classification` - 3 edges
4. `Normalized binary veracity label` - 3 edges
5. `AKCIT-FN/fakenews-data repository` - 2 edges
6. `Fact-checking metadata` - 2 edges
7. `Text normalization and cleaning fields` - 2 edges
8. `Misinformation and fake news detection` - 1 edges
9. `Source type taxonomy (news, tweets, WhatsApp messages)` - 1 edges
10. `Brazilian Portuguese NLP` - 1 edges

## Surprising Connections (you probably didn't know these)
- `AKCIT-FN/fakenews-data repository` --references--> `FakenewsBR_factchecked.csv`  [INFERRED]
  README.md → README.md  _Bridges community 1 → community 0_
- `FakenewsBR` --conceptually_related_to--> `Binary text classification`  [EXTRACTED]
  README.md → README.md  _Bridges community 0 → community 2_
- `FakenewsBR_factchecked.csv` --shares_data_with--> `Text normalization and cleaning fields`  [EXTRACTED]
  README.md → README.md  _Bridges community 1 → community 2_

## Hyperedges (group relationships)
- **FakenewsBR dataset composition and task** — readme_fakenewsbr_factchecked_csv, readme_normalized_veracity_label, readme_source_type_taxonomy, readme_fact_checking_metadata, readme_binary_text_classification [EXTRACTED 0.80]

## Communities (3 total, 1 thin omitted)

### Community 0 - "Brazilian Misinformation Research"
Cohesion: 0.50
Nodes (4): Brazilian Portuguese NLP, AKCIT-FN/fakenews-data repository, FakenewsBR, Misinformation and fake news detection

### Community 1 - "Dataset Structure & Labels"
Cohesion: 0.67
Nodes (4): Fact-checking metadata, FakenewsBR_factchecked.csv, Normalized binary veracity label, Source type taxonomy (news, tweets, WhatsApp messages)

## Knowledge Gaps
- **3 isolated node(s):** `Misinformation and fake news detection`, `Source type taxonomy (news, tweets, WhatsApp messages)`, `Brazilian Portuguese NLP`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 3 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `FakenewsBR_factchecked.csv` connect `Dataset Structure & Labels` to `Brazilian Misinformation Research`, `Classification Task & Text Fields`?**
  _High betweenness centrality (0.514) - this node is a cross-community bridge._
- **Why does `FakenewsBR` connect `Brazilian Misinformation Research` to `Dataset Structure & Labels`, `Classification Task & Text Fields`?**
  _High betweenness centrality (0.463) - this node is a cross-community bridge._
- **Why does `Binary text classification` connect `Classification Task & Text Fields` to `Brazilian Misinformation Research`, `Dataset Structure & Labels`?**
  _High betweenness centrality (0.097) - this node is a cross-community bridge._
- **What connects `Misinformation and fake news detection`, `Source type taxonomy (news, tweets, WhatsApp messages)`, `Brazilian Portuguese NLP` to the rest of the system?**
  _3 weakly-connected nodes found - possible documentation gaps or missing edges._