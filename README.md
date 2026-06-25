---
language:
- pt
tags:
- fake-news
- misinformation
- fact-checking
pretty_name: FakenewsBR
size_categories:
- 10K<n<100K
---

# FakenewsBR 

FakenewsBR is a Brazilian Portuguese dataset for misinformation and fake news research. It aggregates texts from multiple Brazilian sources, including news articles, tweets, WhatsApp messages, and political/news-related content, and provides a normalized binary veracity label for each record.

Repository: [https://github.com/AKCIT-FN/fakenews-data]

## Dataset Details

| Field | Value |
|---|---|
| Dataset name | FakenewsBR Factchecked |
| Language | Brazilian Portuguese |
| Format | CSV |
| Official file | `FakenewsBR_factchecked.csv` |
| Number of records | 51,205 |
| Number of columns | 17 |
| Main task | Binary text classification |
| Labels | `fake`, `true` |

## Dataset Summary
This dataset was built to support research and experimentation on misinformation detection in the Brazilian context. It contains textual content from different source types and includes both original and cleaned text fields.

Each instance contains a normalized label indicating whether the content is classified as fake or true. When available, the dataset also includes review URLs and fact-checking metadata such as ratings, claimants, and fact-check URLs

## Label Distribution
| Label | Count |
|---|---:|
| `fake` | 35,150 |
| `true` | 16,055 

## Columns
| Column | Description |
|---|---|
| `rid` | Row identifier in the released file. |
| `dataset_name` | Name of the source dataset or subset. |
| `source_type` | Type of source, such as news, tweets, or WhatsApp messages. |
| `source_description` | Short description of the source dataset. |
| `label` | Binary veracity label: `fake` or `true`. |
| `date_iso` | Normalized date in ISO format, when available. |
| `url_review` | URL of a review or fact-checking page, when available. |
| `text` | Original consolidated text. |
| `text_clean` | Cleaned and normalized version of the text. |
| `text_no_url` | Text with URLs removed. |
| `extracted_urls` | URLs extracted from the original text. |
| `is_duplicated` | Duplicate-content flag. |
| `is_null` | Null-text flag. |
| `too_short` | Short-text flag. |
| `factcheck_rating` | Fact-checking rating, when available. |
| `factcheck_claimant` | Claimant associated with the fact-checking entry, when available. |
| `factcheck_url` | URL returned or associated with fact-checking metadata, when available. |

## Intended Uses
This dataset can be used for:
- Fake news and misinformation detection.
- Binary text classification in Brazilian Portuguese.
- Evaluation of NLP models and language models in Portuguese.
- Research on Brazilian disinformation patterns.
- Experiments involving fact-checking metadata and source analysis.