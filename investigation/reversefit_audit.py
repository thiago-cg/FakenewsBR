"""Auditoria do dataset atual para identificar lacunas e orientar expansao."""
import pandas as pd
import json

df = pd.read_csv("data/FakenewsBR_sanitized.csv", low_memory=False)
df["date_iso"] = pd.to_datetime(df["date_iso"], errors="coerce")
df["year"] = df["date_iso"].dt.year

# === 1. Sub-representacao temporal
print("=" * 60)
print("1. DISTRIBUICAO TEMPORAL (pre/post GPT-3.5 nov/2022)")
print("=" * 60)
nov22 = pd.Timestamp("2022-11-30")
pre = (df["date_iso"] <= nov22).sum()
pos = (df["date_iso"] > nov22).sum()
nodate = df["date_iso"].isna().sum()
print(f"pre-Nov/2022: {pre} | pos-Nov/2022: {pos} | sem data: {nodate}")
print()
print("Por ano:")
yc = df["year"].value_counts().sort_index()
print(yc.to_string())
print()

# === 2. Lacunas por tipo de midia
print("=" * 60)
print("2. CANAIS: sub-representados")
print("=" * 60)
ch = df["source_type"].value_counts()
print(ch.to_string())
print()

# === 3. Topicos (por TF-IDF simples)
print("=" * 60)
print("3. TOPICOS (palavras-chave dominantes por dataset)")
print("=" * 60)
for name in ["Fake.br", "FakeWhatsApp.BR_2018", "COVID19.BR", "fakes", "true", "LLM4BR_300"]:
    sub = df[df["dataset_name"] == name]
    if len(sub) > 0:
        sample = sub["text_no_url"].dropna().astype(str).str.cat(sep=" ")[:200000]
        words = pd.Series(sample.lower().split()).value_counts().head(15)
        print(f"\n  {name} (n={len(sub)}):")
        for w, c in words.items():
            print(f"    {w:25s} {c}")

# === 4. PT-PT
print("\n" + "=" * 60)
print("4. DISTRIBUICAO PT-PT (Poligrafo/Observador)")
print("=" * 60)
urls = df["url_review"].fillna("") + " " + df["factcheck_url"].fillna("")
df["is_ptpt"] = urls.str.contains("poligrafo|observador", case=False, regex=True)
print(f"Total PT-PT: {df['is_ptpt'].sum()} ({df['is_ptpt'].mean()*100:.1f}%)")
print(f"  fake PT-PT: {((df['is_ptpt']) & (df['label']=='fake')).sum()}")
print(f"  true PT-PT: {((df['is_ptpt']) & (df['label']=='true')).sum()}")
print()

# === 5. Pre GPT-3.5 vs pos
print("=" * 60)
print("5. Volume PRE vs POS IA generativa")
print("=" * 60)
gpt_era = []
for _, row in df.iterrows():
    if pd.isna(row["date_iso"]):
        gpt_era.append("sem_data")
    elif row["date_iso"] <= pd.Timestamp("2022-11-30"):
        gpt_era.append("pre_GPT35")
    else:
        gpt_era.append("pos_GPT35")
df["gpt_era"] = gpt_era
print(pd.crosstab(df["gpt_era"], df["label"]))
print()

# === 6. Cobertura de temas pre-2018
print("=" * 60)
print("6. AMOSTRAS POR ANO (pre-2018 = fraco)")
print("=" * 60)
pre_2018 = df[df["date_iso"] < pd.Timestamp("2018-01-01")]
print(f"pre-2018 com data: {len(pre_2018)}")
print(f"  fake: {(pre_2018['label']=='fake').sum()}")
print(f"  true: {(pre_2018['label']=='true').sum()}")
print(f"2018+: {len(df) - len(pre_2018) - df['date_iso'].isna().sum()}")
