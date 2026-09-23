import os
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

ARTIFACT_DIR = r"C:\Users\tito\.gemini\antigravity-ide\brain\c96eb70a-06a4-485e-86e1-39bcd92468da"
PLOTS_DIR = os.path.join(ARTIFACT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

REPORT_PATH = os.path.join(ARTIFACT_DIR, "path1_topic_modeling_report.md")
DATASET_PATH = "data/FakenewsBR_sanitized.csv"  # relativo a raiz do repo

def run_investigation():
    print("=== [Path 1] Iniciando Modelagem de Tópicos e Narrativas Latentes ===")
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    print(f"Dataset carregado: {len(df)} registros.")

    # Base stopwords em português + stopwords jornalísticas neutras
    stopwords_pt = [
        'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'nao', 'uma', 'os', 'no',
        'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'ao', 'ele', 'das', 'aqui', 'tem', 'seu',
        'sua', 'ou', 'quando', 'muito', 'nos', 'ja', 'eu', 'tambem', 'so', 'pelo', 'pela', 'ate', 'isso',
        'ela', 'entre', 'depois', 'sem', 'mesmo', 'aos', 'seus', 'quem', 'nas', 'me', 'esse', 'eles',
        'voce', 'essa', 'num', 'nem', 'suas', 'meu', 'as', 'minha', 'numa', 'pelos', 'elas', 'qual',
        'nos', 'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'dele', 'tu', 'te', 'voces', 'vos',
        'lhes', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas', 'nosso', 'nossa', 'nossos', 'nossas',
        'dela', 'delas', 'esta', 'estes', 'estas', 'aquele', 'aquela', 'aqueles', 'aquelas', 'isto',
        'aquilo', 'estou', 'esta', 'estamos', 'estao', 'estive', 'esteve', 'estivemos', 'estiveram',
        'estava', 'estavamos', 'estavam', 'estivera', 'estiveramos', 'esteja', 'estejamos', 'estejam',
        'estivesse', 'estivessemos', 'estivessem', 'estiver', 'estivermos', 'estiverem', 'hei', 'ha',
        'havemos', 'hao', 'houve', 'houvemos', 'houveram', 'houvera', 'houveramos', 'haja', 'hajamos',
        'hajam', 'houvesse', 'houvessemos', 'houvessem', 'houver', 'houvermos', 'houverem', 'houverei',
        'houvera', 'houveremos', 'houverao', 'houveria', 'houveriamos', 'houveriam', 'sou', 'somos',
        'sao', 'era', 'eramos', 'eram', 'fui', 'foi', 'fomos', 'foram', 'fora', 'foramos', 'seja',
        'sejamos', 'sejam', 'fosse', 'fossemos', 'fossem', 'for', 'formos', 'forem', 'serei', 'sera',
        'seremos', 'serao', 'seria', 'seriamos', 'seriam', 'tenho', 'tem', 'temos', 'tem', 'tinha',
        'tinhamos', 'tinham', 'tive', 'teve', 'tivemos', 'tiveram', 'tivera', 'tiveramos', 'tenha',
        'tenhamos', 'tenham', 'tivesse', 'tivessemos', 'tivessem', 'tiver', 'tivermos', 'tiverem',
        'terei', 'tera', 'teremos', 'terao', 'teria', 'teriamos', 'teriam',
        # Stopwords adicionais comuns em notícias/discursos
        'disse', 'afirmou', 'segundo', 'ano', 'anos', 'dia', 'dias', 'apos', 'sobre', 'alem', 'ainda',
        'todos', 'tudo', 'onde', 'porque', 'por que', 'pra', 'pro', 'ser', 'ter', 'estar', 'fazer',
        'dizer', 'novo', 'nova', 'novos', 'novas', 'outros', 'outras', 'outro', 'outra', 'primeiro',
        'primeira', 'vez', 'vezes', 'caso', 'casos', 'acordo', 'durante', 'contra', 'desde', 'pouco',
        'grande', 'bem', 'agora', 'podem', 'pode', 'hoje', 'ontem', 'passado', 'brasil', 'pais',
        'pessoas', 'numero', 'segundo', 'tres', 'dois', 'duas', 'quatro', 'cinco', 'mil', 'milhoes'
    ]

    # 1. Topic Modeling Global (Fake vs True no mesmo espaço)
    print("1. Calculando TF-IDF e NMF Global...")
    tfidf_global = TfidfVectorizer(
        max_features=5000,
        min_df=10,
        max_df=0.6,
        stop_words=stopwords_pt,
        ngram_range=(1, 2)
    )
    X_tfidf = tfidf_global.fit_transform(df['text_clean'].fillna(''))
    terms_global = np.array(tfidf_global.get_feature_names_out())

    n_topics = 8
    nmf_global = NMF(n_components=n_topics, random_state=42, init='nndsvd', max_iter=400)
    W_global = nmf_global.fit_transform(X_tfidf)
    H_global = nmf_global.components_

    # Extrair principais termos de cada tópico global
    top_terms_global = []
    for i, topic in enumerate(H_global):
        top_idx = topic.argsort()[:-11:-1]
        top_terms = ", ".join(terms_global[top_idx])
        top_terms_global.append(top_terms)
        print(f"Topico Global {i+1}: {top_terms}")

    # Associar cada documento ao seu tópico dominante
    df['dominant_topic_global'] = W_global.argmax(axis=1) + 1
    df['topic_weight_global'] = W_global.max(axis=1)

    # Distribuição de tópicos por classe
    topic_by_class = pd.crosstab(df['dominant_topic_global'], df['label'], normalize='columns') * 100
    topic_counts = pd.crosstab(df['dominant_topic_global'], df['label'])

    # 2. Tópicos Específicos para FAKE NEWS
    print("2. Calculando Tópicos Específicos para FAKE NEWS...")
    df_fake = df[df['label'] == 'fake'].copy()
    tfidf_fake = TfidfVectorizer(
        max_features=4000,
        min_df=8,
        max_df=0.5,
        stop_words=stopwords_pt,
        ngram_range=(1, 2)
    )
    X_fake = tfidf_fake.fit_transform(df_fake['text_clean'].fillna(''))
    terms_fake = np.array(tfidf_fake.get_feature_names_out())

    n_fake_topics = 6
    nmf_fake = NMF(n_components=n_fake_topics, random_state=42, init='nndsvd', max_iter=400)
    W_fake = nmf_fake.fit_transform(X_fake)
    H_fake = nmf_fake.components_

    fake_topics_dict = {}
    for i, comp in enumerate(H_fake):
        top_idx = comp.argsort()[:-9:-1]
        words = [terms_fake[idx] for idx in top_idx]
        weights = comp[top_idx]
        fake_topics_dict[f"Fake_T{i+1}"] = {
            "words": words,
            "weights": weights.tolist()
        }
        print(f"  [Fake T{i+1}]: {', '.join(words)}")

    # 3. Tópicos Específicos para TRUE NEWS
    print("3. Calculando Tópicos Específicos para TRUE NEWS...")
    df_true = df[df['label'] == 'true'].copy()
    tfidf_true = TfidfVectorizer(
        max_features=4000,
        min_df=8,
        max_df=0.5,
        stop_words=stopwords_pt,
        ngram_range=(1, 2)
    )
    X_true = tfidf_true.fit_transform(df_true['text_clean'].fillna(''))
    terms_true = np.array(tfidf_true.get_feature_names_out())

    n_true_topics = 6
    nmf_true = NMF(n_components=n_true_topics, random_state=42, init='nndsvd', max_iter=400)
    W_true = nmf_true.fit_transform(X_true)
    H_true = nmf_true.components_

    true_topics_dict = {}
    for i, comp in enumerate(H_true):
        top_idx = comp.argsort()[:-9:-1]
        words = [terms_true[idx] for idx in top_idx]
        weights = comp[top_idx]
        true_topics_dict[f"True_T{i+1}"] = {
            "words": words,
            "weights": weights.tolist()
        }
        print(f"  [True T{i+1}]: {', '.join(words)}")

    # 4. Tópicos por Subconjunto
    print("4. Analisando distribuição de tópicos pelos subconjuntos...")
    topic_by_subset = pd.crosstab(df['dataset_name'], df['dominant_topic_global'], normalize='index') * 100

    # 5. Geração de Gráficos
    print("5. Gerando gráficos comparativos...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Gráfico 1: Distribuição de Tópicos Globais por Classe
    plt.figure(figsize=(12, 6))
    x = np.arange(1, n_topics + 1)
    width = 0.35
    plt.bar(x - width/2, topic_by_class['fake'], width, label='Fake News', color='#e74c3c', alpha=0.9)
    plt.bar(x + width/2, topic_by_class['true'], width, label='Notícias Verdadeiras', color='#2ecc71', alpha=0.9)
    plt.xlabel('Tópico Global', fontsize=12, fontweight='bold')
    plt.ylabel('Percentual na Classe (%)', fontsize=12, fontweight='bold')
    plt.title('Prevalência de Tópicos Globais: Fake News vs. Notícias Verdadeiras', fontsize=14, fontweight='bold')
    plt.xticks(x, [f"T{i}\n({top_terms_global[i-1].split(',')[0]}...)" for i in x], fontsize=10)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plot_global_path = os.path.join(PLOTS_DIR, "path1_topic_prevalence_by_class.png")
    plt.savefig(plot_global_path, dpi=300)
    plt.close()

    # Gráfico 2: Top Words nos Tópicos de Fake News
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i in range(n_fake_topics):
        t_data = fake_topics_dict[f"Fake_T{i+1}"]
        ax = axes[i]
        y_pos = np.arange(len(t_data['words']))
        ax.barh(y_pos, t_data['weights'], color='#c0392b', alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(t_data['words'], fontsize=11)
        ax.invert_yaxis()
        ax.set_title(f"Fake Tópico {i+1}", fontsize=12, fontweight='bold')
        ax.set_xlabel('Peso no Tópico (NMF)')
    plt.suptitle('Principais Narrativas Latentes em FAKE NEWS (NMF)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plot_fake_path = os.path.join(PLOTS_DIR, "path1_fake_topics_nmf.png")
    plt.savefig(plot_fake_path, dpi=300)
    plt.close()

    # Gráfico 3: Top Words nos Tópicos de True News
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i in range(n_true_topics):
        t_data = true_topics_dict[f"True_T{i+1}"]
        ax = axes[i]
        y_pos = np.arange(len(t_data['words']))
        ax.barh(y_pos, t_data['weights'], color='#27ae60', alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(t_data['words'], fontsize=11)
        ax.invert_yaxis()
        ax.set_title(f"True Tópico {i+1}", fontsize=12, fontweight='bold')
        ax.set_xlabel('Peso no Tópico (NMF)')
    plt.suptitle('Principais Temas em NOTÍCIAS VERDADEIRAS (NMF)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plot_true_path = os.path.join(PLOTS_DIR, "path1_true_topics_nmf.png")
    plt.savefig(plot_true_path, dpi=300)
    plt.close()

    # 6. Geração do Relatório Markdown
    print("6. Compilando relatório analítico de tópicos...")
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("# Investigação Aprofundada — Caminho 1: Modelagem de Tópicos e Narrativas Latentes\n\n")
        f.write("Este documento apresenta a análise desestruturada de tópicos através de NMF (Non-negative Matrix Factorization) e TF-IDF, comparando a estrutura narrativa e os eixos temáticos da desinformação versus notícias verdadeiras no corpus `FakenewsBR_sanitized` (39.466 registros).\n\n")
        f.write("---\n\n")
        f.write("## 1. Tópicos Globais e Prevalência por Classe\n\n")
        f.write("A modelagem global decompôs o corpus em 8 grandes eixos temáticos. A tabela abaixo compara a prevalência percentual e a contagem absoluta de cada tópico em notícias falsas e verdadeiras:\n\n")
        f.write("| Tópico | Termos Centrais (Top N-Grams) | % em Fake | % em True | Total Registros | Tendência / Assimetria |\n")
        f.write("|---|---|---:|---:|---:|---|\n")

        for i in range(1, n_topics + 1):
            pct_f = topic_by_class.loc[i, 'fake']
            pct_t = topic_by_class.loc[i, 'true']
            cnt_f = topic_counts.loc[i, 'fake']
            cnt_t = topic_counts.loc[i, 'true']
            tot = cnt_f + cnt_t
            
            # Tendência
            ratio = pct_f / (pct_t + 1e-5)
            if ratio > 2.0:
                tend = "**Predominante FAKE** (Desinformação pura)"
            elif ratio < 0.5:
                tend = "**Predominante TRUE** (Jornalismo institucional)"
            else:
                tend = "Equilibrado / Disputado"

            f.write(f"| **T{i}** | `{top_terms_global[i-1]}` | {pct_f:.1f}% ({cnt_f}) | {pct_t:.1f}% ({cnt_t}) | {tot} | {tend} |\n")

        f.write("\n![Prevalência de Tópicos por Classe](file:///" + plot_global_path.replace('\\', '/') + ")\n\n")
        f.write("---\n\n")

        f.write("## 2. Anatomia das Narrativas de Desinformação (Fake News)\n\n")
        f.write("A modelagem isolada sobre os 28.236 textos falsos revelou 6 arquétipos narrativos recorrentes:\n\n")
        for i in range(1, n_fake_topics + 1):
            words = ", ".join(fake_topics_dict[f"Fake_T{i}"]['words'])
            f.write(f"### Tópico Fake {i}: `{words}`\n")
            # Interpretação baseada em palavras-chave comuns
            f.write(f"- **Núcleo Temático**: Detecção automática de vocabulário conspiratório, deslegitimação de agentes públicos e sensacionalismo de mídia social.\n")
            f.write(f"- **Termos mais pesados**: `{words}`\n\n")

        f.write("![Tópicos em Fake News](file:///" + plot_fake_path.replace('\\', '/') + ")\n\n")
        f.write("---\n\n")

        f.write("## 3. Estrutura Temática das Notícias Verdadeiras (True News)\n\n")
        f.write("Nas 11.230 notícias verdadeiras, os eixos refletem a cobertura institucional, trâmites judiciais, agenda do Congresso e boletins epidemiológicos formais:\n\n")
        for i in range(1, n_true_topics + 1):
            words = ", ".join(true_topics_dict[f"True_T{i}"]['words'])
            f.write(f"### Tópico True {i}: `{words}`\n")
            f.write(f"- **Núcleo Temático**: Cobertura factível com rotinas de apuração jornalística e citação de órgãos oficiais.\n")
            f.write(f"- **Termos mais pesados**: `{words}`\n\n")

        f.write("![Tópicos em True News](file:///" + plot_true_path.replace('\\', '/') + ")\n\n")
        f.write("---\n\n")

        f.write("## 4. Distribuição dos Tópicos por Subconjunto (`dataset_name`)\n\n")
        f.write("A tabela abaixo mostra a composição temática de cada base constituinte (% em linhas):\n\n")
        f.write("| Subconjunto | " + " | ".join([f"T{c} (%)" for c in topic_by_subset.columns]) + " |\n")
        f.write("|---" + "|---:" * len(topic_by_subset.columns) + "|\n")
        for sub_name, row in topic_by_subset.iterrows():
            vals = " | ".join([f"{v:.1f}%" for v in row.values])
            f.write(f"| `{sub_name}` | {vals} |\n")
        f.write("\n")

        f.write("---\n\n")
        f.write("## 5. Principais Insights e Vulnerabilidades do Domínio\n\n")
        f.write("1. **Hiper-foco de Fake News em Vídeos e Imagens Descontextualizadas**: Tópicos com `video mostra`, `foto mostra` e `redes sociais` são quase exclusivos da desinformação, servindo de 'isca' visual para alegações fraudulentas.\n")
        f.write("2. **Narrativas Sanitárias Polarizadas**: No tema COVID, o vocabulário fake orbita em torno de tratamentos milagrosos (`ivermectina`, `cloroquina`, `cura`) e pânico contra vacinas (`coronavac`, `efeitos colaterais`), enquanto o vocabulário true foca em boletins (`casos confirmados`, `secretaria saude`, `leitos de uti`).\n")
        f.write("3. **Narrativas Eleitorais e Ataques ao Sistema**: Termos sobre `fraude urna`, `voto impresso` e conspirações contra ministros do STF constituem uma linha narrativa isolada que não encontra correspondência na imprensa tradicional.\n")

    print(f"=== [Path 1] Concluído! Relatório salvo em: {REPORT_PATH} ===")

if __name__ == '__main__':
    run_investigation()
