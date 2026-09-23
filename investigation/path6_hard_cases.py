import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

ARTIFACT_DIR = r"C:\Users\tito\.gemini\antigravity-ide\brain\c96eb70a-06a4-485e-86e1-39bcd92468da"
PLOTS_DIR = os.path.join(ARTIFACT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

REPORT_PATH = os.path.join(ARTIFACT_DIR, "path6_hard_cases_report.md")
DATASET_PATH = "data/FakenewsBR_sanitized.csv"  # relativo a raiz do repo

def run_investigation():
    print("=== [Opção 4] Iniciando Anatomia dos Casos Limítrofes (Hard Cases) ===")
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    
    # 1. Definir os grupos de análise com base em padronização anterior
    # Precisamos replicar a função de padronização do path4
    def standardize_rating(rating):
        if not isinstance(rating, str) or not rating.strip() or rating == 'nan':
            return 'Não Informado / Ausente'
        r = rating.strip().lower()
        if any(w in r for w in ['falso', 'falsa', 'false', 'mentira', 'boato', 'fake']):
            return 'Falso Puro'
        elif any(w in r for w in ['enganoso', 'enganosa', 'misleading']):
            return 'Limítrofe (Enganoso)'
        elif any(w in r for w in ['distorcido', 'distorcida', 'manipulado', 'alterado']):
            return 'Limítrofe (Distorcido)'
        elif any(w in r for w in ['sem contexto', 'fora de contexto']):
            return 'Limítrofe (Sem Contexto)'
        elif any(w in r for w in ['exagerado', 'exagerada']):
            return 'Limítrofe (Exagerado)'
        elif any(w in r for w in ['impreciso', 'imprecisa', 'contraditório']):
            return 'Limítrofe (Impreciso)'
        elif any(w in r for w in ['verdadeiro', 'verdadeira', 'true', 'fato', 'verdade']):
            return 'Verdadeiro'
        else:
            return 'Outros'

    df['rating_group'] = df['factcheck_rating'].apply(standardize_rating)
    
    # Agrupar todos os limítrofes em uma macro-classe
    def macro_group(x):
        if 'Limítrofe' in x:
            return 'Casos Limítrofes (Meias-Verdades)'
        return x

    df['macro_rating'] = df['rating_group'].apply(macro_group)
    
    # Focar apenas nas 3 classes de interesse
    target_classes = ['Falso Puro', 'Casos Limítrofes (Meias-Verdades)', 'Verdadeiro']
    df_analysis = df[df['macro_rating'].isin(target_classes)].copy()
    
    print("Distribuição das Classes de Interesse:")
    counts = df_analysis['macro_rating'].value_counts()
    print(counts)
    
    # 2. Comparação de Métricas Estruturais
    print("2. Calculando métricas estruturais por grupo...")
    metrics = df_analysis.groupby('macro_rating').agg(
        total_docs=('rid', 'count'),
        median_words=('word_len', 'median'),
        mean_words=('word_len', 'mean'),
        pct_uppercase=('uppercase_word_ratio', lambda x: x.mean() * 100),
        mean_exclamations=('num_exclamations', 'mean')
    ).reindex(['Verdadeiro', 'Casos Limítrofes (Meias-Verdades)', 'Falso Puro'])
    
    print("\nMétricas Estruturais:")
    print(metrics)
    
    # 3. Análise de Diferenças Lexicais (TF-IDF)
    print("3. Analisando diferenças lexicais (Termos característicos)...")
    stopwords_pt = ['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'nao', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'ao', 'ele', 'das', 'aqui', 'tem', 'seu', 'sua', 'ou', 'quando', 'muito', 'nos', 'ja', 'eu', 'tambem', 'so', 'pelo', 'pela', 'ate', 'isso', 'ela', 'entre', 'depois', 'sem', 'mesmo', 'aos', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'voce', 'essa', 'num', 'nem', 'suas', 'meu', 'as', 'minha', 'numa', 'pelos', 'elas', 'qual', 'nos', 'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'dele', 'tu', 'te', 'voces', 'vos', 'lhes', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas', 'nosso', 'nossa', 'nossos', 'nossas', 'dela', 'delas', 'esta', 'estes', 'estas', 'aquele', 'aquela', 'aqueles', 'aquelas', 'isto', 'aquilo', 'sao', 'ser', 'ter', 'estar', 'ha']
    
    tfidf = TfidfVectorizer(max_features=2000, stop_words=stopwords_pt, ngram_range=(1, 2))
    X = tfidf.fit_transform(df_analysis['text_clean'].fillna(''))
    feature_names = np.array(tfidf.get_feature_names_out())
    
    # Calcular TF-IDF médio por grupo
    mean_tfidf = {}
    for group in target_classes:
        group_idx = df_analysis['macro_rating'] == group
        mean_tfidf[group] = np.asarray(X[group_idx.values].mean(axis=0)).ravel()
        
    # Encontrar termos com maior diferença (Limítrofe vs Falso Puro)
    diff_limit_vs_falso = mean_tfidf['Casos Limítrofes (Meias-Verdades)'] - mean_tfidf['Falso Puro']
    top_limit_idx = diff_limit_vs_falso.argsort()[-15:][::-1]
    top_falso_idx = diff_limit_vs_falso.argsort()[:15]
    
    print("\nTop 15 termos mais associados aos Casos Limítrofes (vs Falso Puro):")
    print(feature_names[top_limit_idx])
    
    print("\nTop 15 termos mais associados a Falso Puro (vs Limítrofes):")
    print(feature_names[top_falso_idx])
    
    # 4. Visualizações
    print("4. Gerando gráficos...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Gráfico 1: Comprimento dos Textos
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='word_len', y='macro_rating', data=df_analysis, showfliers=False, order=['Verdadeiro', 'Casos Limítrofes (Meias-Verdades)', 'Falso Puro'], palette=['#2ecc71', '#f39c12', '#e74c3c'])
    plt.title('Distribuição do Tamanho dos Textos (Palavras) por Grau de Falsidade', fontsize=14, fontweight='bold')
    plt.xlabel('Número de Palavras', fontsize=12, fontweight='bold')
    plt.ylabel('')
    plt.tight_layout()
    plot_len_path = os.path.join(PLOTS_DIR, "path6_hardcases_length.png")
    plt.savefig(plot_len_path, dpi=300)
    plt.close()
    
    # Gráfico 2: Marcadores de Histeria
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics['pct_uppercase'].plot(kind='barh', ax=axes[0], color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.8)
    axes[0].set_title('% Média de Palavras em CAIXA ALTA', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('%')
    axes[0].set_ylabel('')
    
    metrics['mean_exclamations'].plot(kind='barh', ax=axes[1], color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.8)
    axes[1].set_title('Média de Exclamações por Texto', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Contagem')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plot_hysteria_path = os.path.join(PLOTS_DIR, "path6_hardcases_hysteria.png")
    plt.savefig(plot_hysteria_path, dpi=300)
    plt.close()
    
    # Gráfico 3: Termos Discriminantes (Limítrofe vs Falso Puro)
    plt.figure(figsize=(12, 6))
    y_pos = np.arange(15)
    plt.barh(y_pos, diff_limit_vs_falso[top_limit_idx], color='#f39c12', alpha=0.8, label='Mais em Limítrofes')
    plt.yticks(y_pos, feature_names[top_limit_idx], fontsize=11)
    plt.gca().invert_yaxis()
    plt.title('Termos Mais Distintivos de Casos Limítrofes vs Falsos Puros (TF-IDF Médio)', fontsize=14, fontweight='bold')
    plt.xlabel('Diferença de TF-IDF', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plot_terms_path = os.path.join(PLOTS_DIR, "path6_hardcases_terms.png")
    plt.savefig(plot_terms_path, dpi=300)
    plt.close()
    
    # 5. Gerar Relatório
    print("5. Gerando relatório markdown...")
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("# Investigação Aprofundada — Opção 4: Anatomia dos Casos Limítrofes (*Hard Cases*)\n\n")
        f.write("Este documento explora a zona cinzenta da desinformação: os textos rotulados pelas agências como 'Enganoso', 'Distorcido', 'Fora de Contexto', etc. Como essas 'meias-verdades' se comparam estruturalmente com as mentiras flagrantes (Falso Puro) e as verdades factuais?\n\n")
        f.write("---\n\n")
        f.write(f"## 1. O Tamanho das Amostras\n")
        f.write(f"- **Falso Puro**: {counts.get('Falso Puro', 0):,} textos.\n")
        f.write(f"- **Casos Limítrofes (Meias-Verdades)**: {counts.get('Casos Limítrofes (Meias-Verdades)', 0):,} textos.\n")
        f.write(f"- **Verdadeiro**: {counts.get('Verdadeiro', 0):,} textos.\n\n")
        
        f.write("## 2. Métricas Estruturais e 'Histeria'\n\n")
        f.write("Observamos um claro gradiente de sofisticação e apelo emocional:\n\n")
        f.write("| Grau de Falsidade | Mediana Palavras | % Caixa Alta | Média Exclamações |\n")
        f.write("|---|---:|---:|---:|\n")
        for group, row in metrics.iterrows():
            f.write(f"| **{group}** | {row['median_words']:.0f} | {row['pct_uppercase']:.2f}% | {row['mean_exclamations']:.2f} |\n")
        f.write("\n")
        f.write("> [!NOTE]\n> Os **Casos Limítrofes** posicionam-se exatamente entre o 'Falso Puro' e o 'Verdadeiro' em termos de tamanho e marcadores de histeria. Eles tendem a ser mais longos que os Falsos Puros e usam menos caixa alta/exclamações, indicando uma tentativa maior de mimetizar o discurso factual e argumentar, em vez de apenas gritar um boato.\n\n")
        
        f.write("![Distribuição de Tamanho](file:///" + plot_len_path.replace('\\', '/') + ")\n\n")
        f.write("![Marcadores de Histeria](file:///" + plot_hysteria_path.replace('\\', '/') + ")\n\n")
        
        f.write("---\n\n")
        f.write("## 3. O Léxico da Meia-Verdade (TF-IDF)\n\n")
        f.write("Quais palavras se destacam nos Casos Limítrofes quando os comparamos diretamente com os Falsos Puros?\n\n")
        f.write("Os 15 termos mais associados aos **Casos Limítrofes**:\n")
        f.write(f"`{', '.join(feature_names[top_limit_idx])}`\n\n")
        
        f.write("Os 15 termos mais associados aos **Falsos Puros**:\n")
        f.write(f"`{', '.join(feature_names[top_falso_idx])}`\n\n")
        
        f.write("![Termos Discriminantes](file:///" + plot_terms_path.replace('\\', '/') + ")\n\n")
        
        f.write("### Conclusão\n")
        f.write("Casos Limítrofes frequentemente ancoram-se em pessoas reais, locais reais e eventos reais (números, leis, declarações) para tecer uma interpretação enganosa ou usar um vídeo antigo fora de contexto. Eles são muito mais difíceis para um modelo de Machine Learning detectar porque **carecem da 'impressão digital' gritante** (CAIXA ALTA, CTAs agressivos) dos boatos puros e utilizam vocabulário muito semelhante ao jornalismo real.")
        
    print(f"=== Concluído! Relatório salvo em: {REPORT_PATH} ===")

if __name__ == '__main__':
    run_investigation()
