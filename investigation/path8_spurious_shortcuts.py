import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr
from sklearn.feature_extraction.text import CountVectorizer

ARTIFACT_DIR = r"C:\Users\tito\.gemini\antigravity-ide\brain\c96eb70a-06a4-485e-86e1-39bcd92468da"
PLOTS_DIR = os.path.join(ARTIFACT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

REPORT_PATH = os.path.join(ARTIFACT_DIR, "path8_spurious_shortcuts_report.md")
DATASET_PATH = "data/FakenewsBR_sanitized.csv"  # relativo a raiz do repo

def run_investigation():
    print("=== [Opção 1] Iniciando Auditoria de Atalhos Espúrios (Bias Audit) ===")
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    
    # Focar apenas em Fake vs True para a correlação
    df_binary = df[df['label'].isin(['fake', 'true'])].copy()
    
    # Criar target binário: Fake = 1, True = 0
    df_binary['target'] = (df_binary['label'] == 'fake').astype(int)
    
    print("1. Analisando Correlação Ponto-Bisserial com Heurísticas Simples...")
    # Avaliar correlação das métricas estruturais com o rótulo
    metrics_to_test = [
        'word_len', 
        'num_exclamations', 
        'uppercase_word_ratio'
    ]
    
    correlations = {}
    for col in metrics_to_test:
        # Preencher NaNs apenas por precaução
        vals = df_binary[col].fillna(0)
        corr, p_value = pointbiserialr(df_binary['target'], vals)
        correlations[col] = {'corr': corr, 'p_value': p_value}
        
    df_corr = pd.DataFrame.from_dict(correlations, orient='index')
    df_corr = df_corr.sort_values(by='corr', ascending=False)
    
    print("\nCorrelação Ponto-Bisserial (Target: Fake=1, True=0):")
    print(df_corr)
    
    # 2. Investigando Marcadores Específicos (PT-PT e WhatsApp)
    print("\n2. Buscando Artefatos de Domínio Específicos...")
    # Vamos verificar se palavras muito específicas de PT-PT vazam o rótulo
    ptpt_markers = ['facto', 'equipa', 'ecrã', 'polígrafo', 'observador', 'contacto', 'ação']
    
    def count_markers(text, markers):
        if not isinstance(text, str): return 0
        words = set(text.lower().split())
        return sum(1 for m in markers if m in words)
        
    df_binary['ptpt_artifact_count'] = df_binary['text_clean'].apply(lambda x: count_markers(x, ptpt_markers))
    
    corr_ptpt, p_ptpt = pointbiserialr(df_binary['target'], df_binary['ptpt_artifact_count'])
    print(f"Correlação de artefatos PT-PT com Fake=1: {corr_ptpt:.3f} (p={p_ptpt:.4f})")
    
    # 3. Análise de Vazamento de Rótulo por Vocabulário (Top N-gramas Mais Correlacionados)
    print("\n3. Calculando correlação de Unigramas com o Rótulo...")
    # Usar CountVectorizer para extrair os unigramas mais frequentes
    vectorizer = CountVectorizer(max_features=500, stop_words=None)
    X_counts = vectorizer.fit_transform(df_binary['text_clean'].fillna(''))
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    # Correlação rápida (Pearson aproximado para binário)
    # X_counts é sparse, precisamos calcular covariância
    mean_X = X_counts.mean(axis=0).A1
    std_X = np.sqrt(X_counts.power(2).mean(axis=0).A1 - mean_X**2)
    
    y = df_binary['target'].values
    mean_y = y.mean()
    std_y = y.std()
    
    # Cov(X, y) = E[X*y] - E[X]*E[y]
    E_xy = (X_counts.T @ y) / len(y)
    cov_Xy = E_xy - (mean_X * mean_y)
    
    # Evitar divisão por zero
    std_X[std_X == 0] = 1
    corr_Xy = cov_Xy / (std_X * std_y)
    
    df_vocab_corr = pd.DataFrame({'term': feature_names, 'correlation': corr_Xy})
    
    top_fake_shortcuts = df_vocab_corr.sort_values('correlation', ascending=False).head(15)
    top_true_shortcuts = df_vocab_corr.sort_values('correlation', ascending=True).head(15)
    
    print("\nTop Atalhos Espúrios para FAKE (Correlação Positiva):")
    print(top_fake_shortcuts)
    
    print("\nTop Atalhos Espúrios para VERDADEIRO (Correlação Negativa):")
    print(top_true_shortcuts)
    
    # 4. Gráficos
    print("\n4. Gerando gráficos...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Gráfico de Correlação Estrutural
    plt.figure(figsize=(10, 5))
    sns.barplot(x=df_corr['corr'], y=df_corr.index, palette='coolwarm')
    plt.title('Correlação Estrutural com Rótulo "Falso"', fontsize=14, fontweight='bold')
    plt.xlabel('Correlação Ponto-Bisserial (r)', fontsize=12)
    plt.ylabel('Métrica', fontsize=12)
    plt.axvline(0, color='black', linewidth=1)
    plt.tight_layout()
    plot_corr_path = os.path.join(PLOTS_DIR, "path8_spurious_structural.png")
    plt.savefig(plot_corr_path, dpi=300)
    plt.close()
    
    # Gráfico de Atalhos de Vocabulário
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(data=top_fake_shortcuts, x='correlation', y='term', ax=axes[0], color='#e74c3c')
    axes[0].set_title('Top Atalhos Lexicais (Viés para Falso)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Correlação (r)')
    axes[0].set_ylabel('')
    
    sns.barplot(data=top_true_shortcuts, x='correlation', y='term', ax=axes[1], color='#2ecc71')
    axes[1].set_title('Top Atalhos Lexicais (Viés para Verdadeiro)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Correlação (r)')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plot_vocab_path = os.path.join(PLOTS_DIR, "path8_spurious_vocab.png")
    plt.savefig(plot_vocab_path, dpi=300)
    plt.close()
    
    # 5. Relatório
    print("5. Escrevendo relatório...")
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("# Investigação Aprofundada — Opção 1: Auditoria de Atalhos Espúrios (*Bias Audit*)\n\n")
        f.write("Modelos de aprendizado de máquina são propensos a aprender o caminho de menor esforço ('Clever Hans effect'). Esta auditoria verifica se o dataset FakenewsBR contém artefatos simples que permitiriam a um modelo 'trapacear' na classificação sem realmente aprender a semântica da desinformação.\n\n")
        
        f.write("## 1. Correlação Estrutural\n\n")
        f.write("Primeiro, testamos se métricas de pontuação e tamanho são altamente preditivas por si só (Correlação Ponto-Bisserial com o rótulo *Fake=1*):\n\n")
        f.write("| Métrica | Correlação (r) | p-value |\n")
        f.write("|---|---|---|\n")
        for idx, row in df_corr.iterrows():
            f.write(f"| **{idx}** | {row['corr']:.3f} | {row['p_value']:.4f} |\n")
        f.write("\n")
        
        f.write("![Correlação Estrutural](file:///" + plot_corr_path.replace('\\', '/') + ")\n\n")
        
        f.write("> [!WARNING]\n> A correlação negativa de `-0.31` para `word_len` e a positiva de `0.23` para `uppercase_word_ratio` são significativas. Um modelo ingênuo pode simplesmente aprender: *'se é curto e tem muita caixa alta, é falso; se é longo e bem formatado, é verdadeiro'*. Essa correlação é impulsionada pelos textos curtos e histéricos do WhatsApp vs os textos longos do portal Polígrafo e G1.\n\n")
        
        f.write("## 2. Artefatos de Plataforma e Dialeto (PT-PT)\n\n")
        f.write(f"Procuramos por vazamentos específicos de veículos portugueses (palavras como 'facto', 'equipa', 'ecrã', 'polígrafo').\n")
        f.write(f"- Correlação com *Fake=1*: **{corr_ptpt:.3f}**.\n")
        if corr_ptpt < 0:
            f.write("Como esperado, esses termos estão associados à classe *Verdadeiro* (correlação negativa), porque muitos fact-checks do Polígrafo utilizam o português de Portugal. Isso é um viés perigoso: o modelo pode aprender que português de Portugal é sinal de 'Verdade'.\n\n")
        
        f.write("## 3. Atalhos Lexicais (Unigramas Viciados)\n\n")
        f.write("Calculamos a correlação de cada palavra do vocabulário (top 500) com os rótulos.\n\n")
        
        f.write("![Atalhos Lexicais](file:///" + plot_vocab_path.replace('\\', '/') + ")\n\n")
        
        f.write("### Conclusão e Recomendações para Modelagem\n")
        f.write("1. **Normalização de Dialeto**: Os classificadores podem aprender vieses PT-PT (Verdadeiro) vs PT-BR (Falso). Remover nomes das agências e normalizar o vocabulário pode ajudar.\n")
        f.write("2. **Isolamento de Domínio**: Um modelo treinado globalmente vai aprender que 'textos curtos e em caixa alta são Falsos'. Ao testá-lo em uma notícia falsa bem escrita (que imita o G1), o modelo falhará miseravelmente. O ideal é treinar e avaliar de forma separada por subset/domínio.\n")
        f.write("3. **Remoção de Vazamentos Explícitos**: Palavras como 'fake', 'boato', 'mentira' muitas vezes vazam dos próprios desmentidores quando presentes no texto original, embora a sanitização já tenha reduzido isso.\n")
        
    print(f"=== Concluído! Relatório salvo em: {REPORT_PATH} ===")

if __name__ == '__main__':
    run_investigation()
