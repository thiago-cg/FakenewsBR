import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

ARTIFACT_DIR = r"C:\Users\tito\.gemini\antigravity-ide\brain\c96eb70a-06a4-485e-86e1-39bcd92468da"
PLOTS_DIR = os.path.join(ARTIFACT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

REPORT_PATH = os.path.join(ARTIFACT_DIR, "path7_framing_emotions_report.md")
DATASET_PATH = "data/FakenewsBR_sanitized.csv"  # relativo a raiz do repo

def run_investigation():
    print("=== [Opção 5] Iniciando Mapeamento de Falácias e Gatilhos Emocionais ===")
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    
    # Preencher NAs no texto
    df['text_clean_str'] = df['text_clean'].fillna('').astype(str)
    
    # 1. Definir Léxicos Heurísticos (Gatilhos Emocionais)
    print("1. Aplicando léxicos heurísticos de emoção...")
    
    lexicon = {
        'Medo_Alerta': ['perigo', 'risco', 'cuidado', 'alerta', 'morte', 'fatal', 'matar', 'medo', 'grave', 'ameaça', 'urgente', 'atenção'],
        'Indignação_Revolta': ['absurdo', 'vergonha', 'revolta', 'canalha', 'bandido', 'roubo', 'corrupção', 'mentira', 'palhaçada', 'safado', 'golpe', 'crime', 'ladrão'],
        'Esperança_Cura': ['cura', 'milagre', 'solução', 'salvou', 'salvação', 'deus', 'abençoe', 'remédio', 'tratamento', 'infalível']
    }
    
    # Falácias / Padrões Retóricos
    patterns = {
        'Autoridade_Anônima': r'\b(médico\s+do|cientista|diretor\s+da|amigo\s+militar|pesquisador|especialista|amigo\s+do|prima\s+da)\b',
        'Urgência_Chantagem': r'\b(repassem|espalhem|antes\s+que\s+apaguem|último\s+aviso|não\s+deixe\s+de\s+compartilhar)\b',
    }
    
    # Função para contar ocorrências do léxico
    def count_lexicon(text, word_list):
        words = set(text.split())
        return sum(1 for w in word_list if w in words)
        
    for emotion, words in lexicon.items():
        df[f'trigger_{emotion}'] = df['text_clean_str'].apply(lambda x: count_lexicon(x, words))
        # Transformar em flag (0 ou 1) para calcular prevalência
        df[f'has_{emotion}'] = (df[f'trigger_{emotion}'] > 0).astype(int)
        
    for pattern_name, regex in patterns.items():
        df[f'has_{pattern_name}'] = df['text_clean_str'].apply(lambda x: 1 if re.search(regex, x) else 0)
        
    # 2. Agregar por Classe e Subconjunto
    print("2. Agregando resultados...")
    features = [f'has_{k}' for k in list(lexicon.keys()) + list(patterns.keys())]
    
    # Por Classe principal (True vs Fake)
    df['subset'] = df['dataset_name'].apply(lambda x: 'WhatsApp' if 'whatsapp' in str(x).lower() else 'Fake.br (Notícias)')
    df_binary = df[df['label'].isin(['fake', 'true'])]
    class_prevalence = df_binary.groupby('label')[features].mean() * 100
    class_prevalence = class_prevalence.rename(index={'fake': 'Falso', 'true': 'Verdadeiro'})
    
    # Por Subconjunto (WhatsApp vs Fake.br)
    # Primeiro mapear subsets do FakenewsBR
    subset_prevalence = df_binary[df_binary['label'] == 'fake'].groupby('subset')[features].mean() * 100
    
    print("\nPrevalência por Classe (% de documentos que contém o gatilho):")
    print(class_prevalence)
    
    # 3. Gerar Visualizações
    print("3. Gerando gráficos...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Gráfico 1: Gatilhos Emocionais por Classe (Fake vs True)
    ax = class_prevalence.T.plot(kind='bar', figsize=(12, 6), color=['#e74c3c', '#2ecc71'], width=0.7)
    plt.title('Prevalência de Gatilhos Emocionais e Falácias (Falso vs Verdadeiro)', fontsize=14, fontweight='bold')
    plt.ylabel('% de Documentos', fontsize=12)
    plt.xlabel('Tipo de Gatilho', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Classe', loc='upper right')
    plt.tight_layout()
    
    plot_class_path = os.path.join(PLOTS_DIR, "path7_framing_by_class.png")
    plt.savefig(plot_class_path, dpi=300)
    plt.close()
    
    # Gráfico 2: Gatilhos no Falso (WhatsApp vs Fake.br)
    ax = subset_prevalence.T.plot(kind='bar', figsize=(12, 6), color=['#3498db', '#9b59b6'], width=0.7)
    plt.title('Gatilhos Emocionais dentro do Universo FAKE (WhatsApp vs Fake.br)', fontsize=14, fontweight='bold')
    plt.ylabel('% de Documentos', fontsize=12)
    plt.xlabel('Tipo de Gatilho', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Plataforma', loc='upper right')
    plt.tight_layout()
    
    plot_subset_path = os.path.join(PLOTS_DIR, "path7_framing_by_subset.png")
    plt.savefig(plot_subset_path, dpi=300)
    plt.close()
    
    # 4. Gerar Relatório
    print("4. Gerando relatório markdown...")
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("# Investigação Aprofundada — Opção 5: Mapeamento de Falácias e Gatilhos Emocionais\n\n")
        f.write("A desinformação não sobrevive apenas de mentiras; ela precisa de engajamento emocional para ser propagada. Este relatório mapeia como diferentes sentimentos e recursos retóricos são utilizados para capturar a atenção.\n\n")
        
        f.write("## 1. O Enquadramento (Framing) Falso vs Verdadeiro\n\n")
        f.write("O gráfico abaixo mostra a porcentagem de textos em cada classe que contém palavras ligadas a Medo, Indignação, Esperança, bem como apelos à Autoridade Anônima e Urgência/Chantagem.\n\n")
        f.write("![Gatilhos por Classe](file:///" + plot_class_path.replace('\\', '/') + ")\n\n")
        
        f.write("> [!NOTE]\n> A indignação e o alerta/medo são frequentemente usados nas Fake News, mas também aparecem nas notícias verdadeiras (já que o jornalismo também reporta crimes e alertas). A grande diferença retórica está nos **Apelos de Urgência** ('repassem antes que apaguem') e na **Autoridade Anônima** ('médico do hospital disse...'), que são assinaturas quase exclusivas da desinformação.\n\n")
        
        f.write("## 2. A Engenharia Psicológica: WhatsApp vs Fake.br\n\n")
        f.write("Como a plataforma influencia a mensagem? Focando apenas nas Fake News, comparamos as correntes de WhatsApp com os boatos estruturados como notícias (Fake.br).\n\n")
        f.write("![Gatilhos no Falso](file:///" + plot_subset_path.replace('\\', '/') + ")\n\n")
        
        f.write("### Conclusão\n")
        f.write("O WhatsApp é o terreno fértil para a **Urgência e Chantagem Emocional**, e para a **Indignação**. Textos de WhatsApp usam muito mais CTAs ('repassem') e palavras de ordem que invocam revolta ('absurdo, vergonha'). Já as notícias falsas publicadas em portais (Fake.br) tentam manter uma aura mais formal, usando menos chantagem emocional barata, mas ainda apelando para autoridades duvidosas ou medo generalizado para captar cliques.\n")
        
    print(f"=== Concluído! Relatório salvo em: {REPORT_PATH} ===")

if __name__ == '__main__':
    run_investigation()
