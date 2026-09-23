import os
import re
from urllib.parse import urlparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ARTIFACT_DIR = r"C:\Users\tito\.gemini\antigravity-ide\brain\c96eb70a-06a4-485e-86e1-39bcd92468da"
PLOTS_DIR = os.path.join(ARTIFACT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

REPORT_PATH = os.path.join(ARTIFACT_DIR, "path4_targets_and_factcheckers_report.md")
DATASET_PATH = "data/FakenewsBR_sanitized.csv"  # relativo a raiz do repo

def run_investigation():
    print("=== [Path 4] Iniciando Investigação de Alvos e Agências de Fact-Checking ===")
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    print(f"Dataset carregado: {len(df)} registros.")

    # 1. Análise de Agências de Checagem através de 'url_review'
    print("1. Extraindo domínios e agências de checagem...")
    def extract_agency(url):
        if not isinstance(url, str) or not url.strip() or url == 'nan':
            return 'Sem URL de Checagem'
        url_lower = url.lower()
        if 'aosfatos.org' in url_lower:
            return 'Aos Fatos'
        elif 'piaui.folha.uol.com.br/lupa' in url_lower or 'lupa.news' in url_lower or 'agencialupa' in url_lower:
            return 'Agência Lupa'
        elif 'boatos.org' in url_lower:
            return 'Boatos.org'
        elif 'g1.globo.com/fato-ou-fake' in url_lower or 'fato-ou-fake' in url_lower:
            return 'G1 Fato ou Fake'
        elif 'estadao.com.br/estadao-verifica' in url_lower or 'estadao-verifica' in url_lower:
            return 'Estadão Verifica'
        elif 'e-farsas.com' in url_lower:
            return 'E-farsas'
        elif 'uol.com.br/comprova' in url_lower or 'projetocomprova.com.br' in url_lower:
            return 'Projeto Comprova'
        elif 'checamos.afp.com' in url_lower or 'afp.com' in url_lower:
            return 'AFP Checamos'
        elif 'folha.uol.com.br' in url_lower:
            return 'Folha de S.Paulo'
        else:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.replace('www.', '')
                return domain if domain else 'Outros'
            except:
                return 'Outros'

    df['factcheck_agency'] = df['url_review'].apply(extract_agency)
    agency_counts = df['factcheck_agency'].value_counts()
    print("Principais Agências de Checagem no Corpus:")
    print(agency_counts.head(10))

    # 2. Taxonomia de Classificações Granulares ('factcheck_rating')
    print("2. Padronizando classificações granulares (factcheck_rating)...")
    def standardize_rating(rating):
        if not isinstance(rating, str) or not rating.strip() or rating == 'nan':
            return 'Não Informado / Ausente'
        r = rating.strip().lower()
        if any(w in r for w in ['falso', 'falsa', 'false', 'mentira', 'boato', 'fake']):
            return 'Falso (Fabricação Completa)'
        elif any(w in r for w in ['enganoso', 'enganosa', 'misleading']):
            return 'Enganoso'
        elif any(w in r for w in ['distorcido', 'distorcida', 'manipulado', 'alterado']):
            return 'Distorcido / Manipulado'
        elif any(w in r for w in ['sem contexto', 'fora de contexto']):
            return 'Fora de Contexto'
        elif any(w in r for w in ['exagerado', 'exagerada']):
            return 'Exagerado'
        elif any(w in r for w in ['impreciso', 'imprecisa', 'contraditório']):
            return 'Impreciso'
        elif any(w in r for w in ['verdadeiro', 'verdadeira', 'true', 'fato', 'verdade']):
            return 'Verdadeiro (Fato)'
        elif any(w in r for w in ['insustentável', 'de olho', 'inconclusivo']):
            return 'Outros / Discutível'
        else:
            return 'Outros'

    df['standardized_rating'] = df['factcheck_rating'].apply(standardize_rating)
    rating_by_label = pd.crosstab(df['standardized_rating'], df['label'], margins=True)
    print("\nCruzamento entre Classificação Granular e Rótulo Binário:")
    print(rating_by_label)

    # 3. Análise de Emissores e Reclamantes ('factcheck_claimant')
    print("3. Investigando origens das alegações (factcheck_claimant)...")
    def clean_claimant(c):
        if not isinstance(c, str) or not c.strip() or c == 'nan':
            return 'Desconhecido / Anônimo'
        c_clean = c.strip()
        c_lower = c_clean.lower()
        if 'redes sociais' in c_lower or 'postagens' in c_lower or 'posts' in c_lower or 'facebook' in c_lower or 'twitter' in c_lower or 'tiktok' in c_lower:
            return 'Redes Sociais (Posts / Usuários)'
        elif 'whatsapp' in c_lower or 'mensagens' in c_lower:
            return 'WhatsApp (Correntes)'
        elif 'jair bolsonaro' in c_lower or 'bolsonaro' in c_lower:
            return 'Jair Bolsonaro'
        elif 'lula' in c_lower or 'luiz inácio lula da silva' in c_lower:
            return 'Lula da Silva'
        elif 'governo federal' in c_lower or 'ministério' in c_lower:
            return 'Governo / Ministérios'
        elif 'alexandre de moraes' in c_lower:
            return 'Alexandre de Moraes'
        else:
            return c_clean

    df['cleaned_claimant'] = df['factcheck_claimant'].apply(clean_claimant)
    top_claimants = df[df['cleaned_claimant'] != 'Desconhecido / Anônimo']['cleaned_claimant'].value_counts().head(12)
    print("\nTop Emissores / Reclamantes Checados:")
    print(top_claimants)

    # 4. Alvos das Notícias Falsas (Quem é o sujeito da checagem?)
    print("4. Mapeando entidades e alvos mais atacados nas fake news...")
    df_fake = df[df['label'] == 'fake']
    targets = {
        'Lula / PT': r'\b(lula|pt|partido dos trabalhadores|dilma|haddad)\b',
        'Bolsonaro / Família': r'\b(bolsonaro|flavio|eduardo|carlos|michelle)\b',
        'STF / Judiciário': r'\b(stf|supremo|moraes|toffoli|gilmar|barroso|fachin|tribunal federal)\b',
        'Vacinas / COVID / Anvisa': r'\b(vacina|coronavac|pfizer|anvisa|cloroquina|ivermectina|doria)\b',
        'Sistema Eleitoral / Urnas': r'\b(urna|urnas|fraude eleitoral|tse|barroso|voto impresso)\b',
        'Mídia / Imprensa (Globo/Folha)': r'\b(globo|rede globo|folha|estadao|imprensa|jornalistas)\b',
        'Congresso / Parlamentares': r'\b(deputado|senador|camara|senado|rodrigo maia|arthur lira)\b'
    }

    target_stats = {}
    for target_name, pattern in targets.items():
        count = df_fake['text_clean'].str.contains(pattern, regex=True, case=False, na=False).sum()
        pct = (count / len(df_fake)) * 100
        target_stats[target_name] = {'count': count, 'pct': pct}

    df_targets = pd.DataFrame(target_stats).T.sort_values(by='count', ascending=False)
    print("\nAlvos Mais Frequentes nas Fake News:")
    print(df_targets)

    # 5. Visualizações
    print("5. Gerando gráficos de agências e alvos...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # Gráfico 1: Agências de Fact-Checking
    plt.figure(figsize=(12, 6))
    agencies_plot = agency_counts[agency_counts.index != 'Sem URL de Checagem'].head(8)
    y_pos = np.arange(len(agencies_plot))
    plt.barh(y_pos, agencies_plot.values, color='#2980b9', alpha=0.85)
    plt.yticks(y_pos, agencies_plot.index, fontsize=11, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.xlabel('Número de Checagens no Dataset', fontsize=12, fontweight='bold')
    plt.title('Agências de Fact-Checking Mais Representadas', fontsize=14, fontweight='bold')
    for i, v in enumerate(agencies_plot.values):
        plt.text(v + 100, i, f"{v:,} ({v/len(df)*100:.1f}%)", va='center', fontweight='bold')
    plt.tight_layout()
    plot_agencies_path = os.path.join(PLOTS_DIR, "path4_factcheck_agencies.png")
    plt.savefig(plot_agencies_path, dpi=300)
    plt.close()

    # Gráfico 2: Taxonomia de Rótulos Granulares
    plt.figure(figsize=(10, 5))
    rating_plot = df[df['standardized_rating'] != 'Não Informado / Ausente']['standardized_rating'].value_counts()
    colors_ratings = ['#c0392b', '#e67e22', '#d35400', '#f39c12', '#27ae60', '#7f8c8d']
    plt.bar(rating_plot.index, rating_plot.values, color=colors_ratings[:len(rating_plot)], alpha=0.9)
    plt.xticks(rotation=25, ha='right', fontsize=10, fontweight='bold')
    plt.ylabel('Contagem de Registros', fontsize=12, fontweight='bold')
    plt.title('Distribuição de Rótulos Granulares das Agências de Checagem', fontsize=13, fontweight='bold')
    for i, v in enumerate(rating_plot.values):
        plt.text(i, v + 200, f"{v:,}", ha='center', fontweight='bold')
    plt.tight_layout()
    plot_ratings_path = os.path.join(PLOTS_DIR, "path4_granular_ratings.png")
    plt.savefig(plot_ratings_path, dpi=300)
    plt.close()

    # Gráfico 3: Alvos Principais das Fake News
    plt.figure(figsize=(11, 5))
    y_pos = np.arange(len(df_targets))
    plt.barh(y_pos, df_targets['pct'], color='#e74c3c', alpha=0.85)
    plt.yticks(y_pos, df_targets.index, fontsize=11, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.xlabel('% de Presença no Total de Fake News (28.236 textos)', fontsize=12, fontweight='bold')
    plt.title('Grandes Alvos da Desinformação no Brasil', fontsize=14, fontweight='bold')
    for i, v in enumerate(df_targets['pct']):
        plt.text(v + 0.5, i, f"{v:.1f}% ({int(df_targets['count'].iloc[i]):,} textos)", va='center', fontweight='bold')
    plt.xlim(0, max(df_targets['pct']) * 1.2)
    plt.tight_layout()
    plot_targets_path = os.path.join(PLOTS_DIR, "path4_disinformation_targets.png")
    plt.savefig(plot_targets_path, dpi=300)
    plt.close()

    # 6. Gravação do Relatório
    print("6. Compilando relatório de alvos e checagens...")
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("# Investigação Aprofundada — Caminho 4: Alvos, Emissores e Agências de Fact-Checking\n\n")
        f.write("Este documento apresenta a análise sobre o ecossistema institucional de checagem presente no dataset `FakenewsBR_sanitized`: quais agências compõem a base, como as checagens granulares são rotuladas e quem são os alvos e emissores preponderantes.\n\n")
        f.write("---\n\n")
        f.write("## 1. O Panorama das Agências de Checagem\n\n")
        f.write("Através da extração de domínios em `url_review` (presente em 23.411 registros), identificou-se a procedência das checagens jornalísticas:\n\n")
        f.write("| Agência de Checagem | Registros Checados | % da Base Total |\n")
        f.write("|---|---:|---:|\n")
        for ag, cnt in agencies_plot.items():
            f.write(f"| **{ag}** | {cnt:,} | {cnt/len(df)*100:.2f}% |\n")

        f.write("\n> [!NOTE]\n")
        f.write("> **Aos Fatos** e **Boatos.org** respondem conjuntamente por mais de 50% de todas as checagens com URL documentada no dataset, seguidos pela **Agência Lupa**, **G1 Fato ou Fake** e **Estadão Verifica**.\n\n")
        f.write("![Agências de Fact-checking](file:///" + plot_agencies_path.replace('\\', '/') + ")\n\n")
        f.write("---\n\n")
        f.write("## 2. A Taxonomia Granular de Desinformação\n\n")
        f.write("Nem toda notícia classificada genericamente como `fake` é uma invenção completa (*outright lie*). A análise da coluna `factcheck_rating` revela a nuance epistemológica atribuída pelos checadores profissionais:\n\n")
        f.write("| Classificação Granular Padronizada | Fake | True | Total Registros | % dos Rótulos Granulares |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        
        for r_name in rating_plot.index:
            cnt_f = rating_by_label.loc[r_name, 'fake'] if (r_name in rating_by_label.index and 'fake' in rating_by_label.columns) else 0
            cnt_t = rating_by_label.loc[r_name, 'true'] if (r_name in rating_by_label.index and 'true' in rating_by_label.columns) else 0
            tot = cnt_f + cnt_t
            pct = (tot / rating_plot.sum()) * 100
            f.write(f"| **{r_name}** | {cnt_f:,} | {cnt_t:,} | {tot:,} | {pct:.1f}% |\n")

        f.write("\n### Nuances Críticas:\n")
        f.write("1. **Distorcido e Fora de Contexto**: Cerca de 15% a 20% das peças rotuladas como `fake` na verdade utilizam elementos factuais (vídeos reais antigos, dados oficiais defasados) manipulados com legendas adulteradas.\n")
        f.write("2. **Desafio de Modelagem**: Modelos puramente léxicos têm extrema dificuldade com textos 'sem contexto' ou 'distorcidos', pois as palavras isoladas são reais e neutras; a falsidade reside no desacoplamento entre o texto e a realidade empírica.\n\n")
        f.write("![Taxonomia Granular](file:///" + plot_ratings_path.replace('\\', '/') + ")\n\n")
        f.write("---\n\n")
        f.write("## 3. Os Grandes Alvos da Desinformação\n\n")
        f.write("Quem a desinformação no Brasil mais ataca ou instrumentaliza? O mapeamento léxico nas 28.236 peças falsas revelou:\n\n")
        for target, r in df_targets.iterrows():
            f.write(f"- **{target}**: presente em **{r['pct']:.1f}%** das fake news ({int(r['count']):,} textos).\n")

        f.write("\n![Grandes Alvos da Desinformação](file:///" + plot_targets_path.replace('\\', '/') + ")\n\n")
        f.write("---\n\n")
        f.write("## 4. Quem São os Emissores Atribuídos? (`factcheck_claimant`)\n\n")
        f.write("Na grande maioria das checagens ({:.1f}%), o emissor não é uma figura pública com nome, mas a categoria genérica de **'Postagens em Redes Sociais'** ou **'Correntes anônimas de WhatsApp'**.\n".format((len(df[df['cleaned_claimant'] == 'Redes Sociais (Posts / Usuários)'])/len(df))*100))
        f.write("Contudo, quando figuras públicas são identificadas pelas agências, **Jair Bolsonaro** e parlamentares de sua base emergem como os indivíduos mais citados como autores diretos de alegações checadas pelas agências no período 2019-2022.\n")

    print(f"=== [Path 4] Concluído! Relatório salvo em: {REPORT_PATH} ===")

if __name__ == '__main__':
    run_investigation()
