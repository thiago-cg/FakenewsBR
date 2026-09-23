import os
import re
import base64

def get_base64_encoded_image(image_path):
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_file:
        return "data:image/png;base64," + base64.b64encode(img_file.read()).decode('utf-8')

# Caminhos
ARTIFACT_DIR = r"C:\Users\tito\.gemini\antigravity-ide\brain\c96eb70a-06a4-485e-86e1-39bcd92468da"
MD_PATH = os.path.join(ARTIFACT_DIR, "master_exploratory_synthesis_report.md")
HTML_PATH = r"c:\Users\tito\OneDrive\Documentos\Projetos\FakenewsBR\FakenewsBR_Exploratory_Report.html"

def convert_md_to_html_manual(md_text):
    # Tratamento simplificado de Markdown para HTML (se 'markdown' não estiver disponível)
    # Como queremos uma tabela HTML bonita e suporte a imagens base64, vamos focar 
    # num regex que faz as substituições essenciais, ou instalamos a biblioteca `markdown`.
    try:
        import markdown
        html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code', 'toc'])
    except ImportError:
        print("Módulo 'markdown' não encontrado. Por favor, rode 'pip install markdown' e tente novamente.")
        return None
        
    return html_body

def main():
    if not os.path.exists(MD_PATH):
        print(f"Erro: Arquivo Markdown não encontrado em {MD_PATH}")
        return

    with open(MD_PATH, 'r', encoding='utf-8') as f:
        md_text = f.read()

    # 1. Encontrar e substituir todas as imagens por base64 (para ser self-contained)
    # Formato esperado: ![alt](caminho_absoluto) ou ![alt](file:///caminho)
    img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    
    def img_replacer(match):
        alt_text = match.group(1)
        path = match.group(2)
        
        # Limpar o path de prefixos (file:///)
        clean_path = path.replace('file:///', '').replace('/', '\\')
        # Algumas vezes vem com C:/... e o replace anterior vai deixar C:\... o que está correto
        
        base64_src = get_base64_encoded_image(clean_path)
        if base64_src:
            return f'<div style="text-align: center;"><img src="{base64_src}" alt="{alt_text}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 20px 0;"></div>'
        else:
            return f'<div><em>[Imagem não encontrada: {alt_text}]</em></div>'

    md_text_images_replaced = re.sub(img_pattern, img_replacer, md_text)
    
    # 2. Converter Markdown para HTML (precisaremos instalar a biblioteca no run_command)
    import markdown
    html_body = markdown.markdown(md_text_images_replaced, extensions=['tables', 'fenced_code'])

    # 3. Construir o HTML final com CSS
    css_styles = """
    :root {
        --primary-color: #2c3e50;
        --secondary-color: #34495e;
        --accent-color: #3498db;
        --bg-color: #f8f9fa;
        --text-color: #333;
        --border-color: #e0e0e0;
    }
    body {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.6;
        color: var(--text-color);
        background-color: var(--bg-color);
        max-width: 1000px;
        margin: 0 auto;
        padding: 40px 20px;
    }
    .container {
        background-color: #ffffff;
        padding: 40px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    h1, h2, h3, h4 {
        color: var(--primary-color);
        margin-top: 1.5em;
        margin-bottom: 0.5em;
        font-weight: 600;
    }
    h1 {
        font-size: 2.2em;
        border-bottom: 2px solid var(--accent-color);
        padding-bottom: 10px;
        margin-top: 0;
    }
    h2 {
        font-size: 1.8em;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 8px;
    }
    p {
        margin-bottom: 1.2em;
    }
    blockquote {
        border-left: 5px solid var(--accent-color);
        background-color: #f0f7fb;
        padding: 15px 20px;
        margin: 20px 0;
        border-radius: 0 8px 8px 0;
        font-style: italic;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 0.95em;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.05);
    }
    table thead tr {
        background-color: var(--primary-color);
        color: #ffffff;
        text-align: left;
    }
    table th, table td {
        padding: 12px 15px;
        border: 1px solid var(--border-color);
    }
    table tbody tr {
        border-bottom: 1px solid var(--border-color);
    }
    table tbody tr:nth-of-type(even) {
        background-color: #f9f9f9;
    }
    table tbody tr:last-of-type {
        border-bottom: 2px solid var(--primary-color);
    }
    code {
        background-color: #f1f1f1;
        padding: 2px 5px;
        border-radius: 4px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.9em;
        color: #e74c3c;
    }
    pre {
        background-color: #2c3e50;
        color: #ecf0f1;
        padding: 15px;
        border-radius: 8px;
        overflow-x: auto;
    }
    pre code {
        background-color: transparent;
        color: inherit;
        padding: 0;
    }
    hr {
        border: 0;
        height: 1px;
        background: var(--border-color);
        margin: 40px 0;
    }
    """

    full_html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório Mestre: Exploração FakenewsBR</title>
    <style>
        {css_styles}
    </style>
</head>
<body>
    <div class="container">
        {html_body}
    </div>
</body>
</html>"""

    with open(HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(full_html)
        
    print(f"Relatório HTML gerado com sucesso em: {HTML_PATH}")

if __name__ == "__main__":
    main()
