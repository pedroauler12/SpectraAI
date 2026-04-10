#!/usr/bin/env python3
"""
Prepara o artigo.md limpo para compilação com Pandoc.
O artigo.md é mantido no formato original do projeto.
Este script aplica transformações necessárias para o PDF SBC:

1. Adiciona YAML frontmatter (de metadata.yaml)
2. Remove header markdown (título, autores, abstract/resumo em texto)
3. Remove &emsp;&emsp; (indentação HTML)
4. Converte citações (AUTOR, ANO) para formato pandoc [@key]
5. Remove numeração manual de seções
6. Converte tabelas pipe markdown para raw LaTeX
7. Mapeia caminhos de imagens para nomes locais
8. Remove seção de referências (pandoc gera do .bib)
9. Remove diagrama ASCII (Figura 1)
"""
import re
import sys
import os


# Mapeamento de caminhos de imagens originais -> locais
IMAGE_MAP = {
    '../outputs/a08_transfer_learning/comparacao_modelos_barplot.png': 'imagens/f2a.png',
    '../outputs/a08_transfer_learning/comparacao_modelos_radar.png': 'imagens/f2b.png',
    '../outputs/a08_transfer_learning/training_curves.png': 'imagens/f3a.png',
    '../outputs/a08_transfer_learning/validation_plots.png': 'imagens/f3b.png',
    '../outputs/a08_transfer_learning/overfitting_analysis.png': 'imagens/f4.png',
    '../outputs/a08_transfer_learning/augmentation_histograms.png': 'imagens/f5a.png',
    '../outputs/a08_transfer_learning/augmentation_visual_comparison.png': 'imagens/f5b.png',
    '../outputs/a08_transfer_learning/grid_search_heatmaps.png': 'imagens/f6a.png',
    '../outputs/a08_transfer_learning/grid_search_top_configs.png': 'imagens/f6b.png',
    '../outputs/a08_transfer_learning/tl_sensitivity_analysis.png': 'imagens/f6c.png',
    '../outputs/a09_interpretabilidade_visualizacao/confusion_matrix_threshold_f1.png': 'imagens/f7a.png',
    '../outputs/a09_interpretabilidade_visualizacao/probability_distributions.png': 'imagens/f7b.png',
    '../outputs/a09_interpretabilidade_visualizacao/roc_pr_curves.png': 'imagens/f7c.png',
    '../outputs/a09_interpretabilidade_visualizacao/gradcam_comparativo.png': 'imagens/f8a.png',
    '../outputs/a09_interpretabilidade_visualizacao/gradcam_por_classe.png': 'imagens/f8b.png',
    '../outputs/a09_interpretabilidade_visualizacao/gradcam_erros.png': 'imagens/f8c.png',
    '../outputs/a09_interpretabilidade_visualizacao/spatial_probability_map.png': 'imagens/f9a.png',
    '../outputs/a09_interpretabilidade_visualizacao/spatial_outcome_map.png': 'imagens/f9b.png',
    '../outputs/a09_interpretabilidade_visualizacao/spatial_probability_hexbin.png': 'imagens/f9c.png',
    '../outputs/a11_pipeline_e2e/confusion_matrix.png': 'imagens/f10.png',
    '../outputs/a11_pipeline_e2e/roc_pr_curves.png': 'imagens/f11.png',
}

# Mapeamento de citações textuais -> chaves bibtex
CITATION_MAP = {
    '(IEA, 2021; USGS, 2025)': '[@iea2021; @usgs2025]',
    '(SABINS, 1999; VAN DER MEER et al., 2012)': '[@sabins1999; @vandermeer2012]',
    '(ABRAMS; YAMAGUCHI, 2019; RAMSEY; FLYNN, 2020)': '[@abrams2019; @ramsey2020]',
    '(ABRAMS; YAMAGUCHI, 2019; ROWAN; MARS, 2003)': '[@abrams2019; @rowan2003]',
    '(SHIRMARD et al., 2022)': '[@shirmard2022]',
    '(BAHRAMI et al., 2024)': '[@bahrami2024]',
    '(SUN et al., 2024)': '[@sun2024]',
    '(ZHU et al., 2017)': '[@zhu2017]',
    '(ROWAN; MARS, 2003)': '[@rowan2003]',
    '(ABRAMS; YAMAGUCHI, 2019)': '[@abrams2019]',
    '(RAMSEY; FLYNN, 2020)': '[@ramsey2020]',
    '(NASA, [s.d.])': '[@nasa_aster]',
    '(SANDLER et al., 2018)': '[@sandler2018]',
    # Narrative citations
    'Abrams e Yamaguchi (2019)': '@abrams2019',
    'Bahrami et al. (2024)': '@bahrami2024',
    'Rowan e Mars (2003)': '@rowan2003',
    'Luo et al. (2025)': '@luo2025',
    'Song et al. (2024)': '@song2024',
    'Shirmard et al. (2022)': '@shirmard2022',
    'Chen et al. (2014)': '@chen2014',
}


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artigo_src = sys.argv[1]
    artigo_out = sys.argv[2]

    with open(artigo_src) as f:
        content = f.read()

    with open(os.path.join(script_dir, 'metadata.yaml')) as f:
        metadata = f.read()

    # Step 1: Remove &emsp;&emsp;
    content = content.replace('&emsp;&emsp; ', '')
    content = content.replace('&emsp;&emsp;', '')

    # Step 2: Convert citations
    for old, new in CITATION_MAP.items():
        content = content.replace(old, new)

    # Step 3: Map image paths
    for old_path, new_path in IMAGE_MAP.items():
        content = content.replace(old_path, new_path)

    # Step 4: Remove ASCII diagram (Figura 1) - already a comment in current version
    # Also handle the original ASCII block if present
    content = re.sub(
        r'```\n┌.*?└.*?```',
        '<!-- Figura 1 removida para PDF -->',
        content,
        flags=re.DOTALL
    )

    # Step 4b: Remove duplicate figure captions (italic lines below images)
    # Pattern: *Figura N. description text*
    content = re.sub(r'\n\*Figura \d+\..*?\*\n', '\n', content)

    # Step 4b2: Remove "Figura N: " prefix from image alt text and add "Fonte: Autores (2026)"
    # ![Figura 3: Curvas de Aprendizado...] -> ![Curvas de Aprendizado... Fonte: Autores (2026)]
    def fix_figure_alt(m):
        alt = m.group(1)
        if 'Fonte:' not in alt:
            alt = alt.rstrip('.') + '. Fonte: Autores (2026)'
        return '![' + alt
    content = re.sub(r'!\[Figura \d+: ([^\]]+)', fix_figure_alt, content)

    # Step 4c: Remove lines with long paths that overflow columns
    # Remove "Fonte:" lines with outputs/ paths
    content = re.sub(r'\*\*Fonte:\*\*.*outputs/.*\n', '', content)
    # Remove "Source: outputs/..." from captions
    content = re.sub(r'Source: outputs/[^\*]*', 'Fonte: Autores (2026)', content)
    content = re.sub(r'Fonte: outputs/[^\*]*', 'Fonte: Autores (2026)', content)
    # Remove "Visualizações Relacionadas:" bullet lists with paths
    content = re.sub(r'- `outputs/[^`]+`[^\n]*\n', '', content)
    # Remove standalone "Visualizações Relacionadas:" headers left empty
    content = re.sub(r'\*\*Visualizações Relacionadas:\*\*\s*\n\s*\n', '\n', content)
    content = re.sub(r'Visualizações Relacionadas:\s*\n\s*\n', '\n', content)
    # Remove "Fonte: Pipeline A11..." lines
    content = re.sub(r'\*\*Fonte:\*\* Pipeline A11[^\n]*\n', '', content)

    # Step 5: Find body start (skip header: title, autores, abstract, resumo, keywords)
    lines = content.split('\n')
    body_start = 0
    for i, line in enumerate(lines):
        if re.match(r'^## \d+\.\s', line):  # First numbered section like "## 1. Introdução"
            body_start = i
            break

    body = '\n'.join(lines[body_start:])

    # Step 6: Remove section numbers
    body = re.sub(r'^(#{2,4}) \d+(\.\d+)*\.?\s+', r'\1 ', body, flags=re.MULTILINE)

    # Step 7: Remove references section at end
    body = re.sub(r'\n## Referências\n.*$', '', body, flags=re.DOTALL)

    # Step 8: Convert pipe tables to raw LaTeX
    body = convert_pipe_tables(body)

    # Step 9: Build final with YAML
    output = '---\n' + metadata.strip() + '\n---\n\n' + body.strip() + '\n'

    with open(artigo_out, 'w') as f:
        f.write(output)

    print(f"    Prepared: {artigo_out}")


def convert_pipe_tables(content):
    """Convert markdown pipe tables to raw LaTeX table environments."""
    lines = content.split('\n')
    result = []
    i = 0

    while i < len(lines):
        if lines[i].strip().startswith('|') and i + 1 < len(lines) and '---' in lines[i + 1]:
            table_lines = []
            caption = ''
            # Check for caption line before table
            if result and result[-1].strip().startswith('Tabela'):
                caption = result.pop().strip()
            elif len(result) >= 2 and result[-1].strip() == '' and result[-2].strip().startswith('Tabela'):
                result.pop()  # empty line
                caption = result.pop().strip()

            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i])
                i += 1

            latex = pipe_table_to_latex(table_lines, caption)
            result.append('')
            result.append('```{=latex}')
            result.append(latex)
            result.append('```')
            result.append('')
            continue

        result.append(lines[i])
        i += 1

    return '\n'.join(result)


def pipe_table_to_latex(table_lines, caption=''):
    """Convert a pipe table to a LaTeX table[H] with resizebox."""
    header_cells = [c.strip() for c in table_lines[0].strip('|').split('|')]
    ncols = len(header_cells)

    data_rows = []
    for line in table_lines[2:]:
        cells = [c.strip() for c in line.strip('|').split('|')]
        data_rows.append(cells)

    colspec = '|' + '|'.join(['l'] * ncols) + '|'

    lines = []
    lines.append('\\begin{table}[H]')
    lines.append('\\centering')
    if caption:
        # Remove "Tabela N – " prefix (LaTeX auto-numbers with \caption)
        clean_caption = re.sub(r'^Tabela \d+\s*[–—-]\s*', '', caption)
        safe_caption = clean_caption.replace('_', '\\_').replace('%', '\\%')
        label = re.sub(r'[^a-z0-9-]', '', caption.lower().replace(' ', '-'))[:30]
        if 'Fonte:' not in safe_caption:
            safe_caption = safe_caption.rstrip('.') + '. Fonte: Autores (2026)'
        lines.append(f'\\caption{{{safe_caption}}}')
        lines.append(f'\\label{{tab:{label}}}')
    lines.append('\\resizebox{\\columnwidth}{!}{%')
    lines.append(f'\\begin{{tabular}}{{{colspec}}}')
    lines.append('\\hline')

    # Header (bold, escaped)
    header_escaped = [escape_latex(c) for c in header_cells]
    lines.append(' & '.join([f'\\textbf{{{c}}}' for c in header_escaped]) + ' \\\\')
    lines.append('\\hline')

    # Data rows
    for row in data_rows:
        while len(row) < ncols:
            row.append('')
        row = row[:ncols]
        lines.append(' & '.join([escape_latex(c) for c in row]) + ' \\\\')

    lines.append('\\hline')
    lines.append('\\end{tabular}%')
    lines.append('}')
    lines.append('\\end{table}')

    return '\n'.join(lines)


def escape_latex(text):
    """Escape special LaTeX characters in table cells."""
    text = text.replace('\\', '\\textbackslash{}')
    text = text.replace('_', '\\_')
    text = text.replace('%', '\\%')
    text = text.replace('&', '\\&')
    text = text.replace('#', '\\#')
    text = text.replace('~', '\\textasciitilde{}')
    text = text.replace('{', '\\{')
    text = text.replace('}', '\\}')
    # Restore \textbf, \textasciitilde, \textbackslash that we just broke
    text = text.replace('\\\\textbf', '\\textbf')
    text = text.replace('\\\\textasciitilde', '\\textasciitilde')
    text = text.replace('\\\\textbackslash', '\\textbackslash')
    text = text.replace('\\\\&', '\\&')
    return text


if __name__ == '__main__':
    main()
