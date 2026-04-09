#!/bin/bash
# =============================================================
# gerar_pdf.sh — Gera o PDF do artigo SBC a partir do Markdown
#
# Fonte única: ../../artigo/artigo.md (markdown limpo)
# Saída: output/artigo.pdf + cópia em ../../artigo/artigo.pdf
# =============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARTIGO_MD="$PROJECT_ROOT/artigo/artigo.md"
OUTPUT_DIR="$SCRIPT_DIR/output"

cd "$SCRIPT_DIR"

echo "=== SpectraAI — Geração de PDF do artigo SBC ==="
echo "Fonte: $ARTIGO_MD"
echo "Saída: $OUTPUT_DIR/"

# 1. Verificar dependências
for cmd in pandoc tectonic python3; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERRO: '$cmd' não encontrado. Instale com: brew install $cmd"
        exit 1
    fi
done

if [ ! -f "$ARTIGO_MD" ]; then
    echo "ERRO: $ARTIGO_MD não encontrado."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# 2. Preparar markdown para Pandoc (adiciona YAML, converte tabelas)
echo "[1/4] Preparando Markdown..."
python3 preparar_md.py "$ARTIGO_MD" "$OUTPUT_DIR/prepared.md"

# 3. Gerar .tex a partir do Markdown preparado
echo "[2/4] Convertendo Markdown → LaTeX..."
pandoc "$OUTPUT_DIR/prepared.md" \
    --template=template-sbc.tex \
    --natbib \
    --number-sections \
    --shift-heading-level-by=-1 \
    --wrap=none \
    --columns=40 \
    --resource-path="$SCRIPT_DIR" \
    -f markdown+yaml_metadata_block+tex_math_dollars+raw_tex+pipe_tables+implicit_figures \
    -V tables=true \
    -o "$OUTPUT_DIR/artigo.tex"

# 4. Pós-processamento do .tex
echo "    Pós-processando .tex..."
cd "$OUTPUT_DIR"

# Remover natbib (SBC usa \cite nativo)
sed -i '' '/\\usepackage.*natbib/d' artigo.tex
sed -i '' '/\\bibliographystyle{plainnat}/d' artigo.tex
sed -i '' '/^\\bibliography{references\.bib}$/d' artigo.tex
sed -i '' 's/\\citep{/\\cite{/g' artigo.tex
sed -i '' 's/\\citet{/\\cite{/g' artigo.tex

# Remover inputenc (XeTeX não precisa)
sed -i '' '/\\usepackage.*inputenc/d' artigo.tex

# Fix imagens pandoc (converter \pandocbounded para figure)
cd "$SCRIPT_DIR"
python3 fix_longtable.py "$OUTPUT_DIR/artigo.tex"

# 5. Copiar dependências para output e compilar
echo "[3/4] Compilando LaTeX → PDF..."
cp "$SCRIPT_DIR/sbc-template.sty" "$OUTPUT_DIR/"
cp "$SCRIPT_DIR/sbc.bst" "$OUTPUT_DIR/"
cp "$SCRIPT_DIR/references.bib" "$OUTPUT_DIR/"
cp -r "$SCRIPT_DIR/imagens" "$OUTPUT_DIR/" 2>/dev/null || true

cd "$OUTPUT_DIR"
tectonic artigo.tex

echo "=== Pronto! ==="
echo "PDF: $OUTPUT_DIR/artigo.pdf"
echo "================================================"
