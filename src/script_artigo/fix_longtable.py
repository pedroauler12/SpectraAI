#!/usr/bin/env python3
"""
Fix tables and images for twocolumn LaTeX layout.
- Tables with <=3 cols: fit in single column with resizebox
- Tables with >3 cols: use table* (span both columns) with resizebox
- Images: ensure they fit in column width
"""
import re
import sys


def process(content):
    # Remove packages we don't need
    content = re.sub(r'\\usepackage\{longtable\}', '', content)
    content = re.sub(r'\\usepackage\{booktabs\}', '', content)

    # Step 1: Convert longtable -> tabular
    lines = content.split('\n')
    result = []
    in_longtable = False
    table_lines = []

    for line in lines:
        if '\\begin{longtable}' in line:
            in_longtable = True
            table_lines = [line]
            continue
        elif in_longtable and '\\end{longtable}' in line:
            table_lines.append(line)
            in_longtable = False
            converted = convert_longtable(table_lines)
            result.append(converted)
            table_lines = []
            continue

        if in_longtable:
            table_lines.append(line)
        else:
            result.append(line)

    content = '\n'.join(result)

    # Step 2: Wrap all images to fit column width
    # Replace \pandocbounded{\includegraphics[...]{...}} with proper figure
    content = fix_images(content)

    return content


def convert_longtable(table_lines):
    """Convert a longtable block to a properly sized tabular."""
    # Count columns by finding data rows (lines with &)
    ncols = 0
    for line in table_lines:
        if '&' in line and '\\begin' not in line and '\\end' not in line:
            count = line.count('&') + 1
            if count > ncols:
                ncols = count

    if ncols == 0:
        ncols = 2

    # Skip the colspec header lines
    data_start = 0
    found_begin = False
    for i, line in enumerate(table_lines):
        if '\\begin{longtable}' in line:
            found_begin = True
            if '@{}}' in line or (line.rstrip().endswith('}') and 'arraybackslash' not in line):
                data_start = i + 1
                break
            continue
        if found_begin:
            if '@{}}' in line:
                data_start = i + 1
                break
            if 'arraybackslash' in line or 'raggedright' in line or 'raggedleft' in line:
                continue
            data_start = i
            break

    data_lines = table_lines[data_start:-1]

    # Clean data lines
    cleaned = []
    for line in data_lines:
        s = line.strip()
        if s in ('\\endhead', '\\endfirsthead', '\\endfoot', '\\endlastfoot', ''):
            continue
        if s.startswith('\\noalign'):
            continue
        line = line.replace('\\toprule', '\\hline')
        line = line.replace('\\midrule', '\\hline')
        line = line.replace('\\bottomrule', '\\hline')
        line = re.sub(r'\\begin\{minipage\}\[.\]\{[^}]+\}\\raggedright\s*', '', line)
        line = re.sub(r'\\begin\{minipage\}\[.\]\{[^}]+\}\\raggedleft\s*', '', line)
        line = re.sub(r'\\begin\{minipage\}\[.\]\{[^}]+\}', '', line)
        line = line.replace('\\end{minipage}', '')
        cleaned.append(line)

    body = '\n'.join(cleaned)
    body = re.sub(r'(\\hline\s*\n?){2,}', '\\\\hline\n', body)

    # Join multiline cells: pandoc splits table rows across multiple lines.
    # A complete row ends with \\. Merge continuation lines into their row.
    merged_lines = []
    current = ''
    for bline in body.split('\n'):
        stripped = bline.strip()
        if not stripped:
            continue
        if stripped == '\\\\hline' or stripped == '\\hline':
            if current:
                # Row didn't end with \\, add it as-is (probably header without \\)
                if not current.rstrip().endswith('\\\\'):
                    current = current.rstrip() + ' \\\\'
                merged_lines.append(current)
                current = ''
            merged_lines.append(stripped)
        elif stripped.endswith('\\\\'):
            current = (current + ' ' + stripped).strip()
            merged_lines.append(current)
            current = ''
        else:
            # Continuation of current row
            current = (current + ' ' + stripped).strip()
    if current:
        if not current.rstrip().endswith('\\\\'):
            current = current.rstrip() + ' \\\\'
        merged_lines.append(current)
    body = '\n'.join(merged_lines)

    # Choose layout based on column count
    colspec = '|' + '|'.join(['l'] * ncols) + '|'

    if ncols > 3:
        # Wide table: span both columns using table*
        out = '\\begin{table*}[!ht]\n'
        out += '\\centering\\scriptsize\n'
        out += '\\begin{tabular}{' + colspec + '}\n'
        out += '\\hline\n'
        out += body.strip() + '\n'
        out += '\\hline\n'
        out += '\\end{tabular}\n'
        out += '\\end{table*}\n'
    else:
        # Narrow table: fit in single column
        out = '\\begingroup\\scriptsize\n'
        out += '\\noindent\\begin{tabular}{' + colspec + '}\n'
        out += '\\hline\n'
        out += body.strip() + '\n'
        out += '\\hline\n'
        out += '\\end{tabular}\n'
        out += '\\endgroup\n'

    return out


def fix_images(content):
    """
    Wrap standalone images in figure environments that fit the column.
    Convert \pandocbounded{\includegraphics[...]{path}}
    to proper \begin{figure}[H] with \columnwidth.
    """
    # Pattern: \pandocbounded{\includegraphics[keepaspectratio,alt={caption}]{path}}
    # followed by optional italic caption text
    def replace_image(m):
        path = m.group('path')
        # Just output the includegraphics — pandoc already wraps in \begin{figure}
        return '\\includegraphics[width=0.85\\columnwidth]{' + path + '}'

    # Match pandocbounded images with optional following \emph{caption}
    content = re.sub(
        r'\\pandocbounded\{\\includegraphics\[keepaspectratio(?:,alt=\{(?P<alt>[^}]*)\})?\]\{(?P<path>[^}]+)\}\}\s*(?P<caption>\\emph\{[^}]*\})?',
        replace_image,
        content
    )

    # Add [H] placement to pandoc-generated \begin{figure} without placement
    content = re.sub(r'\\begin\{figure\}\s*\n', '\\\\begin{figure}[H]\n', content)

    return content


if __name__ == '__main__':
    path = sys.argv[1]
    with open(path, 'r') as f:
        content = f.read()
    content = process(content)
    with open(path, 'w') as f:
        f.write(content)
    print(f"    Fixed tables and images in {path}")
