"""
File: md-to-html.py
Date: 2026-10-15

Purpose:
    Turn the markdown manuscript into the html that html-to-docx.py reads.

    The manuscript is written in markdown because that is what gets edited and
    diffed; the Word file is a build product. The subset understood here is the
    one the manuscript uses: ATX headings, paragraphs, bullet and numbered
    lists, pipe tables, images with a caption, block quotes, horizontal rules,
    and inline bold / italic / code.

    Images become <figure><img><figcaption>, which is the shape html-to-docx.py
    already knows how to place.

Usage:
    python python/md-to-html.py --input doc/paper-2026-10-15.md \
        --output doc/paper-2026-10-15.html --title "..."
"""

import argparse
import html as html_mod
import re
from pathlib import Path

ap = argparse.ArgumentParser(description='Markdown -> HTML for '
                                         'html-to-docx.py')
ap.add_argument('--input', required=True)
ap.add_argument('--output', required=True)
ap.add_argument('--title', default='Manuscript')
ap.add_argument('--subtitle', default='')
ap.add_argument('--meta', action='append', default=[],
                help='Extra cover lines, repeatable')
args = ap.parse_args()

SRC = Path(args.input)
OUT = Path(args.output)


def inline(text: str) -> str:
    '''
    Inline markdown to inline html. Code spans are escaped first so that
    characters inside them are not read as markdown.
    '''
    spans = []

    def stash(m):
        spans.append(m.group(1))
        return f'\x00{len(spans) - 1}\x00'

    text = re.sub(r'`([^`]+)`', stash, text)
    text = html_mod.escape(text, quote=False)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<!\*)\*(?!\*)([^*]+?)\*(?!\*)', r'<em>\1</em>', text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)',
                  r'<a href="\2">\1</a>', text)
    for i, s in enumerate(spans):
        text = text.replace(f'\x00{i}\x00',
                            f'<code>{html_mod.escape(s, quote=False)}</code>')
    return text


def is_table_row(line: str) -> bool:
    return line.startswith('|') and line.rstrip().endswith('|')


def split_row(line: str):
    return [c.strip() for c in line.strip().strip('|').split('|')]


def main():
    lines = SRC.read_text(encoding='utf-8').split('\n')
    out = []
    i = 0
    para = []

    def flush():
        if para:
            out.append('<p>' + inline(' '.join(para)) + '</p>')
            para.clear()

    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            flush()
            i += 1
            continue

        if stripped.startswith('<!--'):
            while i < len(lines) and '-->' not in lines[i]:
                i += 1
            i += 1
            continue

        if stripped.startswith('!['):
            flush()
            m = re.match(r'!\[(.*?)\]\((.*?)\)', stripped)
            cap, src = m.group(1), m.group(2)
            out.append('<figure>')
            out.append(f'<img src="{src}" alt="{html_mod.escape(cap)}">')
            out.append(f'<figcaption>{inline(cap)}</figcaption>')
            out.append('</figure>')
            i += 1
            continue

        if stripped.startswith('#'):
            flush()
            level = len(stripped) - len(stripped.lstrip('#'))
            text = stripped.lstrip('#').strip()
            cls = ' class="part"' if text.startswith('Part ') else ''
            out.append(f'<h{level}{cls}>{inline(text)}</h{level}>')
            i += 1
            continue

        if stripped in ('---', '***', '___'):
            flush()
            out.append('<hr>')
            i += 1
            continue

        if is_table_row(stripped):
            flush()
            rows = []
            while i < len(lines) and is_table_row(lines[i].strip()):
                rows.append(split_row(lines[i].strip()))
                i += 1
            sep = rows[1] if len(rows) > 1 else None
            if sep and all(re.fullmatch(r':?-{2,}:?', c) for c in sep):
                head, body = rows[0], rows[2:]
            else:
                head, body = None, rows
            out.append('<table>')
            if head:
                out.append('<tr>' + ''.join(f'<th>{inline(c)}</th>'
                                            for c in head) + '</tr>')
            for r in body:
                out.append('<tr>' + ''.join(f'<td>{inline(c)}</td>'
                                            for c in r) + '</tr>')
            out.append('</table>')
            continue

        if re.match(r'^[-*]\s+', stripped):
            flush()
            items = []
            while i < len(lines) and re.match(r'^[-*]\s+', lines[i].strip()):
                items.append(re.sub(r'^[-*]\s+', '', lines[i].strip()))
                i += 1
            out.append('<ul>' + ''.join(f'<li>{inline(t)}</li>'
                                        for t in items) + '</ul>')
            continue

        if re.match(r'^\d+\.\s+', stripped):
            flush()
            items = []
            while i < len(lines) and re.match(r'^\d+\.\s+', lines[i].strip()):
                items.append(re.sub(r'^\d+\.\s+', '', lines[i].strip()))
                i += 1
            out.append('<ol>' + ''.join(f'<li>{inline(t)}</li>'
                                        for t in items) + '</ol>')
            continue

        if stripped.startswith('>'):
            flush()
            block = []
            while i < len(lines) and lines[i].strip().startswith('>'):
                block.append(re.sub(r'^>\s?', '', lines[i].strip()))
                i += 1
            out.append('<blockquote>' + inline(' '.join(block))
                       + '</blockquote>')
            continue

        para.append(stripped)
        i += 1

    flush()

    meta = ''.join(f'<p>{inline(m)}</p>' for m in args.meta)
    sub = (f'<p class="cover-subtitle">{inline(args.subtitle)}</p>'
           if args.subtitle else '')
    doc = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{html_mod.escape(args.title)}</title>
<style>
  :root {{ --primary:#1F3864; --accent:#2E75B6; --muted:#595959; }}
  body {{ font-family:"Times New Roman", serif; color:#1A1A1A; }}
  figure {{ margin:12pt 0; }}
  figure img {{ max-width:100%; }}
  figcaption {{ font-size:9pt; color:#595959; }}
  blockquote {{ color:#404040; font-style:italic; }}
</style>
</head>
<body>

<section class="report-cover">
  <p class="cover-title">{inline(args.title)}</p>
  {sub}
  <div>
    {meta}
  </div>
</section>

{chr(10).join(out)}

</body>
</html>
'''
    OUT.write_text(doc, encoding='utf-8')
    print(f'wrote {OUT} ({len(doc)} bytes, '
          f'{len(re.findall(r"<figure>", doc))} figures, '
          f'{len(re.findall(r"<table>", doc))} tables)')


if __name__ == '__main__':
    main()
