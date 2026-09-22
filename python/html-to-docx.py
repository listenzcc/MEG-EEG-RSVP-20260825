#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert the Stage-2 formatted HTML report into a .docx (OOXML) using only the stdlib.

Motivation: the html-to-docx skill / python-docx are not available offline, so the
.docx is assembled directly as an OOXML package.

Usage:
    python python/html-to-docx.py --input <html> --output <docx>

Supported input dialect (the business-report template produced by Stage 2):
    section.report-cover   -> title page
    nav.doc-toc            -> static table of contents
    div.executive-summary  -> summary box, inner table.card-table -> 2x3 card grid
    h1 / h2 / h3           -> Heading 1-3
    p  (p.caption)         -> body text / small gray caption
    ul / ol / li           -> bullet / numbered paragraphs
    table.data-table       -> bordered table with repeating header row
    div.callout            -> shaded single-cell box
    div.divider            -> horizontal rule paragraph
    strong / em / code     -> bold / italic / monospace runs
"""

import argparse
import os
import re
import sys
import zipfile
from html.parser import HTMLParser

# ---------------------------------------------------------------- geometry
TWIP_PER_CM = 1440 / 2.54
CONTENT_WIDTH = int(round(15.6 * TWIP_PER_CM))  # 15.6 cm usable width on A4
PAGE_W, PAGE_H = 11906, 16838                    # A4 portrait
MARGIN = int(round(2.7 * TWIP_PER_CM))

# ---------------------------------------------------------------- palette
PRIMARY = "1F3864"
ACCENT = "2E75B6"
MUTED = "595959"
TEXT = "1A1A1A"
BORDER = "BFBFBF"
HEAD_FILL = "D9E2F3"
CALLOUT_FILL = "F2F6FC"
WARN_FILL = "FDF2E9"

BODY_FONT = "微软雅黑"
HEAD_FONT = "微软雅黑"
MONO_FONT = "Consolas"

VOID_TAGS = {"br", "hr", "img", "meta", "link", "input"}


# ---------------------------------------------------------------- mini DOM
class Node:
    __slots__ = ("tag", "attrs", "children")

    def __init__(self, tag, attrs=None):
        self.tag = tag
        self.attrs = dict(attrs or {})
        self.children = []

    @property
    def classes(self):
        return self.attrs.get("class", "").split()

    def has(self, name):
        return name in self.classes


class TreeBuilder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("#root")
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].children.append(Node(tag, attrs))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def find(node, tag):
    """First descendant with the given tag."""
    for child in node.children:
        if isinstance(child, Node):
            if child.tag == tag:
                return child
            got = find(child, tag)
            if got is not None:
                return got
    return None


# ---------------------------------------------------------------- text utils
def esc(text):
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def norm(text):
    """Collapse HTML source whitespace into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def inline_runs(node, bold=False, italic=False, mono=False):
    """Flatten inline content into a list of (text, bold, italic, mono)."""
    out = []
    if isinstance(node, str):
        cleaned = node.replace("\n", " ")
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        if cleaned:
            out.append((cleaned, bold, italic, mono))
        return out
    b = bold or node.tag in ("strong", "b", "th")
    i = italic or node.tag in ("em", "i")
    m = mono or node.tag in ("code", "kbd", "samp")
    if node.tag == "br":
        out.append(("\n", b, i, m))
    for child in node.children:
        out.extend(inline_runs(child, b, i, m))
    return out


def plain_text(node):
    return "".join(t for t, *_ in inline_runs(node))


def display_width(text):
    """CJK glyphs are roughly twice as wide as latin ones."""
    w = 0
    for ch in text:
        w += 2 if ord(ch) > 0x2E80 else 1
    return w


def run_xml(text, bold=False, italic=False, mono=False, color=None, size=None):
    rpr = ['<w:rFonts w:ascii="%s" w:hAnsi="%s" w:eastAsia="%s" w:cs="%s"/>'
           % (MONO_FONT if mono else BODY_FONT,
              MONO_FONT if mono else BODY_FONT,
              BODY_FONT, BODY_FONT)]
    if bold:
        rpr.append("<w:b/>")
    if italic:
        rpr.append("<w:i/>")
    if color:
        rpr.append('<w:color w:val="%s"/>' % color)
    if size:
        rpr.append('<w:sz w:val="%d"/><w:szCs w:val="%d"/>' % (size, size))
    return '<w:r><w:rPr>%s</w:rPr><w:t xml:space="preserve">%s</w:t></w:r>' % (
        "".join(rpr), esc(text))


def para_xml(runs, style=None, align=None, space_after=None,
             space_before=None, indent=None, line=None, borders=None,
             size=None, color=None):
    ppr = []
    if style:
        ppr.append('<w:pStyle w:val="%s"/>' % style)
    if align:
        ppr.append('<w:jc w:val="%s"/>' % align)
    if indent is not None:
        ppr.append('<w:ind w:firstLine="%d"/>' % indent)
    spacing = {}
    if line:
        spacing["line"] = str(line)
        spacing["lineRule"] = "auto"
    if space_before is not None:
        spacing["before"] = str(space_before)
    if space_after is not None:
        spacing["after"] = str(space_after)
    if spacing:
        ppr.append("<w:spacing %s/>" % " ".join('w:%s="%s"' % kv for kv in spacing.items()))
    if borders:
        ppr.append(borders)
    if size is not None or color is not None:
        body = "".join(run_xml(t, bold=b, italic=i, mono=m, color=color, size=size)
                       for t, b, i, m in runs)
    else:
        body = "".join(run_xml(*r) for r in runs)
    return "<w:p>%s%s</w:p>" % ("<w:pPr>%s</w:pPr>" % "".join(ppr) if ppr else "", body)


def runs_from(node, **kw):
    return [(t, b, i, m) for t, b, i, m in inline_runs(node)]


# ---------------------------------------------------------------- blocks
def para_from_node(node, style=None, **kw):
    runs = inline_runs(node)
    if not runs:
        return ""
    return para_xml(runs, style=style, **kw)


def cell_xml(node, width, is_header=False, fill=None):
    runs = inline_runs(node)
    align = "center"
    if isinstance(node, Node):
        if node.has("cell-left"):
            align = "left"
        elif node.has("cell-data"):
            align = "right"
    props = '<w:tcW w:w="%d" w:type="dxa"/>' % width
    if fill:
        props += '<w:shd w:val="clear" w:color="auto" w:fill="%s"/>' % fill
    props += '<w:vAlign w:val="center"/>'
    body = "".join(run_xml(t, bold=is_header or b, italic=i, mono=m,
                           size=18, color=PRIMARY if is_header else None)
                   for t, b, i, m in runs)
    ppr = '<w:pPr><w:jc w:val="%s"/><w:spacing w:before="20" w:after="20"/></w:pPr>' % align
    return '<w:tc><w:tcPr>%s</w:tcPr><w:p>%s%s</w:p></w:tc>' % (props, ppr, body)


def table_xml(node):
    rows = []
    for child in node.children:
        if isinstance(child, Node) and child.tag == "tr":
            cells = [c for c in child.children if isinstance(c, Node) and c.tag in ("td", "th")]
            if cells:
                rows.append(cells)
    if not rows:
        return ""

    ncol = max(len(r) for r in rows)
    weights = [1] * ncol
    for row in rows:
        for j, cell in enumerate(row):
            if j >= ncol:
                continue
            span = int(cell.attrs.get("colspan", "1"))
            w = max(display_width(plain_text(cell)) // span, 4)
            weights[j] = max(weights[j], w)
    total = sum(weights)
    widths = [max(int(CONTENT_WIDTH * w / total), 600) for w in weights]

    xml = ['<w:tbl>']
    xml.append("<w:tblPr>"
               '<w:tblW w:w="%d" w:type="dxa"/>' % CONTENT_WIDTH
               + '<w:jc w:val="center"/>'
               '<w:tblLayout w:type="fixed"/>'
               '<w:tblBorders>'
               + "".join('<w:%s w:val="single" w:sz="4" w:space="0" w:color="%s"/>'
                         % (side, BORDER)
                         for side in ("top", "left", "bottom", "right", "insideH", "insideV"))
               + "</w:tblBorders>"
               '<w:tblCellMar><w:top w:w="60" w:type="dxa"/><w:left w:w="90" w:type="dxa"/>'
               '<w:bottom w:w="60" w:type="dxa"/><w:right w:w="90" w:type="dxa"/></w:tblCellMar>'
               "</w:tblPr>")
    xml.append('<w:tblGrid>' + "".join('<w:gridCol w:w="%d"/>' % w for w in widths) + "</w:tblGrid>")

    for ri, row in enumerate(rows):
        is_head = ri == 0 and all(c.tag == "th" for c in row)
        trpr = "<w:trPr><w:tblHeader/></w:trPr>" if is_head else ""
        cells = []
        for j, cell in enumerate(row):
            width = widths[j] if j < len(widths) else widths[-1]
            cells.append(cell_xml(cell, width, is_header=is_head,
                                  fill=HEAD_FILL if is_head else None))
        xml.append("<w:tr>%s%s</w:tr>" % (trpr, "".join(cells)))
    xml.append("</w:tbl>")
    return "".join(xml)


def card_grid_xml(node):
    """table.card-table -> one card per cell, rendered as a bordered grid."""
    cells = []
    for child in node.children:
        if isinstance(child, Node) and child.tag == "tr":
            for c in child.children:
                if isinstance(c, Node) and c.tag in ("td", "th"):
                    cells.append(c)
    if not cells:
        return ""
    ncols = 3
    widths = [CONTENT_WIDTH // ncols] * ncols

    def card_cell(card, width):
        title = find(card, "p")  # first <p> is the title
        paras = [p for p in card.children if isinstance(p, Node) and p.tag == "p"]
        out = []
        for k, p in enumerate(paras):
            runs = inline_runs(p)
            if not runs:
                continue
            if k == 0:
                out.append(para_xml([(t, True, False, False) for t, *_ in runs],
                                    align="center", size=18, color=MUTED, space_after=20))
            elif k == 1:
                out.append(para_xml([(t, True, False, False) for t, *_ in runs],
                                    align="center", size=44, color=PRIMARY, space_after=20))
            else:
                out.append(para_xml([(t, False, False, False) for t, *_ in runs],
                                    align="center", size=16, color=MUTED, space_after=60))
        tcpr = ('<w:tcW w:w="%d" w:type="dxa"/>'
                '<w:shd w:val="clear" w:color="auto" w:fill="%s"/>'
                '<w:vAlign w:val="center"/>' % (width, "FFFFFF"))
        return "<w:tc><w:tcPr>%s</w:tcPr>%s</w:tc>" % (tcpr, "".join(out))

    rows = [cells[i:i + ncols] for i in range(0, len(cells), ncols)]
    xml = ['<w:tbl>',
           '<w:tblPr><w:tblW w:w="%d" w:type="dxa"/><w:jc w:val="center"/>'
           '<w:tblLayout w:type="fixed"/>'
           '<w:tblCellMar><w:top w:w="120" w:type="dxa"/><w:left w:w="120" w:type="dxa"/>'
           '<w:bottom w:w="120" w:type="dxa"/><w:right w:w="120" w:type="dxa"/></w:tblCellMar>'
           "</w:tblPr>" % CONTENT_WIDTH,
           '<w:tblGrid>' + "".join('<w:gridCol w:w="%d"/>' % w for w in widths) + "</w:tblGrid>"]
    for row in rows:
        xml.append("<w:tr>" + "".join(card_cell(c, widths[i]) for i, c in enumerate(row)) + "</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml)


def callout_xml(node):
    fill = WARN_FILL if node.has("callout-warning") else CALLOUT_FILL
    accent = "C55A11" if node.has("callout-warning") else ACCENT
    paras = []
    for child in node.children:
        if isinstance(child, Node) and child.tag in ("p", "h3", "h4"):
            runs = inline_runs(child)
            if not runs:
                continue
            bold = child.tag != "p" or child.has("callout-title")
            paras.append(para_xml([(t, bold, False, m) for t, _, _, m in runs],
                                  size=20, space_after=60, line=300))
    if not paras:
        return ""
    tcpr = ('<w:tcW w:w="%d" w:type="dxa"/>'
            '<w:shd w:val="clear" w:color="auto" w:fill="%s"/>'
            '<w:tcBorders><w:left w:val="single" w:sz="18" w:space="0" w:color="%s"/>'
            '<w:top w:val="nil"/><w:bottom w:val="nil"/><w:right w:val="nil"/></w:tcBorders>'
            '<w:tcMar><w:top w:w="120" w:type="dxa"/><w:left w:w="180" w:type="dxa"/>'
            '<w:bottom w:w="120" w:type="dxa"/><w:right w:w="180" w:type="dxa"/></w:tcMar>'
            % (CONTENT_WIDTH, fill, accent))
    return ('<w:tbl><w:tblPr><w:tblW w:w="%d" w:type="dxa"/>'
            '<w:tblLayout w:type="fixed"/></w:tblPr>'
            '<w:tblGrid><w:gridCol w:w="%d"/></w:tblGrid>'
            '<w:tr><w:tc><w:tcPr>%s</w:tcPr>%s</w:tc></w:tr></w:tbl>'
            % (CONTENT_WIDTH, CONTENT_WIDTH, tcpr, "".join(paras)))


def divider_xml():
    borders = ('<w:pBdr><w:bottom w:val="single" w:sz="6" w:space="1" w:color="%s"/></w:pBdr>'
               % BORDER)
    return '<w:p><w:pPr><w:spacing w:before="120" w:after="120"/>%s</w:pPr></w:p>' % borders


def page_break_xml():
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


# ---------------------------------------------------------------- walk
class DocxBuilder:
    def __init__(self):
        self.blocks = []

    def emit(self, xml):
        if xml:
            self.blocks.append(xml)

    # ---- cover -------------------------------------------------------
    def cover(self, node):
        self.emit('<w:p><w:pPr><w:spacing w:before="1400" w:after="0"/></w:pPr></w:p>')
        for child in node.children:
            if not isinstance(child, Node):
                continue
            cls = child.classes
            if "cover-category" in cls:
                self.emit(para_from_node(child, align="left", size=24, color=PRIMARY,
                                         space_after=240))
            elif "cover-title" in cls:
                self.emit(para_from_node(child, align="left", size=56, color=PRIMARY,
                                         space_after=200))
            elif "cover-subtitle" in cls:
                self.emit(para_from_node(child, align="left", size=28, color=MUTED,
                                         space_after=600))
        meta = find(node, "div")
        if meta is not None:
            for p in meta.children:
                if isinstance(p, Node) and p.tag == "p":
                    self.emit(para_from_node(p, align="left", size=20, color=MUTED,
                                             space_after=80))
        self.emit(page_break_xml())

    # ---- toc ---------------------------------------------------------
    def toc(self, node):
        title = find(node, "p")
        if title is not None:
            self.emit(para_from_node(title, style="Heading2", align="left"))
        for i, li in enumerate(self._list_items(node), start=1):
            self.emit(para_xml([("%d. " % i, False, False, False)] + inline_runs(li),
                               style="TOCEntry", space_after=60))
        self.emit(page_break_xml())

    def _list_items(self, node):
        out = []
        for child in node.children:
            if isinstance(child, Node):
                if child.tag == "li":
                    out.append(child)
                else:
                    out.extend(self._list_items(child))
        return out

    # ---- generic -----------------------------------------------------
    def walk(self, node, ordered_counter=None):
        for child in node.children:
            if isinstance(child, str):
                continue
            tag = child.tag
            cls = child.classes

            if tag in ("style", "script", "head", "title", "meta", "link"):
                continue
            if tag == "section":
                if "report-cover" in cls:
                    self.cover(child)
                else:
                    self.walk(child)
                continue
            if tag == "nav":
                self.toc(child)
                continue
            if tag == "main":
                self.walk(child)
                continue
            if tag == "div":
                if "callout" in cls or "callout-warning" in cls:
                    self.emit(callout_xml(child))
                    continue
                if "divider" in cls:
                    self.emit(divider_xml())
                    continue
                if "card-table" in cls:
                    self.emit(card_grid_xml(child))
                    continue
                if "summary-header" in cls:
                    self.walk(child)
                    continue
                self.walk(child)
                continue
            if tag == "table":
                if "card-table" in cls:
                    self.emit(card_grid_xml(child))
                else:
                    self.emit(table_xml(child))
                self.emit(para_xml([("", False, False, False)], space_after=80))
                continue
            if tag in ("h1", "h2", "h3"):
                style = {"h1": "Heading1", "h2": "Heading2", "h3": "Heading3"}[tag]
                self.emit(para_from_node(child, style=style, align="left"))
                continue
            if tag == "p":
                if any(c.startswith("cover-") for c in cls):
                    continue
                if "caption" in cls:
                    self.emit(para_from_node(child, align="left", size=18, color=MUTED,
                                             space_after=140))
                elif "card-title" in cls or "card-value" in cls or "card-unit" in cls:
                    continue
                elif "toc-title" in cls:
                    continue
                else:
                    self.emit(para_from_node(child, style="BodyText", align="both"))
                continue
            if tag in ("ul", "ol"):
                items = self._list_items(child)
                for i, li in enumerate(items, start=1):
                    marker = "%d. " % i if tag == "ol" else "• "
                    runs = [(marker, False, False, False)] + inline_runs(li)
                    self.emit(para_xml(runs, style="ListText", align="left"))
                continue
            if tag == "hr":
                self.emit(divider_xml())
                continue
            # unknown container
            self.walk(child)


# ---------------------------------------------------------------- package
def styles_xml():
    def heading(sid, name, size, color, before, after, outline):
        return ('<w:style w:type="paragraph" w:styleId="%s"><w:name w:val="%s"/>'
                '<w:basedOn w:val="Normal"/><w:qFormat/>'
                '<w:pPr><w:keepNext/><w:outlineLvl w:val="%d"/>'
                '<w:spacing w:before="%d" w:after="%d"/></w:pPr>'
                '<w:rPr><w:rFonts w:ascii="%s" w:hAnsi="%s" w:eastAsia="%s" w:cs="%s"/>'
                '<w:b/><w:color w:val="%s"/><w:sz w:val="%d"/><w:szCs w:val="%d"/></w:rPr>'
                '</w:style>' % (sid, name, outline, before, after,
                                HEAD_FONT, HEAD_FONT, HEAD_FONT, HEAD_FONT,
                                color, size, size))

    def simple(sid, name, indent=None, size=21, color=TEXT, bold=False,
               after=120, line=None, align=None):
        spacing = {"after": str(after)}
        if line:
            spacing["line"] = str(line)
            spacing["lineRule"] = "auto"
        ppr = ["<w:spacing %s/>" % " ".join('w:%s="%s"' % kv for kv in spacing.items())]
        if indent is not None:
            ppr.append('<w:ind w:firstLine="%d"/>' % indent)
        if align:
            ppr.append('<w:jc w:val="%s"/>' % align)
        rpr = ['<w:rFonts w:ascii="%s" w:hAnsi="%s" w:eastAsia="%s" w:cs="%s"/>'
               % (BODY_FONT, BODY_FONT, BODY_FONT, BODY_FONT)]
        if bold:
            rpr.append("<w:b/>")
        rpr.append('<w:color w:val="%s"/>' % color)
        rpr.append('<w:sz w:val="%d"/><w:szCs w:val="%d"/>' % (size, size))
        return ('<w:style w:type="paragraph" w:styleId="%s"><w:name w:val="%s"/>'
                '<w:basedOn w:val="Normal"/><w:qFormat/>'
                '<w:pPr>%s</w:pPr><w:rPr>%s</w:rPr></w:style>'
                % (sid, name, "".join(ppr), "".join(rpr)))

    parts = [
        '<w:docDefaults><w:rPrDefault><w:rPr>'
        '<w:rFonts w:ascii="%s" w:hAnsi="%s" w:eastAsia="%s" w:cs="%s"/>'
        '<w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr></w:rPrDefault>'
        '<w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="288" w:lineRule="auto"/>'
        '</w:pPr></w:pPrDefault></w:docDefaults>' % (BODY_FONT, BODY_FONT, BODY_FONT, BODY_FONT),
        '<w:style w:type="paragraph" w:default="1" w:styleId="Normal">'
        '<w:name w:val="Normal"/><w:qFormat/>'
        '<w:pPr><w:spacing w:after="120" w:line="288" w:lineRule="auto"/></w:pPr>'
        '<w:rPr><w:rFonts w:ascii="%s" w:hAnsi="%s" w:eastAsia="%s" w:cs="%s"/>'
        '<w:color w:val="%s"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr></w:style>'
        % (BODY_FONT, BODY_FONT, BODY_FONT, BODY_FONT, TEXT),
        heading("Heading1", "heading 1", 32, PRIMARY, 320, 160, 0),
        heading("Heading2", "heading 2", 26, PRIMARY, 240, 120, 1),
        heading("Heading3", "heading 3", 23, ACCENT, 200, 100, 2),
        simple("BodyText", "Body Text", indent=420, size=21, after=120),
        simple("TOCEntry", "TOC Entry", size=21, after=60),
        simple("ListText", "List Text", size=21, after=60),
        simple("Caption", "caption", size=18, color=MUTED, after=140),
    ]
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            + "".join(parts) + "</w:styles>")


def footer_xml():
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:p><w:pPr><w:jc w:val="center"/>'
            '<w:rPr><w:rFonts w:ascii="%s" w:hAnsi="%s" w:eastAsia="%s" w:cs="%s"/>'
            '<w:color w:val="%s"/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr></w:pPr>'
            '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
            '<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
            '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>'
            % (BODY_FONT, BODY_FONT, BODY_FONT, BODY_FONT, MUTED))


def document_xml(blocks):
    sect = ('<w:sectPr><w:footerReference w:type="default" r:id="rId2"/>'
            '<w:pgSz w:w="%d" w:h="%d"/>'
            '<w:pgMar w:top="%d" w:right="%d" w:bottom="%d" w:left="%d" '
            'w:header="851" w:footer="851" w:gutter="0"/>'
            '<w:cols w:space="425"/><w:docGrid w:linePitch="312"/></w:sectPr>'
            % (PAGE_W, PAGE_H, MARGIN, MARGIN, MARGIN, MARGIN))
    body = "".join(blocks) + '<w:p><w:pPr><w:spacing w:after="0"/></w:pPr></w:p>'
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<w:body>%s%s</w:body></w:document>' % (body, sect))


CONTENT_TYPES = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                 '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                 '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                 '<Default Extension="xml" ContentType="application/xml"/>'
                 '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                 '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
                 '<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>'
                 '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
                 '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
                 '</Types>')

ROOT_RELS = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
             '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
             '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
             '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
             '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
             '</Relationships>')

DOC_RELS = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>'
            '</Relationships>')


def core_xml(title):
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties '
            'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            '<dc:title>%s</dc:title><dc:creator>RSVP pipeline</dc:creator>'
            '<cp:lastModifiedBy>RSVP pipeline</cp:lastModifiedBy>'
            '<dcterms:created xsi:type="dcterms:W3CDTF">2026-09-22T00:00:00Z</dcterms:created>'
            '<dcterms:modified xsi:type="dcterms:W3CDTF">2026-09-22T00:00:00Z</dcterms:modified>'
            '</cp:coreProperties>' % esc(title))


APP_XML = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
           '<Application>html-to-docx.py</Application></Properties>')


# ---------------------------------------------------------------- main
def convert(html_path, docx_path):
    with open(html_path, "r", encoding="utf-8") as fh:
        raw = fh.read()
    if raw.startswith("\ufeff"):
        raw = raw[1:]

    parser = TreeBuilder()
    parser.feed(raw)
    body = find(parser.root, "body")
    if body is None:
        body = parser.root

    builder = DocxBuilder()
    builder.walk(body)

    title = "RSVP 目标识别分析报告"
    doc = document_xml(builder.blocks)

    with zipfile.ZipFile(docx_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", ROOT_RELS)
        z.writestr("word/document.xml", doc)
        z.writestr("word/_rels/document.xml.rels", DOC_RELS)
        z.writestr("word/styles.xml", styles_xml())
        z.writestr("word/footer1.xml", footer_xml())
        z.writestr("docProps/core.xml", core_xml(title))
        z.writestr("docProps/app.xml", APP_XML)

    return len(builder.blocks)


def main():
    ap = argparse.ArgumentParser(description="HTML -> DOCX (stdlib only)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    n = convert(args.input, args.output)
    size = os.path.getsize(args.output)
    print("blocks: %d" % n)
    print("output: %s" % args.output)
    print("size  : %d bytes" % size)


if __name__ == "__main__":
    main()
