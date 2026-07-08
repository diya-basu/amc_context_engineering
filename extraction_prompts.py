"""
extraction_prompts.py
=======================
Format-specific extraction prompts. XLSX/CSV/TXT never appear here — they're
already structured/plain text and never touch the LLM at all.
"""

PDF_PAGE_PROMPT = """You are extracting ALL content from a page of a mutual fund brochure.
Your output builds a semantic search index — extract EVERYTHING on the page.

1. PROSE: all body text, reading order, paragraph breaks preserved, nothing omitted.
2. TABLES: GitHub-flavoured markdown, all rows/columns, never truncated.
3. CHARTS/IMAGES: describe axes, every data point, legends. Icons: "Icon: <label>".
4. HEADERS/FOOTERS: verbatim, marked [HEADER]/[FOOTER].
5. BOLD/HIGHLIGHTED TEXT: wrap in **asterisks**.
6. NUMBERS/PERCENTAGES: exact, never rounded or paraphrased.
7. DISCLAIMERS: extract completely, even fine print.

Output ONLY the extracted content — no commentary, no "this page contains...".
Extract everything from the page shown:"""

DOCX_IMAGE_PROMPT = """You are extracting content from an image embedded in a Word
document (likely a chart, diagram, or table screenshot). Describe all axes,
data points, labels, and numbers exactly as shown. Output only the extracted
content, no commentary."""

PPTX_SLIDE_IMAGE_PROMPT = """You are extracting content from an image embedded in a
presentation slide (likely a chart, diagram, or infographic). Describe all
axes, data points, labels, icons, and numbers exactly as shown. Output only
the extracted content, no commentary."""