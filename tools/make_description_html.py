"""README.md -> standalone HTML (pictures embedded) for the Windows zip.

    python tools/make_description_html.py README.md Описание.html
"""
import base64
import re
import sys
from pathlib import Path

import markdown

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
html = markdown.markdown(src.read_text(encoding="utf-8"), extensions=["tables", "fenced_code"])


def embed(m):
    p = src.parent / m.group(1)
    if not p.exists():
        return m.group(0)
    return 'src="data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode() + '"'


html = re.sub(r'src="([^"]+\.png)"', embed, html)
css = """body{font-family:Segoe UI,Arial,sans-serif;max-width:1000px;margin:30px auto;padding:0 20px;
line-height:1.55;color:#222} img{max-width:100%;border:1px solid #ccc;margin:8px 0}
table{border-collapse:collapse} td,th{border:1px solid #bbb;padding:4px 8px}
code{background:#f0f0f0;padding:1px 4px} blockquote{border-left:4px solid #2a6;margin:0;padding:4px 14px;background:#f4f7f4}"""
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(f'<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8"><title>RingFlow - описание</title>'
               f'<style>{css}</style></head><body>{html}</body></html>', encoding="utf-8")
print("written", ascii(str(dst)))  # ascii(): Windows consoles may not show Cyrillic
