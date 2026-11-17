"""Genera results/presentacion/presentacion.pdf (16:9) a partir de results/presentacion/slides/*.html.

Cada fichero es una diapositiva: un fragmento HTML que usa las clases de `CSS`. Se juntan en orden
alfabético, se numeran y se imprimen a PDF con Chrome o Edge sin interfaz. Con --png guarda además
cada diapositiva como imagen en results/presentacion/png/ (las usa el README).

Todas las diapositivas siguen el mismo esquema: título, una zona de gráfico y una zona de texto.

Uso: python results/presentacion/build.py [--png]
Las figuras salen de `python results/presentacion/figuras.py` (y esta, de `ej1_cache.py`).
"""
from pathlib import Path
import shutil
import subprocess
import sys

import fitz  # PyMuPDF

DIR = Path(__file__).resolve().parent
SLIDES, HTML, PDF, PNG = DIR / "slides", DIR / "presentacion.html", DIR / "presentacion.pdf", DIR / "png"
CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chrome",
]

CSS = """
@page { size: 254mm 142.875mm; margin: 0; }
* { box-sizing: border-box; }
html, body { margin: 0; }
.slide { width: 254mm; height: 142.875mm; position: relative; overflow: hidden; break-after: page;
         font-family: Bahnschrift, "DIN Alternate", "Segoe UI", sans-serif; font-weight: 300;
         color: #EAF0F4; background: #10273A; padding: 10mm 13mm 0; }
.slide p { margin: 0; }
img { display: block; }
b { font-weight: 600; color: #fff; }
.tema { font-size: 11pt; color: #9FB3C2; }
h1 { font-size: 23pt; font-weight: 600; line-height: 1.1; margin: 1mm 0 4mm; min-height: 17.5mm; }
.cuerpo { display: flex; gap: 9mm; align-items: flex-start; }
.fig { width: 150mm; flex: none; }
.txt { flex: 1; min-width: 0; font-size: 11.5pt; line-height: 1.3; }
.txt p + p { margin-top: 2mm; }
.f { font-size: 12pt; color: #fff; white-space: nowrap; }
.gris { color: #9FB3C2; font-size: 10pt; line-height: 1.28; }

/* lista de cifras: la cifra en ámbar y, debajo, qué mide */
.nums { margin-top: 2mm; }
.nums .n { font-size: 19pt; font-weight: 600; color: #F5B301; line-height: 1.05; margin-top: 2mm; }
.nums .n small { font-size: 12pt; font-weight: 300; color: #C5D3DD; }
.nums .d { font-size: 10.5pt; line-height: 1.26; color: #C5D3DD; }

/* ej. 4: gráfico a lo ancho y, debajo, tres columnas de texto alineadas con los paneles */
.ancha { width: 228mm; }
.tres { display: flex; gap: 8mm; margin-top: 1mm; }
.tres > div { flex: 1; min-width: 0; padding-left: 12mm; }
.tres .n { margin-top: 0; }

table { border-collapse: collapse; width: 100%; font-size: 11pt; line-height: 1.25; }
th { white-space: nowrap; font-weight: 300; color: #9FB3C2; text-align: left; border-bottom: 0.6pt solid #9FB3C2; padding: 0 2mm 1.2mm 0; }
td { padding: 1.6mm 2mm 1.6mm 0; border-bottom: 0.4pt solid #3A566B; vertical-align: top; }
td.c { color: #F5B301; font-weight: 600; white-space: nowrap; }
tr.elegida td { color: #fff; font-weight: 600; }

.foot { position: absolute; left: 13mm; right: 13mm; bottom: 4.5mm; display: flex; justify-content: space-between;
        font-size: 9pt; color: #7F97A8; }
.portada { padding-top: 14mm; }
.portada h1 { font-size: 30pt; min-height: 0; margin: 1.5mm 0 2mm; }
.portada .sub { font-size: 14pt; color: #C5D3DD; margin-bottom: 7mm; }
.portada .preg { font-size: 19pt; font-weight: 600; color: #F5B301; line-height: 1.18; }
.portada .aut { font-size: 12.5pt; margin-top: 9mm; line-height: 1.4; }
"""

TITLE = "Aprendizaje por refuerzo para algoritmos de ejecución"
FOOT = "Romero, Dapena y García, MIAX 2026"


def find_chrome():
    for c in CHROME_CANDIDATES:
        path = c if Path(c).exists() else shutil.which(c)
        if path:
            return path
    raise FileNotFoundError("No se encuentra Chrome/Edge para imprimir el PDF")


def main():
    files = sorted(SLIDES.glob("*.html"))
    if not files:
        raise SystemExit("No hay diapositivas en results/presentacion/slides/")
    parts = []
    for i, f in enumerate(files, 1):
        body = f.read_text(encoding="utf-8").replace(" %", "&nbsp;%")   # el % no se separa de su cifra
        footer = "" if i == 1 else f'<div class="foot"><span>{FOOT}</span><span>{i}</span></div>'
        parts.append(body.replace("</section>", footer + "</section>", 1))
    HTML.write_text(f'<!doctype html><html lang="es"><head><meta charset="utf-8"><title>{TITLE}</title>'
                    f"<style>{CSS}</style></head><body>{''.join(parts)}</body></html>", encoding="utf-8")
    subprocess.run([find_chrome(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", HTML.as_uri()], check=True, capture_output=True, timeout=120)
    doc = fitz.open(PDF)
    if "--png" in sys.argv:
        PNG.mkdir(exist_ok=True)
        for old in PNG.glob("diapo_*.png"):
            old.unlink()
        for i, page in enumerate(doc, 1):
            page.get_pixmap(dpi=110).save(PNG / f"diapo_{i:02d}.png")
    print(f"{PDF.name}: {doc.page_count} diapositivas ({len(files)} ficheros)")
    if doc.page_count != len(files):
        raise SystemExit("Alguna diapositiva se desborda: hay más páginas que ficheros")


if __name__ == "__main__":
    main()
