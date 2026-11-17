"""Genera presentacion/presentacion.pdf (16:9) a partir de presentacion/slides/*.html.

Cada fichero es una diapositiva: un fragmento HTML que usa las clases de `CSS`. Se juntan en orden
alfabético, se numeran y se imprimen a PDF con Chrome o Edge sin interfaz. Con --png guarda además
cada diapositiva como imagen en presentacion/png/ para revisarla.

Uso: python presentacion/build.py [--png]
Las figuras salen de `python presentacion/figuras.py` (y esta, de `ej1_cache.py`).
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
body { font-family: Cambria, Georgia, "Times New Roman", serif; color: #111; }
.slide { width: 254mm; height: 142.875mm; padding: 7mm 12mm 9mm; position: relative; overflow: hidden;
         break-after: page; display: flex; flex-direction: column; font-size: 11pt; line-height: 1.25; }
.slide h1 { font-size: 20pt; margin: 0 0 2mm; font-weight: bold; color: #102a43; line-height: 1.1; }
.slide .kicker { font-size: 10pt; color: #2B6CB0; letter-spacing: .04em; text-transform: uppercase; margin-bottom: 0.8mm; }
.slide .lead { font-size: 13pt; color: #333; margin: 0 0 3mm; }
.slide p { margin: 0 0 1.5mm; }
.slide ul { margin: 0; padding-left: 5mm; }
.slide li { margin: 0 0 1.2mm; }
.slide b, .slide strong { color: #102a43; }
.row { display: flex; gap: 3mm; align-items: stretch; }
.col { display: flex; flex-direction: column; min-width: 0; }
.fig { width: 100%; display: block; }
.cap { font-size: 9pt; color: #555; margin: 0.8mm 0 0; }
.note { font-size: 9.5pt; color: #555; }
.tag { font-size: 9pt; color: #2B6CB0; letter-spacing: .04em; text-transform: uppercase; margin-bottom: 0.6mm; }
.box { border: 0.8pt solid #9fb3c8; border-top: 1.2mm solid #2B6CB0; padding: 1.5mm 2.5mm; background: #fff; }
.box.g { border-top-color: #2F9E6E; } .box.g .tag { color: #2F9E6E; }
.box.o { border-top-color: #D9822B; } .box.o .tag { color: #B8691A; }
.box.r { border-top-color: #C53030; } .box.r .tag { color: #C53030; }
.arrow { align-self: center; color: #2B6CB0; font-size: 15pt; line-height: 1; }
.kpi { flex: 1; border: 0.8pt solid #9fb3c8; border-radius: 1.5mm; padding: 1.5mm 2.5mm; background: #f7fafc; }
.kpi .l { font-size: 9pt; color: #555; }
.kpi .v { font-size: 17pt; font-weight: bold; color: #2B6CB0; line-height: 1.05; }
.kpi .v small { font-size: 10pt; white-space: nowrap; color: #555; font-weight: normal; }
.kpi .n { font-size: 9pt; color: #555; }
.kpi.bad .v { color: #C53030; } .kpi.ok .v { color: #2F9E6E; } .kpi.o .v { color: #D9822B; }
.verdict { border-left: 2.2mm solid #2F9E6E; padding: 1.5mm 3mm; background: #f1f8f4; font-size: 11pt; }
.verdict.warn { border-left-color: #D9822B; background: #fdf6ee; }
.par { flex: 1; padding: 1.2mm 2.2mm; }
.par .pv { font-size: 13pt; font-weight: bold; color: #2F9E6E; line-height: 1.15; white-space: nowrap; }
.par .pv s { color: #C53030; text-decoration: none; font-weight: normal; }
.par .pv small { font-size: 9pt; color: #555; font-weight: normal; }
.idea { flex: 1; font-size: 9.8pt; line-height: 1.22; justify-content: space-between; }
.idea .tag { letter-spacing: .02em; white-space: nowrap; }
.idea .que { min-height: 8.6mm; }
.idea .res { display: flex; align-items: baseline; gap: 2.5mm; }
.idea .res .v { font-size: 17pt; font-weight: bold; color: #2F9E6E; line-height: 1.05; white-space: nowrap; }
.idea .res .v small { font-size: 10pt; color: #555; font-weight: normal; }
.f { font-family: "Cambria Math", Cambria, serif; font-style: italic; }
table { border-collapse: collapse; width: 100%; font-size: 9.5pt; border-top: 1pt solid #000; border-bottom: 1pt solid #000; }
thead th { border-bottom: 0.6pt solid #000; font-weight: normal; color: #555; }
th, td { padding: 0.9mm 1.5mm; text-align: right; }
th:first-child, td:first-child { text-align: left; }
td.win { background: #d6efe2; font-weight: bold; color: #102a43; }
td.near { background: #fbe9d2; }
.num { position: absolute; right: 12mm; bottom: 3mm; font-size: 8.5pt; color: #888; }
.foot { position: absolute; left: 12mm; bottom: 3mm; font-size: 8.5pt; color: #888; }
.title { justify-content: center; padding-top: 9mm; }
.title h1 { font-size: 30pt; }
"""

TITLE = "Aprendizaje por refuerzo para algoritmos de ejecución"
FOOT = "RL para ejecución de órdenes · Romero, Dapena y García · MIAX · github.com/Romequinco/rl-execution-agents"


def find_chrome():
    for c in CHROME_CANDIDATES:
        path = c if Path(c).exists() else shutil.which(c)
        if path:
            return path
    raise FileNotFoundError("No se encuentra Chrome/Edge para imprimir el PDF")


def main():
    files = sorted(SLIDES.glob("*.html"))
    if not files:
        raise SystemExit("No hay diapositivas en presentacion/slides/")
    parts = []
    for i, f in enumerate(files, 1):
        body = f.read_text(encoding="utf-8")
        footer = "" if i == 1 else f'<div class="foot">{FOOT}</div><div class="num">{i}</div>'
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
