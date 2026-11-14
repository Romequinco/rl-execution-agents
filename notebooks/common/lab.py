"""Utilidades de presentación para ``execution_lab.ipynb``: estilo común, tarjetas de cifras,
fuente de funciones y figuras ya guardadas. No entrena ni escribe en ``results/``."""
import inspect
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import HTML, Image, Markdown, display

from .protocol import FIGURES

ROOT = Path(__file__).resolve().parents[2]

# Colores de la casa (los mismos que plots.py): algoritmo, mercado, referencia.
ALGO, MKT, MID, OK, KO, GREY = "#D9822B", "#2B6CB0", "#C53030", "#2F9E6E", "#C53030", "#718096"


def setup():
    """Estilo de figuras y tablas común a todo el notebook."""
    plt.rcParams.update({
        "figure.dpi": 110, "figure.figsize": (11, 4.2), "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "axes.titleweight": "bold", "axes.titlesize": 12, "legend.frameon": False,
        "font.size": 10,
    })
    pd.set_option("display.precision", 3)


def es(texto):
    """Formato español: coma decimal y espacio antes de %."""
    texto = re.sub(r"(?<=\d)\.(?=\d)", ",", str(texto))
    return re.sub(r"(?<=\d)%", " %", texto)


def kpis(items, title=None):
    """Fila de tarjetas con cifras clave. ``items``: lista de (etiqueta, valor[, nota])."""
    cards = []
    for it in items:
        label, value = it[0], es(it[1])
        note = es(it[2]) if len(it) > 2 else ""
        cards.append(
            "<div style='flex:1;min-width:150px;border:1px solid #8884;border-radius:10px;"
            "padding:10px 14px'>"
            f"<div style='font-size:12px;opacity:.75'>{label}</div>"
            f"<div style='font-size:24px;font-weight:700;margin:2px 0'>{value}</div>"
            f"<div style='font-size:11px;opacity:.65'>{note}</div></div>")
    head = f"<div style='font-weight:700;margin:6px 0'>{title}</div>" if title else ""
    display(HTML(head + "<div style='display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 12px'>"
                 + "".join(cards) + "</div>"))


def codigo(*objetos, desde=None, hasta=None):
    """Muestra el código real de funciones o clases de ``common/`` (opcionalmente un fragmento)."""
    for obj in objetos:
        src = textwrap.dedent(inspect.getsource(obj))
        if desde is not None:
            src = src[src.rfind("\n", 0, src.index(desde)) + 1:]   # desde el inicio de esa línea
        if hasta is not None:
            src = src[:src.index(hasta)]
        src = textwrap.dedent(src)
        path = Path(inspect.getfile(obj)).resolve().relative_to(ROOT).as_posix()
        display(Markdown(f"**`{obj.__qualname__}`** · `{path}`\n\n```python\n{src.rstrip()}\n```"))


def figura(nombre, width=900):
    """Muestra una figura ya guardada en ``results/figures/<nombre>.png``."""
    display(Image(filename=str(FIGURES / f"{nombre}.png"), width=width))
