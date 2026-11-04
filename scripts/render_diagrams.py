"""Re-render docs/diagrams (sources -> PNG + SVG) and build docs/diagrams/Network_Attack_Forecasting_Diagrams.pdf.

Mermaid sources (.mmd: class, sequence, ER) need the Mermaid CLI:  npm i -g @mermaid-js/mermaid-cli
Graphviz sources (.dot: architecture, use case, activity, DFD) need Graphviz:  https://graphviz.org/download/
The PDF needs matplotlib (already in requirements). Usage:  python scripts/render_diagrams.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "docs" / "diagrams"
CAPTIONS = {"01_system_architecture": "Figure 1: System Architecture", "02_use_case": "Figure 2: Use Case Diagram", "03_class_diagram": "Figure 3: Class Diagram",
            "04_sequence_diagram": "Figure 4: Sequence Diagram", "05_activity_diagram": "Figure 5: Activity Diagram", "06_er_diagram": "Figure 6: ER Diagram (logical data model)",
            "07_level1_dfd": "Figure 7: Level-1 Data Flow Diagram"}
MERMAID_CFG = '{"theme":"base","themeVariables":{"fontFamily":"Segoe UI, Arial, sans-serif","fontSize":"15px","primaryColor":"#f6e9e3","primaryBorderColor":"#c96442","lineColor":"#4a4946","background":"#ffffff"},"flowchart":{"useMaxWidth":false},"sequence":{"useMaxWidth":false,"actorMargin":40},"er":{"useMaxWidth":false},"class":{"useMaxWidth":false}}'


def render() -> None:
    cfg = D / ".mermaid-config.json"; cfg.write_text(MERMAID_CFG)
    for src in sorted(D.glob("*.mmd")):
        if not shutil.which("mmdc"):
            print("mmdc not found - skipping", src.name); continue
        pp = ["-p", os.environ["MMDC_PUPPETEER_CONFIG"]] if os.environ.get("MMDC_PUPPETEER_CONFIG") else []   # e.g. --no-sandbox in containers
        for ext, extra in (("png", ["-s", "3"]), ("svg", [])):
            subprocess.run(["mmdc", "-i", str(src), "-o", str(src.with_suffix("." + ext)), "-c", str(cfg), "-b", "white", *pp, *extra], check=True)
    for src in sorted(D.glob("*.dot")):
        if not shutil.which("dot"):
            print("dot not found - skipping", src.name); continue
        subprocess.run(["dot", "-Tpng", "-Gdpi=220", str(src), "-o", str(src.with_suffix(".png"))], check=True, cwd=D)
        subprocess.run(["dot", "-Tsvg", str(src), "-o", str(src.with_suffix(".svg"))], check=True, cwd=D)
    cfg.unlink(missing_ok=True)


def pdf() -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    out = D / "Network_Attack_Forecasting_Diagrams.pdf"
    with PdfPages(out) as pp:
        for stem, cap in CAPTIONS.items():
            png = D / f"{stem}.png"
            if not png.exists():
                continue
            img = mpimg.imread(png); h, w = img.shape[:2]
            landscape = w / h > 1.1
            fig = plt.figure(figsize=(11.69, 8.27) if landscape else (8.27, 11.69))
            ax = fig.add_axes([0.04, 0.07, 0.92, 0.86]); ax.imshow(img); ax.axis("off")
            fig.text(0.5, 0.025, cap, ha="center", fontsize=12, family="DejaVu Sans")
            pp.savefig(fig, dpi=200); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    try:
        render(); pdf()
    except subprocess.CalledProcessError as e:
        sys.exit(f"render failed: {e}")
