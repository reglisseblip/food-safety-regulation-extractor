"""
parser — turns a document (PDF, DOCX…) into clean markdown (tables included).

First stage of the extraction (before the LLM that produces the `Rule` objects).

Two paths:
  * **Docling** (DEFAULT): ML models (layout, tables via TableFormer, OCR). Robust and accurate on
    tables — the measured quality (benchmark) relies on it. Slower (~20 s/doc).
  * **pymupdf4llm** (`fast=True` option): digital PDF -> markdown WITHOUT any ML model. ~10× faster
    (2 s vs 20 s, and none of Docling's ~56 s import), keeps tables. Use it when speed matters more
    than accuracy (dev, large batches); `ocr=True` forces Docling (a scan has no text layer).

By default we keep the models: a run's speed comes from loading once (cached converter) and from
parallelism across documents, not from dropping the models. The fast path stays available.

Usage:
    uv run python src/parser.py "source/reglementations/India/GMP additives.pdf" --out out.md
    uv run python src/parser.py "<digital PDF>" --fast --out out.md      # fast path, no models
    uv run python src/parser.py "<scanned file>" --ocr --out out.md      # OCR (Docling)
"""
from __future__ import annotations

import logging
import pathlib
import sys
import threading

import logs

log = logging.getLogger(__name__)


def to_markdown(path: str | pathlib.Path, ocr: bool = False, fast: bool = False) -> str:
    """Convert a local document to markdown. Default = Docling (ML models); `fast=True` = fast
    pymupdf4llm path (digital PDF, no models). `ocr=True` (scan) forces Docling."""
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    use_fast = fast and not ocr and p.suffix.lower() == ".pdf"
    log.info("parsing [bold]%s[/] (ocr=%s, fast=%s)…", p.name, ocr, use_fast)
    md = _pymupdf_markdown(p) if use_fast else _docling_markdown(p, ocr)
    log.info("→ markdown : %d caractères", len(md))
    return md


# ------------------------------------------------------------------ fast path (opt-in)
def _pymupdf_markdown(p: pathlib.Path) -> str:
    """Digital PDF -> markdown (tables included) without any ML model. Lazy import:
    the library is loaded only when this path is taken."""
    import pymupdf4llm
    return pymupdf4llm.to_markdown(str(p))


# ------------------------------------------------------------------ robust path (default: Docling)
_CONVERTERS: dict = {}                 # one Docling converter per OCR mode, kept once
_CONVERTERS_LOCK = threading.Lock()    # two documents may parse at the same time


def _converter(ocr: bool):
    """Docling converter, built ONCE per OCR mode then reused (its models load on the first
    conversion, ~heavy). Lazy import: Docling is loaded only on first use — a run that takes only
    the fast path never pays its import (~56 s)."""
    with _CONVERTERS_LOCK:
        if ocr not in _CONVERTERS:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
            opts = PdfPipelineOptions(do_ocr=ocr, do_table_structure=True)
            _CONVERTERS[ocr] = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
            )
        return _CONVERTERS[ocr]


def _docling_markdown(p: pathlib.Path, ocr: bool) -> str:
    """Default path: Docling (ML models). Handles PDF, DOCX, and scans (`ocr=True`)."""
    return _converter(ocr).convert(str(p)).document.export_to_markdown()


if __name__ == "__main__":
    logs.setup()
    USAGE = 'usage: uv run python src/parser.py "<fichier>" [--fast] [--ocr] [--out sortie.md]'
    args = sys.argv[1:]

    # read --out (and its path) first, so we don't mistake it for the source file
    dst = None
    if "--out" in args:
        i = args.index("--out")
        if i + 1 >= len(args) or args[i + 1].startswith("--"):  # --out with no path after it
            print(USAGE)
            raise SystemExit(1)
        dst = args[i + 1]
        args = args[:i] + args[i + 2:]

    # the source file = the first remaining argument that isn't an option (order is free)
    sources = [a for a in args if not a.startswith("--")]
    if not sources:
        print(USAGE)
        raise SystemExit(1)

    src = sources[0]
    ocr = "--ocr" in args
    fast = "--fast" in args
    md = to_markdown(src, ocr=ocr, fast=fast)

    if dst:
        pathlib.Path(dst).write_text(md, encoding="utf-8")
        print(f"OK -> {dst}  ({len(md)} caractères, ocr={ocr}, fast={fast})")
    else:
        print(f"[{len(md)} caractères, ocr={ocr}, fast={fast}]\n")
        print(md[:3000])
