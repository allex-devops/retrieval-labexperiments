"""The paper corpus: download it, pull the text out page by page, and cut it into chunks.

Chunks never cross a page boundary, so every chunk knows which page it came from and the
benchmark can score retrieval at page level, not just "found the right paper".
"""
from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import pymupdf

from .paths import BENCHMARK_DIR, DATA_DIR

PAPERS = BENCHMARK_DIR / "papers.jsonl"
PDF_DIR = DATA_DIR / "pdfs"
PAGES = DATA_DIR / "pages.jsonl"
ARXIV_PDF = "https://arxiv.org/pdf/{id}"
# arXiv asks automated clients to keep to about one request every three seconds
POLITE_DELAY = 3.0


@dataclass(frozen=True)
class Chunk:
    id: str  # "<paper>:<page>:<n>"
    paper: str
    page: int
    text: str

    @property
    def page_id(self) -> str:
        return f"{self.paper}#{self.page}"


def load_jsonl(path: Path) -> list[dict]:
    # split on "\n" only: str.splitlines() also breaks on U+2028, U+0085 and friends, which PDF text
    # contains and json.dumps(ensure_ascii=False) leaves unescaped inside strings
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def paper_ids() -> list[str]:
    return [p["id"] for p in load_jsonl(PAPERS)]


def fetch(
    ids: list[str],
    dest: Path = PDF_DIR,
    *,
    from_dir: Path | None = None,
    delay: float = POLITE_DELAY,
    client: httpx.Client | None = None,
    sleep=time.sleep,
) -> dict[str, int]:
    """Get every PDF into `dest`. Resumable: files already there are skipped.
    With `from_dir`, copy from a local folder instead of downloading."""
    dest.mkdir(parents=True, exist_ok=True)
    counts = {"have": 0, "copied": 0, "downloaded": 0, "failed": 0}
    client = client or httpx.Client(
        timeout=60, follow_redirects=True, headers={"User-Agent": "retrieval-lab/0.1 (research benchmark)"}
    )
    first = True
    for pid in ids:
        target = dest / f"{pid}.pdf"
        if target.exists() and target.stat().st_size > 0:
            counts["have"] += 1
            continue
        if from_dir and (from_dir / f"{pid}.pdf").exists():
            shutil.copyfile(from_dir / f"{pid}.pdf", target)
            counts["copied"] += 1
            continue
        if not first:
            sleep(delay)
        first = False
        try:
            resp = client.get(ARXIV_PDF.format(id=pid))
            resp.raise_for_status()
            if not resp.content.startswith(b"%PDF"):
                raise ValueError("not a PDF")
        except (httpx.HTTPError, ValueError) as e:
            print(f"  {pid}: {e}")
            counts["failed"] += 1
            continue
        tmp = target.with_suffix(".part")
        tmp.write_bytes(resp.content)
        tmp.replace(target)
        counts["downloaded"] += 1
    return counts


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    """Split text into chunks of at most `size` characters, preferring to break at sentence ends."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # only look for a break in the back half, otherwise a stray early period makes tiny chunks
            floor = start + size // 2
            cut = max(text.rfind(". ", floor, end), text.rfind("\n", floor, end))
            if cut != -1:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)  # the max guarantees progress
    return chunks


def read_pages(path: Path) -> list[tuple[int, str]]:
    with pymupdf.open(path) as doc:
        return [(i + 1, page.get_text()) for i, page in enumerate(doc)]


def extract_pages(ids: list[str], pdf_dir: Path = PDF_DIR, out: Path = PAGES) -> int:
    """Pull the text out of every PDF once, so experiments don't re-parse PDFs."""
    rows = []
    for pid in ids:
        path = pdf_dir / f"{pid}.pdf"
        if not path.exists():
            continue
        for page, text in read_pages(path):
            if text.strip():
                rows.append({"paper": pid, "page": page, "text": text})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return len(rows)


def load_pages(path: Path = PAGES) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `retrieval-lab fetch` then `retrieval-lab prepare`")
    return load_jsonl(path)


def make_chunks(pages: list[dict], size: int = 800, overlap: int = 120) -> list[Chunk]:
    out = []
    for p in pages:
        for n, text in enumerate(chunk_text(p["text"], size=size, overlap=overlap)):
            out.append(Chunk(f"{p['paper']}:{p['page']}:{n}", p["paper"], p["page"], text))
    return out
