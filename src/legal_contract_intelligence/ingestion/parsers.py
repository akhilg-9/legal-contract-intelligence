"""PDF and HTML parsers.

Returns a `ParsedDocument` with text content split into paragraphs and metadata
that we keep all the way through to citations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from bs4 import BeautifulSoup
from pypdf import PdfReader


@dataclass
class ParsedDocument:
    source_path: str
    doc_id: str
    title: Optional[str]
    paragraphs: List[str]
    page_breaks: List[int] = field(default_factory=list)  # paragraph indices that start a new page

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)


_WS_RE = re.compile(r"[ \t]+")
_NEWLINES_RE = re.compile(r"\n{2,}")


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = _NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def _split_paragraphs(text: str) -> List[str]:
    """Split on blank lines, drop empties, keep paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p for p in paragraphs if p]


def parse_pdf(path: Path) -> ParsedDocument:
    reader = PdfReader(str(path))
    paragraphs: List[str] = []
    page_breaks: List[int] = []
    title = (reader.metadata or {}).get("/Title") if reader.metadata else None

    for page in reader.pages:
        page_breaks.append(len(paragraphs))
        text = _normalize(page.extract_text() or "")
        for p in _split_paragraphs(text):
            paragraphs.append(p)

    doc_id = path.stem
    return ParsedDocument(
        source_path=str(path),
        doc_id=doc_id,
        title=title or doc_id,
        paragraphs=paragraphs,
        page_breaks=page_breaks,
    )


def parse_html(path: Path) -> ParsedDocument:
    html = path.read_text(errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else None

    paragraphs: List[str] = []
    # Prefer block-level elements as paragraph boundaries.
    blocks = soup.find_all(["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "td", "pre"])
    if blocks:
        for block in blocks:
            text = _normalize(block.get_text(" ", strip=True))
            if text:
                paragraphs.append(text)
    else:
        text = _normalize(soup.get_text("\n"))
        paragraphs = _split_paragraphs(text)

    doc_id = path.stem
    return ParsedDocument(
        source_path=str(path),
        doc_id=doc_id,
        title=title or doc_id,
        paragraphs=paragraphs,
    )


def parse_text(path: Path) -> ParsedDocument:
    text = _normalize(path.read_text(errors="ignore"))
    paragraphs = _split_paragraphs(text)
    doc_id = path.stem
    return ParsedDocument(
        source_path=str(path),
        doc_id=doc_id,
        title=doc_id,
        paragraphs=paragraphs,
    )


def parse_document(path: str | Path) -> ParsedDocument:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix in {".html", ".htm"}:
        return parse_html(path)
    if suffix in {".txt", ".md"}:
        return parse_text(path)
    raise ValueError(f"unsupported file type: {suffix}")
