"""SEC EDGAR fetcher for contract exhibits.

EDGAR's fair-access policy requires identifying ourselves in the User-Agent
header and limiting request rate to 10/sec. We use httpx with a small token-
bucket and on-disk caching so re-running ingestion is free.

Reference: https://www.sec.gov/os/accessing-edgar-data
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import httpx


EDGAR_BASE = "https://www.sec.gov"
EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"


@dataclass
class EdgarFiling:
    accession_no: str
    cik: str
    form_type: str
    filing_date: str
    primary_doc: str  # path/filename inside the filing
    company_name: str
    description: Optional[str] = None

    @property
    def archive_dir(self) -> str:
        # CIK without leading zeros, accession with dashes removed
        clean = self.accession_no.replace("-", "")
        cik_no_pad = str(int(self.cik))
        return f"{EDGAR_BASE}/Archives/edgar/data/{cik_no_pad}/{clean}"


class EdgarClient:
    def __init__(
        self,
        user_agent: str,
        cache_dir: str | Path = "data/edgar_raw",
        rate_limit_per_sec: int = 8,
    ):
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "EDGAR requires a User-Agent that includes a contact email "
                "(see https://www.sec.gov/os/accessing-edgar-data)."
            )
        self.user_agent = user_agent
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._min_interval = 1.0 / max(1, rate_limit_per_sec)
        self._last_request_at: float = 0.0
        self._client = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=30.0,
            follow_redirects=True,
        )

    def __enter__(self) -> "EdgarClient":
        return self

    def __exit__(self, *exc) -> None:
        self._client.close()

    def _throttle(self) -> None:
        now = time.monotonic()
        wait = self._min_interval - (now - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    def _get(self, url: str) -> httpx.Response:
        self._throttle()
        response = self._client.get(url)
        response.raise_for_status()
        return response

    def search_exhibits(
        self,
        query: str = "material contract",
        forms: Iterable[str] = ("8-K", "10-K"),
        limit: int = 10,
    ) -> List[EdgarFiling]:
        """Full-text search EDGAR for material-contract exhibits."""
        params = {
            "q": query,
            "dateRange": "custom",
            "forms": ",".join(forms),
        }
        url = httpx.URL(EDGAR_SEARCH, params=params)
        response = self._get(str(url))
        data = response.json()
        hits = data.get("hits", {}).get("hits", [])[:limit]
        filings: List[EdgarFiling] = []
        for hit in hits:
            src = hit.get("_source", {})
            adsh = (src.get("adsh") or "").strip()
            ciks = src.get("ciks") or []
            display_names = src.get("display_names") or []
            if not adsh or not ciks:
                continue
            filings.append(
                EdgarFiling(
                    accession_no=adsh,
                    cik=str(ciks[0]),
                    form_type=src.get("form", "?"),
                    filing_date=src.get("file_date", "?"),
                    primary_doc=hit.get("_id", "").split(":")[-1],
                    company_name=display_names[0] if display_names else "?",
                    description=src.get("description"),
                )
            )
        return filings

    def fetch_document(self, filing: EdgarFiling) -> Path:
        """Download a filing's primary document into the cache, return local path."""
        cache_subdir = self.cache_dir / filing.accession_no.replace("-", "")
        cache_subdir.mkdir(parents=True, exist_ok=True)
        local = cache_subdir / filing.primary_doc.replace("/", "_")
        if local.exists() and local.stat().st_size > 0:
            return local
        url = f"{filing.archive_dir}/{filing.primary_doc}"
        response = self._get(url)
        local.write_bytes(response.content)
        manifest = cache_subdir / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "accession_no": filing.accession_no,
                    "cik": filing.cik,
                    "form": filing.form_type,
                    "company": filing.company_name,
                    "filing_date": filing.filing_date,
                    "primary_doc": filing.primary_doc,
                },
                indent=2,
            )
        )
        return local
