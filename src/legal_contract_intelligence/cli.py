"""`lci` command-line entry point.

Commands:
- lci ingest <path>           Parse + chunk + embed + upsert a file or directory
- lci ask "<question>"        Run the full RAG pipeline; print answer + citations
- lci search "<query>"        Retrieval-only; print top-k chunks with scores
- lci info                    Show Qdrant collection stats
- lci fetch-samples           Pull a handful of public SEC contracts into data/sample/
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import settings
from .ingestion.edgar import EdgarClient
from .pipeline import RagPipeline, ingest_path
from .prompts import load_prompt
from .retrieval import RetrievalMode, retrieve as retrieve_chunks
from .vectorstore import collection_info, get_vector_store

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.command()
def ingest(
    path: str = typer.Argument(..., help="File or directory to ingest"),
    chunk_tokens: int = typer.Option(650, help="Target tokens per chunk"),
    overlap_tokens: int = typer.Option(100, help="Token overlap between adjacent chunks"),
) -> None:
    """Parse, chunk, embed, and upsert into Qdrant."""
    n = ingest_path(path, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
    console.print(f"[green]ingested {n} chunks from {path}[/green]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Natural-language question"),
    provider: Optional[str] = typer.Option(None, help="Override prompt config's provider (ollama|openai)"),
    prompt_version: Optional[str] = typer.Option(None, "--prompt", help="Specific prompt version, e.g. v1"),
) -> None:
    """Answer a question using the RAG pipeline."""
    prompt_config = load_prompt(version=prompt_version)
    pipeline = RagPipeline(prompt_config=prompt_config, provider_override=provider)
    result = pipeline.ask(question)

    console.print(Panel(result.answer, title=f"Answer  (prompts: {result.prompt_version})", expand=False))

    table = Table(title=f"Retrieved chunks (top {len(result.retrieved)})", show_lines=False)
    table.add_column("chunk_id", style="cyan", no_wrap=True)
    table.add_column("score", justify="right")
    table.add_column("doc / page", style="dim")
    table.add_column("preview")
    for c in result.retrieved:
        page = c.metadata.get("page_number")
        loc = f"{c.metadata.get('title') or c.doc_id}" + (f"  p.{page}" if page else "")
        preview = c.text.replace("\n", " ")[:140] + ("..." if len(c.text) > 140 else "")
        table.add_row(c.chunk_id, f"{c.score:.3f}", loc, preview)
    console.print(table)

    if result.citations:
        console.print(f"[bold]cited:[/bold] {', '.join(result.citations)}")
    if result.insufficient_context:
        console.print("[yellow]flagged: insufficient context[/yellow]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Retrieval-only query"),
    k: int = typer.Option(8, help="Top-k chunks to return"),
    mode: str = typer.Option("reranked", help="dense | sparse | hybrid | reranked"),
    candidate_k: int = typer.Option(20, help="Per-source breadth before fusion/rerank"),
) -> None:
    """Run retrieval only (no generation); inspect what the retriever surfaces."""
    pairs = retrieve_chunks(query, mode=mode, top_k=k, candidate_k=candidate_k)  # type: ignore[arg-type]
    table = Table(title=f"Top-{k} for: {query}  (mode={mode})")
    table.add_column("chunk_id", style="cyan", no_wrap=True)
    table.add_column("score", justify="right")
    table.add_column("doc / page", style="dim")
    table.add_column("preview")
    for doc, score in pairs:
        page = doc.metadata.get("page_number")
        loc = f"{doc.metadata.get('title') or doc.metadata.get('doc_id', '?')}" + (f"  p.{page}" if page else "")
        preview = doc.page_content.replace("\n", " ")[:140] + ("..." if len(doc.page_content) > 140 else "")
        table.add_row(str(doc.metadata.get("chunk_id", "?")), f"{score:.3f}", loc, preview)
    console.print(table)


@app.command()
def info() -> None:
    """Print Qdrant collection stats."""
    console.print(collection_info())


@app.command("fetch-samples")
def fetch_samples(
    query: str = typer.Option("material contract", help="EDGAR full-text query"),
    forms: str = typer.Option("8-K,10-K", help="Comma-separated form types"),
    limit: int = typer.Option(5, help="Number of filings to pull"),
    dest: str = typer.Option("data/sample", help="Where to drop the downloaded files"),
) -> None:
    """Pull a few public SEC contract exhibits into data/sample/."""
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    with EdgarClient(user_agent=settings.edgar_user_agent) as client:
        filings = client.search_exhibits(query=query, forms=tuple(forms.split(",")), limit=limit)
        if not filings:
            console.print("[yellow]no filings matched.[/yellow]")
            return
        for filing in filings:
            try:
                local = client.fetch_document(filing)
            except Exception as exc:  # pragma: no cover
                console.print(f"[red]skip {filing.accession_no}: {exc}[/red]")
                continue
            target = dest_path / f"{filing.accession_no}_{local.name}"
            target.write_bytes(local.read_bytes())
            console.print(f"[green]ok[/green] {target}")


if __name__ == "__main__":
    app()
