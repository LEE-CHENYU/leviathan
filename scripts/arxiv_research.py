#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

try:
    import arxiv
except Exception as exc:
    raise SystemExit("Missing dependency: arxiv (pip install arxiv)") from exc

try:
    from pypdf import PdfReader
except Exception as exc:
    raise SystemExit("Missing dependency: pypdf (pip install pypdf)") from exc

try:
    from openai import OpenAI
except Exception as exc:
    raise SystemExit("Missing dependency: openai (pip install openai)") from exc

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "arxiv_loop.yaml"
STATE_FILE = ROOT / "research" / "arxiv" / "state.json"
LATEST_MD = ROOT / "research" / "arxiv" / "latest.md"
DEFAULT_STOPFILE = ROOT / "arxiv_24h.stop"


def load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def normalize_keyword(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _term_query(term: str) -> str:
    term = term.strip()
    if not term:
        return ""
    if " " in term:
        return f'ti:"{term}" OR abs:"{term}"'
    return f"ti:{term} OR abs:{term}"


def build_query(
    config: Dict[str, Any],
    topic: Optional[str] = None,
    query_override: Optional[str] = None,
) -> str:
    if query_override:
        return query_override

    topic = (topic or "").strip()
    keywords = [normalize_keyword(k) for k in (config.get("keywords") or []) if k]
    categories = [c.strip() for c in (config.get("categories") or []) if c]

    required_query = ""
    if topic:
        required_query = f"({_term_query(topic)})"

    kw_parts = []
    for kw in keywords:
        term = _term_query(kw)
        if term:
            kw_parts.append(f"({term})")

    kw_query = " OR ".join(kw_parts) if kw_parts else ""
    cat_query = " OR ".join([f"cat:{c}" for c in categories]) if categories else ""

    if required_query and kw_query and cat_query:
        return f"{required_query} AND ({kw_query}) AND ({cat_query})"
    if required_query and kw_query:
        return f"{required_query} AND ({kw_query})"
    if required_query and cat_query:
        return f"{required_query} AND ({cat_query})"
    if required_query:
        return required_query
    if kw_query and cat_query:
        return f"({kw_query}) AND ({cat_query})"
    if kw_query:
        return kw_query
    if cat_query:
        return cat_query
    return "all:agent"


def score_paper(result: arxiv.Result, keywords: List[str]) -> float:
    title = (result.title or "").lower()
    abstract = (result.summary or "").lower()

    score = 0.0
    for kw in keywords:
        kw = kw.lower()
        if kw in title:
            score += 3.0
        if kw in abstract:
            score += 1.0

    if result.categories:
        score += 0.5

    if result.published:
        days = (dt.datetime.now(dt.timezone.utc) - result.published).days
        if days <= 0:
            days = 0
        score += max(0.0, 30 - days) / 30.0

    return score


def download_pdf(result: arxiv.Result, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    result.download_pdf(filename=str(dest))
    return dest


def extract_text(pdf_path: Path, max_chars: int = 30000) -> str:
    reader = PdfReader(str(pdf_path))
    chunks = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text:
            chunks.append(text)
        if sum(len(c) for c in chunks) >= max_chars:
            break
    text = "\n".join(chunks)
    return text[:max_chars]


def init_client() -> Tuple[OpenAI, str]:
    provider = os.environ.get("ARXIV_LLM_PROVIDER", "openrouter").strip().lower()
    model = os.environ.get("ARXIV_LLM_MODEL", "minimax/minimax-m2.1").strip()
    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY missing for OpenRouter provider")
        base_url = os.environ.get("ARXIV_LLM_BASE_URL", "https://openrouter.ai/api/v1")
        client = OpenAI(api_key=api_key, base_url=base_url)
        return client, model

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY missing for OpenAI provider")
    client = OpenAI(api_key=api_key)
    return client, model


def summarize_paper(
    client: OpenAI,
    model: str,
    paper: Dict[str, Any],
    content: str,
    max_tokens: int,
    topic: Optional[str] = None,
) -> str:
    topic_line = f"Research topic: {topic}" if topic else ""
    prompt = f"""
You are summarizing a research paper for the Leviathan project (multi-agent, emergent strategy, and system design).
{topic_line}

Title: {paper['title']}
Authors: {paper['authors']}
Published: {paper['published']}
Categories: {paper['categories']}

Abstract:
{paper['abstract']}

Extracted text (partial):
{content}

Write a concise, decision‑oriented summary with:
1) Core contribution and method
2) Key findings/claims
3) Why it matters for Leviathan (explicitly tie to self‑improving strategies, diversity, coordination, mechanisms, or evaluation)
4) Concrete experiments or changes we could try
5) Risks/limitations

Keep it under 400 words.
""".strip()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")


def write_summary_md(output_dir: Path, paper: Dict[str, Any], summary: str, score: float) -> Path:
    safe_id = paper["arxiv_id"].replace("/", "_")
    date = paper["published"].split("T")[0]
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", paper["title"].strip().lower())[:80]
    filename = f"{date}_{slug}_{safe_id}.md"
    path = output_dir / filename

    md = f"""
# {paper['title']}

- **arXiv**: {paper['arxiv_id']}
- **Published**: {paper['published']}
- **Authors**: {paper['authors']}
- **Categories**: {paper['categories']}
- **Relevance score**: {score:.2f}
- **PDF**: {paper['pdf_url']}
- **Link**: {paper['abs_url']}

## Summary
{summary}

## Abstract
{paper['abstract']}
""".strip() + "\n"

    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(md)
    return path


def update_latest_md(output_dir: Path, picks: List[Tuple[Dict[str, Any], float]]) -> None:
    lines = ["# Latest arXiv relevance picks", ""]
    for paper, score in picks:
        lines.append(f"- **{paper['title']}** ({paper['arxiv_id']}, score {score:.2f})")
        lines.append(f"  - {paper['abs_url']}")
    lines.append("")
    LATEST_MD.write_text("\n".join(lines))


def run_once(
    config: Dict[str, Any],
    topic_override: Optional[str] = None,
    query_override: Optional[str] = None,
) -> None:
    query = build_query(config, topic=topic_override, query_override=query_override)
    keywords = [normalize_keyword(k) for k in (config.get("keywords") or []) if k]
    if topic_override:
        keywords.insert(0, normalize_keyword(topic_override))
    max_results = int(config.get("max_results") or 50)
    top_k = int(config.get("TopK") or 5)
    min_score = float(config.get("min_score") or 0.0)
    output_dir = ROOT / (config.get("output_dir") or "research/arxiv")
    summary_model = config.get("summary_model") or os.environ.get("ARXIV_LLM_MODEL", "minimax/minimax-m2.1")
    summary_max_tokens = int(config.get("summary_max_tokens") or 1024)
    recency_days = int(config.get("recency_days") or 0)

    client = arxiv.Client(page_size=25, delay_seconds=3, num_retries=2)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    state = load_state()
    seen = set(state.get("seen", []))

    results = list(client.results(search))
    ranked: List[Tuple[arxiv.Result, float]] = []
    for result in results:
        if recency_days and result.published:
            age_days = (dt.datetime.now(dt.timezone.utc) - result.published).days
            if age_days > recency_days:
                continue
        score = score_paper(result, keywords)
        if score < min_score:
            continue
        ranked.append((result, score))

    ranked.sort(key=lambda item: item[1], reverse=True)
    picks = ranked[:top_k]

    if not picks:
        return

    llm_client, model_id = init_client()
    if summary_model:
        model_id = summary_model

    summary_items: List[Tuple[Dict[str, Any], float]] = []
    for result, score in picks:
        arxiv_id = result.get_short_id()
        if arxiv_id in seen:
            continue

        pdf_name = arxiv_id.replace("/", "_") + ".pdf"
        pdf_path = output_dir / "pdfs" / pdf_name
        try:
            download_pdf(result, pdf_path)
            content = extract_text(pdf_path)
        except Exception:
            content = result.summary or ""

        paper = {
            "title": result.title.strip(),
            "authors": ", ".join(a.name for a in result.authors),
            "published": result.published.isoformat(),
            "abstract": result.summary.strip(),
            "categories": ", ".join(result.categories),
            "pdf_url": result.pdf_url,
            "abs_url": result.entry_id,
            "arxiv_id": arxiv_id,
        }

        summary = summarize_paper(
            llm_client,
            model_id,
            paper,
            content,
            summary_max_tokens,
            topic=topic_override,
        )
        write_summary_md(output_dir, paper, summary, score)
        summary_items.append((paper, score))
        seen.add(arxiv_id)

    if summary_items:
        update_latest_md(output_dir, summary_items)

    state["seen"] = sorted(seen)
    state["last_run_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    save_state(state)


def main() -> int:
    parser = argparse.ArgumentParser(description="Search arXiv, rank papers, summarize and store Markdown summaries.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--once", action="store_true", help="Run a single cycle.")
    parser.add_argument("--topic", default=None, help="Override research topic to focus search.")
    parser.add_argument("--query", default=None, help="Override the full arXiv query string.")
    parser.add_argument("--duration-hours", type=float, default=24.0)
    parser.add_argument("--interval-minutes", type=float, default=None)
    args = parser.parse_args()

    load_env()
    config = load_config(Path(args.config))

    interval_minutes = args.interval_minutes or float(config.get("interval_minutes") or 360)
    stopfile = Path(os.environ.get("ARXIV_STOPFILE", str(DEFAULT_STOPFILE)))

    if args.once:
        run_once(config, topic_override=args.topic, query_override=args.query)
        return 0

    end_time = time.time() + args.duration_hours * 3600
    while time.time() < end_time:
        if stopfile.exists():
            break
        run_once(config, topic_override=args.topic, query_override=args.query)
        for _ in range(int(interval_minutes * 60)):
            if stopfile.exists():
                return 0
            time.sleep(1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
