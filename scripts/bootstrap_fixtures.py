#!/usr/bin/env python3
"""Bootstrap local fixtures for the hep_cot demo + offline tests.

Fetches the default demo paper (arXiv:2206.08956) from arXiv and extracts
its LaTeX source into ``paper_cache/<id>_source/`` so that:

  * the README "Quick start" examples have a paper to load, and
  * ``tests/test_paper_tools_offline.py`` — which is ``pytest.mark.skipif``
    when the cached source is missing — can exercise the full paper-tool
    pipeline without network access.

Usage
-----
    python scripts/bootstrap_fixtures.py                # default paper
    python scripts/bootstrap_fixtures.py 1811.10461     # any arXiv id
    python scripts/bootstrap_fixtures.py --cache-dir /tmp/pc 2206.08956

Requires network on first run only. Re-running is a no-op if the source
directory already exists (arXiv fetch is cached on disk).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hep_cot.paper_fetcher import fetch_paper, normalize_arxiv_id  # noqa: E402

DEFAULT_PAPER = "2206.08956"
DEFAULT_CACHE = REPO_ROOT / "paper_cache"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "arxiv_id",
        nargs="?",
        default=DEFAULT_PAPER,
        help=f"arXiv id to bootstrap (default: {DEFAULT_PAPER})",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE),
        help=f"Destination cache directory (default: {DEFAULT_CACHE})",
    )
    args = parser.parse_args(argv)

    arxiv_id = normalize_arxiv_id(args.arxiv_id) or args.arxiv_id
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    source_dir = cache_dir / f"{arxiv_id.replace('/', '_')}_source"
    if source_dir.is_dir() and any(source_dir.iterdir()):
        print(f"[skip] {source_dir} already populated")
        return 0

    print(f"[fetch] arxiv:{arxiv_id} -> {cache_dir}")
    try:
        info = fetch_paper(arxiv_id, cache_dir=str(cache_dir))
    except Exception as e:
        print(f"[error] fetch_paper failed: {type(e).__name__}: {e}", file=sys.stderr)
        print(
            "        Check network access to arxiv.org / inspirehep.net "
            "and retry.",
            file=sys.stderr,
        )
        return 1

    print(f"[ok]    title:     {info.title or '(unknown)'}")
    print(f"[ok]    authors:   {len(info.authors)}")
    print(f"[ok]    chars:     {len(info.text_content)}")
    print(f"[ok]    source:    {source_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
