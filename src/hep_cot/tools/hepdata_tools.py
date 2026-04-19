"""HEPData tools: record discovery + structured table access.

HEPData exposes each submission as a JSON document plus per-table
resources. The API surface has evolved over time; we stick to the
stable endpoints:

  * ``https://www.hepdata.net/search/?q=arxiv:<id>&format=json``
    returns search results including the HEPData record id
  * ``https://www.hepdata.net/record/ins<inspire_id>?format=json`` or
    ``/record/<publication_recid>?format=json`` returns a submission
  * A submission carries a list of ``data_tables``; each carries a
    ``csv_file`` / ``json`` URL for raw values.

Graceful degradation: many older papers have no HEPData record.
Callers must check ``error`` in the returned dict.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ..agent.tool_registry import ToolSchema


HEPDATA_BASE = "https://www.hepdata.net"
CACHE_TTL_SECONDS = 30 * 24 * 3600  # HEPData records are immutable once published


def _http_get_json(url: str, cache_dir: str) -> dict[str, Any] | None:
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    cache_path = Path(cache_dir) / f"hepdata_{key}.json"

    if cache_path.exists():
        try:
            age = time.time() - cache_path.stat().st_mtime
            if age < CACHE_TTL_SECONDS:
                with open(cache_path, encoding="utf-8") as f:
                    return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass

    data = _fetch_raw(url, expect_json=True)
    if data is None:
        return None

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except OSError:
        pass
    return data


def _fetch_raw(url: str, expect_json: bool = True) -> Any:
    try:
        import httpx  # type: ignore
    except ImportError:
        from urllib.request import Request, urlopen

        req = Request(
            url,
            headers={
                "User-Agent": "hep-copilot/0.2",
                "Accept": "application/json" if expect_json else "*/*",
            },
        )
        try:
            with urlopen(req, timeout=45) as resp:
                raw = resp.read()
            return json.loads(raw) if expect_json else raw
        except Exception:
            return None

    try:
        with httpx.Client(
            timeout=45.0,
            headers={
                "User-Agent": "hep-copilot/0.2",
                "Accept": "application/json" if expect_json else "*/*",
            },
            follow_redirects=True,
        ) as c:
            r = c.get(url)
            r.raise_for_status()
            return r.json() if expect_json else r.content
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Record discovery
# ---------------------------------------------------------------------------


def _search_record(arxiv_id: str, cache_dir: str) -> dict[str, Any] | None:
    """Return the first HEPData search hit for an arxiv id, or None."""
    url = f"{HEPDATA_BASE}/search/?q=arxiv:{quote(arxiv_id)}&format=json"
    data = _http_get_json(url, cache_dir)
    if not data:
        return None
    results = data.get("results") or []
    if not results:
        return None
    return results[0]


def _get_record(recid_or_ins: str, cache_dir: str) -> dict[str, Any] | None:
    """Fetch a HEPData record by internal recid or ins<inspire_id>."""
    url = f"{HEPDATA_BASE}/record/{recid_or_ins}?format=json"
    return _http_get_json(url, cache_dir)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _tool_fetch_hepdata(
    arxiv_id: str, cache_dir: str = "./paper_cache"
) -> dict[str, Any]:
    """Return record metadata + list of tables for ``arxiv_id``."""
    hit = _search_record(arxiv_id, cache_dir)
    if not hit:
        return {
            "error": f"no HEPData record found for arxiv:{arxiv_id}",
            "suggestion": "this paper may pre-date HEPData coverage, or the "
            "submission may be unpublished",
        }

    recid = hit.get("recid") or hit.get("record_id") or hit.get("id")
    inspire_id = hit.get("inspire_id") or hit.get("parent_inspire_id")

    lookup = None
    if inspire_id:
        lookup = _get_record(f"ins{inspire_id}", cache_dir)
    if lookup is None and recid:
        lookup = _get_record(str(recid), cache_dir)
    if lookup is None:
        return {
            "error": "search hit found but could not fetch full record",
            "search_hit": hit,
        }

    tables = lookup.get("data_tables") or []
    table_summaries = []
    for t in tables:
        table_summaries.append(
            {
                "name": t.get("name") or t.get("title") or "",
                "description": (t.get("description") or "")[:400],
                "doi": t.get("doi") or "",
                "has_numeric": bool(t.get("data_schema") or t.get("values_loader")),
            }
        )

    return {
        "arxiv_id": arxiv_id,
        "hepdata_recid": recid,
        "inspire_id": inspire_id,
        "title": (lookup.get("title") or hit.get("title") or "").strip(),
        "journal": lookup.get("journal_info", ""),
        "doi": lookup.get("hepdata_doi") or lookup.get("doi") or "",
        "total_tables": len(tables),
        "tables": table_summaries,
    }


def _tool_get_hepdata_table(
    arxiv_id: str,
    table_name: str,
    cache_dir: str = "./paper_cache",
    max_rows: int = 100,
) -> dict[str, Any]:
    """Return the numerical content of a single HEPData table by name."""
    hit = _search_record(arxiv_id, cache_dir)
    if not hit:
        return {"error": f"no HEPData record for arxiv:{arxiv_id}"}

    inspire_id = hit.get("inspire_id") or hit.get("parent_inspire_id")
    recid = hit.get("recid") or hit.get("record_id") or hit.get("id")
    lookup = None
    if inspire_id:
        lookup = _get_record(f"ins{inspire_id}", cache_dir)
    if lookup is None and recid:
        lookup = _get_record(str(recid), cache_dir)
    if lookup is None:
        return {"error": "record fetch failed"}

    tables = lookup.get("data_tables") or []
    tgt = None
    table_name_lc = table_name.lower()
    for t in tables:
        name = (t.get("name") or "") + " " + (t.get("title") or "")
        if table_name_lc in name.lower():
            tgt = t
            break
    if tgt is None:
        return {
            "error": f"table '{table_name}' not found",
            "available": [t.get("name") for t in tables[:30]],
        }

    # Try to get numeric data. HEPData typically exposes a data loader URL
    # or inline ``data_schema`` + ``values``.
    values_url = tgt.get("values_loader") or tgt.get("data_values_url")
    inline = tgt.get("values") or tgt.get("data")

    values: Any = None
    if values_url:
        url = (
            values_url
            if values_url.startswith("http")
            else f"{HEPDATA_BASE}{values_url}"
        )
        values = _http_get_json(url, cache_dir)
    if values is None and inline is not None:
        values = inline

    if values is None:
        return {
            "error": "table found but no numeric content could be resolved",
            "name": tgt.get("name", ""),
            "description": tgt.get("description", ""),
            "doi": tgt.get("doi", ""),
        }

    # Trim overly large payloads to protect the context window.
    trimmed = False
    if isinstance(values, list) and len(values) > max_rows:
        values = values[:max_rows]
        trimmed = True
    elif (
        isinstance(values, dict)
        and "dependent_variables" in values
    ):
        for dv in values.get("dependent_variables", []) or []:
            vs = dv.get("values") or []
            if len(vs) > max_rows:
                dv["values"] = vs[:max_rows]
                trimmed = True
        for iv in values.get("independent_variables", []) or []:
            vs = iv.get("values") or []
            if len(vs) > max_rows:
                iv["values"] = vs[:max_rows]
                trimmed = True

    return {
        "arxiv_id": arxiv_id,
        "name": tgt.get("name", ""),
        "description": (tgt.get("description") or "")[:1000],
        "doi": tgt.get("doi", ""),
        "truncated": trimmed,
        "max_rows": max_rows,
        "values": values,
    }


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------


def build_hepdata_tools(cache_dir: str = "./paper_cache") -> list[ToolSchema]:
    def discover(arxiv_id: str) -> dict[str, Any]:
        return _tool_fetch_hepdata(arxiv_id, cache_dir)

    def table(arxiv_id: str, table_name: str, max_rows: int = 100) -> dict[str, Any]:
        return _tool_get_hepdata_table(arxiv_id, table_name, cache_dir, max_rows)

    return [
        ToolSchema(
            name="fetch_hepdata",
            description=(
                "Look up the HEPData record for a paper by arxiv id and list "
                "its data tables (cross sections, cut flows, likelihood "
                "envelopes, etc). Returns 'error' if the paper has no "
                "HEPData submission (common for older or theory papers)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "arxiv_id": {
                        "type": "string",
                        "description": "arXiv id of the experimental paper.",
                    }
                },
                "required": ["arxiv_id"],
            },
            func=discover,
        ),
        ToolSchema(
            name="get_hepdata_table",
            description=(
                "Fetch the numerical content of a specific HEPData table for "
                "a paper. Matches table_name as a case-insensitive substring "
                "against table names/titles. Large tables are truncated; "
                "independent/dependent variables and values are returned."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "arxiv_id": {
                        "type": "string",
                        "description": "arXiv id of the paper.",
                    },
                    "table_name": {
                        "type": "string",
                        "description": (
                            "Substring to match against HEPData table names. "
                            "E.g. 'Table 5', 'Systematic breakdown', "
                            "'Differential cross section'."
                        ),
                    },
                    "max_rows": {
                        "type": "integer",
                        "description": "Truncate per-variable value list at this many rows.",
                        "default": 100,
                    },
                },
                "required": ["arxiv_id", "table_name"],
            },
            func=table,
        ),
    ]
