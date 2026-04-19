"""Fetch papers from arXiv/INSPIRE and extract text content."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.request import urlopen, Request
from urllib.error import URLError


@dataclass
class PaperInfo:
    """Metadata and content of a fetched paper."""
    arxiv_id: str = ""
    inspire_id: str = ""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    categories: list[str] = field(default_factory=list)
    publication_info: str = ""
    doi: str = ""
    pdf_path: str = ""
    text_content: str = ""
    figures_dir: str = ""
    source_url: str = ""


def normalize_arxiv_id(identifier: str) -> str | None:
    """Extract arXiv ID from various input formats."""
    identifier = identifier.strip()

    # Direct arXiv ID: 2012.12345 or hep-ex/0612015
    m = re.match(r"^(\d{4}\.\d{4,5})(v\d+)?$", identifier)
    if m:
        return m.group(1)

    m = re.match(r"^([\w-]+/\d{7})(v\d+)?$", identifier)
    if m:
        return m.group(1)

    # arXiv URL
    m = re.search(r"arxiv\.org/abs/([\d.]+|[\w-]+/\d{7})", identifier)
    if m:
        return m.group(1)

    m = re.search(r"arxiv\.org/pdf/([\d.]+|[\w-]+/\d{7})", identifier)
    if m:
        return m.group(1)

    return None


def normalize_inspire_id(identifier: str) -> str | None:
    """Extract INSPIRE literature ID or texkey from input."""
    identifier = identifier.strip()

    # INSPIRE URL: inspirehep.net/literature/12345
    m = re.search(r"inspirehep\.net/literature/(\d+)", identifier)
    if m:
        return m.group(1)

    # INSPIRE texkey style: CMS:2024xyz
    m = re.match(r"^[A-Z]+:\d{4}[a-z]+$", identifier)
    if m:
        return identifier

    return None


def fetch_arxiv_metadata(arxiv_id: str) -> PaperInfo:
    """Fetch metadata from arXiv API."""
    info = PaperInfo(arxiv_id=arxiv_id)
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"

    try:
        req = Request(url, headers={"User-Agent": "hep-cot/0.1"})
        with urlopen(req, timeout=30) as resp:
            data = resp.read().decode("utf-8")

        # Parse arXiv Atom feed with ElementTree
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        try:
            root = ET.fromstring(data)
            entry = root.find("atom:entry", ns)
            if entry is not None:
                title_el = entry.find("atom:title", ns)
                if title_el is not None and title_el.text:
                    info.title = re.sub(r"\s+", " ", title_el.text.strip())

                summary_el = entry.find("atom:summary", ns)
                if summary_el is not None and summary_el.text:
                    info.abstract = re.sub(r"\s+", " ", summary_el.text.strip())

                info.authors = [
                    name_el.text
                    for author_el in entry.findall("atom:author", ns)
                    if (name_el := author_el.find("atom:name", ns)) is not None
                    and name_el.text
                ]

                categories = [
                    cat_el.get("term", "")
                    for cat_el in entry.findall("arxiv:primary_category", ns)
                    + entry.findall("atom:category", ns)
                    if cat_el.get("term")
                ]
                info.categories = list(dict.fromkeys(categories))

                doi_el = entry.find("arxiv:doi", ns)
                if doi_el is not None and doi_el.text:
                    info.doi = doi_el.text.strip()
                else:
                    for link_el in entry.findall("atom:link", ns):
                        if link_el.get("title") == "doi":
                            href = link_el.get("href", "")
                            if href.startswith("http://dx.doi.org/"):
                                info.doi = href[len("http://dx.doi.org/"):]
                            elif href.startswith("https://doi.org/"):
                                info.doi = href[len("https://doi.org/"):]
                            break

        except ET.ParseError:
            # Fallback to regex if XML is malformed
            entry_m = re.search(r"<entry>(.*?)</entry>", data, re.DOTALL)
            if entry_m:
                entry_text = entry_m.group(1)
                title_m = re.search(r"<title>(.*?)</title>", entry_text, re.DOTALL)
                if title_m:
                    info.title = re.sub(r"\s+", " ", title_m.group(1).strip())

            abstract_m = re.search(r"<summary>(.*?)</summary>", data, re.DOTALL)
            if abstract_m:
                info.abstract = re.sub(r"\s+", " ", abstract_m.group(1).strip())

            info.authors = re.findall(r"<name>(.*?)</name>", data)

            categories = re.findall(r'category term="(.*?)"', data)
            info.categories = list(dict.fromkeys(categories))

            doi_m = re.search(r'doi="(.*?)"', data)
            if doi_m:
                info.doi = doi_m.group(1)

        info.source_url = f"https://arxiv.org/abs/{arxiv_id}"

    except (URLError, OSError) as e:
        print(f"Warning: Could not fetch arXiv metadata: {e}")

    # Fallback: try to extract metadata from LaTeX source if API failed
    if not info.title:
        info.title = ""  # will be filled from tex if available
    info.source_url = f"https://arxiv.org/abs/{arxiv_id}"

    return info


def extract_metadata_from_tex(info: PaperInfo, tex_content: str) -> None:
    """Fill in missing metadata fields by parsing the LaTeX source."""
    if not info.title:
        m = re.search(r"\\title\{(.+?)\}", tex_content, re.DOTALL)
        if m:
            title = m.group(1)
            title = re.sub(r"\\texorpdfstring\{[^}]*\}\{([^}]*)\}", r"\1", title)
            title = re.sub(r"[~\n]+", " ", title)
            title = re.sub(r"\\[a-zA-Z]+\s*", "", title)
            title = re.sub(r"[{}$]", "", title)
            info.title = re.sub(r"\s+", " ", title).strip()

    if not info.abstract:
        m = re.search(r"\\abstract\{(.+?)\}(?:\s*\\)", tex_content, re.DOTALL)
        if not m:
            m = re.search(r"\\begin\{abstract\}(.+?)\\end\{abstract\}", tex_content, re.DOTALL)
        if m:
            info.abstract = re.sub(r"\s+", " ", m.group(1).strip())

    if not info.doi:
        m = re.search(r"\\doi\{(10\.\d{4,}/[^\}]+)\}", tex_content)
        if m:
            info.doi = m.group(1)


def resolve_inspire_to_arxiv(inspire_id: str) -> str | None:
    """Resolve an INSPIRE ID to an arXiv ID."""
    if re.match(r"^\d+$", inspire_id):
        url = f"https://inspirehep.net/api/literature/{inspire_id}"
    else:
        url = f"https://inspirehep.net/api/literature?q=texkey:{inspire_id}&size=1"

    try:
        req = Request(url, headers={"User-Agent": "hep-cot/0.1", "Accept": "application/json"})
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if "metadata" in data:
            record = data
        elif "hits" in data and data["hits"].get("hits"):
            record = data["hits"]["hits"][0]
        else:
            return None

        arxiv_eprints = record.get("metadata", {}).get("arxiv_eprints", [])
        if arxiv_eprints:
            return arxiv_eprints[0].get("value")
    except (URLError, OSError, json.JSONDecodeError, KeyError):
        pass

    return None


def download_pdf(arxiv_id: str, output_dir: str) -> str:
    """Download PDF from arXiv."""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    output_path = os.path.join(output_dir, f"{arxiv_id.replace('/', '_')}.pdf")

    if os.path.exists(output_path):
        return output_path

    try:
        req = Request(pdf_url, headers={"User-Agent": "hep-cot/0.1"})
        with urlopen(req, timeout=120) as resp:
            with open(output_path, "wb") as f:
                f.write(resp.read())
        return output_path
    except (URLError, OSError) as e:
        raise RuntimeError(f"Failed to download PDF: {e}") from e


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using available tools."""
    # Try pdftotext first (poppler-utils)
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", pdf_path, "-"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: try python-based extraction
    try:
        import importlib
        pymupdf = importlib.import_module("fitz")  # PyMuPDF
        doc = pymupdf.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        if text.strip():
            return text
    except (ImportError, Exception):
        pass

    return ""


def extract_text_mineru(pdf_path: str, cache_dir: str) -> str | None:
    """Use MinerU API to extract text from PDF (fallback method).
    Requires MINERU_API_TOKEN environment variable or config.
    """
    api_token = os.environ.get("MINERU_API_TOKEN", "")
    api_base = os.environ.get("MINERU_API_BASE", "https://mineru.net/api/v4")

    if not api_token:
        return None

    try:
        # Import from the hep-rag-v2 MinerU client
        import sys
        rag_path = os.path.expanduser("~/.codex/hep-rag-v2-workspace/src")
        if rag_path not in sys.path:
            sys.path.insert(0, rag_path)

        from hep_rag_v2.providers.mineru_api import MinerUClient

        client = MinerUClient(api_base=api_base, api_token=api_token)
        task = client.submit_local_pdf(Path(pdf_path))

        # Download and extract the result
        zip_path = Path(cache_dir) / f"mineru_{Path(pdf_path).stem}.zip"
        client.download_result_zip(task, output_path=zip_path)

        import zipfile
        extract_dir = Path(cache_dir) / f"mineru_{Path(pdf_path).stem}"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

        # Look for markdown output
        for root, dirs, files in os.walk(extract_dir):
            for f in files:
                if f.endswith(".md"):
                    md_path = os.path.join(root, f)
                    with open(md_path, "r", encoding="utf-8") as fh:
                        content = fh.read()
                    if len(content) > 500:
                        print(f"  MinerU: extracted {len(content)} chars from {f}")
                        return content

    except Exception as e:
        print(f"  MinerU fallback failed: {e}")

    return None


def fetch_paper(
    identifier: str,
    cache_dir: str = "./paper_cache",
    force_method: str | None = None,
) -> PaperInfo:
    """Fetch a paper from arXiv or INSPIRE and extract text content.

    Text extraction priority:
      1. arXiv LaTeX source (best quality for HEP papers)
      2. MinerU API (if MINERU_API_TOKEN is set)
      3. pdftotext / PyMuPDF (last resort)

    Args:
        identifier: arXiv ID, arXiv URL, INSPIRE ID, or INSPIRE URL
        cache_dir: directory to cache downloaded files
        force_method: override extraction method ("latex", "mineru", "pdf")
    """
    os.makedirs(cache_dir, exist_ok=True)

    arxiv_id = normalize_arxiv_id(identifier)

    if not arxiv_id:
        inspire_id = normalize_inspire_id(identifier)
        if inspire_id:
            arxiv_id = resolve_inspire_to_arxiv(inspire_id)
            if not arxiv_id:
                raise ValueError(f"Could not resolve INSPIRE ID '{inspire_id}' to arXiv ID")
        else:
            raise ValueError(
                f"Cannot parse identifier: '{identifier}'. "
                "Provide an arXiv ID (e.g., 2012.12345), arXiv URL, "
                "or INSPIRE URL/texkey."
            )

    info = fetch_arxiv_metadata(arxiv_id)

    # --- Text extraction chain ---
    extraction_method = ""

    # Method 1: arXiv LaTeX source (best for HEP)
    if force_method in (None, "latex"):
        from .tex_extractor import extract_text_from_source
        print("  Trying arXiv LaTeX source...", end=" ", flush=True)
        tex_text = extract_text_from_source(arxiv_id, cache_dir)
        if tex_text and len(tex_text) > 1000:
            info.text_content = tex_text
            extraction_method = "latex_source"
            print(f"OK ({len(tex_text)} chars)")

    # Method 2: MinerU API (if source not available)
    if not info.text_content and force_method in (None, "mineru"):
        print("  Trying MinerU API...", end=" ", flush=True)
        try:
            pdf_path = download_pdf(arxiv_id, cache_dir)
            info.pdf_path = pdf_path
            mineru_text = extract_text_mineru(pdf_path, cache_dir)
            if mineru_text and len(mineru_text) > 500:
                info.text_content = mineru_text
                extraction_method = "mineru"
                print(f"OK ({len(mineru_text)} chars)")
            else:
                print("skipped (no API token or failed)")
        except Exception as e:
            print(f"failed ({e})")

    # Method 3: pdftotext / PyMuPDF (last resort)
    if not info.text_content and force_method in (None, "pdf"):
        print("  Falling back to pdftotext...", end=" ", flush=True)
        try:
            if not info.pdf_path:
                info.pdf_path = download_pdf(arxiv_id, cache_dir)
            pdf_text = extract_text_from_pdf(info.pdf_path)
            if pdf_text:
                info.text_content = pdf_text
                extraction_method = "pdftotext"
                print(f"OK ({len(pdf_text)} chars)")
            else:
                print("failed")
        except Exception as e:
            print(f"failed ({e})")

    if not info.text_content:
        print("  WARNING: No text could be extracted from this paper.")

    # Fill in missing metadata from LaTeX source if available
    if info.text_content and extraction_method == "latex_source":
        extract_metadata_from_tex(info, info.text_content)

    # Store extraction method as extra info
    info.publication_info = extraction_method

    return info
