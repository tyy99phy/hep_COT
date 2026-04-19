"""Extract readable text from arXiv LaTeX source packages."""

from __future__ import annotations

import gzip
import io
import os
import re
import tarfile
from pathlib import Path


def download_arxiv_source(arxiv_id: str, cache_dir: str) -> str | None:
    """Download arXiv e-print source and extract to cache_dir.
    Returns path to extracted directory, or None on failure.
    """
    from urllib.request import urlopen, Request
    from urllib.error import URLError

    extract_dir = os.path.join(cache_dir, f"{arxiv_id.replace('/', '_')}_source")
    if os.path.isdir(extract_dir) and any(
        f.endswith(".tex") for f in os.listdir(extract_dir)
    ):
        return extract_dir

    url = f"https://arxiv.org/e-print/{arxiv_id}"
    try:
        req = Request(url, headers={"User-Agent": "hep-cot/0.1"})
        data = urlopen(req, timeout=120).read()
    except (URLError, OSError) as e:
        print(f"Warning: Could not download arXiv source: {e}")
        return None

    os.makedirs(extract_dir, exist_ok=True)

    # Try tar.gz first (most common)
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            # Security: filter out absolute paths, parent refs, and symlinks
            safe_members = [
                m for m in tar.getmembers()
                if not m.name.startswith("/")
                and ".." not in m.name
                and not m.issym()
                and not m.islnk()
            ]
            import sys
            if sys.version_info >= (3, 12):
                tar.extractall(extract_dir, members=safe_members, filter="data")
            else:
                tar.extractall(extract_dir, members=safe_members)
        return extract_dir
    except tarfile.TarError:
        pass

    # Try single gzipped file
    try:
        content = gzip.decompress(data)
        with open(os.path.join(extract_dir, "paper.tex"), "wb") as f:
            f.write(content)
        return extract_dir
    except (gzip.BadGzipFile, OSError):
        pass

    # Raw TeX (rare)
    try:
        text = data.decode("utf-8", errors="replace")
        if "\\begin{document}" in text or "\\documentclass" in text:
            with open(os.path.join(extract_dir, "paper.tex"), "w") as f:
                f.write(text)
            return extract_dir
    except Exception:
        pass

    return None


def find_main_tex(source_dir: str) -> str | None:
    """Find the main .tex file in an extracted arXiv source directory."""
    tex_files = []
    for f in os.listdir(source_dir):
        if f.endswith(".tex") and not f.endswith("-authorlist.tex"):
            full = os.path.join(source_dir, f)
            tex_files.append((f, os.path.getsize(full)))

    if not tex_files:
        # Check one level deeper
        for d in os.listdir(source_dir):
            subdir = os.path.join(source_dir, d)
            if os.path.isdir(subdir):
                for f in os.listdir(subdir):
                    if f.endswith(".tex") and not f.endswith("-authorlist.tex"):
                        full = os.path.join(subdir, f)
                        tex_files.append((os.path.join(d, f), os.path.getsize(full)))

    if not tex_files:
        return None

    # Heuristic: the main file is usually the one with \begin{document}
    for fname, _ in tex_files:
        full = os.path.join(source_dir, fname)
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read(10000)
                if "\\begin{document}" in content:
                    return full
        except OSError:
            continue

    # Fallback: largest .tex file (excluding authorlist)
    tex_files.sort(key=lambda x: x[1], reverse=True)
    return os.path.join(source_dir, tex_files[0][0])


def read_tex_with_inputs(tex_path: str) -> str:
    """Read a .tex file and recursively resolve \\input{} and \\include{} commands."""
    base_dir = os.path.dirname(tex_path)
    seen: set[str] = set()

    def _resolve(path: str, depth: int = 0) -> str:
        if depth > 10 or path in seen:
            return ""
        seen.add(path)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            return ""

        def _replace_input(m: re.Match) -> str:
            ref = m.group(1)
            if not ref.endswith(".tex"):
                ref += ".tex"
            ref_path = os.path.join(base_dir, ref)
            if os.path.isfile(ref_path):
                return _resolve(ref_path, depth + 1)
            return m.group(0)

        text = re.sub(r"\\(?:input|include)\{([^}]+)\}", _replace_input, text)
        return text

    return _resolve(tex_path)


def clean_latex_for_llm(tex: str) -> str:
    """Light cleaning of LaTeX for LLM consumption.
    
    Preserves most structure including equations, but removes
    low-value boilerplate. Does NOT try to convert LaTeX to plain text --
    modern LLMs can read LaTeX natively.
    """
    # Remove comments (but keep %-escapes in URLs)
    tex = re.sub(r"(?<!\\)%.*$", "", tex, flags=re.MULTILINE)

    # Remove author list blocks (very long in CMS papers)
    tex = re.sub(
        r"\\begin\{LARGE\}.*?\\end\{LARGE\}",
        "[Author list removed]",
        tex,
        flags=re.DOTALL,
    )

    # Remove \hypersetup, \pdfoutput, and similar preamble noise
    tex = re.sub(r"\\hypersetup\{[^}]*\}", "", tex)
    tex = re.sub(r"\\pdfoutput\s*=\s*\d+", "", tex)

    # Remove excessive blank lines
    tex = re.sub(r"\n{3,}", "\n\n", tex)

    return tex.strip()


def extract_text_from_source(arxiv_id: str, cache_dir: str) -> str | None:
    """Full pipeline: download arXiv source → find main .tex → extract clean text."""
    source_dir = download_arxiv_source(arxiv_id, cache_dir)
    if not source_dir:
        return None

    main_tex = find_main_tex(source_dir)
    if not main_tex:
        print(f"Warning: No main .tex file found in source for {arxiv_id}")
        return None

    full_tex = read_tex_with_inputs(main_tex)
    if not full_tex:
        return None

    cleaned = clean_latex_for_llm(full_tex)

    # Also try to include bibliography
    bbl_files = [
        f for f in os.listdir(source_dir)
        if f.endswith(".bbl")
    ]
    if bbl_files:
        bbl_path = os.path.join(source_dir, bbl_files[0])
        try:
            with open(bbl_path, "r", encoding="utf-8", errors="replace") as f:
                bbl = f.read()
            cleaned += "\n\n% === Bibliography ===\n" + bbl
        except OSError:
            pass

    return cleaned
