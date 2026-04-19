"""Extract figures, captions, and labels from arXiv LaTeX source."""

from __future__ import annotations

import base64
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FigureInfo:
    """A single figure extracted from the paper."""
    figure_number: int
    label: str = ""
    caption_tex: str = ""
    image_files: list[str] = field(default_factory=list)
    png_paths: list[str] = field(default_factory=list)
    is_subfigure: bool = False
    tex_block: str = ""
    tex_position: int = 0  # character offset in the full tex
    context_before: str = ""  # text snippet before the figure
    phase_key: str = ""  # which analysis phase this figure belongs to

    @property
    def caption_short(self) -> str:
        """First sentence of caption, lightly cleaned."""
        text = self.caption_tex
        text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
        text = re.sub(r"[{}\\~]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        # First sentence
        m = re.match(r"(.+?\.)\s", text)
        return m.group(1) if m else text[:150]


def extract_figures_from_tex(tex_content: str) -> list[FigureInfo]:
    """Parse LaTeX to extract all figure environments with metadata and position."""
    figures: list[FigureInfo] = []
    fig_num = 0

    pattern = r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}"
    for m in re.finditer(pattern, tex_content, re.DOTALL):
        block = m.group(1)

        # Skip logo-only figures
        imgs = re.findall(r"\\includegraphics.*?\{([^}]+)\}", block)
        physics_imgs = [
            img for img in imgs
            if not any(skip in img.lower() for skip in
                       ["logo", "orcid", "icon", "cms-bw", "cern"])
        ]
        if not physics_imgs:
            continue

        fig_num += 1

        # Extract caption (handle nested braces)
        cap_text = ""
        cap_start = block.find("\\caption{")
        if cap_start >= 0:
            depth = 0
            i = cap_start + len("\\caption{")
            start = i
            while i < len(block):
                if block[i] == "{":
                    depth += 1
                elif block[i] == "}":
                    if depth == 0:
                        cap_text = block[start:i]
                        break
                    depth -= 1
                i += 1

        # Extract label
        label = ""
        label_m = re.search(r"\\label\{([^}]+)\}", block)
        if label_m:
            label = label_m.group(1)

        # Capture surrounding context for phase assignment
        pos = m.start()
        ctx_start = max(0, pos - 500)
        context_before = tex_content[ctx_start:pos].strip()
        # Keep only last meaningful paragraph
        paragraphs = re.split(r"\n\s*\n", context_before)
        context_before = paragraphs[-1].strip() if paragraphs else ""

        figures.append(FigureInfo(
            figure_number=fig_num,
            label=label,
            caption_tex=cap_text.strip(),
            image_files=physics_imgs,
            is_subfigure=len(physics_imgs) > 1,
            tex_block=block.strip(),
            tex_position=pos,
            context_before=context_before[-300:],
        ))

    return figures


def build_figure_inventory(figures: list[FigureInfo]) -> str:
    """Build a concise text listing all figures for inclusion in prompts."""
    if not figures:
        return ""
    lines = ["The paper contains the following figures (all attached as images):"]
    for fig in figures:
        panels = f" ({len(fig.image_files)} panels)" if fig.is_subfigure else ""
        lines.append(
            f"  - Figure {fig.figure_number}{panels}: {fig.caption_short}"
        )
    lines.append(
        "\nWhen a figure is relevant to the current discussion, refer to it "
        "by number and analyze its content in detail."
    )
    return "\n".join(lines)


def convert_figures_to_png(
    figures: list[FigureInfo],
    source_dir: str,
    output_dir: str | None = None,
    dpi: int = 200,
) -> None:
    """Convert figure PDFs/EPS to PNG files for vision input."""
    if output_dir is None:
        output_dir = os.path.join(source_dir, "figures_png")
    os.makedirs(output_dir, exist_ok=True)

    for fig in figures:
        fig.png_paths = []
        for img_file in fig.image_files:
            src_path = os.path.join(source_dir, img_file)
            if not os.path.isfile(src_path):
                # Try without extension or with different extension
                for ext in [".pdf", ".png", ".eps", ".jpg", ""]:
                    candidate = os.path.join(source_dir, img_file.rsplit(".", 1)[0] + ext)
                    if os.path.isfile(candidate):
                        src_path = candidate
                        break
                else:
                    continue

            stem = Path(img_file).stem
            png_path = os.path.join(output_dir, f"{stem}.png")

            if os.path.isfile(png_path):
                fig.png_paths.append(png_path)
                continue

            ext = Path(src_path).suffix.lower()

            if ext == ".png":
                fig.png_paths.append(src_path)
                continue

            if ext == ".pdf":
                try:
                    subprocess.run(
                        ["pdftoppm", "-png", "-r", str(dpi), "-singlefile",
                         src_path, png_path.rsplit(".", 1)[0]],
                        capture_output=True, timeout=30,
                    )
                    if os.path.isfile(png_path):
                        fig.png_paths.append(png_path)
                        continue
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass

            if ext == ".eps":
                try:
                    subprocess.run(
                        ["convert", "-density", str(dpi), src_path, png_path],
                        capture_output=True, timeout=30,
                    )
                    if os.path.isfile(png_path):
                        fig.png_paths.append(png_path)
                        continue
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass


def figure_to_base64(png_path: str) -> str:
    """Read a PNG and return base64-encoded string for API input.

    Currently unused -- retained as a utility for future multimodal API input
    that requires inline base64 images rather than file paths.
    """
    with open(png_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def build_figure_analysis_prompt(fig: FigureInfo, paper_title: str = "") -> str:
    """Build a detailed analysis prompt for a single figure."""
    panels = ""
    if fig.is_subfigure:
        panel_labels = "abcdefgh"
        for i, img in enumerate(fig.image_files):
            label = panel_labels[i] if i < len(panel_labels) else str(i + 1)
            panels += f"  - Panel ({label}): {img}\n"

    return f"""\
Analyze Figure {fig.figure_number} of this paper in detail.
{f"Paper: {paper_title}" if paper_title else ""}

Original caption: {fig.caption_tex}

{f"This is a multi-panel figure:{chr(10)}{panels}" if panels else ""}

Please provide a thorough analysis:

1. DESCRIPTION: What exactly does this figure show? Identify all axes,
   curves, data points, bands, and legend entries.

2. PHYSICS CONTENT: What physics information is conveyed? What are the
   key quantitative takeaways from this figure?

3. DATA vs PREDICTION: If this is a data/MC comparison, comment on the
   level of agreement. Are there any notable discrepancies? Do the
   uncertainties look reasonable?

4. METHODOLOGY: What analysis choices are reflected in this figure?
   (Binning, variable choice, normalization, signal scaling, etc.)
   Are these choices well-motivated?

5. CRITICAL ASSESSMENT: Is this figure effective at communicating the
   result? Is there anything misleading, missing, or that could be
   improved? How does it compare to similar figures in other analyses?

Be quantitative -- read off specific values, ratios, and features
from the plot where possible."""


def extract_and_prepare_figures(
    tex_content: str,
    source_dir: str,
    cache_dir: str | None = None,
) -> list[FigureInfo]:
    """Full pipeline: parse tex → extract figures → convert to PNG."""
    figures = extract_figures_from_tex(tex_content)

    if not figures:
        return []

    png_dir = cache_dir or os.path.join(source_dir, "figures_png")
    convert_figures_to_png(figures, source_dir, png_dir)

    # Report
    total_pngs = sum(len(f.png_paths) for f in figures)
    print(f"  Extracted {len(figures)} figures ({total_pngs} image panels)")
    for fig in figures:
        status = "OK" if fig.png_paths else "MISSING"
        print(f"    Fig {fig.figure_number}: {fig.image_files} [{status}] -- {fig.caption_short[:80]}")

    return figures
