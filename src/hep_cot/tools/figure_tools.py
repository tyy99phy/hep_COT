"""Figure tools: list/get figure metadata for the loaded paper.

Figures are lazy-loaded the first time any figure tool is called.

Note: these tools return *metadata* (caption, label, PNG path). The
agent loop does not currently attach images to message turns — use
:func:`hep_cot.figure_extractor.build_figure_analysis_prompt` + the
upstream app if you want multimodal input. For MVP-1 the model
interacts via captions + file paths only.
"""

from __future__ import annotations

import os
from typing import Any

from ..agent.tool_registry import ToolSchema
from ..figure_extractor import FigureInfo, extract_and_prepare_figures
from .paper_tools import PaperContext


def _ensure_figures_loaded(ctx: PaperContext) -> str | None:
    """Load figures into ``ctx.figures`` lazily. Returns error string or None."""
    if ctx.figures:
        return None
    if not ctx.is_loaded:
        return "no paper loaded — call fetch_paper first"
    p = ctx.paper
    assert p is not None
    if p.publication_info != "latex_source":
        return (
            f"figures require LaTeX source extraction; current method is "
            f"'{p.publication_info or 'unknown'}'"
        )
    if not ctx.source_dir or not os.path.isdir(ctx.source_dir):
        return f"source directory not found: {ctx.source_dir!r}"

    figures = extract_and_prepare_figures(p.text_content, ctx.source_dir)
    ctx.figures = figures
    return None


def _fig_dict(fig: FigureInfo) -> dict[str, Any]:
    return {
        "figure_number": fig.figure_number,
        "label": fig.label,
        "caption": fig.caption_tex,
        "caption_short": fig.caption_short,
        "image_files": fig.image_files,
        "png_paths": fig.png_paths,
        "is_subfigure": fig.is_subfigure,
        "has_image": bool(fig.png_paths),
    }


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _tool_list_figures(ctx: PaperContext) -> dict[str, Any]:
    err = _ensure_figures_loaded(ctx)
    if err:
        return {"error": err}

    return {
        "count": len(ctx.figures),
        "figures": [
            {
                "figure_number": f.figure_number,
                "label": f.label,
                "caption_short": f.caption_short,
                "has_image": bool(f.png_paths),
                "is_subfigure": f.is_subfigure,
            }
            for f in ctx.figures
        ],
    }


def _tool_get_figure(ctx: PaperContext, figure_number: int) -> dict[str, Any]:
    err = _ensure_figures_loaded(ctx)
    if err:
        return {"error": err}

    for fig in ctx.figures:
        if fig.figure_number == figure_number:
            return _fig_dict(fig)
    return {
        "error": f"figure {figure_number} not found",
        "available": [f.figure_number for f in ctx.figures],
    }


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------


def build_figure_tools(ctx: PaperContext) -> list[ToolSchema]:
    def lst() -> dict[str, Any]:
        return _tool_list_figures(ctx)

    def get(figure_number: int) -> dict[str, Any]:
        return _tool_get_figure(ctx, figure_number)

    return [
        ToolSchema(
            name="list_figures",
            description=(
                "List all physics figures (excluding logos) in the loaded paper, "
                "with figure number, label, short caption, and whether an image "
                "file is available. Figures are lazy-loaded on first call."
            ),
            parameters={"type": "object", "properties": {}},
            func=lst,
        ),
        ToolSchema(
            name="get_figure",
            description=(
                "Return full metadata for a specific figure: full caption, "
                "label, image file paths (PNG), and whether it is a multi-panel "
                "(subfigure) composite."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "figure_number": {
                        "type": "integer",
                        "description": "1-based figure number as listed by list_figures.",
                    }
                },
                "required": ["figure_number"],
            },
            func=get,
        ),
    ]
