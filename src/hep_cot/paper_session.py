"""Interactive paper analysis session -- drives Codex through the analysis protocol."""

from __future__ import annotations

import os
import threading
import time

from .app_server_client import CodexSession
from .cot_store import CotStore
from .event_router import EventRouter, TurnRecord
from .paper_fetcher import PaperInfo, fetch_paper
from .analysis_protocol import (
    ANALYSIS_PHASES,
    AnalysisPhase,
    HepExpMetadata,
    build_initial_context,
    extract_phase_summary,
    format_phase_prompt,
)
from .figure_extractor import (
    FigureInfo,
    extract_and_prepare_figures,
    build_figure_inventory,
    build_figure_analysis_prompt,
)
from .ui import (
    TerminalPrinter,
    blue, bold, cyan, dim, green, red, yellow,
)


class PaperAnalysisSession:
    """Drives a structured paper analysis through Codex with CoT extraction."""

    def __init__(
        self,
        paper_identifier: str,
        codex_binary: str = "codex",
        model: str = "gpt-5.4",
        reasoning_effort: str = "high",
        output_dir: str = "./cot_sessions",
        cache_dir: str = "./paper_cache",
        sandbox: str = "read-only",
    ):
        self.paper_identifier = paper_identifier
        self.codex_binary = codex_binary
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.sandbox = sandbox

        self.paper: PaperInfo | None = None
        self.figures: list[FigureInfo] = []
        self.session: CodexSession | None = None
        self.router = EventRouter()
        self.store: CotStore | None = None
        self.hep_meta = HepExpMetadata()

        self._turn_done = threading.Event()
        self._printer = TerminalPrinter()
        self._last_turn_failed = False
        self._current_phase: AnalysisPhase | None = None
        self._phase_index = 0

        self._setup_router_callbacks()
        self._phase_summaries: list[str] = []  # accumulated summaries from completed phases

    def _setup_router_callbacks(self) -> None:
        self.router.on_reasoning_delta.append(self._printer.print_reasoning_delta)
        self.router.on_agent_message_delta.append(self._printer.print_agent_delta)
        self.router.on_tool_started.append(self._printer.print_tool_started)
        self.router.on_tool_completed.append(self._printer.print_tool_completed)
        self.router.on_turn_started.append(self._on_turn_started)
        self.router.on_turn_completed.append(self._on_turn_completed)
        self.router.on_error.append(self._on_error)

    # ------------------------------------------------------------------
    # Event handlers (same pattern as CotRepl)
    # ------------------------------------------------------------------

    def _on_notification(self, msg: dict) -> None:
        if self.store:
            self.store.log_raw_event(msg)
        self.router.handle_event(msg)

    def _on_server_request(self, msg: dict) -> dict | None:
        if self.store:
            self.store.log_raw_event(msg)
        method = msg.get("method", "")
        if "requestApproval" in method:
            params = msg.get("params", {})
            cmd = params.get("command", params.get("reason", "action"))
            print(f"\n{yellow('[APPROVAL]')} {cmd}")
            print(f"  {dim('Auto-accepting for paper analysis session')}")
            return {"decision": "accept"}
        return None

    def _on_turn_started(self, turn: TurnRecord) -> None:
        self._turn_done.clear()
        self._last_turn_failed = False
        self._printer.reset()

    def _on_error(self, message: str, will_retry: bool) -> None:
        self._printer.end_blocks()
        retry_note = " (retrying...)" if will_retry else " (giving up)"
        print(f"\n{red(f'[ERROR] {message}{retry_note}')}", flush=True)

    def _on_turn_completed(self, turn: TurnRecord) -> None:
        self._printer.end_blocks()

        turn.phase_key = self._current_phase.key if self._current_phase else "free"

        if turn.status == "failed":
            self._last_turn_failed = True
            print(f"\n{red('[Turn failed] The turn did not complete successfully.')}")
        else:
            self._last_turn_failed = False
            if self.store:
                self.store.log_turn(turn)
            # Extract summary for anti-repetition in subsequent phases
            if self._current_phase and turn.agent_response:
                summary = extract_phase_summary(
                    turn.agent_response, self._current_phase.name,
                )
                self._phase_summaries.append(summary)

        n_r = len(turn.reasoning_steps)
        usage = turn.usage
        duration = round(turn.end_time - turn.start_time, 1)
        reasoning_tokens = usage.get("output_tokens_details", {}).get("reasoning_tokens", 0)

        phase_label = self._current_phase.name if self._current_phase else "free"
        print(f"\n{dim('---')}")
        print(dim(
            f"  [{phase_label}] reasoning_steps={n_r} "
            f"duration={duration}s reasoning_tokens={reasoning_tokens}"
        ))
        print(dim("---"))
        self._turn_done.set()

    # ------------------------------------------------------------------
    # Core flow
    # ------------------------------------------------------------------

    def run(self) -> None:
        print(bold("=" * 60))
        print(bold("  HEP Paper Analysis -- CoT Extraction"))
        print(bold("=" * 60))
        print()

        # Step 1: Fetch paper
        print(f"{blue('[1/3]')} Fetching paper: {cyan(self.paper_identifier)}")
        try:
            self.paper = fetch_paper(self.paper_identifier, cache_dir=self.cache_dir)
        except Exception as e:
            print(f"{red(f'Error fetching paper: {e}')}")
            return

        print(f"  Title: {bold(self.paper.title)}")
        print(f"  arXiv: {self.paper.arxiv_id}")
        if self.paper.authors:
            display_authors = ", ".join(self.paper.authors[:3])
            if len(self.paper.authors) > 3:
                display_authors += f" et al. ({len(self.paper.authors)} authors)"
            print(f"  Authors: {display_authors}")
        if self.paper.text_content:
            print(f"  Text extracted: {len(self.paper.text_content)} chars")
            print(f"  Method: {self.paper.publication_info}")
        else:
            print(f"  {yellow('Warning: No text extracted from PDF.')}")
            print(f"  {dim('The analysis will rely on abstract and web search.')}")

        # Extract figures
        if self.paper.text_content and self.paper.publication_info == "latex_source":
            source_dir = os.path.join(
                self.cache_dir,
                f"{self.paper.arxiv_id.replace('/', '_')}_source",
            )
            if os.path.isdir(source_dir):
                self.figures = extract_and_prepare_figures(
                    self.paper.text_content, source_dir,
                )
        if self.figures:
            n_with_png = sum(1 for f in self.figures if f.png_paths)
            print(f"  Figures ready for vision analysis: {n_with_png}/{len(self.figures)}")
        print()

        # Step 2: Initialize session
        print(f"{blue('[2/3]')} Connecting to Codex...")
        session_id = f"paper_{self.paper.arxiv_id.replace('/', '_')}_{int(time.time())}"
        self.store = CotStore(output_dir=self.output_dir, session_id=session_id)

        self.session = CodexSession(
            codex_binary=self.codex_binary,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            sandbox=self.sandbox,
            approval_policy="never",
        )
        self.session.client.on_notification(self._on_notification)
        self.session.client.on_server_request(self._on_server_request)

        try:
            self.session.connect()
            thread_id = self.session.start_thread()
            self.store.set_meta(self.model, thread_id)
            if self.paper and self.paper.text_content:
                self.store.set_context(self.paper.text_content)
            from .analysis_protocol import REASONING_EXTERNALIZATION_INSTRUCTION
            self.store.set_system_prompt(REASONING_EXTERNALIZATION_INSTRUCTION)
            self.hep_meta.arxiv_id = self.paper.arxiv_id
            self.hep_meta.title = self.paper.title
            print(f"  Thread: {cyan(thread_id)}")
        except Exception as e:
            print(f"{red(f'Error connecting: {e}')}")
            return

        # Step 3: Run analysis
        print(f"{blue('[3/3]')} Starting structured analysis")
        print()
        self._print_protocol_overview()

        try:
            self._run_protocol()
        except KeyboardInterrupt:
            print(f"\n{yellow('Session interrupted.')}")
        except Exception as e:
            print(f"\n{red(f'Error: {e}')}")
        finally:
            self._finalize()

    def _print_protocol_overview(self) -> None:
        print(dim("Analysis phases:"))
        for i, phase in enumerate(ANALYSIS_PHASES):
            print(f"  {dim(f'{i+1}.')} {phase.name} {dim(f'-- {phase.description}')}")
        if self.figures:
            n_panels = sum(len(f.png_paths) for f in self.figures)
            print(f"\n  {yellow(f'{len(self.figures)} figures ({n_panels} panels) attached to every turn.')}")
            print(dim("  The model will reference figures naturally during discussion."))
            print(dim("  Any undiscussed figures will be flagged before the final assessment."))
        print()
        print(dim("Commands:"))
        print(dim("  [Enter]     = run next phase"))
        print(dim("  /skip       = skip current phase"))
        print(dim("  /ask <text> = free-form follow-up"))
        print(dim("  /deeper     = follow-up questions for current phase"))
        print(dim("  /status     = show progress"))
        print(dim("  /quit       = end session"))
        print()

    def _run_protocol(self) -> None:
        text_snippet = ""
        if self.paper:
            text_snippet = build_initial_context(self.paper)

        fig_inventory = build_figure_inventory(self.figures) if self.figures else ""
        all_png_paths = [
            p for fig in self.figures for p in fig.png_paths
        ]

        self._phase_index = 0

        while self._phase_index < len(ANALYSIS_PHASES):
            phase = ANALYSIS_PHASES[self._phase_index]
            self._current_phase = phase

            print(f"\n{'='*50}")
            print(f"{bold(blue(f'Phase {self._phase_index+1}/{len(ANALYSIS_PHASES)}: {phase.name}'))}")
            print(f"{'='*50}")

            try:
                action = input(
                    f"\n{dim('[Enter]=run / /skip / /ask <text> / /deeper / /quit]')} "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                break

            if action.lower() in ("/quit", "/exit", "/q"):
                break
            elif action.lower() == "/skip":
                print(dim(f"  Skipping {phase.name}"))
                self._phase_index += 1
                continue
            elif action.lower() == "/status":
                self._print_status()
                continue
            elif action.lower().startswith("/ask "):
                free_text = action[5:].strip()
                if free_text:
                    self._send_turn_with_figures(free_text, all_png_paths)
                continue
            elif action.lower() == "/deeper":
                self._run_follow_ups(phase, all_png_paths)
                continue
            elif action:
                self._send_turn_with_figures(action, all_png_paths)
                continue

            # Run the phase prompt
            if self.paper:
                prev_summary = "\n".join(self._phase_summaries) if self._phase_summaries else ""
                prompt = format_phase_prompt(
                    phase, self.paper, text_snippet, fig_inventory,
                    previous_phases_summary=prev_summary,
                )
            else:
                prompt = phase.prompt_template

            self._send_turn_with_figures(prompt, all_png_paths)

            if self._last_turn_failed:
                print(f"\n{yellow('Turn failed. Press Enter to retry, /skip to move on.')}")
                try:
                    retry = input(f"  {dim('[Enter]=retry / /skip]')} ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if retry.lower() != "/skip":
                    self._send_turn_with_figures(prompt, all_png_paths)

            if phase.follow_up_prompts:
                print(f"\n{dim(f'  {len(phase.follow_up_prompts)} follow-up questions available. Type /deeper to explore.')}")

            # Before the final assessment phase, check figure coverage
            next_idx = self._phase_index + 1
            if (next_idx < len(ANALYSIS_PHASES)
                    and ANALYSIS_PHASES[next_idx].key == "assessment"
                    and self.figures):
                self._ensure_figure_coverage(all_png_paths)

            self._phase_index += 1

        self._free_form_mode(all_png_paths)

    def _send_turn_with_figures(
        self,
        text: str,
        png_paths: list[str],
        timeout: float = 600,
    ) -> None:
        """Send a turn with text and all figure images attached."""
        if not self.session or not self.session.thread_id:
            return

        input_items: list[dict] = [{"type": "text", "text": text}]
        for png_path in png_paths:
            abs_path = os.path.abspath(png_path)
            input_items.append({
                "type": "localImage",
                "path": abs_path,
            })

        self._turn_done.clear()
        self.session.client.send_request("turn/start", {
            "threadId": self.session.thread_id,
            "input": input_items,
            "effort": self.session.reasoning_effort,
            "summary": "detailed",
        })
        self._turn_done.wait(timeout=timeout)

    def _run_follow_ups(self, phase: AnalysisPhase, png_paths: list[str] | None = None) -> None:
        for i, prompt in enumerate(phase.follow_up_prompts):
            print(f"\n{dim(f'Follow-up {i+1}/{len(phase.follow_up_prompts)}:')}")
            print(f"  {cyan(prompt)}")
            try:
                action = input(f"  {dim('[Enter]=ask / /skip]')} ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if action.lower() == "/skip":
                continue
            if png_paths:
                self._send_turn_with_figures(prompt, png_paths)
            else:
                self._send_and_wait(prompt)

    def _ensure_figure_coverage(self, all_png_paths: list[str]) -> None:
        """Check which figures have been discussed; force analysis of any missed ones."""
        # Scan all completed turn responses for figure references
        discussed: set[int] = set()
        for turn in self.router.completed_turns:
            text = turn.agent_response.lower()
            for fig in self.figures:
                # Check for "figure 1", "fig. 1", "fig 1", "Figure 1" etc.
                patterns = [
                    f"figure {fig.figure_number}",
                    f"fig. {fig.figure_number}",
                    f"fig {fig.figure_number}",
                    f"fig.~{fig.figure_number}",
                ]
                if any(p in text for p in patterns):
                    discussed.add(fig.figure_number)
                # Also check label references
                if fig.label and fig.label.lower() in text:
                    discussed.add(fig.figure_number)

        missed = [f for f in self.figures if f.figure_number not in discussed and f.png_paths]

        if not missed:
            print(f"\n{dim(f'  All {len(self.figures)} figures have been discussed.')}")
            return

        print(f"\n{'='*50}")
        print(f"{bold(yellow('Figure Coverage Check'))}")
        print(f"  {yellow(f'{len(missed)} figure(s) not yet discussed:')}")
        for fig in missed:
            print(f"    - Figure {fig.figure_number}: {fig.caption_short[:70]}")
        print(f"{'='*50}")

        for fig in missed:
            print(f"\n{bold(yellow(f'Analyzing Figure {fig.figure_number}...'))}")

            try:
                action = input(f"  {dim('[Enter]=analyze / /skip]')} ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if action.lower() == "/skip":
                continue

            title = self.paper.title if self.paper else ""
            prompt = build_figure_analysis_prompt(fig, title)
            # Send with only this figure's images for focused analysis
            self._send_turn_with_figures(prompt, fig.png_paths)

    def _free_form_mode(self, all_png_paths: list[str] | None = None) -> None:
        print(f"\n{'='*50}")
        print(bold(blue("Free-form discussion")))
        print(f"{'='*50}")
        print(dim("All phases complete. Ask any additional questions, or /quit to end."))

        while True:
            try:
                user_input = input(f"\n{bold(cyan('You > '))}").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input:
                continue
            if user_input.lower() in ("/quit", "/exit", "/q"):
                break
            if user_input.lower() == "/status":
                self._print_status()
                continue
            if user_input.lower() == "/export":
                if self.store:
                    path = self.store.finalize()
                    print(f"{green('Exported:')} {path}")
                continue
            if all_png_paths:
                self._send_turn_with_figures(user_input, all_png_paths)
            else:
                self._send_and_wait(user_input)

    def _send_and_wait(self, text: str, timeout: float = 600) -> None:
        if not self.session:
            return
        self._turn_done.clear()
        self.session.send_turn(text)
        self._turn_done.wait(timeout=timeout)

    def _print_status(self) -> None:
        turns = self.router.completed_turns
        total_reasoning = sum(len(t.reasoning_steps) for t in turns)
        total_tools = sum(len(t.tool_calls) for t in turns)
        print(f"\n{bold('Session Status:')}")
        if self.paper:
            print(f"  Paper: {self.paper.title}")
            print(f"  arXiv: {self.paper.arxiv_id}")
        print(f"  Phase: {self._phase_index}/{len(ANALYSIS_PHASES)}")
        print(f"  Turns: {len(turns)}")
        print(f"  Reasoning steps: {total_reasoning}")
        print(f"  Tool calls: {total_tools}")

    def _finalize(self) -> None:
        if not self.store:
            return

        # Add HEP metadata to the session summary
        self.store.update_meta("hep_metadata", self.hep_meta.to_dict())
        self.store.update_meta("paper", {
            "arxiv_id": self.paper.arxiv_id if self.paper else "",
            "title": self.paper.title if self.paper else "",
            "abstract": self.paper.abstract if self.paper else "",
            "authors_count": len(self.paper.authors) if self.paper else 0,
            "categories": self.paper.categories if self.paper else [],
            "doi": self.paper.doi if self.paper else "",
        })
        self.store.update_meta("analysis_protocol", {
            "total_phases": len(ANALYSIS_PHASES),
            "phases_completed": min(self._phase_index, len(ANALYSIS_PHASES)),
            "phase_names": [p.name for p in ANALYSIS_PHASES],
        })
        self.store.update_meta("figures", [
            {
                "number": fig.figure_number,
                "label": fig.label,
                "caption": fig.caption_tex[:500],
                "image_files": fig.image_files,
                "has_vision_analysis": bool(fig.png_paths),
            }
            for fig in self.figures
        ])

        path = self.store.finalize()
        print(f"\n{green('Session saved:')} {path}")

        turns = self.router.completed_turns
        total_reasoning = sum(len(t.reasoning_steps) for t in turns)
        print(f"  Turns: {len(turns)}")
        print(f"  Total reasoning steps: {total_reasoning}")
        print(f"  Raw events: {self.store.raw_file}")

        if self.session:
            self.session.disconnect()
