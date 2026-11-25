"""
Standalone PySide6 dashboard for inspecting System 1 (fast decision) and
System 2 (longer thinking) prompts, exact sent minimap images, tool calls,
and structured outputs.

Live mode (polls the host policy server over HTTP in a background thread):
    uv run python -m starcraft_diffusion_gemma.dashboard --url http://localhost:8080

Offline replay mode (reads a ``--log-dir`` recorded by ``server.app``):
    uv run python -m starcraft_diffusion_gemma.dashboard --log-dir ./logs/match1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import httpx
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from starcraft_diffusion_gemma.brain.traces import (
    S1Trace,
    S2Trace,
    TracesSnapshot,
)
from starcraft_diffusion_gemma.state.models import CommandBundle

MAX_HISTORY_ITEMS = 100
MAX_PIXMAP_CACHE = 120


def format_timestamp(ts: float) -> str:
    """Format an epoch timestamp as ``HH:MM:SS`` local time."""
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def format_bundle_summary(bundle: CommandBundle | None, frame: int | None) -> str:
    """Render a compact one-line summary of the active S0 ``CommandBundle``."""
    if bundle is None:
        return "Waiting for game state (/tick)..."
    tgt = (
        f"({bundle.army_target.x},{bundle.army_target.y})"
        if bundle.army_target
        else "-"
    )
    step = (
        f"{bundle.active_build_step.supply} {bundle.active_build_step.action} {bundle.active_build_step.target}"
        if bundle.active_build_step
        else "done"
    )
    s1_age = f"{bundle.s1_age_secs:.1f}s" if bundle.s1_age_secs is not None else "-"
    s2_age = f"{bundle.s2_age_secs:.1f}s" if bundle.s2_age_secs is not None else "-"
    return (
        f"Frame {frame if frame is not None else bundle.frame_issued}  |  "
        f"Rev {bundle.revision}  |  "
        f"Posture: {bundle.posture.value} -> {tgt}  |  "
        f"Mix: {bundle.zealot_ratio:.0%}Z/{bundle.dragoon_ratio:.0%}D  |  "
        f"Gates: {bundle.target_gateways}  Probes: {bundle.target_probes}  |  "
        f"Next: {step}  |  "
        f"S1 age: {s1_age}  S2 age: {s2_age}"
    )


def format_s1_list_item(trace: S1Trace) -> str:
    """One-line summary for the System 1 history list."""
    ts = format_timestamp(trace.timestamp)
    if trace.error:
        return (
            f"#{trace.id}  f{trace.frame}  {ts}  [ERROR]  ({trace.latency_secs:.2f}s)"
        )
    if trace.output is None:
        return f"#{trace.id}  f{trace.frame}  {ts}  [no output]  ({trace.latency_secs:.2f}s)"
    out = trace.output
    esc = "  [ESCALATE]" if out.escalate_to_s2 else ""
    emg = (
        f"  [{out.emergency_action.value}]"
        if out.emergency_action.value != "none"
        else ""
    )
    return (
        f"#{trace.id}  f{trace.frame}  {ts}  "
        f"{out.posture.value} -> ({out.army_target.x},{out.army_target.y})  "
        f"threat={out.threat_level.value}{emg}{esc}  ({trace.latency_secs:.2f}s)"
    )


def format_s2_list_item(trace: S2Trace) -> str:
    """One-line summary for the System 2 history list."""
    ts = format_timestamp(trace.timestamp)
    n_tools = len(trace.tool_calls)
    if trace.error:
        return (
            f"#{trace.id}  f{trace.frame}  {ts}  [{trace.reason}]  "
            f"{n_tools} tools  [ERROR]  ({trace.latency_secs:.2f}s)"
        )
    phase = trace.directive.phase.value if trace.directive else "-"
    return (
        f"#{trace.id}  f{trace.frame}  {ts}  [{trace.reason}]  "
        f"phase={phase}  {n_tools} tools  ({trace.latency_secs:.2f}s)"
    )


def load_log_dir(
    log_dir: Path,
) -> tuple[list[S1Trace], list[S2Trace], dict[str, bytes]]:
    """Load persisted ``traces.jsonl`` and ``images/*.png`` from a ``--log-dir``."""
    s1_traces: list[S1Trace] = []
    s2_traces: list[S2Trace] = []
    images: dict[str, bytes] = {}
    jsonl_path = log_dir / "traces.jsonl"
    if jsonl_path.is_file():
        for raw_line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            kind = payload.get("kind")
            if kind == "s1":
                trace_s1 = S1Trace.model_validate(payload)
                s1_traces.append(trace_s1)
                if trace_s1.image_id:
                    img_path = log_dir / "images" / f"{trace_s1.image_id}.png"
                    if img_path.is_file():
                        images[trace_s1.image_id] = img_path.read_bytes()
            elif kind == "s2":
                trace_s2 = S2Trace.model_validate(payload)
                s2_traces.append(trace_s2)
                if trace_s2.image_id:
                    img_path = log_dir / "images" / f"{trace_s2.image_id}.png"
                    if img_path.is_file():
                        images[trace_s2.image_id] = img_path.read_bytes()
    return s1_traces, s2_traces, images


def _make_readonly_text() -> QPlainTextEdit:
    """Create a monospaced read-only text viewer."""
    widget = QPlainTextEdit()
    widget.setReadOnly(True)
    font = QFont("Monospace")
    font.setStyleHint(QFont.StyleHint.TypeWriter)
    font.setPointSize(9)
    widget.setFont(font)
    return widget


class ImageView(QLabel):
    """Aspect-preserving minimap image viewer."""

    def __init__(self, placeholder: str = "No image yet") -> None:
        super().__init__(placeholder)
        self._placeholder = placeholder
        self._source: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(220, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("background-color: #11161d; color: #8b949e;")

    def set_pixmap_source(
        self, pixmap: QPixmap | None, missing_text: str | None = None
    ) -> None:
        """Set the full-resolution source pixmap and rescale to fit."""
        self._source = pixmap
        if pixmap is None or pixmap.isNull():
            self.setPixmap(QPixmap())
            self.setText(missing_text or self._placeholder)
            return
        self._refresh_scaled()

    def _refresh_scaled(self) -> None:
        if self._source is None or self._source.isNull():
            return
        target = self.size()
        if target.width() <= 0 or target.height() <= 0:
            return
        scaled = self._source.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_scaled()


class TracePoller(QThread):
    """Background HTTP poller for ``/debug/traces`` and ``/debug/images/{id}.png``."""

    snapshot_received = Signal(object)
    connection_changed = Signal(bool, str)

    def __init__(self, base_url: str, poll_ms: int = 500) -> None:
        super().__init__()
        self._base_url = base_url.rstrip("/")
        self._poll_secs = max(0.1, poll_ms / 1000.0)
        self._stop_requested = False
        self._last_trace_id = 0
        self._seen_images: set[str] = set()

    def request_stop(self) -> None:
        """Signal the polling loop to exit cleanly."""
        self._stop_requested = True

    def run(self) -> None:
        with httpx.Client(base_url=self._base_url, timeout=1.5) as client:
            while not self._stop_requested:
                try:
                    resp = client.get(
                        "/debug/traces",
                        params={"since_id": self._last_trace_id, "limit": 100},
                    )
                    resp.raise_for_status()
                    snap = TracesSnapshot.model_validate(resp.json())
                    # Server restarted with a lower sequence id -> reset cursor.
                    if snap.latest_trace_id < self._last_trace_id:
                        self._last_trace_id = 0
                        self._seen_images.clear()
                        resp = client.get(
                            "/debug/traces",
                            params={"since_id": 0, "limit": 100},
                        )
                        resp.raise_for_status()
                        snap = TracesSnapshot.model_validate(resp.json())

                    new_images: dict[str, bytes] = {}
                    image_ids = [
                        t.image_id
                        for t in (*snap.s1_traces, *snap.s2_traces)
                        if t.image_id and t.image_id not in self._seen_images
                    ]
                    for img_id in image_ids:
                        if self._stop_requested:
                            break
                        img_resp = client.get(f"/debug/images/{img_id}.png")
                        if img_resp.status_code == 200:
                            new_images[img_id] = img_resp.content
                            self._seen_images.add(img_id)

                    if snap.latest_trace_id > self._last_trace_id:
                        self._last_trace_id = snap.latest_trace_id

                    self.connection_changed.emit(True, f"Connected ({self._base_url})")
                    self.snapshot_received.emit((snap, new_images))
                except Exception as exc:  # noqa: BLE001
                    self.connection_changed.emit(False, f"Disconnected: {exc}")

                deadline = time.monotonic() + self._poll_secs
                while not self._stop_requested and time.monotonic() < deadline:
                    self.msleep(50)


class System1Pane(QWidget):
    """Left pane displaying System 1 (fast decision) history, image, prompt, and output."""

    def __init__(self) -> None:
        super().__init__()
        self._traces: dict[int, S1Trace] = {}
        self._pixmaps: OrderedDict[str, QPixmap] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        header = QLabel("System 1 — Fast Decision")
        header.setStyleSheet("font-size: 14px; font-weight: bold; color: #58a6ff;")
        layout.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter, stretch=1)

        self._history = QListWidget()
        self._history.setMinimumHeight(110)
        self._history.currentItemChanged.connect(self._on_selection_changed)
        splitter.addWidget(self._history)

        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)
        detail_layout.setContentsMargins(0, 0, 0, 0)

        self._image_caption = QLabel("Sent image: (none)")
        self._image_caption.setStyleSheet("color: #8b949e; font-size: 11px;")
        detail_layout.addWidget(self._image_caption)

        self._image_view = ImageView("No System 1 image yet")
        detail_layout.addWidget(self._image_view, stretch=2)

        self._tabs = QTabWidget()
        self._output_view = _make_readonly_text()
        self._prompt_view = _make_readonly_text()
        self._error_view = _make_readonly_text()
        self._tabs.addTab(self._output_view, "Decision Output")
        self._tabs.addTab(self._prompt_view, "Prompt Sent")
        self._tabs.addTab(self._error_view, "Error")
        detail_layout.addWidget(self._tabs, stretch=3)

        splitter.addWidget(detail_widget)
        splitter.setSizes([160, 560])

    def bind_pixmap_cache(self, cache: OrderedDict[str, QPixmap]) -> None:
        self._pixmaps = cache

    def add_traces(self, traces: Sequence[S1Trace], auto_select: bool) -> None:
        for trace in traces:
            if trace.id in self._traces:
                continue
            self._traces[trace.id] = trace
            item = QListWidgetItem(format_s1_list_item(trace))
            item.setData(Qt.ItemDataRole.UserRole, trace.id)
            self._history.insertItem(0, item)
            while self._history.count() > MAX_HISTORY_ITEMS:
                old = self._history.takeItem(self._history.count() - 1)
                if old is not None:
                    old_id = old.data(Qt.ItemDataRole.UserRole)
                    self._traces.pop(old_id, None)
        if auto_select and self._history.count() > 0:
            self._history.setCurrentRow(0)

    def clear_all(self) -> None:
        self._traces.clear()
        self._history.clear()
        self._image_caption.setText("Sent image: (none)")
        self._image_view.set_pixmap_source(None)
        self._output_view.clear()
        self._prompt_view.clear()
        self._error_view.clear()

    def _on_selection_changed(
        self, current: QListWidgetItem | None, _previous: QListWidgetItem | None
    ) -> None:
        if current is None:
            return
        trace_id = current.data(Qt.ItemDataRole.UserRole)
        trace = self._traces.get(trace_id)
        if trace is None:
            return
        self._render_trace(trace)

    def _render_trace(self, trace: S1Trace) -> None:
        if trace.image_id and self._pixmaps is not None:
            pix = self._pixmaps.get(trace.image_id)
            self._image_caption.setText(
                f"Sent minimap image: {trace.image_id} (frame {trace.frame})"
            )
            self._image_view.set_pixmap_source(
                pix, missing_text=f"Image {trace.image_id} unavailable"
            )
        else:
            self._image_caption.setText("Sent image: (none)")
            self._image_view.set_pixmap_source(None)

        if trace.output is not None:
            self._output_view.setPlainText(
                json.dumps(trace.output.model_dump(), indent=2)
            )
        else:
            self._output_view.setPlainText("(no valid S1Output — see Error tab)")

        self._prompt_view.setPlainText(
            f"=== SYSTEM PROMPT ===\n{trace.system_prompt}\n\n"
            f"=== USER PROMPT (with minimap image) ===\n{trace.user_prompt}"
        )
        self._error_view.setPlainText(trace.error or "(no error)")


class System2Pane(QWidget):
    """Right pane displaying System 2 (longer thinking) history, image, tools, prompts, and directive."""

    def __init__(self) -> None:
        super().__init__()
        self._traces: dict[int, S2Trace] = {}
        self._pixmaps: OrderedDict[str, QPixmap] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        header = QLabel("System 2 — Longer Thinking")
        header.setStyleSheet("font-size: 14px; font-weight: bold; color: #d2a8ff;")
        layout.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter, stretch=1)

        self._history = QListWidget()
        self._history.setMinimumHeight(110)
        self._history.currentItemChanged.connect(self._on_selection_changed)
        splitter.addWidget(self._history)

        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)
        detail_layout.setContentsMargins(0, 0, 0, 0)

        self._image_caption = QLabel("Sent image: (none)")
        self._image_caption.setStyleSheet("color: #8b949e; font-size: 11px;")
        detail_layout.addWidget(self._image_caption)

        self._image_view = ImageView("No System 2 image yet")
        detail_layout.addWidget(self._image_view, stretch=2)

        self._tabs = QTabWidget()
        self._directive_view = _make_readonly_text()
        self._tools_view = _make_readonly_text()
        self._step1_prompt_view = _make_readonly_text()
        self._step2_prompt_view = _make_readonly_text()
        self._error_view = _make_readonly_text()
        self._tabs.addTab(self._directive_view, "Directive")
        self._tabs.addTab(self._tools_view, "Tools & Thinking")
        self._tabs.addTab(self._step1_prompt_view, "Step 1 Prompt")
        self._tabs.addTab(self._step2_prompt_view, "Step 2 Prompt")
        self._tabs.addTab(self._error_view, "Error")
        detail_layout.addWidget(self._tabs, stretch=3)

        splitter.addWidget(detail_widget)
        splitter.setSizes([160, 560])

    def bind_pixmap_cache(self, cache: OrderedDict[str, QPixmap]) -> None:
        self._pixmaps = cache

    def add_traces(self, traces: Sequence[S2Trace], auto_select: bool) -> None:
        for trace in traces:
            if trace.id in self._traces:
                continue
            self._traces[trace.id] = trace
            item = QListWidgetItem(format_s2_list_item(trace))
            item.setData(Qt.ItemDataRole.UserRole, trace.id)
            self._history.insertItem(0, item)
            while self._history.count() > MAX_HISTORY_ITEMS:
                old = self._history.takeItem(self._history.count() - 1)
                if old is not None:
                    old_id = old.data(Qt.ItemDataRole.UserRole)
                    self._traces.pop(old_id, None)
        if auto_select and self._history.count() > 0:
            self._history.setCurrentRow(0)

    def clear_all(self) -> None:
        self._traces.clear()
        self._history.clear()
        self._image_caption.setText("Sent image: (none)")
        self._image_view.set_pixmap_source(None)
        self._directive_view.clear()
        self._tools_view.clear()
        self._step1_prompt_view.clear()
        self._step2_prompt_view.clear()
        self._error_view.clear()

    def _on_selection_changed(
        self, current: QListWidgetItem | None, _previous: QListWidgetItem | None
    ) -> None:
        if current is None:
            return
        trace_id = current.data(Qt.ItemDataRole.UserRole)
        trace = self._traces.get(trace_id)
        if trace is None:
            return
        self._render_trace(trace)

    def _render_trace(self, trace: S2Trace) -> None:
        if trace.image_id and self._pixmaps is not None:
            pix = self._pixmaps.get(trace.image_id)
            self._image_caption.setText(
                f"Sent minimap image (Step 1 & Step 2): {trace.image_id} (frame {trace.frame})"
            )
            self._image_view.set_pixmap_source(
                pix, missing_text=f"Image {trace.image_id} unavailable"
            )
        else:
            self._image_caption.setText("Sent image: (none)")
            self._image_view.set_pixmap_source(None)

        if trace.directive is not None:
            self._directive_view.setPlainText(
                json.dumps(trace.directive.model_dump(), indent=2)
            )
        else:
            self._directive_view.setPlainText("(no valid S2Directive — see Error tab)")

        sections: list[str] = [f"Trigger: {trace.reason}"]
        if trace.step1_reasoning:
            sections.append(f"\n=== REASONING ===\n{trace.step1_reasoning}")
        if trace.tool_calls:
            tool_lines = []
            for idx, tc in enumerate(trace.tool_calls, start=1):
                args_str = json.dumps(tc.arguments, indent=2)
                tool_lines.append(
                    f"[{idx}] {tc.name}\nArgs:\n{args_str}\nResult:\n{tc.result}"
                )
            sections.append("\n=== TOOL CALLS ===\n" + "\n\n".join(tool_lines))
        else:
            sections.append("\n=== TOOL CALLS ===\n(none)")
        if trace.step1_summary:
            sections.append(f"\n=== STEP 1 SUMMARY ===\n{trace.step1_summary}")
        self._tools_view.setPlainText("\n".join(sections))

        self._step1_prompt_view.setPlainText(
            f"=== STEP 1 SYSTEM PROMPT ===\n{trace.step1_system_prompt}\n\n"
            f"=== STEP 1 USER PROMPT (with minimap image) ===\n{trace.step1_user_prompt}"
        )
        self._step2_prompt_view.setPlainText(
            f"=== STEP 2 SYSTEM PROMPT ===\n{trace.step2_system_prompt or '(not reached)'}\n\n"
            f"=== STEP 2 USER PROMPT (with minimap image) ===\n"
            f"{trace.step2_user_prompt or '(not reached)'}"
        )
        self._error_view.setPlainText(trace.error or "(no error)")


class DashboardWindow(QMainWindow):
    """Main PySide6 window showing System 1 and System 2 side-by-side."""

    def __init__(
        self,
        server_url: str = "http://localhost:8080",
        poll_ms: int = 500,
        log_dir: Path | None = None,
        start_poller: bool = True,
    ) -> None:
        super().__init__()
        self.setWindowTitle("StarCraft DiffusionGemma — System 1 & System 2 Dashboard")
        self.resize(1440, 900)
        self._live_follow = True
        self._pixmaps: OrderedDict[str, QPixmap] = OrderedDict()
        self._poller: TracePoller | None = None

        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # Top status bar.
        top_bar = QHBoxLayout()
        self._status_badge = QLabel("Connecting...")
        self._status_badge.setStyleSheet(
            "padding: 3px 8px; border-radius: 4px; background-color: #30363d; color: #f0f6fc;"
        )
        top_bar.addWidget(self._status_badge)

        self._bundle_label = QLabel("Waiting for game state...")
        self._bundle_label.setStyleSheet("color: #c9d1d9; padding-left: 8px;")
        top_bar.addWidget(self._bundle_label, stretch=1)

        self._live_btn = QPushButton("Live: ON")
        self._live_btn.setCheckable(True)
        self._live_btn.setChecked(True)
        self._live_btn.toggled.connect(self._on_live_toggled)
        top_bar.addWidget(self._live_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._on_clear)
        top_bar.addWidget(clear_btn)
        main_layout.addLayout(top_bar)

        # Main side-by-side panes: System 1 (left) | System 2 (right).
        self.s1_pane = System1Pane()
        self.s2_pane = System2Pane()
        self.s1_pane.bind_pixmap_cache(self._pixmaps)
        self.s2_pane.bind_pixmap_cache(self._pixmaps)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.s1_pane)
        splitter.addWidget(self.s2_pane)
        splitter.setSizes([720, 720])
        main_layout.addWidget(splitter, stretch=1)

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0d1117;
                color: #e6edf3;
            }
            QListWidget, QPlainTextEdit {
                background-color: #161b22;
                color: #e6edf3;
                border: 1px solid #30363d;
            }
            QListWidget::item:selected {
                background-color: #1f6feb;
                color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid #30363d;
            }
            QTabBar::tab {
                background-color: #161b22;
                color: #8b949e;
                padding: 5px 10px;
                border: 1px solid #30363d;
            }
            QTabBar::tab:selected {
                background-color: #21262d;
                color: #f0f6fc;
            }
            QPushButton {
                background-color: #21262d;
                color: #f0f6fc;
                border: 1px solid #30363d;
                padding: 4px 10px;
                border-radius: 4px;
            }
            QPushButton:checked {
                background-color: #238636;
                border-color: #2ea043;
            }
            """)

        if log_dir is not None:
            self.load_from_log_dir(log_dir)
        elif start_poller:
            self._poller = TracePoller(server_url, poll_ms=poll_ms)
            self._poller.snapshot_received.connect(self._on_snapshot)
            self._poller.connection_changed.connect(self._on_connection_changed)
            self._poller.start()

    def _store_pixmap(self, image_id: str, png_bytes: bytes) -> None:
        pix = QPixmap()
        if pix.loadFromData(png_bytes, "PNG"):
            self._pixmaps[image_id] = pix
            self._pixmaps.move_to_end(image_id)
            while len(self._pixmaps) > MAX_PIXMAP_CACHE:
                self._pixmaps.popitem(last=False)

    def apply_snapshot(
        self, snapshot: TracesSnapshot, new_images: dict[str, bytes] | None = None
    ) -> None:
        """Apply an incoming ``TracesSnapshot`` and any newly fetched PNGs."""
        if new_images:
            for img_id, png_bytes in new_images.items():
                self._store_pixmap(img_id, png_bytes)
        self._bundle_label.setText(
            format_bundle_summary(snapshot.command_bundle, snapshot.frame)
        )
        if snapshot.s1_traces:
            self.s1_pane.add_traces(snapshot.s1_traces, auto_select=self._live_follow)
        if snapshot.s2_traces:
            self.s2_pane.add_traces(snapshot.s2_traces, auto_select=self._live_follow)

    def load_from_log_dir(self, log_dir: Path) -> None:
        """Load a recorded ``--log-dir`` for offline replay."""
        s1_list, s2_list, images = load_log_dir(log_dir)
        for img_id, png_bytes in images.items():
            self._store_pixmap(img_id, png_bytes)
        self.s1_pane.add_traces(s1_list, auto_select=True)
        self.s2_pane.add_traces(s2_list, auto_select=True)
        self._status_badge.setText(f"Offline log: {log_dir.name}")
        self._status_badge.setStyleSheet(
            "padding: 3px 8px; border-radius: 4px; background-color: #1f6feb; color: #ffffff;"
        )
        self._bundle_label.setText(
            f"Loaded {len(s1_list)} System 1 traces and {len(s2_list)} System 2 traces from {log_dir}"
        )

    def _on_snapshot(self, payload: object) -> None:
        snap, new_images = payload  # type: ignore[misc]
        self.apply_snapshot(snap, new_images)

    def _on_connection_changed(self, connected: bool, text: str) -> None:
        self._status_badge.setText(text)
        color = "#238636" if connected else "#da3633"
        self._status_badge.setStyleSheet(
            f"padding: 3px 8px; border-radius: 4px; background-color: {color}; color: #ffffff;"
        )

    def _on_live_toggled(self, checked: bool) -> None:
        self._live_follow = checked
        self._live_btn.setText("Live: ON" if checked else "Live: PAUSED")

    def _on_clear(self) -> None:
        self._pixmaps.clear()
        self.s1_pane.clear_all()
        self.s2_pane.clear_all()

    def closeEvent(self, event: object) -> None:  # noqa: N802
        if self._poller is not None:
            self._poller.request_stop()
            self._poller.wait(2000)
        super().closeEvent(event)  # type: ignore[arg-type]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for launching the PySide6 dashboard."""
    parser = argparse.ArgumentParser(
        description="PySide6 dashboard for System 1 (Decision) & System 2 (Thinking) traces."
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help="Host policy server base URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--poll-ms",
        type=int,
        default=500,
        help="Polling interval in milliseconds (default: 500)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Optional recorded --log-dir to open offline instead of polling a live server",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the PySide6 dashboard application."""
    opts = parse_args(argv)
    app = QApplication(sys.argv)
    window = DashboardWindow(
        server_url=opts.url,
        poll_ms=opts.poll_ms,
        log_dir=opts.log_dir,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
