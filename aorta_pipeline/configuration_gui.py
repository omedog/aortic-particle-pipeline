"""Interactive filter, projected-geometry, line, and cohort configuration."""
from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from copy import deepcopy
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import PolygonSelector, SpanSelector

from .preview import FilterPreview, compute_filter_preview
from .settings import load_settings

PROJECTIONS = {
    "xy_polygon": (0, 1, "XY", "x (m)", "y (m)"),
    "xz_polygon": (0, 2, "XZ", "x (m)", "z (m)"),
    "yz_polygon": (1, 2, "YZ", "y (m)", "z (m)"),
}
OUTPUTS = ["full_paths", "chamber_to_exit", "line_to_line", "chamber_stasis", "line_conditions"]


def _read_projected_geometry(path: Path) -> dict[str, list[list[float]] | None]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {name: deepcopy(payload.get(name)) for name in PROJECTIONS}


class ProjectionPanel:
    def __init__(self, parent, positions: np.ndarray, polygons: dict, *, instruction: str):
        self.frame = ttk.Frame(parent, padding=8)
        self.positions = positions
        self.polygons = {name: deepcopy(polygons.get(name)) for name in PROJECTIONS}
        ttk.Label(self.frame, text=instruction, wraplength=1050).pack(anchor="w", pady=(0, 6))
        figure = Figure(figsize=(10.8, 5.4), dpi=100)
        self.figure = figure
        self.axes, self.artists, self.selectors = {}, {}, []
        for index, (name, (horizontal, vertical, title, xlabel, ylabel)) in enumerate(PROJECTIONS.items(), 1):
            axis = figure.add_subplot(1, 3, index)
            axis.scatter(positions[:, horizontal], positions[:, vertical], s=2, alpha=0.12,
                         color="#375a7f", rasterized=True)
            axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(True, color="#e6e9ed", linewidth=0.5)
            self.axes[name] = axis
            selector = PolygonSelector(
                axis, lambda vertices, key=name: self._polygon_selected(key, vertices),
                useblit=True, props={"color": "#e67e22", "linewidth": 2, "alpha": 0.9},
                handle_props={"markerfacecolor": "#e67e22"},
            )
            self.selectors.append(selector)
            self._render_polygon(name)
        figure.tight_layout()
        self.canvas = FigureCanvasTkAgg(figure, master=self.frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        buttons = ttk.Frame(self.frame)
        buttons.pack(fill="x", pady=(6, 0))
        for name, (_, _, title, _, _) in PROJECTIONS.items():
            ttk.Button(buttons, text=f"Clear {title}", command=lambda key=name: self.clear(key)).pack(
                side="left", padx=(0, 6)
            )

    def _polygon_selected(self, name: str, vertices) -> None:
        if len(vertices) >= 3:
            self.polygons[name] = [[float(x), float(y)] for x, y in vertices]
            self._render_polygon(name)
            self.canvas.draw_idle()

    def _render_polygon(self, name: str) -> None:
        old = self.artists.pop(name, None)
        if old is not None:
            old.remove()
        vertices = self.polygons.get(name)
        if vertices:
            closed = np.vstack([np.asarray(vertices, float), np.asarray(vertices[0], float)])
            artist, = self.axes[name].plot(closed[:, 0], closed[:, 1], color="#d35400", linewidth=2.2)
            self.artists[name] = artist

    def clear(self, name: str) -> None:
        self.polygons[name] = None
        self._render_polygon(name)
        self.canvas.draw_idle()

    def get(self) -> dict[str, list[list[float]] | None]:
        return deepcopy(self.polygons)


class ChamberDialog:
    def __init__(self, parent, positions: np.ndarray, polygons: dict):
        self.result = None
        window = tk.Toplevel(parent)
        self.window = window
        window.title("Initial chamber polygons")
        window.geometry("1120x680")
        window.transient(parent)
        panel = ProjectionPanel(
            window, positions, polygons,
            instruction="Draw the initial-chamber inclusion polygon independently in any projection. "
                        "A particle belongs to the chamber only when it is inside every enabled projection.",
        )
        panel.frame.pack(fill="both", expand=True)
        actions = ttk.Frame(window, padding=8)
        actions.pack(fill="x")
        ttk.Button(actions, text="Cancel", command=window.destroy).pack(side="right")
        ttk.Button(actions, text="Use chamber polygons", command=lambda: self._save(panel)).pack(
            side="right", padx=(0, 8)
        )
        window.grab_set()
        parent.wait_window(window)

    def _save(self, panel):
        polygons = panel.get()
        if not any(polygons.values()):
            messagebox.showerror("Initial chamber", "Draw at least one chamber polygon.", parent=self.window)
            return
        self.result = polygons
        self.window.destroy()


class RuleDialog:
    def __init__(self, parent, rule: dict | None, available_lines: list[str], default_index: int):
        self.result = None
        self.window = tk.Toplevel(parent)
        self.window.title("Line-cohort rule")
        self.window.transient(parent)
        self.window.resizable(False, False)
        body = ttk.Frame(self.window, padding=14)
        body.pack(fill="both", expand=True)
        source = rule or {}
        self.id_var = tk.StringVar(value=str(source.get("id", f"rule_{default_index}")))
        self.name_var = tk.StringVar(value=str(source.get("name", f"line_cohort_{default_index}")))
        self.all_var = tk.StringVar(value=", ".join(source.get("all_of", [])))
        self.any_var = tk.StringVar(value=", ".join(source.get("any_of", [])))
        self.exclude_var = tk.BooleanVar(value=bool(source.get("exclude_crossing", True)))
        rows = [
            ("Rule ID", self.id_var), ("Output name", self.name_var),
            ("All required predicates", self.all_var), ("At least one predicate", self.any_var),
        ]
        for row, (label, variable) in enumerate(rows):
            ttk.Label(body, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=5)
            ttk.Entry(body, textvariable=variable, width=62).grid(row=row, column=1, sticky="ew", pady=5)
        allowed = [f"cross:{name}" for name in available_lines] + ["no_exit", "first_frame"]
        ttk.Label(body, text="Available predicates: " + (", ".join(allowed) or "no_exit, first_frame"),
                  wraplength=600).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 3))
        ttk.Label(body, text="Leave a predicate group blank when it should impose no constraint.",
                  foreground="#59636e").grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(body, text="Reject paths crossing any exclusion line",
                        variable=self.exclude_var).grid(row=6, column=0, columnspan=2, sticky="w", pady=8)
        actions = ttk.Frame(body)
        actions.grid(row=7, column=0, columnspan=2, sticky="e")
        ttk.Button(actions, text="Cancel", command=self.window.destroy).pack(side="right")
        ttk.Button(actions, text="Save rule", command=lambda: self._save(set(allowed))).pack(
            side="right", padx=(0, 8)
        )
        self.window.grab_set()
        parent.wait_window(self.window)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [token.strip() for token in text.split(",") if token.strip()]

    def _save(self, allowed: set[str]):
        rule_id = self.id_var.get().strip()
        name = self.name_var.get().strip()
        all_of, any_of = self._tokens(self.all_var.get()), self._tokens(self.any_var.get())
        unknown = sorted(set(all_of + any_of).difference(allowed))
        if not rule_id or not name:
            messagebox.showerror("Rule", "Rule ID and output name are required.", parent=self.window)
            return
        if unknown:
            messagebox.showerror("Rule", f"Unknown predicates: {', '.join(unknown)}", parent=self.window)
            return
        self.result = {"id": rule_id, "name": name, "all_of": all_of, "any_of": any_of,
                       "exclude_crossing": self.exclude_var.get()}
        self.window.destroy()


class ConfigurationDialog:
    def __init__(self, parent, raw_input: Path, run_dir: Path, config_path: Path | None, on_saved):
        self.parent, self.raw_input, self.run_dir, self.on_saved = parent, raw_input, run_dir, on_saved
        self.window = tk.Toplevel(parent)
        self.window.title("Configure filtration, geometry, and cohorts")
        self.window.geometry("1220x840")
        self.window.minsize(1000, 700)
        self.window.transient(parent)
        self.settings = load_settings(config_path)
        self.roi_geometry = _read_projected_geometry(Path(self.settings["geometry"]["roi"]))
        self.chamber_geometry = _read_projected_geometry(Path(self.settings["geometry"]["initial_chamber"]))
        self.crossing_lines = self._load_line_payload(Path(self.settings["geometry"]["crossing_lines"]))
        self.exclusion_lines = self._load_line_payload(Path(self.settings["geometry"]["exclusion_lines"]))
        self.rules = deepcopy(self.settings["cohorts"].get("line_rules", []))
        self.preview_queue: queue.Queue = queue.Queue()
        self.preview: FilterPreview | None = None
        self.first_line_point = None
        self.line_draw_kind = None
        self.line_marker = None

        loading = ttk.Frame(self.window, padding=30)
        loading.pack(fill="both", expand=True)
        self.loading_frame = loading
        ttk.Label(loading, text="Reading trajectories and computing distributions…",
                  font=("Segoe UI", 13, "bold")).pack(pady=(160, 12))
        progress = ttk.Progressbar(loading, mode="indeterminate", length=360)
        progress.pack()
        progress.start(10)
        ttk.Label(loading, text="This is read-only. Large experimental datasets may take a little time.",
                  foreground="#59636e").pack(pady=10)
        threading.Thread(target=self._load_preview, daemon=True).start()
        self.window.after(100, self._poll_preview)

    @staticmethod
    def _load_line_payload(path: Path) -> list[dict]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return deepcopy(payload.get("lines", []))

    def _load_preview(self):
        try:
            self.preview_queue.put(("ok", compute_filter_preview(self.raw_input)))
        except Exception as exc:
            self.preview_queue.put(("error", str(exc)))

    def _poll_preview(self):
        try:
            kind, payload = self.preview_queue.get_nowait()
        except queue.Empty:
            if self.window.winfo_exists():
                self.window.after(100, self._poll_preview)
            return
        if kind == "error":
            messagebox.showerror("Cannot analyze trajectories", payload, parent=self.window)
            self.window.destroy()
            return
        self.preview = payload
        self.loading_frame.destroy()
        self._build_editor()

    def _build_editor(self):
        header = ttk.Frame(self.window, padding=(14, 10))
        header.pack(fill="x")
        ttk.Label(header, text=f"{self.preview.track_count:,} trajectories · {self.preview.point_count:,} points",
                  font=("Segoe UI", 11, "bold")).pack(side="left")
        ttk.Label(header, text=str(self.raw_input), foreground="#59636e").pack(side="left", padx=14)
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill="both", expand=True, padx=12)
        threshold_tab, roi_tab, cohort_tab = ttk.Frame(notebook), ttk.Frame(notebook), ttk.Frame(notebook)
        notebook.add(threshold_tab, text="1  Threshold filters")
        notebook.add(roi_tab, text="2  ROI polygons")
        notebook.add(cohort_tab, text="3  Cohorts and lines")
        self._build_thresholds(threshold_tab)
        self.roi_panel = ProjectionPanel(
            roi_tab, self.preview.sample_positions, self.roi_geometry,
            instruction="Draw an inclusion polygon on each desired projection. A trajectory survives only when "
                        "every point is inside every polygon that is defined. An empty plane imposes no constraint.",
        )
        self.roi_panel.frame.pack(fill="both", expand=True)
        self._build_cohorts(cohort_tab)
        footer = ttk.Frame(self.window, padding=12)
        footer.pack(fill="x")
        ttk.Button(footer, text="Cancel", command=self.window.destroy).pack(side="right")
        ttk.Button(footer, text="Save setup and use in main window", command=self._save,
                   style="Run.TButton").pack(side="right", padx=(0, 8))

    def _build_thresholds(self, parent):
        settings = self.settings["filtration"]
        self.min_length_var = tk.StringVar(value=str(settings["min_length"]))
        self.max_length_var = tk.StringVar(value=str(settings["max_length"]))
        self.min_speed_var = tk.StringVar(value=str(settings.get("minimum_speed_um_per_frame", 0.0)))
        self.max_speed_var = tk.StringVar(value=str(settings.get(
            "maximum_speed_um_per_frame", settings.get("speed_cutoff_um_per_frame", 1300.0)
        )))
        form = ttk.Frame(parent, padding=10)
        form.pack(fill="x")
        fields = [
            ("Minimum length (points)", self.min_length_var), ("Maximum length (points)", self.max_length_var),
            ("Minimum speed (µm/frame)", self.min_speed_var), ("Maximum speed (µm/frame)", self.max_speed_var),
        ]
        for index, (label, variable) in enumerate(fields):
            ttk.Label(form, text=label).grid(row=0, column=index * 2, sticky="e", padx=(10, 5))
            entry = ttk.Entry(form, textvariable=variable, width=11)
            entry.grid(row=0, column=index * 2 + 1, sticky="w")
            entry.bind("<FocusOut>", lambda _event: self._sync_spans())
        ttk.Label(form, text="Drag horizontally on either histogram to set its accepted interval. "
                             "Counts use a logarithmic scale so sparse tails remain visible.",
                  foreground="#59636e").grid(row=1, column=0, columnspan=8, sticky="w", pady=(8, 0))
        figure = Figure(figsize=(10.5, 5.3), dpi=100)
        length_axis, speed_axis = figure.subplots(1, 2)
        self._plot_histogram(length_axis, self.preview.length, "Trajectory-length distribution", "length (points)")
        self._plot_histogram(speed_axis, self.preview.speed_um_per_frame, "Finite-difference speed distribution",
                             "speed (µm/frame)")
        length_axis.set_xlim(
            min(self.preview.length.edges[0], float(self.min_length_var.get())),
            max(self.preview.length.edges[-1], float(self.max_length_var.get())),
        )
        speed_axis.set_xlim(
            min(self.preview.speed_um_per_frame.edges[0], float(self.min_speed_var.get())),
            max(self.preview.speed_um_per_frame.edges[-1], float(self.max_speed_var.get())),
        )
        self.length_span = SpanSelector(length_axis, self._length_selected, "horizontal", interactive=True,
                                        props={"facecolor": "#3498db", "alpha": 0.25})
        self.speed_span = SpanSelector(speed_axis, self._speed_selected, "horizontal", interactive=True,
                                       props={"facecolor": "#e67e22", "alpha": 0.25})
        self.length_span.extents = (float(self.min_length_var.get()), float(self.max_length_var.get()))
        self.speed_span.extents = (float(self.min_speed_var.get()), float(self.max_speed_var.get()))
        figure.tight_layout()
        self.threshold_canvas = FigureCanvasTkAgg(figure, master=parent)
        self.threshold_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

    @staticmethod
    def _plot_histogram(axis, histogram, title, xlabel):
        axis.stairs(histogram.counts, histogram.edges, fill=True, alpha=0.55, color="#376f9e")
        axis.axvline(histogram.median, color="#273746", linestyle="--", linewidth=1.2, label="median")
        axis.set(title=title, xlabel=xlabel, ylabel="count (log scale)")
        axis.set_yscale("symlog", linthresh=1)
        axis.grid(True, color="#e6e9ed", linewidth=0.5)
        axis.legend(frameon=False)

    def _length_selected(self, lower, upper):
        self.min_length_var.set(str(max(2, round(lower))))
        self.max_length_var.set(str(max(2, round(upper))))

    def _speed_selected(self, lower, upper):
        self.min_speed_var.set(f"{max(0.0, lower):.3f}")
        self.max_speed_var.set(f"{max(0.0, upper):.3f}")

    def _sync_spans(self):
        try:
            self.length_span.extents = (float(self.min_length_var.get()), float(self.max_length_var.get()))
            self.speed_span.extents = (float(self.min_speed_var.get()), float(self.max_speed_var.get()))
            self.threshold_canvas.draw_idle()
        except ValueError:
            pass

    def _build_cohorts(self, parent):
        top = ttk.Frame(parent, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="Outputs", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0, 10))
        enabled = set(self.settings["cohorts"].get("enabled_outputs", OUTPUTS[1:]))
        self.output_vars = {}
        labels = {
            "full_paths": "Full paths (no retrospective filter)", "chamber_to_exit": "Chamber → exit",
            "line_to_line": "Line → line", "chamber_stasis": "Chamber stasis",
            "line_conditions": "Custom line cohorts",
        }
        for name in OUTPUTS:
            variable = tk.BooleanVar(value=name in enabled)
            self.output_vars[name] = variable
            ttk.Checkbutton(top, text=labels[name], variable=variable).pack(side="left", padx=5)
        ttk.Button(top, text="Full paths only", command=self._full_only).pack(side="right", padx=(6, 0))
        ttk.Button(top, text="Select all", command=lambda: self._set_all_outputs(True)).pack(side="right")

        panes = ttk.Panedwindow(parent, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        line_frame = ttk.LabelFrame(panes, text="Crossing and exclusion lines", padding=8)
        rule_frame = ttk.LabelFrame(panes, text="Custom line-cohort definitions", padding=8)
        panes.add(line_frame, weight=3)
        panes.add(rule_frame, weight=2)
        self._build_line_editor(line_frame)
        self._build_rule_editor(rule_frame)

        chamber_bar = ttk.Frame(parent, padding=(10, 0, 10, 8))
        chamber_bar.pack(fill="x")
        self.chamber_summary_var = tk.StringVar()
        self._update_chamber_summary()
        ttk.Button(chamber_bar, text="Edit initial chamber polygons…", command=self._edit_chamber).pack(side="left")
        ttk.Label(chamber_bar, textvariable=self.chamber_summary_var, foreground="#59636e").pack(
            side="left", padx=10
        )

    def _set_all_outputs(self, value):
        for variable in self.output_vars.values():
            variable.set(value)

    def _full_only(self):
        self._set_all_outputs(False)
        self.output_vars["full_paths"].set(True)

    def _build_line_editor(self, parent):
        self.line_tree = ttk.Treeview(parent, columns=("kind", "id", "start", "end", "side"), show="headings", height=7)
        widths = {"kind": 75, "id": 100, "start": 145, "end": 145, "side": 65}
        for column in self.line_tree["columns"]:
            self.line_tree.heading(column, text=column.replace("_", " ").title())
            self.line_tree.column(column, width=widths[column], anchor="w")
        self.line_tree.pack(fill="x")
        controls = ttk.Frame(parent)
        controls.pack(fill="x", pady=5)
        ttk.Button(controls, text="Draw crossing line", command=lambda: self._begin_line("crossing")).pack(side="left")
        ttk.Button(controls, text="Draw exclusion line", command=lambda: self._begin_line("exclusion")).pack(
            side="left", padx=5
        )
        ttk.Button(controls, text="Delete selected", command=self._delete_line).pack(side="left")
        self.line_status_var = tk.StringVar(value="Choose a draw button, then click two endpoints in the XY plot.")
        ttk.Label(parent, textvariable=self.line_status_var, foreground="#59636e", wraplength=620).pack(anchor="w")
        figure = Figure(figsize=(6.2, 4.0), dpi=100)
        self.line_axis = figure.add_subplot(111)
        positions = self.preview.sample_positions
        self.line_axis.scatter(positions[:, 0], positions[:, 1], s=2, alpha=0.1, color="#657786", rasterized=True)
        self.line_axis.set(xlabel="x (m)", ylabel="y (m)", title="XY line geometry")
        self.line_axis.set_aspect("equal", adjustable="box")
        self.line_axis.grid(True, color="#e6e9ed", linewidth=0.5)
        figure.tight_layout()
        self.line_canvas = FigureCanvasTkAgg(figure, master=parent)
        self.line_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.line_canvas.mpl_connect("button_press_event", self._line_click)
        self._refresh_lines()

    def _begin_line(self, kind):
        self.line_draw_kind, self.first_line_point = kind, None
        if self.line_marker is not None:
            self.line_marker.remove()
            self.line_marker = None
        self.line_status_var.set(f"Drawing {kind} line: click the first endpoint in the XY plot.")

    def _line_click(self, event):
        if self.line_draw_kind is None or event.inaxes is not self.line_axis or event.xdata is None:
            return
        point = [float(event.xdata), float(event.ydata)]
        if self.first_line_point is None:
            self.first_line_point = point
            self.line_marker, = self.line_axis.plot(point[0], point[1], "o", color="#f39c12")
            self.line_canvas.draw_idle()
            self.line_status_var.set("Click the second endpoint.")
            return
        kind, start, end = self.line_draw_kind, self.first_line_point, point
        if np.allclose(start, end):
            messagebox.showerror("Line", "The two line endpoints must be different.", parent=self.window)
            self._cancel_line_draw()
            return
        default_id = f"line_{len(self.crossing_lines) + 1}" if kind == "crossing" else f"excl_{len(self.exclusion_lines) + 1}"
        line_id = simpledialog.askstring("Line identifier", "Unique line identifier:", initialvalue=default_id,
                                         parent=self.window)
        if not line_id:
            self._cancel_line_draw()
            return
        existing = {line.get("id") for line in self.crossing_lines + self.exclusion_lines}
        if line_id in existing:
            messagebox.showerror("Line", f"A line named {line_id!r} already exists.", parent=self.window)
            self._cancel_line_draw()
            return
        row = {"id": line_id, "start": start, "end": end}
        if kind == "crossing":
            side = simpledialog.askstring(
                "Interior side", "Which side points toward the interior? Enter left or right.",
                initialvalue="left", parent=self.window,
            )
            if side is None or side.lower().strip() not in {"left", "right"}:
                messagebox.showerror("Line", "Interior side must be left or right.", parent=self.window)
                self._cancel_line_draw()
                return
            row["interior_side"] = side.lower().strip()
            self.crossing_lines.append(row)
        else:
            self.exclusion_lines.append(row)
        self._cancel_line_draw()
        self._refresh_lines()

    def _cancel_line_draw(self):
        self.line_draw_kind, self.first_line_point = None, None
        if self.line_marker is not None:
            self.line_marker.remove()
            self.line_marker = None
        self.line_status_var.set("Choose a draw button, then click two endpoints in the XY plot.")
        self.line_canvas.draw_idle()

    def _refresh_lines(self):
        self.line_tree.delete(*self.line_tree.get_children())
        for kind, rows in (("crossing", self.crossing_lines), ("exclusion", self.exclusion_lines)):
            for index, line in enumerate(rows):
                fmt = lambda point: f"({point[0]:.5f}, {point[1]:.5f})"
                self.line_tree.insert("", "end", iid=f"{kind}:{index}", values=(
                    kind, line["id"], fmt(line["start"]), fmt(line["end"]), line.get("interior_side", "—")
                ))
        for artist in list(self.line_axis.lines):
            artist.remove()
        for artist in list(self.line_axis.texts):
            artist.remove()
        for line in self.crossing_lines:
            start, end = np.asarray(line["start"]), np.asarray(line["end"])
            self.line_axis.annotate(
                "", xy=end, xytext=start,
                arrowprops={"arrowstyle": "->", "color": "#2471a3", "linewidth": 2},
            )
            self.line_axis.text(end[0], end[1], f"{line['id']} ({line.get('interior_side', 'left')})",
                                fontsize=8, color="#1b4f72")
        for line in self.exclusion_lines:
            start, end = np.asarray(line["start"]), np.asarray(line["end"])
            self.line_axis.plot([start[0], end[0]], [start[1], end[1]], color="#c0392b", linewidth=1.6)
        self.line_canvas.draw_idle()

    def _delete_line(self):
        selection = self.line_tree.selection()
        if not selection:
            return
        kind, raw_index = selection[0].split(":")
        rows = self.crossing_lines if kind == "crossing" else self.exclusion_lines
        rows.pop(int(raw_index))
        self._refresh_lines()

    def _build_rule_editor(self, parent):
        self.rule_tree = ttk.Treeview(parent, columns=("id", "name", "all", "any", "exclude"), show="headings")
        for column, width in (("id", 70), ("name", 160), ("all", 190), ("any", 190), ("exclude", 65)):
            self.rule_tree.heading(column, text=column.title())
            self.rule_tree.column(column, width=width, anchor="w")
        self.rule_tree.pack(fill="both", expand=True)
        controls = ttk.Frame(parent)
        controls.pack(fill="x", pady=(6, 0))
        ttk.Button(controls, text="Add rule", command=lambda: self._edit_rule(None)).pack(side="left")
        ttk.Button(controls, text="Edit selected", command=self._edit_selected_rule).pack(side="left", padx=5)
        ttk.Button(controls, text="Delete selected", command=self._delete_rule).pack(side="left")
        ttk.Label(parent, text="Rules combine every ‘All’ predicate with at least one ‘Any’ predicate. "
                               "Crossing predicates use the line IDs defined on the left.",
                  wraplength=470, foreground="#59636e").pack(anchor="w", pady=(7, 0))
        self._refresh_rules()

    def _refresh_rules(self):
        self.rule_tree.delete(*self.rule_tree.get_children())
        for index, rule in enumerate(self.rules):
            self.rule_tree.insert("", "end", iid=str(index), values=(
                rule.get("id", f"rule_{index + 1}"), rule.get("name", ""),
                ", ".join(rule.get("all_of", [])), ", ".join(rule.get("any_of", [])),
                "yes" if rule.get("exclude_crossing", True) else "no",
            ))

    def _edit_rule(self, index: int | None):
        source = self.rules[index] if index is not None else None
        dialog = RuleDialog(self.window, source, [line["id"] for line in self.crossing_lines], len(self.rules) + 1)
        if dialog.result is None:
            return
        if index is None:
            self.rules.append(dialog.result)
        else:
            self.rules[index] = dialog.result
        self._refresh_rules()

    def _edit_selected_rule(self):
        selection = self.rule_tree.selection()
        if selection:
            self._edit_rule(int(selection[0]))

    def _delete_rule(self):
        selection = self.rule_tree.selection()
        if selection:
            self.rules.pop(int(selection[0]))
            self._refresh_rules()

    def _edit_chamber(self):
        dialog = ChamberDialog(self.window, self.preview.sample_positions, self.chamber_geometry)
        if dialog.result is not None:
            self.chamber_geometry = dialog.result
            self._update_chamber_summary()

    def _update_chamber_summary(self):
        views = [PROJECTIONS[name][2] for name, polygon in self.chamber_geometry.items() if polygon]
        self.chamber_summary_var.set("Enabled chamber projections: " + (", ".join(views) or "none"))

    @staticmethod
    def _geometry_payload(polygons):
        return {"point_rule": "all_points_inside", "combine_rule": "and", **polygons}

    def _save(self):
        try:
            min_length, max_length = int(self.min_length_var.get()), int(self.max_length_var.get())
            min_speed, max_speed = float(self.min_speed_var.get()), float(self.max_speed_var.get())
        except ValueError:
            messagebox.showerror("Thresholds", "Threshold entries must be numeric.", parent=self.window)
            return
        if min_length < 2 or min_length > max_length:
            messagebox.showerror("Thresholds", "Length must satisfy 2 ≤ minimum ≤ maximum.", parent=self.window)
            return
        if min_speed < 0 or min_speed > max_speed:
            messagebox.showerror("Thresholds", "Speed must satisfy 0 ≤ minimum ≤ maximum.", parent=self.window)
            return
        roi = self.roi_panel.get()
        if not any(roi.values()):
            messagebox.showerror("ROI", "Draw at least one ROI inclusion polygon.", parent=self.window)
            return
        enabled = [name for name in OUTPUTS if self.output_vars[name].get()]
        if not enabled:
            messagebox.showerror("Outputs", "Select at least one Lagrangian output.", parent=self.window)
            return
        if set(enabled) != {"full_paths"} and not any(self.chamber_geometry.values()):
            messagebox.showerror("Cohorts", "Retrospective cohorts require an initial chamber polygon.",
                                 parent=self.window)
            return
        if {"chamber_to_exit", "line_to_line"}.intersection(enabled) and not self.crossing_lines:
            messagebox.showerror("Cohorts", "Chamber-to-exit and line-to-line cohorts require a crossing line.",
                                 parent=self.window)
            return
        if "line_conditions" in enabled and not self.rules:
            messagebox.showerror("Cohort rules", "Add at least one custom line-cohort rule.", parent=self.window)
            return
        known_predicates = {"no_exit", "first_frame"} | {f"cross:{line['id']}" for line in self.crossing_lines}
        line_ids = [line["id"] for line in self.crossing_lines + self.exclusion_lines]
        if len(line_ids) != len(set(line_ids)):
            messagebox.showerror("Lines", "Every crossing and exclusion line must have a unique ID.",
                                 parent=self.window)
            return
        rule_ids = [str(rule.get("id", "")).strip() for rule in self.rules]
        rule_names = [str(rule.get("name", "")).strip() for rule in self.rules]
        if len(rule_ids) != len(set(rule_ids)) or len(rule_names) != len(set(rule_names)):
            messagebox.showerror("Cohort rules", "Every rule ID and output name must be unique.",
                                 parent=self.window)
            return
        for rule in self.rules if "line_conditions" in enabled else []:
            unknown = set(rule.get("all_of", []) + rule.get("any_of", [])).difference(known_predicates)
            if unknown:
                messagebox.showerror("Cohort rules", f"Rule {rule.get('id')} uses undefined predicates: "
                                     f"{', '.join(sorted(unknown))}", parent=self.window)
                return

        setup_dir = self.run_dir.resolve() / "setup"
        setup_dir.mkdir(parents=True, exist_ok=True)
        (setup_dir / "roi.json").write_text(json.dumps(self._geometry_payload(roi), indent=2) + "\n", encoding="utf-8")
        (setup_dir / "initial_chamber.json").write_text(
            json.dumps(self._geometry_payload(self.chamber_geometry), indent=2) + "\n", encoding="utf-8"
        )
        (setup_dir / "crossing_lines.json").write_text(
            json.dumps({"lines": self.crossing_lines}, indent=2) + "\n", encoding="utf-8"
        )
        (setup_dir / "exclusion_lines.json").write_text(
            json.dumps({"lines": self.exclusion_lines}, indent=2) + "\n", encoding="utf-8"
        )
        output_settings = deepcopy(self.settings)
        output_settings["filtration"] = {
            "min_length": min_length, "max_length": max_length,
            "minimum_speed_um_per_frame": min_speed, "maximum_speed_um_per_frame": max_speed,
        }
        output_settings["geometry"] = {
            "roi": "roi.json", "initial_chamber": "initial_chamber.json",
            "crossing_lines": "crossing_lines.json", "exclusion_lines": "exclusion_lines.json",
        }
        output_settings["cohorts"]["enabled_outputs"] = enabled
        output_settings["cohorts"]["line_rules"] = self.rules
        config_path = setup_dir / "pipeline_settings.json"
        config_path.write_text(json.dumps(output_settings, indent=2) + "\n", encoding="utf-8")
        self.on_saved(config_path)
        self.window.destroy()
