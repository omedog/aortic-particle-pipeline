"""Tk desktop front end for the self-contained processing pipeline."""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .settings import PACKAGE_ROOT
from .synthetic import generate_synthetic_dataset

STAGE_HELP = {
    "all": "Run filtration, Eulerian extraction, Lagrangian extension/cohorts, and validation.",
    "filter": "Convert raw trajectories, calculate velocity, and apply length, speed, and ROI filters.",
    "eulerian": "Create the sliding-window regular-grid velocity time series from filtered data.",
    "extend": "Extend measured paths with the mesh-free Lagrangian flow map.",
    "cohorts": "Export the selected full-path and/or retrospective cohort time series.",
    "validate": "Test chained reconstruction against artificially hidden measured trajectories.",
}


class PipelineGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Aortic Particle Processing Pipeline")
        self.root.geometry("1040x760")
        self.root.minsize(880, 650)
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.worker_thread: threading.Thread | None = None

        sample_dir = PACKAGE_ROOT / "sample_data"
        sample_input = sample_dir / "trajectories.zarr"
        sample_config = sample_dir / "synthetic_settings.json"
        self.input_var = tk.StringVar(value=str(sample_input) if sample_input.exists() else "")
        self.run_dir_var = tk.StringVar(value=str(PACKAGE_ROOT / "demo_run"))
        self.config_var = tk.StringVar(value=str(sample_config) if sample_config.exists() else "")
        self.stage_var = tk.StringVar(value="all")
        self.workers_var = tk.StringVar(value="1")
        self.track_limit_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")
        self.stage_help_var = tk.StringVar(value=STAGE_HELP["all"])

        self._configure_style()
        self._build()
        self.root.after(100, self._poll_messages)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10), foreground="#495463")
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("Run.TButton", font=("Segoe UI", 10, "bold"), padding=(16, 8))

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Aortic Particle Processing", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="Prepare filtered trajectories, Eulerian fields, Lagrangian path maps, cohorts, and validation outputs.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 14))

        paths = ttk.LabelFrame(outer, text="Data and output", style="Section.TLabelframe", padding=12)
        paths.pack(fill="x")
        paths.columnconfigure(1, weight=1)
        self._path_row(paths, 0, "Raw trajectories", self.input_var, self._browse_input_file,
                       extra=("Zarr…", self._browse_input_directory))
        self._path_row(paths, 1, "Run directory", self.run_dir_var, self._browse_run_dir)
        self._path_row(paths, 2, "Settings override", self.config_var, self._browse_config, optional=True)

        controls = ttk.LabelFrame(outer, text="Processing", style="Section.TLabelframe", padding=12)
        controls.pack(fill="x", pady=(12, 0))
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="Stage").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=5)
        stage_box = ttk.Combobox(controls, textvariable=self.stage_var, values=list(STAGE_HELP),
                                 state="readonly", width=18)
        stage_box.grid(row=0, column=1, sticky="w", pady=5)
        stage_box.bind("<<ComboboxSelected>>", self._stage_changed)
        ttk.Label(controls, textvariable=self.stage_help_var, style="Subtitle.TLabel",
                  wraplength=670).grid(row=1, column=1, columnspan=3, sticky="w", pady=(0, 7))
        ttk.Label(controls, text="Workers").grid(row=0, column=2, sticky="e", padx=(20, 6), pady=5)
        ttk.Spinbox(controls, from_=1, to=64, textvariable=self.workers_var, width=7).grid(
            row=0, column=3, sticky="w", pady=5
        )
        ttk.Label(controls, text="Track limit").grid(row=0, column=4, sticky="e", padx=(20, 6), pady=5)
        ttk.Entry(controls, textvariable=self.track_limit_var, width=10).grid(row=0, column=5, sticky="w", pady=5)
        ttk.Label(controls, text="Leave the track limit blank for a complete run.",
                  style="Subtitle.TLabel").grid(row=1, column=4, columnspan=2, sticky="w", pady=(0, 7))

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=12)
        self.demo_button = ttk.Button(actions, text="Create demo data…", command=self._create_demo)
        self.demo_button.pack(side="left")
        self.configure_button = ttk.Button(actions, text="Configure filters & cohorts…",
                                           command=self._configure_processing)
        self.configure_button.pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Open results", command=self._open_results).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Clear log", command=self._clear_log).pack(side="left", padx=(8, 0))
        self.stop_button = ttk.Button(actions, text="Stop", command=self._stop, state="disabled")
        self.stop_button.pack(side="right")
        self.run_button = ttk.Button(actions, text="Run pipeline", command=self._run, style="Run.TButton")
        self.run_button.pack(side="right", padx=(0, 8))

        log_frame = ttk.LabelFrame(outer, text="Run log", style="Section.TLabelframe", padding=8)
        log_frame.pack(fill="both", expand=True)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(log_frame, wrap="word", height=18, font=("Consolas", 9), relief="flat",
                           background="#111820", foreground="#dbe4ee", insertbackground="white",
                           padx=10, pady=8, state="disabled")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(8, 0))
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=180)
        self.progress.pack(side="right")
        ttk.Label(footer, textvariable=self.status_var).pack(side="left")

    def _path_row(self, parent, row, label, variable, command, *, extra=None, optional=False):
        text = f"{label} (optional)" if optional else label
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=5)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=5)
        ttk.Button(parent, text="Browse…", command=command).grid(row=row, column=2, padx=(8, 0), pady=5)
        if extra:
            ttk.Button(parent, text=extra[0], command=extra[1]).grid(row=row, column=3, padx=(5, 0), pady=5)

    def _browse_input_file(self):
        path = filedialog.askopenfilename(
            title="Select trajectory table", filetypes=[("Trajectory tables", "*.parquet *.csv"), ("All files", "*.*")]
        )
        if path:
            self.input_var.set(path)

    def _browse_input_directory(self):
        path = filedialog.askdirectory(title="Select trajectories.zarr directory")
        if path:
            self.input_var.set(path)

    def _browse_run_dir(self):
        path = filedialog.askdirectory(title="Select or create a run directory", mustexist=False)
        if path:
            self.run_dir_var.set(path)

    def _browse_config(self):
        path = filedialog.askopenfilename(title="Select settings JSON", filetypes=[("JSON", "*.json")])
        if path:
            self.config_var.set(path)

    def _stage_changed(self, _event=None):
        self.stage_help_var.set(STAGE_HELP[self.stage_var.get()])

    def _create_demo(self):
        parent = filedialog.askdirectory(title="Choose where to create the synthetic demo")
        if not parent:
            return
        destination = Path(parent) / "synthetic_aorta_demo"
        try:
            created = generate_synthetic_dataset(destination)
        except Exception as exc:
            messagebox.showerror("Demo creation failed", str(exc))
            return
        self.input_var.set(str(created["zarr"]))
        self.config_var.set(str(created["config"]))
        self.run_dir_var.set(str(destination / "run"))
        self.stage_var.set("all")
        self._stage_changed()
        self._append_log(
            f"Created deterministic demo: {created['tracks']} tracks, {created['rows']} rows\n"
            f"Input: {created['zarr']}\nSettings: {created['config']}\n"
        )
        self.status_var.set("Demo data ready")

    def _configure_processing(self):
        raw_text, run_text = self.input_var.get().strip(), self.run_dir_var.get().strip()
        if not raw_text or not Path(raw_text).exists():
            messagebox.showerror("Configuration", "Select an existing raw trajectory input first.")
            return
        if not run_text:
            messagebox.showerror("Configuration", "Select a run directory first.")
            return
        config_text = self.config_var.get().strip()
        config_path = Path(config_text) if config_text else None
        if config_path is not None and not config_path.is_file():
            messagebox.showerror("Configuration", "The current settings override does not exist.")
            return
        from .configuration_gui import ConfigurationDialog
        ConfigurationDialog(
            self.root, Path(raw_text).resolve(), Path(run_text).resolve(), config_path,
            self._configuration_saved,
        )

    def _configuration_saved(self, path: Path):
        self.config_var.set(str(path))
        self.stage_var.set("all")
        self._stage_changed()
        try:
            enabled = json.loads(path.read_text(encoding="utf-8"))["cohorts"]["enabled_outputs"]
            if "full_paths" in enabled:
                self.track_limit_var.set("")
        except (KeyError, OSError, json.JSONDecodeError):
            pass
        self._append_log(f"Saved interactive setup: {path}\n")
        self.status_var.set("Custom filtration and cohort setup ready")

    def _validate_fields(self) -> tuple[bool, str]:
        stage = self.stage_var.get()
        run_dir = Path(self.run_dir_var.get().strip()) if self.run_dir_var.get().strip() else None
        if run_dir is None:
            return False, "Choose a run directory."
        if stage in {"all", "filter"}:
            raw = Path(self.input_var.get().strip()) if self.input_var.get().strip() else None
            if raw is None or not raw.exists():
                return False, "Choose an existing raw Zarr, Parquet, or CSV input."
        elif not (run_dir / "work" / "filtered" / "trajectories.parquet").exists():
            return False, "This stage requires filtered data in the selected run directory. Run filtration first."
        config_text = self.config_var.get().strip()
        if config_text and not Path(config_text).is_file():
            return False, "The settings override does not exist."
        for value, name in ((self.workers_var.get().strip(), "Workers"),
                            (self.track_limit_var.get().strip(), "Track limit")):
            if value:
                try:
                    if int(value) < 1:
                        raise ValueError
                except ValueError:
                    return False, f"{name} must be a positive integer."
        return True, ""

    def _command(self) -> list[str]:
        stage = self.stage_var.get()
        command = [sys.executable, "-u", "-m", "aorta_pipeline", stage,
                   "--run-dir", str(Path(self.run_dir_var.get()).resolve())]
        if stage in {"all", "filter"}:
            command.extend(["--input", str(Path(self.input_var.get()).resolve())])
        config = self.config_var.get().strip()
        if config:
            command.extend(["--config", str(Path(config).resolve())])
        if stage in {"all", "extend"}:
            workers = self.workers_var.get().strip()
            limit = self.track_limit_var.get().strip()
            if workers:
                command.extend(["--workers", workers])
            if limit:
                command.extend(["--track-limit", limit])
        return command

    def _run(self):
        if self.process is not None:
            return
        valid, message = self._validate_fields()
        if not valid:
            messagebox.showerror("Cannot start", message)
            return
        command = self._command()
        self._append_log("\n$ " + subprocess.list2cmdline(command) + "\n")
        self.run_button.configure(state="disabled")
        self.demo_button.configure(state="disabled")
        self.configure_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.progress.start(12)
        self.status_var.set(f"Running {self.stage_var.get()}…")
        self.worker_thread = threading.Thread(target=self._run_subprocess, args=(command,), daemon=True)
        self.worker_thread.start()

    def _run_subprocess(self, command: list[str]):
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self.process = subprocess.Popen(
                command, cwd=str(PACKAGE_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1, creationflags=creationflags,
            )
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.messages.put(("log", line))
            code = self.process.wait()
            self.messages.put(("done", code))
        except Exception as exc:
            self.messages.put(("error", str(exc)))

    def _poll_messages(self):
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "done":
                    self._finish(int(payload))
                elif kind == "error":
                    self._append_log(f"ERROR: {payload}\n")
                    self._finish(1)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)

    def _finish(self, return_code: int):
        self.process = None
        self.progress.stop()
        self.run_button.configure(state="normal")
        self.demo_button.configure(state="normal")
        self.configure_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        if return_code == 0:
            self.status_var.set("Completed successfully")
            self._append_log("Pipeline completed successfully.\n")
        else:
            self.status_var.set(f"Stopped or failed (exit code {return_code})")
            self._append_log(f"Pipeline exited with code {return_code}.\n")

    def _stop(self):
        process = self.process
        if process is None or process.poll() is not None:
            return
        if not messagebox.askyesno("Stop pipeline", "Stop the active pipeline and its worker processes?"):
            return
        self._terminate_process(process)

    def _terminate_process(self, process: subprocess.Popen[str]):
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW, check=False)
            else:
                process.terminate()
            self.status_var.set("Stopping…")
        except Exception as exc:
            messagebox.showerror("Could not stop", str(exc))

    def _open_results(self):
        result_dir = Path(self.run_dir_var.get().strip()) / "results"
        if not result_dir.exists():
            messagebox.showinfo("Results", "No results directory exists for this run yet.")
            return
        if os.name == "nt":
            os.startfile(result_dir)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(result_dir)])
        else:
            subprocess.Popen(["xdg-open", str(result_dir)])

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _append_log(self, text: str):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _on_close(self):
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno("Pipeline running", "A pipeline is still running. Stop it and close?"):
                return
            self._terminate_process(self.process)
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    PipelineGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
