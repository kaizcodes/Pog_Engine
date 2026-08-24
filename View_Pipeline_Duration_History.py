"""Show average RunAll durations for the five user-facing pipeline steps.

Run from the Pog Engine project folder:
    python View_Pipeline_Duration_History.py

The RunAll GUI appends one row per attempted step to
``View_Pipeline_Duration_History.csv`` beside this script. Only successful
rows are included in the averages; failed and stopped attempts stay visible
for diagnosis, while skipped attempts retain the previous successful duration
when one exists.
"""

from __future__ import annotations

import csv
import statistics
import tkinter as tk
from collections import defaultdict
from pathlib import Path
from tkinter import ttk

from pipeline_config import BIG_STEP_LABELS, STEP_HISTORY_FILENAME

SCRIPT_DIR = Path(__file__).resolve().parent
HISTORY_PATH = SCRIPT_DIR / STEP_HISTORY_FILENAME


def load_duration_rows(path: Path = HISTORY_PATH) -> list[dict[str, object]]:
    """Read valid duration rows, ignoring incomplete/corrupt CSV records."""
    if not path.is_file():
        return []

    rows: list[dict[str, object]] = []
    try:
        with path.open("r", newline="", encoding="utf-8") as history_file:
            for raw_row in csv.DictReader(history_file):
                try:
                    step_number = int(raw_row.get("step_number", ""))
                    duration_seconds = float(raw_row.get("duration_seconds", ""))
                except (TypeError, ValueError):
                    continue
                if not 1 <= step_number <= len(BIG_STEP_LABELS):
                    continue
                rows.append(
                    {
                        "run_timestamp": raw_row.get("run_timestamp", ""),
                        "run_id": raw_row.get("run_id", ""),
                        "stream_folder": raw_row.get("stream_folder", ""),
                        "step_number": step_number,
                        "step_label": raw_row.get("step_label") or BIG_STEP_LABELS[step_number - 1],
                        "status": raw_row.get("status", ""),
                        "duration_seconds": max(duration_seconds, 0.0),
                    }
                )
    except OSError:
        return []
    return rows


def summarize_duration_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return one average/min/max/latest summary for each big step."""
    completed: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "Done":
            completed[int(row["step_number"])].append(row)

    summaries: list[dict[str, object]] = []
    for step_number, step_label in enumerate(BIG_STEP_LABELS, start=1):
        step_rows = completed.get(step_number, [])
        durations = [float(row["duration_seconds"]) for row in step_rows]
        latest = step_rows[-1] if step_rows else None
        summaries.append(
            {
                "step_number": step_number,
                "step_label": step_label,
                "completed_runs": len(durations),
                "average_seconds": statistics.fmean(durations) if durations else None,
                "fastest_seconds": min(durations) if durations else None,
                "slowest_seconds": max(durations) if durations else None,
                "latest_seconds": float(latest["duration_seconds"]) if latest else None,
            }
        )
    return summaries


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    total_seconds = max(seconds, 0.0)
    rounded_seconds = round(total_seconds, 1)
    whole_seconds = int(total_seconds)
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, display_seconds = divmod(remainder, 60)
    if hours:
        readable = f"{hours}h {minutes}m {display_seconds}s"
    elif minutes:
        readable = f"{minutes}m {display_seconds}s"
    else:
        readable = f"{display_seconds}s"
    return f"{rounded_seconds:.1f}s ({readable})"


def build_gui() -> tk.Tk:
    root = tk.Tk()
    root.title("Pog Engine - Pipeline Step Durations")
    root.geometry("1120x680")
    root.configure(bg="#121212")

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background="#121212", foreground="#f2f2f2", fieldbackground="#1e1e1e")
    style.configure("TFrame", background="#121212")
    style.configure("TLabel", background="#121212", foreground="#f2f2f2")
    style.configure("TLabelframe", background="#121212", foreground="#f2f2f2")
    style.configure("TLabelframe.Label", background="#121212", foreground="#f2f2f2")
    style.configure("Treeview", background="#1e1e1e", fieldbackground="#1e1e1e", foreground="#f2f2f2")
    style.configure("Treeview.Heading", background="#2b2b2b", foreground="#f2f2f2")
    style.map("Treeview", background=[("selected", "#465d75")])

    container = ttk.Frame(root, padding=14)
    container.pack(fill="both", expand=True)
    container.rowconfigure(2, weight=1)
    container.rowconfigure(4, weight=2)
    container.columnconfigure(0, weight=1)

    ttk.Label(container, text="Pipeline step durations", font=("Segoe UI", 17, "bold")).grid(
        row=0, column=0, sticky="w"
    )
    header = ttk.Frame(container)
    header.grid(row=1, column=0, sticky="ew", pady=(4, 10))
    header.columnconfigure(0, weight=1)
    path_var = tk.StringVar()
    ttk.Label(header, textvariable=path_var).grid(row=0, column=0, sticky="w")

    summary_frame = ttk.LabelFrame(container, text="Average successful step times", padding=8)
    summary_frame.grid(row=2, column=0, sticky="nsew")
    summary_frame.columnconfigure(1, weight=1)
    summary_columns = ("step", "runs", "average", "fastest", "slowest", "latest")
    summary = ttk.Treeview(summary_frame, columns=summary_columns, show="headings", height=6)
    headings = {
        "step": "Step",
        "runs": "Completed runs",
        "average": "Average",
        "fastest": "Fastest",
        "slowest": "Slowest",
        "latest": "Latest",
    }
    widths = {"step": 330, "runs": 120, "average": 190, "fastest": 190, "slowest": 190, "latest": 190}
    for column in summary_columns:
        summary.heading(column, text=headings[column])
        summary.column(column, width=widths[column], anchor="w")
    summary.grid(row=0, column=0, sticky="nsew")
    summary_scroll = ttk.Scrollbar(summary_frame, orient="vertical", command=summary.yview)
    summary_scroll.grid(row=0, column=1, sticky="ns")
    summary.configure(yscrollcommand=summary_scroll.set)

    history_frame = ttk.LabelFrame(container, text="Logged attempts", padding=8)
    history_frame.grid(row=4, column=0, sticky="nsew", pady=(12, 0))
    history_frame.rowconfigure(0, weight=1)
    history_frame.columnconfigure(0, weight=1)
    history_columns = ("timestamp", "step", "status", "duration", "folder")
    history = ttk.Treeview(history_frame, columns=history_columns, show="headings")
    history_headings = {
        "timestamp": "Finished",
        "step": "Step",
        "status": "Status",
        "duration": "Duration",
        "folder": "VOD folder",
    }
    history_widths = {"timestamp": 160, "step": 260, "status": 90, "duration": 190, "folder": 420}
    for column in history_columns:
        history.heading(column, text=history_headings[column])
        history.column(column, width=history_widths[column], anchor="w")
    history.grid(row=0, column=0, sticky="nsew")
    history_scroll = ttk.Scrollbar(history_frame, orient="vertical", command=history.yview)
    history_scroll.grid(row=0, column=1, sticky="ns")
    history.configure(yscrollcommand=history_scroll.set)

    status_var = tk.StringVar()
    ttk.Label(container, textvariable=status_var).grid(row=5, column=0, sticky="w", pady=(8, 0))

    def refresh() -> None:
        rows = load_duration_rows()
        for tree in (summary, history):
            tree.delete(*tree.get_children())
        for item in summarize_duration_rows(rows):
            summary.insert(
                "",
                "end",
                values=(
                    item["step_label"],
                    item["completed_runs"],
                    format_duration(item["average_seconds"]),
                    format_duration(item["fastest_seconds"]),
                    format_duration(item["slowest_seconds"]),
                    format_duration(item["latest_seconds"]),
                ),
            )
        for row in reversed(rows):
            history.insert(
                "",
                "end",
                values=(
                    row["run_timestamp"],
                    row["step_label"],
                    row["status"],
                    format_duration(float(row["duration_seconds"])),
                    row["stream_folder"],
                ),
            )
        completed_count = sum(row.get("status") == "Done" for row in rows)
        path_var.set(f"Data: {HISTORY_PATH}    Completed step timings: {completed_count}")
        status_var.set("Averages use successful Done rows only. Refresh to load new runs.")

    ttk.Button(header, text="Refresh", command=refresh).grid(row=0, column=1, padx=(10, 0))
    refresh()
    return root


def main() -> None:
    build_gui().mainloop()


if __name__ == "__main__":
    main()
