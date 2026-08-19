"""Tk GUI for choosing Ollama models and pipeline defaults.

The form reads its fields and presets from pipeline_config.py. Saving updates
the defaults in that file; it does not install packages or run the installer.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Allow direct launch from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pipeline_config as cfg  # noqa: E402


def _current_values() -> dict:
    return {
        param["key"]: getattr(cfg, param["key"], None)
        for param in cfg.EDITABLE_PARAMS
    }


def _find_matching_preset(discovery_model: str, judge_model: str) -> tuple[str, dict] | None:
    """Return the preset for an exact discovery/judge model pairing."""
    discovery_model = str(discovery_model or "").strip()
    judge_model = str(judge_model or "").strip()
    if not discovery_model or not judge_model:
        return None
    for name, preset in cfg.PRESETS.items():
        values = preset.get("values", {})
        if (
            str(values.get("MODEL", "")).strip() == discovery_model
            and str(values.get("JUDGE_MODEL", "")).strip() == judge_model
        ):
            return name, preset
    return None




def run_gui() -> int:
    import tkinter as tk
    from tkinter import messagebox, ttk

    root = tk.Tk()
    root.title("Pog_Engine - Configure Pog Engine")
    root.geometry("1080x820")
    root.configure(bg="#121212")
    root.minsize(880, 620)

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background="#121212", foreground="#f2f2f2", fieldbackground="#1e1e1e")
    style.configure("TFrame", background="#121212")
    style.configure("TLabelframe", background="#121212", foreground="#f2f2f2")
    style.configure("TLabelframe.Label", background="#121212", foreground="#f2f2f2", font=("Segoe UI", 10, "bold"))
    style.configure("TLabel", background="#121212", foreground="#f2f2f2")
    style.configure("TButton", background="#242424", foreground="#f2f2f2", bordercolor="#3a3a3a")
    style.map("TButton", background=[("active", "#333333")], foreground=[("disabled", "#666666")])
    style.configure("TEntry", fieldbackground="#1e1e1e", foreground="#f2f2f2")
    style.configure("TCombobox", fieldbackground="#1e1e1e", foreground="#f2f2f2", background="#1e1e1e")
    style.map("TCombobox", fieldbackground=[("readonly", "#1e1e1e")], foreground=[("readonly", "#f2f2f2")])
    style.configure("Accent.TButton", background="#2a6a4a", foreground="#f2f2f2", bordercolor="#3a8a5a")
    style.map("Accent.TButton", background=[("active", "#357a55")])

    var_state: dict[str, tk.Variable] = {}
    phrase_editors: dict[str, tk.Text] = {}
    installed_models: list[str] = []
    config_path = Path(cfg.__file__).resolve()
    last_judge_model = str(getattr(cfg, "JUDGE_MODEL", "") or "").strip()


    def phrase_text(value) -> str:
        if isinstance(value, (list, tuple)):
            return "\n".join(str(item) for item in value)
        return "" if value is None else str(value)


    def make_var(param: dict, value) -> tk.Variable:
        kind = param["kind"]
        if kind == "bool":
            variable = tk.BooleanVar(value=bool(value))
        elif kind == "int":
            try:
                value = int(value) if value not in (None, "") else 0
            except (TypeError, ValueError):
                value = 0
            variable = tk.IntVar(value=value)
        elif kind == "float":
            try:
                value = float(value) if value not in (None, "") else 0.0
            except (TypeError, ValueError):
                value = 0.0
            variable = tk.DoubleVar(value=value)
        elif kind == "phrases":
            variable = tk.StringVar(value=phrase_text(value))
        else:
            variable = tk.StringVar(value="" if value is None else str(value))
        return variable

    def init_state_from(values: dict) -> None:
        for param in cfg.EDITABLE_PARAMS:
            key = param["key"]
            if param["kind"] == "phrases" and key in phrase_editors:
                editor = phrase_editors[key]
                editor.delete("1.0", "end")
                editor.insert("1.0", phrase_text(values.get(key)))
                continue

            value_var = make_var(param, values.get(key))
            if key in var_state:
                var_state[key].set(value_var.get())
            else:
                var_state[key] = value_var

    def collect_state() -> dict:
        values: dict = {}
        for param in cfg.EDITABLE_PARAMS:
            key = param["key"]
            if param["kind"] == "phrases":
                values[key] = phrase_editors[key].get("1.0", "end-1c")
                continue
            variable = var_state[key]
            if param["kind"] == "bool":
                values[key] = bool(variable.get())
            elif param["kind"] == "int":
                raw = str(variable.get()).strip()
                values[key] = int(raw) if raw else 0
            elif param["kind"] == "float":
                raw = str(variable.get()).strip()
                values[key] = float(raw) if raw else 0.0
            else:
                values[key] = variable.get()
        return values

    init_state_from(_current_values())

    top = ttk.Frame(root, padding=(12, 12, 12, 6))
    top.pack(fill="x")
    ttk.Label(top, text="Configure Pog Engine", font=("Segoe UI", 16, "bold")).pack(anchor="w")
    ttk.Label(
        top,
        text=("Choose the Ollama models, apply a preset for this machine, then save. "
              "Changes go directly to pipeline_config.py."),
    ).pack(anchor="w", pady=(2, 8))


    def refresh_models() -> None:
        nonlocal installed_models
        installed_models, warning = cfg.list_ollama_models()
        model_values = installed_models
        if warning:
            model_values = []
            models_status_var.set(warning)
        else:
            count = len(installed_models)
            models_status_var.set(
                f"{count} model(s) detected by `ollama list`."
                if count
                else "No models found. Pull one with `ollama pull <name>`."
            )
        for key in ("MODEL", "JUDGE_MODEL"):
            for combo in model_combos.get(key, []):
                combo.configure(values=model_values)
        root.update_idletasks()

    models_status_var = tk.StringVar(value="Click 'Scan models' to detect installed Ollama models.")

    model_combos: dict[str, list[ttk.Combobox]] = {}
    model_params = [param for param in cfg.EDITABLE_PARAMS if param["kind"] == "model"]
    for param in model_params:
        row_frame = ttk.Frame(top)
        row_frame.pack(fill="x", pady=2)
        row_frame.columnconfigure(1, weight=1)
        ttk.Label(row_frame, text=param["label"], width=42, anchor="w").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        combo = ttk.Combobox(
            row_frame,
            textvariable=var_state[param["key"]],
            values=installed_models,
        )
        combo.grid(row=0, column=1, sticky="ew")
        ttk.Label(row_frame, text=param["help"], foreground="#8a8a8a").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(2, 0)
        )
        model_combos.setdefault(param["key"], []).append(combo)

    scan_row = ttk.Frame(top)
    scan_row.pack(fill="x", pady=(6, 0))
    ttk.Button(scan_row, text="Scan models", command=refresh_models).pack(side="left")
    ttk.Label(scan_row, textvariable=models_status_var, width=80, anchor="w").pack(side="left", padx=10)

    preset_row = ttk.Frame(root, padding=(12, 4, 12, 4))
    preset_row.pack(fill="x")
    ttk.Label(preset_row, text="Preset:").pack(side="left")
    preset_names = list(cfg.PRESETS)
    preset_var = tk.StringVar()
    preset_cb = ttk.Combobox(
        preset_row,
        textvariable=preset_var,
        values=preset_names,
        width=72,
        state="readonly",
    )
    preset_cb.pack(side="left", padx=(6, 6))

    recommendation_var = tk.StringVar(value="")


    def set_state_value(key: str, value) -> None:
        editor = phrase_editors.get(key)
        if editor is not None:
            editor.delete("1.0", "end")
            editor.insert("1.0", phrase_text(value))
        else:
            var_state[key].set(value)

    def apply_preset() -> None:
        nonlocal last_judge_model
        name = preset_var.get()
        if not name:
            messagebox.showinfo("Pog_Engine", "Pick a preset first.", parent=root)
            return
        if name not in cfg.PRESETS:
            messagebox.showerror("Pog_Engine", f"Unknown preset: {name}", parent=root)
            return
        values = cfg.PRESETS[name].get("values", {})
        for key, value in values.items():
            if key in var_state or key in phrase_editors:
                set_state_value(key, value)
        last_judge_model = str(var_state["JUDGE_MODEL"].get()).strip()
        recommendation_var.set("")
        preset = cfg.PRESETS[name]
        log(f"Loaded preset '{preset.get('name_short', name)}' into the form.")
        log(preset.get("description", ""))
        log("Review the values, then click Save.")


    ttk.Button(
        preset_row,
        text="Apply preset to form",
        style="Accent.TButton",
        command=apply_preset,
    ).pack(side="left", padx=(0, 6))

    recommendation_row = ttk.Frame(root, padding=(12, 0, 12, 4))
    recommendation_row.pack(fill="x")
    ttk.Label(
        recommendation_row,
        textvariable=recommendation_var,
        foreground="#d8b86a",
        wraplength=1040,
        justify="left",
    ).pack(fill="x")

    current_preset = _find_matching_preset(
        str(var_state["MODEL"].get()).strip(),
        str(var_state["JUDGE_MODEL"].get()).strip(),
    )
    if current_preset is not None:
        preset_cb.set(current_preset[0])
    elif preset_names:
        preset_cb.current(0)

    middle_container = ttk.Frame(root, padding=(12, 4, 12, 4))
    middle_container.pack(fill="both", expand=True)
    canvas = tk.Canvas(middle_container, bg="#121212", highlightthickness=0)
    scrollbar = ttk.Scrollbar(middle_container, orient="vertical", command=canvas.yview)
    params_frame = ttk.Frame(canvas)
    params_frame.bind("<Configure>", lambda event: canvas.configure(
        scrollregion=canvas.bbox("all")
    ))
    window_id = canvas.create_window((0, 0), window=params_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    def stretch_canvas(_event=None) -> None:
        canvas.itemconfigure(window_id, width=canvas.winfo_width())

    canvas.bind("<Configure>", stretch_canvas)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    by_stage: dict[str, list[dict]] = {}
    for param in cfg.EDITABLE_PARAMS:
        by_stage.setdefault(param["stage"], []).append(param)

    section_row = 0
    for stage, params in by_stage.items():
        section = ttk.LabelFrame(params_frame, text=stage, padding=(10, 6))
        section.grid(row=section_row, column=0, sticky="ew", pady=(0, 8))
        section.columnconfigure(1, weight=1)
        section_row += 1

        for row_index, param in enumerate(params):
            row = row_index * 2
            ttk.Label(section, text=param["label"], anchor="w").grid(
                row=row, column=0, sticky="ew", padx=(0, 8), pady=(2, 0)
            )
            kind = param["kind"]
            if kind == "bool":
                widget = ttk.Checkbutton(section, variable=var_state[param["key"]])
            elif kind == "model":
                widget = ttk.Combobox(
                    section,
                    textvariable=var_state[param["key"]],
                    values=installed_models,
                    width=28,
                )
                model_combos.setdefault(param["key"], []).append(widget)
            elif kind == "phrases":
                widget = tk.Text(
                    section,
                    height=8,
                    width=58,
                    wrap="word",
                    bg="#1e1e1e",
                    fg="#f2f2f2",
                    insertbackground="#f2f2f2",
                    relief="flat",
                )
                widget.insert("1.0", var_state[param["key"]].get())
                phrase_editors[param["key"]] = widget
            else:
                widget = ttk.Entry(section, textvariable=var_state[param["key"]], width=14)
            widget.grid(
                row=row,
                column=1,
                sticky="ew" if kind == "phrases" else "w",
                pady=(2, 0),
            )
            ttk.Label(section, text=param["help"], foreground="#8a8a8a", wraplength=720).grid(
                row=row + 1, column=0, columnspan=2, sticky="w", pady=(0, 4)
            )

    bottom = ttk.Frame(root, padding=(12, 6, 12, 12))
    bottom.pack(fill="both", expand=False)
    ttk.Label(bottom, text="Log").pack(anchor="w")
    log_box = tk.Text(
        bottom,
        height=8,
        wrap="word",
        state="disabled",
        bg="#0f0f0f",
        fg="#f2f2f2",
        insertbackground="#f2f2f2",
    )
    log_box.pack(fill="both", expand=True)

    def log(message: str) -> None:
        log_box.configure(state="normal")
        log_box.insert("end", message.rstrip("\n") + "\n")
        log_box.see("end")
        log_box.configure(state="disabled")

    def on_judge_model_selected(_event=None) -> None:
        nonlocal last_judge_model
        judge_model = str(var_state["JUDGE_MODEL"].get()).strip()
        if judge_model == last_judge_model:
            return
        previous_model = last_judge_model
        last_judge_model = judge_model
        match = _find_matching_preset(var_state["MODEL"].get(), judge_model)
        if match is None:
            recommendation_var.set("")
            return

        preset_name, preset = match
        preset_var.set(preset_name)
        recommendation_var.set(
            f"Recommended preset for {var_state['MODEL'].get()} + {judge_model}: "
            f"{preset_name}. Apply it to tune the context and output settings for a better experience."
        )
        log(
            f"Recommended preset '{preset.get('name_short', preset_name)}' after "
            f"changing the judge model from {previous_model or '(empty)'} to {judge_model}."
        )
        messagebox.showinfo(
            "Recommended preset",
            f"You changed the judge model to {judge_model}.\n\n"
            f"A matching discovery + judge preset is available:\n{preset_name}\n\n"
            "Using this preset is recommended for a better experience because it "
            "tunes the model context and output settings for this pairing.\n\n"
            "The preset is selected above; click 'Apply preset to form' to use it.",
            parent=root,
        )

    for combo in model_combos.get("JUDGE_MODEL", []):
        combo.bind("<<ComboboxSelected>>", on_judge_model_selected, add="+")


    status_var = tk.StringVar(value="Ready.")
    ttk.Label(root, textvariable=status_var, anchor="w").pack(fill="x", side="bottom")

    save_bar = ttk.Frame(root, padding=(12, 4, 12, 0))
    save_bar.pack(fill="x")
    ttk.Label(save_bar, text="Target:").pack(side="left")
    ttk.Label(
        save_bar,
        text=str(config_path),
        foreground="#8a8a8a",
    ).pack(side="left", padx=(4, 12))

    def do_save() -> None:
        values = collect_state()
        problems: list[str] = []
        for param in cfg.EDITABLE_PARAMS:
            key = param["key"]
            kind = param["kind"]
            if kind in ("int", "float"):
                try:
                    if kind == "int":
                        int(values[key])
                    else:
                        float(values[key])
                except (TypeError, ValueError):
                    problems.append(f"{key}: {values[key]!r} is not a valid {kind}")
            elif kind == "model" and not values[key]:
                problems.append(f"{key}: choose a model")

        if problems:
            for problem in problems:
                log(f"[!] {problem}")
            messagebox.showerror(
                "Pog_Engine",
                "Fix these values before saving:\n\n" + "\n".join(problems),
                parent=root,
            )
            return

        ok, message = cfg.apply_config_values(values, str(config_path))
        if ok:
            log(f"[OK] {message}")
            status_var.set(f"Saved: {message}")
            messagebox.showinfo(
                "Pog_Engine",
                "Saved. The next pipeline run will use these defaults.",
                parent=root,
            )
        else:
            log(f"[!] {message}")
            status_var.set(f"Save failed: {message}")
            messagebox.showerror("Pog_Engine", f"Save failed:\n{message}", parent=root)

    def do_revert() -> None:
        nonlocal last_judge_model
        importlib.reload(cfg)
        init_state_from(_current_values())
        last_judge_model = str(getattr(cfg, "JUDGE_MODEL", "") or "").strip()
        recommendation_var.set("")
        log("Reverted to the values currently in pipeline_config.py.")

    ttk.Button(
        save_bar,
        text="Save to pipeline_config.py",
        style="Accent.TButton",
        command=do_save,
    ).pack(side="left")
    ttk.Button(save_bar, text="Revert form", command=do_revert).pack(side="left", padx=6)

    root.after(80, refresh_models)
    log(f"Loaded {len(cfg.EDITABLE_PARAMS)} editable parameters.")
    log(f"Config: {config_path}")
    log(f"Presets available: {len(cfg.PRESETS)}")

    root.mainloop()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = argv[1:] if argv is not None else sys.argv[1:]
    if "--cli" in args:
        print("Edit pipeline_config.py directly or use the GUI.")
        return 0
    try:
        import tkinter  # noqa: F401
    except ImportError:
        print("[INFO] tkinter is not available.")
        return 1
    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
