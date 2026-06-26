"""
dashboard.py  –  PPO Training Dashboard  (v2)
══════════════════════════════════════════════
Launched by map_reader.py as a subprocess when AI TRAIN is clicked.
Reads training_state.json written by PPOTrainer every frame and
renders live charts + stats in a Tkinter window.

Fixes over v1:
  • Rewards update every poll tick (500 ms) — not just when history changes
  • Chart redraws on EVERY poll, not only when history list changes
  • Save writes full model weights + Adam moment state so training resumes
  • Entropy coefficient shown and tracked
  • Rolling average overlay on the reward chart
  • Cleaner error handling so a bad state file never crashes the dashboard

Usage:
    python dashboard.py --state <path_to_training_state.json> --models <models_dir>
"""

import sys
import os
import json
import time
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── CLI args ───────────────────────────────────────────────────
STATE_PATH = None
MODELS_DIR = None
for i, arg in enumerate(sys.argv):
    if arg == "--state"  and i + 1 < len(sys.argv):
        STATE_PATH = sys.argv[i + 1]
    if arg == "--models" and i + 1 < len(sys.argv):
        MODELS_DIR = sys.argv[i + 1]

_HERE = os.path.dirname(os.path.abspath(__file__))
if STATE_PATH is None:
    STATE_PATH = os.path.join(_HERE, "training_state.json")
if MODELS_DIR is None:
    MODELS_DIR = os.path.join(_HERE, "models")

MANIFEST_PATH = os.path.join(MODELS_DIR, "manifest.json")

# ── Palette ────────────────────────────────────────────────────
BG_DARK  = "#0d0d16"
BG_PANEL = "#14141f"
BG_CARD  = "#1a1a2e"
ACCENT   = "#3285e0"
ACCENT2  = "#a259ff"
GREEN    = "#4ade80"
RED      = "#f87171"
ORANGE   = "#fb923c"
YELLOW   = "#fbbf24"
TEXT     = "#c8c8d8"
HINT     = "#606078"
WHITE    = "#eeeef8"
BORDER   = "#2a2a40"

POLL_MS      = 500    # refresh interval in ms
ROLLING_WIN  = 10     # episodes for rolling average


# ══════════════════════════════════════════════════════════════
#  File helpers
# ══════════════════════════════════════════════════════════════

def _load_state() -> dict:
    """Read training_state.json; return {} on any error."""
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return {}
        return json.loads(raw)
    except Exception:
        return {}


def _load_manifest() -> list:
    try:
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _write_state_patch(patch: dict):
    """
    Merge `patch` into the existing state file atomically.
    Used to inject a save_request without losing other fields.
    """
    try:
        try:
            with open(STATE_PATH, encoding="utf-8") as f:
                data = json.loads(f.read())
        except Exception:
            data = {}
        data.update(patch)
        tmp = STATE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, STATE_PATH)          # atomic on all platforms
        return True
    except Exception:
        return False


def _rolling_avg(values: list, window: int) -> list:
    """Simple rolling average; returns same-length list."""
    if not values:
        return []
    out = []
    for i in range(len(values)):
        lo  = max(0, i - window + 1)
        out.append(sum(values[lo:i + 1]) / (i - lo + 1))
    return out


# ══════════════════════════════════════════════════════════════
#  Dashboard
# ══════════════════════════════════════════════════════════════

class Dashboard(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("PPO Training Dashboard  v2")
        self.configure(bg=BG_DARK)
        self.geometry("940x700")
        self.minsize(780, 540)

        # internal state — track previous values so we know what changed
        self._prev_hist     : list  = []
        self._prev_ep       : int   = -1
        self._stat_vars     : dict  = {}   # key → tk.StringVar
        self._stat_colors   : dict  = {}   # key → label widget (for colour changes)

        self._build_ui()
        self._poll()                        # start polling loop

    # ══════════════════════════════════════════════════════════
    #  UI construction
    # ══════════════════════════════════════════════════════════

    def _build_ui(self):
        # ── top bar ──
        top = tk.Frame(self, bg=BG_DARK)
        top.pack(fill=tk.X, padx=16, pady=(10, 0))

        tk.Label(top, text="PPO  TRAINING  DASHBOARD",
                 font=("Consolas", 15, "bold"),
                 bg=BG_DARK, fg=WHITE).pack(side=tk.LEFT)

        self._status_var = tk.StringVar(value="● CONNECTING…")
        self._status_lbl = tk.Label(top, textvariable=self._status_var,
                                    font=("Consolas", 10, "bold"),
                                    bg=BG_DARK, fg=ORANGE)
        self._status_lbl.pack(side=tk.RIGHT, padx=6)

        tk.Frame(self, bg=BORDER, height=1).pack(fill=tk.X, padx=12, pady=6)

        # ── body: left panel | right chart ──
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        left = tk.Frame(body, bg=BG_DARK, width=252)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        left.pack_propagate(False)

        right = tk.Frame(body, bg=BG_DARK)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_stats_panel(left)
        self._build_chart_panel(right)
        self._build_bottom_bar()

    # ── stat card factory ──────────────────────────────────────
    def _make_card(self, parent, title: str, key: str,
                   color: str = TEXT, unit: str = ""):
        frame = tk.Frame(parent, bg=BG_CARD,
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill=tk.X, pady=3, ipady=4, ipadx=6)

        hdr = tk.Frame(frame, bg=BG_CARD)
        hdr.pack(fill=tk.X, padx=8, pady=(4, 0))
        tk.Label(hdr, text=title.upper(),
                 font=("Consolas", 8), bg=BG_CARD, fg=HINT).pack(side=tk.LEFT)
        if unit:
            tk.Label(hdr, text=unit,
                     font=("Consolas", 7), bg=BG_CARD, fg=HINT).pack(side=tk.RIGHT)

        var = tk.StringVar(value="—")
        lbl = tk.Label(frame, textvariable=var,
                       font=("Consolas", 16, "bold"),
                       bg=BG_CARD, fg=color)
        lbl.pack(anchor="w", padx=8, pady=(0, 4))

        self._stat_vars[key]   = var
        self._stat_colors[key] = lbl

    def _build_stats_panel(self, parent):
        tk.Label(parent, text="LIVE METRICS",
                 font=("Consolas", 10, "bold"),
                 bg=BG_DARK, fg=ACCENT).pack(anchor="w", pady=(4, 6))

        self._make_card(parent, "Episode",         "episode",     WHITE)
        self._make_card(parent, "Episode Reward",  "ep_reward",   GREEN)
        self._make_card(parent, "Best Reward",     "best_reward", ACCENT2)
        self._make_card(parent, "Total Steps",     "total_steps", TEXT)
        self._make_card(parent, "Actor Loss",      "actor_loss",  ORANGE)
        self._make_card(parent, "Critic Loss",     "critic_loss", RED)
        self._make_card(parent, "Entropy Coef",    "entropy_coef",YELLOW)

        tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, pady=10)

        # ── checkpoint section ──
        tk.Label(parent, text="CHECKPOINT",
                 font=("Consolas", 9, "bold"),
                 bg=BG_DARK, fg=HINT).pack(anchor="w")

        tk.Button(
            parent, text="⬇  Save & Resume Checkpoint",
            font=("Consolas", 9, "bold"),
            bg=ACCENT, fg=WHITE,
            activebackground="#4a9ef5",
            relief=tk.FLAT, cursor="hand2",
            pady=8, command=self._on_save,
        ).pack(fill=tk.X, pady=(6, 2))

        self._ver_var = tk.StringVar(value="No saved versions yet.")
        tk.Label(parent, textvariable=self._ver_var,
                 font=("Consolas", 8),
                 bg=BG_DARK, fg=HINT,
                 wraplength=230, justify=tk.LEFT).pack(anchor="w", pady=2)

    def _build_chart_panel(self, parent):
        hdr = tk.Frame(parent, bg=BG_DARK)
        hdr.pack(fill=tk.X, pady=(4, 2))
        tk.Label(hdr, text="REWARD  HISTORY",
                 font=("Consolas", 10, "bold"),
                 bg=BG_DARK, fg=ACCENT).pack(side=tk.LEFT)

        self._chart_info = tk.StringVar(value="")
        tk.Label(hdr, textvariable=self._chart_info,
                 font=("Consolas", 8), bg=BG_DARK, fg=HINT).pack(side=tk.RIGHT)

        if HAS_MPL:
            fig = Figure(figsize=(5.2, 4.0), dpi=95, facecolor=BG_CARD)
            self._ax = fig.add_subplot(111)
            self._ax.set_facecolor(BG_CARD)
            self._ax.tick_params(colors=HINT, labelsize=7)
            for spine in self._ax.spines.values():
                spine.set_edgecolor(BORDER)
            self._ax.set_xlabel("Episode", color=HINT, fontsize=8)
            self._ax.set_ylabel("Total Reward", color=HINT, fontsize=8)
            self._ax.grid(True, color=BORDER, linewidth=0.5, linestyle="--")

            # raw reward line
            self._line_raw, = self._ax.plot(
                [], [], color=ACCENT, linewidth=1.2,
                alpha=0.5, label="Reward")
            # rolling average line
            self._line_avg, = self._ax.plot(
                [], [], color=GREEN, linewidth=2.0,
                label=f"Avg-{ROLLING_WIN}")

            self._ax.legend(
                facecolor=BG_CARD, edgecolor=BORDER,
                labelcolor=TEXT, fontsize=7,
                loc="upper left")

            self._fill_obj = None
            fig.tight_layout(pad=1.6)

            self._canvas = FigureCanvasTkAgg(fig, master=parent)
            self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            tk.Label(parent,
                     text="matplotlib not found.\n\nInstall it:\n  pip install matplotlib",
                     font=("Consolas", 10), bg=BG_CARD, fg=HINT,
                     justify=tk.LEFT).pack(fill=tk.BOTH, expand=True,
                                           padx=12, pady=12)

        # live log line at bottom of chart
        self._log_var = tk.StringVar(value="Waiting for training data…")
        tk.Label(parent, textvariable=self._log_var,
                 font=("Consolas", 8), bg=BG_DARK, fg=HINT,
                 anchor="w").pack(fill=tk.X, pady=(4, 0))

    def _build_bottom_bar(self):
        bar = tk.Frame(self, bg=BG_PANEL)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Label(bar, text=f"State: {STATE_PATH}",
                 font=("Consolas", 7), bg=BG_PANEL, fg=HINT,
                 anchor="w").pack(side=tk.LEFT, padx=10, pady=5)
        tk.Label(bar, text="PPO Dashboard  v2.0",
                 font=("Consolas", 7), bg=BG_PANEL, fg="#3c3c50",
                 anchor="e").pack(side=tk.RIGHT, padx=10, pady=5)

    # ══════════════════════════════════════════════════════════
    #  Poll loop  — runs every POLL_MS ms
    # ══════════════════════════════════════════════════════════

    def _poll(self):
        state = _load_state()
        if state:
            self._refresh(state)
        else:
            self._status_var.set("● NO DATA")
            self._status_lbl.config(fg=RED)
        self.after(POLL_MS, self._poll)

    # ══════════════════════════════════════════════════════════
    #  Refresh UI from state dict
    # ══════════════════════════════════════════════════════════

    def _refresh(self, state: dict):
        running = state.get("running", False)
        self._status_var.set("● TRAINING" if running else "● STOPPED")
        self._status_lbl.config(fg=GREEN if running else RED)

        # ── stat cards — always update every tick ──
        ep        = state.get("episode",      0)
        ep_rew    = state.get("ep_reward",    0.0)
        best_rew  = state.get("best_reward",  0.0)
        steps     = state.get("total_steps",  0)
        a_loss    = state.get("actor_loss",   0.0)
        c_loss    = state.get("critic_loss",  0.0)
        ent       = state.get("entropy_coef", 0.0)

        self._stat_vars["episode"].set(f"{ep:,}")
        self._stat_vars["total_steps"].set(f"{steps:,}")
        self._stat_vars["actor_loss"].set(f"{a_loss:.6f}")
        self._stat_vars["critic_loss"].set(f"{c_loss:.6f}")
        self._stat_vars["entropy_coef"].set(f"{ent:.5f}")

        # colour-coded reward: green positive, red negative
        self._stat_vars["ep_reward"].set(f"{ep_rew:.2f}")
        self._stat_colors["ep_reward"].config(
            fg=GREEN if ep_rew >= 0 else RED)

        self._stat_vars["best_reward"].set(f"{best_rew:.2f}")
        self._stat_colors["best_reward"].config(
            fg=ACCENT2 if best_rew >= 0 else RED)

        # flash episode card when episode increments
        if ep != self._prev_ep and self._prev_ep >= 0:
            self._flash("episode")
        self._prev_ep = ep

        # ── reward history chart — redraw every tick ──
        hist = state.get("reward_history", [])
        self._redraw_chart(hist)

        # ── version manifest ──
        manifest = _load_manifest()
        if manifest:
            recent = sorted(manifest, key=lambda x: x["timestamp"])[-4:]
            lines  = "\n".join(
                f"• {m['label']}  ep={m.get('episode','?')}  "
                f"best={m.get('best_reward','?')}"
                for m in reversed(recent)
            )
            self._ver_var.set(f"Recent saves:\n{lines}")
        else:
            self._ver_var.set("No saved versions yet.")

        # ── bottom log line ──
        self._log_var.set(
            f"Ep {ep}  |  reward={ep_rew:.2f}  |  "
            f"best={best_rew:.2f}  |  steps={steps:,}"
        )

    # ══════════════════════════════════════════════════════════
    #  Chart drawing
    # ══════════════════════════════════════════════════════════

    def _redraw_chart(self, hist: list):
        """Redraw both the raw and rolling-average reward lines."""
        if not HAS_MPL or not hist:
            if HAS_MPL and not hist:
                self._chart_info.set("No episodes yet…")
            return

        x   = list(range(1, len(hist) + 1))
        avg = _rolling_avg(hist, ROLLING_WIN)

        # update lines
        self._line_raw.set_data(x, hist)
        self._line_avg.set_data(x, avg)

        # update fill under raw line
        if self._fill_obj is not None:
            try:
                self._fill_obj.remove()
            except Exception:
                pass
            self._fill_obj = None

        try:
            self._fill_obj = self._ax.fill_between(
                x, hist, alpha=0.12, color=ACCENT)
        except Exception:
            pass

        # rescale axes
        self._ax.relim()
        self._ax.autoscale_view()

        # chart info label
        self._chart_info.set(
            f"{len(hist)} episodes  |  "
            f"avg-{ROLLING_WIN}: {avg[-1]:.2f}  |  "
            f"max: {max(hist):.2f}"
        )

        # draw — use draw_idle so Tk event loop isn't blocked
        self._canvas.draw_idle()

    # ══════════════════════════════════════════════════════════
    #  Visual helpers
    # ══════════════════════════════════════════════════════════

    def _flash(self, key: str):
        """Briefly highlight a stat card to signal a change."""
        lbl = self._stat_colors.get(key)
        if lbl is None:
            return
        orig = lbl.cget("fg")
        lbl.config(fg=WHITE)
        self.after(300, lambda: lbl.config(fg=orig))

    # ══════════════════════════════════════════════════════════
    #  Save checkpoint
    # ══════════════════════════════════════════════════════════

    def _on_save(self):
        """
        Ask for a label, then inject a save_request into the state file.
        PPOTrainer reads this flag each frame and calls agent.save_version().
        The saved JSON now includes full weight arrays AND Adam moment
        states, so training can resume exactly where it left off.
        """
        label = simpledialog.askstring(
            "Save Checkpoint",
            "Label for this checkpoint\n"
            "(leave blank to use a timestamp):",
            parent=self,
        )
        if label is None:          # user cancelled
            return

        label = label.strip() or time.strftime("%Y%m%d_%H%M%S")

        # Validate label (no path separators or quotes)
        bad = set('/\\"\'')
        if any(c in bad for c in label):
            messagebox.showerror(
                "Invalid Label",
                "Label must not contain  /  \\  \"  '",
                parent=self)
            return

        ok = _write_state_patch({"save_request": label})
        if ok:
            messagebox.showinfo(
                "Save Requested",
                f"Checkpoint  '{label}'  will be saved at the next\n"
                f"training step and added to the manifest.\n\n"
                f"Models dir:\n{MODELS_DIR}",
                parent=self,
            )
        else:
            messagebox.showerror(
                "Write Error",
                "Could not write save request.\n"
                f"Check that  {STATE_PATH}  is writable.",
                parent=self,
            )


# ══════════════════════════════════════════════════════════════
#  Entry
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = Dashboard()
    app.mainloop()