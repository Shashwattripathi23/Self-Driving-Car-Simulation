"""
map_reader.py  –  Autonomous Car Simulator  |  Map Viewer
══════════════════════════════════════════════════════════
Opens a .json map file produced by environment.py and
renders the road exactly as it was drawn.

Controls
────────
  O / Ctrl+O   open another map file
  R            reset / clear current map  (or reset car if active)
  ESC          quit
  ── When car is active ──
  W            throttle
  S            reverse
  A / D        steer left / right
  SPACE        brake
"""

import pygame
import sys
import json
import os
import math
import random
import subprocess
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox

# ── PPO model (optional – graceful fallback if numpy missing) ──
try:
    from model import PPOAgent, PPOTrainer, get_latest_model_path, MODELS_DIR
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False
    MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# ── Window config ──────────────────────────────────────────────
WIN_W, WIN_H  = 960, 650
SIDEBAR_W     = 220
CANVAS_W      = WIN_W - SIDEBAR_W

# ── Colours ────────────────────────────────────────────────────
BG           = (18,  18,  24)
ROAD_SURF    = (45,  45,  55)
EDGE_COL     = (220, 220, 220)
CENTER_COL   = (255, 220,  60)
SIDEBAR_BG   = (26,  26,  34)
TEXT_COL     = (180, 180, 190)
HINT_COL     = (100, 100, 115)
ACCENT_COL   = (50,  130, 220)
BTN_COL      = (50,  130, 220)
BTN_HOV      = (70,  160, 255)
BTN_TXT      = (255, 255, 255)
OK_COL       = (80,  200, 120)
ERR_COL      = (220,  80,  80)
GRID_COL     = (28,  28,  38)
CAR_BTN_COL  = (160,  60, 220)
CAR_BTN_HOV  = (190,  90, 255)
WARN_COL     = (230, 160,  40)
AI_BTN_COL   = ( 30, 160, 120)
AI_BTN_HOV   = ( 50, 200, 150)
AI_BTN_ACT   = (220, 160,  30)

# ── Car physics constants ──────────────────────────────────────
FPS          = 60
CAR_SCALE    = 1.0
WHEELBASE    = 28.0
MAX_SPEED    = 180.0
AI_MAX_SPEED = 480.0   # faster in AI training mode
MAX_STEER    = math.radians(28)
STEER_RATE   = math.radians(90)
STEER_RTN    = math.radians(150)
ACCEL        = 110.0
AI_ACCEL     = 280.0   # stronger acceleration in AI mode
BRAKE_F      = 280.0
DRAG_K       = 1.6
AI_SUBSTEPS  = 6       # physics steps per render frame in AI mode

# Car drawing size (half-lengths in px)
CAR_FL, CAR_RL, CAR_HW = 12, 9, 5   # front-len, rear-len, half-width

# ── Road-boundary helper ───────────────────────────────────────

def point_segment_dist(px, py, ax, ay, bx, by):
    """Closest distance from point P to segment AB, and the foot point."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay), (ax, ay)
    t = max(0.0, min(1.0, ((px - ax)*dx + (py - ay)*dy) / (dx*dx + dy*dy)))
    fx, fy = ax + t*dx, ay + t*dy
    return math.hypot(px - fx, py - fy), (fx, fy)


def closest_point_on_polyline(px, py, pts):
    """Return (min_dist, foot_x, foot_y) from point to polyline."""
    best = (1e18, px, py)
    for i in range(len(pts) - 1):
        d, (fx, fy) = point_segment_dist(px, py, *pts[i], *pts[i+1])
        if d < best[0]:
            best = (d, fx, fy)
    return best


def enforce_road_boundary(car, left_pts, right_pts, margin=6):
    """Push the car back if it has crossed either road edge."""
    if not left_pts or not right_pts:
        return

    for edge_pts in (left_pts, right_pts):
        d, fx, fy = closest_point_on_polyline(car.x, car.y, edge_pts)
        if d < margin:
            # push car away from the edge along the normal
            if d > 0.001:
                nx = (car.x - fx) / d
                ny = (car.y - fy) / d
            else:
                nx, ny = 0.0, -1.0
            push = margin - d + 1
            car.x += nx * push
            car.y += ny * push
            # cancel velocity component into the wall
            car.speed *= 0.3


# ── Ray-casting ────────────────────────────────────────────────

RAY_MAX_DIST  = 220    # px  – maximum ray length on the map canvas
RAY_N         = 10     # rays per side
RAY_FOV_FRONT = 160    # total front arc degrees
RAY_FOV_BACK  = 160    # total rear  arc degrees


def _ray_seg_intersect(ox, oy, dx, dy, ax, ay, bx, by):
    """Return t >= 0 if ray (O+t*D) hits segment AB, else None."""
    ex, ey = bx - ax, by - ay
    denom  = dx * ey - dy * ex
    if abs(denom) < 1e-10:
        return None
    t = ((ax - ox) * ey - (ay - oy) * ex) / denom
    u = ((ax - ox) * dy - (ay - oy) * dx) / denom
    if t >= 0 and 0.0 <= u <= 1.0:
        return t
    return None


def _cast_single_ray(ox, oy, angle, left_pts, right_pts,
                     max_dist=RAY_MAX_DIST):
    """Cast one ray; return (dist, end_x, end_y, frac)."""
    dx, dy   = math.cos(angle), math.sin(angle)
    best_t   = max_dist
    for pts in (left_pts, right_pts):
        for i in range(len(pts) - 1):
            t = _ray_seg_intersect(ox, oy, dx, dy,
                                   pts[i][0], pts[i][1],
                                   pts[i+1][0], pts[i+1][1])
            if t is not None and 0 < t < best_t:
                best_t = t
    ex   = ox + dx * best_t
    ey   = oy + dy * best_t
    frac = best_t / max_dist          # 0=wall touching, 1=open air
    return best_t, ex, ey, frac


def compute_rays(car, left_pts, right_pts):
    """
    Return list of ray dicts.  First RAY_N are front, last RAY_N are rear.
    Each dict: rel_angle, dist, end_x, end_y, frac
    """
    if not left_pts or not right_pts:
        return []
    rays = []
    h    = car.heading
    # Front: spread evenly over front arc, centred on heading
    for i in range(RAY_N):
        off  = math.radians(-RAY_FOV_FRONT/2 + i * RAY_FOV_FRONT / (RAY_N - 1))
        dist, ex, ey, frac = _cast_single_ray(
            car.x, car.y, h + off, left_pts, right_pts)
        rays.append(dict(rel_angle=off, dist=dist,
                         end_x=ex, end_y=ey, frac=frac))
    # Rear: spread over rear arc (centred on heading+180°)
    for i in range(RAY_N):
        off  = math.radians(180 - RAY_FOV_BACK/2 + i * RAY_FOV_BACK / (RAY_N - 1))
        dist, ex, ey, frac = _cast_single_ray(
            car.x, car.y, h + off, left_pts, right_pts)
        rays.append(dict(rel_angle=off, dist=dist,
                         end_x=ex, end_y=ey, frac=frac))
    return rays


def _ray_color(frac):
    """Green(far) → Orange(medium) → Red(close)."""
    if frac < 0.30:
        return (220,  50,  50)   # red
    elif frac < 0.60:
        # lerp orange between 0.30-0.60
        t = (frac - 0.30) / 0.30
        return (220, int(50 + 110*t), 40)
    else:
        # lerp green between 0.60-1.00
        t = (frac - 0.60) / 0.40
        return (int(220 - 160*t), int(160 + 40*t), 40)


def draw_rays_map(surf, car, rays):
    """Draw rays on the map canvas (world-space coords)."""
    cx, cy = int(car.x), int(car.y)
    for ray in rays:
        col = _ray_color(ray['frac'])
        ex, ey = int(ray['end_x']), int(ray['end_y'])
        pygame.draw.line(surf, (*col, 140), (cx, cy), (ex, ey), 1)
        # dot at the hit point
        pygame.draw.circle(surf, col, (ex, ey), 2)


# ── Car class (self-contained, scaled for the map canvas) ─────

class MapCar:
    def __init__(self, x, y, heading=0.0):
        self.x       = float(x)
        self.y       = float(y)
        self.heading = float(heading)
        self.speed   = 0.0
        self.steer   = 0.0
        self.thr     = 0.0
        self.brake   = False

    def update(self, thr, brake, steer_dir, dt, ai_mode=False):
        self.thr   = thr
        self.brake = brake
        ms = AI_MAX_SPEED if ai_mode else MAX_SPEED
        ac = AI_ACCEL    if ai_mode else ACCEL

        speed_factor = min(1.0, abs(self.speed) / 20.0)
        target = steer_dir * MAX_STEER
        diff   = target - self.steer
        rate   = STEER_RATE * speed_factor if abs(steer_dir) > 0.01 \
                 else STEER_RTN * max(0.1, speed_factor)
        self.steer += math.copysign(min(abs(diff), rate * dt), diff)

        if brake:
            if self.speed > 0:
                self.speed = max(0, self.speed - BRAKE_F * dt)
            elif self.speed < 0:
                self.speed = min(0, self.speed + BRAKE_F * dt)
        else:
            if thr > 0.01:
                self.speed += thr * ac * dt
            elif thr < -0.01:
                self.speed -= abs(thr) * ac * 0.5 * dt
            drag = DRAG_K * (self.speed**2) / ms
            total_drag = (drag + 15.0) * dt
            if self.speed > 0:
                self.speed = max(0, self.speed - total_drag)
            elif self.speed < 0:
                self.speed = min(0, self.speed + total_drag)

        self.speed = max(-ms * 0.35, min(ms, self.speed))

        if abs(self.speed) > 0.1 and abs(self.steer) > 0.0005:
            R = WHEELBASE / math.tan(self.steer)
            self.heading += (self.speed / R) * dt

        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt

    def draw(self, surf):
        h  = self.heading
        cx, cy = int(self.x), int(self.y)

        def rot(lx, ly):
            c, s = math.cos(h), math.sin(h)
            return (cx + lx*c - ly*s, cy + lx*s + ly*c)

        def poly(pts, col, w=0):
            ip = [(int(x), int(y)) for x, y in pts]
            if len(ip) >= 3:
                if w: pygame.draw.polygon(surf, col, ip, w)
                else:  pygame.draw.polygon(surf, col, ip)

        FL, RL, HW = CAR_FL, CAR_RL, CAR_HW

        # rear wing
        rx = -(RL + 3)
        poly([rot(rx,-8), rot(rx,8), rot(rx+2,7), rot(rx+2,-7)], (18,18,28))

        # body
        body = [
            rot(FL, -2), rot(FL, 2),
            rot(FL-4, HW+1), rot(FL-9, HW+2),
            rot(0, HW), rot(-RL+4, HW+2), rot(-RL, HW),
            rot(-RL, -HW), rot(-RL+4, -HW-2), rot(0, -HW),
            rot(FL-9, -HW-2), rot(FL-4, -HW-1),
        ]
        poly(body, (210, 0, 35))
        poly([rot(FL-1,-1.5), rot(FL-1,1.5), rot(-RL+3,3.5), rot(-RL+3,-3.5)], (230, 25, 55))

        # front wing
        fx = FL + 2
        poly([rot(fx,-10), rot(fx,10), rot(fx+2,9), rot(fx+2,-9)], (18,18,28))

        # cockpit
        poly([rot(5,-2.5), rot(5,2.5), rot(-1,2), rot(-1,-2)], (25,25,38))
        poly([rot(4,-3), rot(4,3), rot(1,2.5), rot(1,-2.5)], (200,175,50))

        # wheels  (tiny rectangles)
        WL, WW = 4, 2
        wheel_defs = [
            (FL-1, -(HW+3), self.steer),
            (FL-1,  (HW+3), self.steer),
            (-RL+2, -(HW+3), 0.0),
            (-RL+2,  (HW+3), 0.0),
        ]
        for lx, ly, sang in wheel_defs:
            wc  = rot(lx, ly)
            wa  = h + sang
            cw, sw = math.cos(wa), math.sin(wa)
            pts = [(wc[0]+px*cw - py*sw, wc[1]+px*sw + py*cw)
                   for px, py in [(-WL,-WW),(WL,-WW),(WL,WW),(-WL,WW)]]
            poly(pts, (32, 32, 42))
            poly(pts, (90, 90, 110), 1)

        # velocity arrow
        if abs(self.speed) > 3:
            s  = 1 if self.speed > 0 else -1
            al = min(abs(self.speed) * 0.12, 18)
            ax = cx + math.cos(h) * s * al
            ay = cy + math.sin(h) * s * al
            pygame.draw.line(surf, (255, 200, 50),
                             (cx, cy), (int(ax), int(ay)), 1)


# ── Shared-state helpers (car.py subprocess) ──────────────────

_car_proc        = None
_shared_state_path = None


def _launch_car_gui():
    """Start car.py as a subprocess in shared-display mode."""
    global _car_proc, _shared_state_path
    if _car_proc and _car_proc.poll() is None:
        _car_proc.terminate()
    fd, _shared_state_path = tempfile.mkstemp(suffix=".json", prefix="car_state_")
    os.close(fd)
    # Write a placeholder so car.py can open the file immediately
    with open(_shared_state_path, "w") as f:
        json.dump({"heading": 0.0, "speed": 0.0, "steer": 0.0,
                   "thr": 0.0, "brake": False}, f)
    car_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "car.py")
    _car_proc = subprocess.Popen([sys.executable, car_script,
                                  "--shared", _shared_state_path])


def _write_shared_state(car, rays=None):
    """Dump current car physics state + ray data to the shared temp file."""
    if _shared_state_path is None:
        return
    try:
        ray_payload = []
        if rays:
            for r in rays:
                ray_payload.append([round(r['rel_angle'], 5),
                                    round(r['frac'],      4)])
        with open(_shared_state_path, "w") as f:
            json.dump({
                "heading": car.heading,
                "speed":   car.speed,
                "steer":   car.steer,
                "thr":     car.thr,
                "brake":   car.brake,
                "rays":    ray_payload,
            }, f)
    except OSError:
        pass


def _kill_car_gui():
    global _car_proc
    if _car_proc and _car_proc.poll() is None:
        _car_proc.terminate()
    _car_proc = None


# ── Drawing helpers ────────────────────────────────────────────

def draw_dashed_line(surf, color, pts, dash=14, gap=8, width=2):
    if len(pts) < 2:
        return
    accumulated = 0.0
    drawing = True
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        seg_len = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
        if seg_len == 0:
            continue
        dx, dy = (bx - ax) / seg_len, (by - ay) / seg_len
        pos = 0.0
        while pos < seg_len:
            remaining = seg_len - pos
            chunk = dash if drawing else gap
            step = min(chunk - accumulated, remaining)
            if drawing:
                sx = ax + dx * pos; sy = ay + dy * pos
                ex = ax + dx * (pos + step); ey = ay + dy * (pos + step)
                pygame.draw.line(surf, color,
                                 (int(sx), int(sy)), (int(ex), int(ey)), width)
            accumulated += step
            pos += step
            if accumulated >= chunk:
                accumulated = 0.0
                drawing = not drawing


def draw_grid(surf):
    spacing = 30
    for gx in range(0, CANVAS_W, spacing):
        for gy in range(0, WIN_H, spacing):
            pygame.draw.circle(surf, GRID_COL, (gx, gy), 1)


def draw_road(surf, data):
    left   = [tuple(p) for p in data.get("left",   [])]
    right  = [tuple(p) for p in data.get("right",  [])]
    center = [tuple(p) for p in data.get("center", [])]

    if len(left) >= 2 and len(right) >= 2:
        poly = left + list(reversed(right))
        pygame.draw.polygon(surf, ROAD_SURF,
                            [(int(x), int(y)) for x, y in poly])

    if len(left)   >= 2:
        pygame.draw.lines(surf, EDGE_COL, False,
                          [(int(x), int(y)) for x, y in left], 2)
    if len(right)  >= 2:
        pygame.draw.lines(surf, EDGE_COL, False,
                          [(int(x), int(y)) for x, y in right], 2)

    if len(center) >= 2:
        draw_dashed_line(surf, CENTER_COL, center)


# ── File dialog helpers ────────────────────────────────────────

def open_file_dialog():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    filepath = filedialog.askopenfilename(
        title="Open Map File",
        filetypes=[("Map JSON", "*.json"), ("All files", "*.*")],
    )
    root.destroy()
    return filepath or None


def load_map(filepath):
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        required = {"center", "left", "right"}
        missing  = required - data.keys()
        if missing:
            return None, f"Missing keys: {', '.join(missing)}"
        if not data["center"]:
            return None, "Map has no road data."
        return data, None
    except json.JSONDecodeError as e:
        return None, f"JSON error: {e}"
    except OSError as e:
        return None, f"File error: {e}"


def spawn_car_on_road(map_data):
    """Pick a random center-line segment, spawn car there, heading along the road."""
    center = [tuple(p) for p in map_data.get("center", [])]
    if len(center) < 2:
        return None
    idx = random.randint(0, len(center) - 2)
    ax, ay = center[idx]
    bx, by = center[idx + 1]
    t  = random.random()
    sx = ax + t * (bx - ax)
    sy = ay + t * (by - ay)
    # heading along the road segment
    heading = math.atan2(by - ay, bx - ax)
    return MapCar(sx, sy, heading)


# ── Sidebar button rects (updated each frame) ──────────────────
OPEN_BTN = pygame.Rect(0, 0, 1, 1)
CAR_BTN  = pygame.Rect(0, 0, 1, 1)
AI_BTN   = pygame.Rect(0, 0, 1, 1)

# ── Training state globals ───────────────────────────────────
_trainer           = None      # PPOTrainer instance
_ai_active         = False     # True while AI is driving
_dashboard_proc    = None      # dashboard.py subprocess
_training_state_path = None    # shared JSON for dashboard

# ── Periodic checkpoint saving ──────────────────────────────
SAVE_EVERY_EPISODES = 150   # rotate spawn point & auto-save checkpoint this often
_last_save_episode   = 0
_ai_spawn          = None      # fixed (x, y, heading) used for AI episode resets

# ── Stall detection (ends episode if car barely moves for N seconds) ──
STALL_TIMEOUT  = 5.0    # seconds of (near-)no net movement before ending the episode
STALL_DIST_EPS = 4.0    # px — net movement below this doesn't reset the stall timer
_stall_x       = None
_stall_y       = None
_stall_timer   = 0.0


def _ask_model_version():
    """
    Tkinter dialog: pick any saved checkpoint from the full manifest,
    or start a Fresh Model.  Returns a PPOAgent or None (cancel).
    """
    if not HAS_MODEL:
        messagebox.showerror("Missing dependency",
                             "numpy is required for the AI model.\n"
                             "Run: pip install numpy")
        return None

    from model import load_manifest as _load_mf
    manifest = _load_mf()          # list of dicts, oldest → newest

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    chosen_path = {"val": None}    # None = cancelled, "__fresh__" = new agent

    # ── colour palette ──
    BG      = "#0d0d16"
    BG_CARD = "#14141f"
    BG_ROW  = "#1a1a2e"
    BG_SEL  = "#1e3a6e"
    FG_MAIN = "#eeeef8"
    FG_HINT = "#6b7280"
    FG_BLUE = "#3285e0"
    BORDER  = "#2a2a40"

    dlg = tk.Toplevel(root)
    dlg.title("Load AI Checkpoint")
    dlg.configure(bg=BG)
    dlg.resizable(True, True)
    dlg.minsize(560, 340)
    dlg.grab_set()

    # ── header ──
    tk.Label(dlg, text="SELECT CHECKPOINT",
             font=("Consolas", 12, "bold"),
             bg=BG, fg=FG_MAIN).pack(pady=(16, 2), padx=20, anchor="w")
    tk.Label(dlg, text="Pick a saved version to resume, or start fresh.",
             font=("Consolas", 9), bg=BG, fg=FG_HINT).pack(padx=20, anchor="w")
    tk.Frame(dlg, bg=BORDER, height=1).pack(fill=tk.X, padx=14, pady=8)

    # ── column headers ──
    hdr = tk.Frame(dlg, bg=BG_CARD)
    hdr.pack(fill=tk.X, padx=14)
    for text, w in [("#", 3), ("Label", 22), ("Episode", 9),
                    ("Best Reward", 12), ("Saved at", 16)]:
        tk.Label(hdr, text=text, font=("Consolas", 8, "bold"),
                 bg=BG_CARD, fg=FG_HINT,
                 width=w, anchor="w").pack(side=tk.LEFT, padx=(4, 0), pady=3)

    # ── listbox ──
    lb_frame = tk.Frame(dlg, bg=BG)
    lb_frame.pack(fill=tk.BOTH, expand=True, padx=14)

    sb = tk.Scrollbar(lb_frame, orient=tk.VERTICAL)
    lb = tk.Listbox(lb_frame, yscrollcommand=sb.set,
                    font=("Consolas", 10),
                    bg=BG_ROW, fg=FG_MAIN,
                    selectbackground=BG_SEL, selectforeground=FG_MAIN,
                    highlightbackground=BORDER, highlightthickness=1,
                    activestyle="none",
                    height=min(max(len(manifest), 3), 10))
    sb.config(command=lb.yview)
    sb.pack(side=tk.RIGHT, fill=tk.Y)
    lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # populate newest-first
    entries = list(reversed(manifest))
    if entries:
        for i, m in enumerate(entries):
            ep   = m.get("episode",     "?")
            best = m.get("best_reward", "?")
            ts   = m.get("timestamp",   "")
            lbl  = m.get("label",       m.get("file", "?"))
            best_str = (f"{float(best):>10.2f}"
                        if isinstance(best, (int, float)) else f"{'?':>10}")
            row = f"  {i+1:<3}  {lbl:<24}  ep={str(ep):<7}  {best_str}   {ts}"
            lb.insert(tk.END, row)
        lb.selection_set(0)
    else:
        lb.insert(tk.END, "   (no saved checkpoints found)")
        lb.config(state=tk.DISABLED)

    # ── file path hint below list ──
    info_var = tk.StringVar(
        value=f"File: {entries[0].get('file','')}" if entries else "")
    tk.Label(dlg, textvariable=info_var,
             font=("Consolas", 8), bg=BG, fg=FG_HINT,
             anchor="w").pack(fill=tk.X, padx=20, pady=(2, 0))

    def on_select(evt=None):
        sel = lb.curselection()
        if sel and entries:
            info_var.set(f"File: {entries[sel[0]].get('file','')}")

    lb.bind("<<ListboxSelect>>", on_select)

    # ── buttons ──
    tk.Frame(dlg, bg=BORDER, height=1).pack(fill=tk.X, padx=14, pady=6)
    btn_row = tk.Frame(dlg, bg=BG)
    btn_row.pack(fill=tk.X, padx=14, pady=(0, 14))

    def do_load():
        sel = lb.curselection()
        if not sel or not entries:
            return
        m    = entries[sel[0]]
        path = os.path.join(MODELS_DIR, m["file"])
        if not os.path.exists(path):
            messagebox.showerror("Not Found",
                                 f"File not found:\n{path}", parent=dlg)
            return
        chosen_path["val"] = path
        dlg.destroy()

    def do_fresh():
        chosen_path["val"] = "__fresh__"
        dlg.destroy()

    def do_cancel():
        dlg.destroy()

    lb.bind("<Double-Button-1>", lambda e: do_load())

    tk.Button(btn_row, text="⬇  Load Selected",
              font=("Consolas", 10, "bold"),
              bg=FG_BLUE, fg="white", relief=tk.FLAT,
              cursor="hand2", padx=12, pady=7,
              state=tk.NORMAL if entries else tk.DISABLED,
              command=do_load).pack(side=tk.LEFT, padx=(0, 6))

    tk.Button(btn_row, text="✦  Fresh Model",
              font=("Consolas", 10, "bold"),
              bg="#1e7a50", fg="white", relief=tk.FLAT,
              cursor="hand2", padx=12, pady=7,
              command=do_fresh).pack(side=tk.LEFT, padx=(0, 6))

    tk.Button(btn_row, text="Cancel",
              font=("Consolas", 9), bg="#1a1a2e", fg=FG_HINT,
              relief=tk.FLAT, cursor="hand2", padx=10, pady=7,
              command=do_cancel).pack(side=tk.RIGHT)

    dlg.protocol("WM_DELETE_WINDOW", do_cancel)
    root.wait_window(dlg)
    root.destroy()

    val = chosen_path["val"]
    if val is None:
        return None                   # cancelled
    if val == "__fresh__":
        return PPOAgent()             # brand-new agent
    try:
        return PPOAgent.load_version(val)
    except Exception as e:
        messagebox.showerror("Load Error",
                             f"Failed to load checkpoint:\n{e}")
        return PPOAgent()             # fall back to fresh


def _launch_dashboard(state_path):
    """Start dashboard.py as a detached subprocess."""
    global _dashboard_proc
    if _dashboard_proc and _dashboard_proc.poll() is None:
        return  # already running
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.py")
    if os.path.exists(script):
        _dashboard_proc = subprocess.Popen(
            [sys.executable, script, "--state", state_path,
             "--models", MODELS_DIR]
        )


def _kill_dashboard():
    global _dashboard_proc
    if _dashboard_proc and _dashboard_proc.poll() is None:
        _dashboard_proc.terminate()
    _dashboard_proc = None


def draw_sidebar(surf, font, small_font, map_data, filename,
                 status, car_active, ai_active=False):
    global OPEN_BTN, CAR_BTN, AI_BTN
    x0 = CANVAS_W

    pygame.draw.rect(surf, SIDEBAR_BG, (x0, 0, SIDEBAR_W, WIN_H))
    pygame.draw.line(surf, (60, 60, 75), (x0, 0), (x0, WIN_H), 1)

    # ── title ──
    surf.blit(font.render("MAP VIEWER", True, TEXT_COL), (x0 + 14, 20))

    # ── map info ──
    if map_data:
        tag_col, tag_text = OK_COL, "MAP LOADED"
    else:
        tag_col, tag_text = (200, 160, 60), "NO MAP"
    surf.blit(small_font.render(tag_text, True, tag_col), (x0 + 14, 48))

    y = 82
    if map_data:
        stats = [
            ("File",       filename[:18] + ("…" if len(filename) > 18 else "")),
            ("Center pts", str(len(map_data.get("center", [])))),
            ("Lane W",     str(map_data.get("lane_width", "?")) + " px"),
            ("Canvas",     f"{map_data.get('canvas', {}).get('width', '?')}"
                           f"×{map_data.get('canvas', {}).get('height', '?')}"),
        ]
    else:
        stats = [("File","—"), ("Center pts","—"), ("Lane W","—"), ("Canvas","—")]

    for label, val in stats:
        surf.blit(small_font.render(label, True, HINT_COL), (x0 + 14, y))
        surf.blit(small_font.render(val,   True, TEXT_COL), (x0 + 14, y + 16))
        y += 42

    # ── Open button ──
    OPEN_BTN = pygame.Rect(x0 + 14, y + 6, SIDEBAR_W - 28, 34)
    mx, my   = pygame.mouse.get_pos()
    hov      = OPEN_BTN.collidepoint(mx, my)
    pygame.draw.rect(surf, BTN_HOV if hov else BTN_COL, OPEN_BTN, border_radius=6)
    lbl = font.render("📂  OPEN MAP", True, BTN_TXT)
    surf.blit(lbl, (OPEN_BTN.centerx - lbl.get_width() // 2,
                    OPEN_BTN.centery - lbl.get_height() // 2))

    # ── Import Car button ──
    CAR_BTN = pygame.Rect(x0 + 14, OPEN_BTN.bottom + 10, SIDEBAR_W - 28, 34)
    if map_data:
        c_col = CAR_BTN_HOV if CAR_BTN.collidepoint(mx, my) else CAR_BTN_COL
        alpha_surf = pygame.Surface((CAR_BTN.width, CAR_BTN.height), pygame.SRCALPHA)
        pygame.draw.rect(alpha_surf, (*c_col, 255), alpha_surf.get_rect(), border_radius=6)
        surf.blit(alpha_surf, CAR_BTN.topleft)
        if car_active:
            car_lbl = font.render("🔄  RESPAWN CAR", True, BTN_TXT)
        else:
            car_lbl = font.render("🚗  IMPORT CAR", True, BTN_TXT)
        surf.blit(car_lbl, (CAR_BTN.centerx - car_lbl.get_width() // 2,
                            CAR_BTN.centery - car_lbl.get_height() // 2))
    else:
        pygame.draw.rect(surf, (50, 50, 65), CAR_BTN, border_radius=6)
        disabled_lbl = font.render("🚗  IMPORT CAR", True, (80, 80, 100))
        surf.blit(disabled_lbl, (CAR_BTN.centerx - disabled_lbl.get_width() // 2,
                                 CAR_BTN.centery - disabled_lbl.get_height() // 2))

    # ── AI TRAIN button ──
    AI_BTN = pygame.Rect(x0 + 14, CAR_BTN.bottom + 8, SIDEBAR_W - 28, 34)
    can_ai = car_active and HAS_MODEL
    if can_ai:
        if ai_active:
            ai_c = AI_BTN_ACT
            ai_txt = font.render("⏹  STOP AI", True, BTN_TXT)
        else:
            ai_c = AI_BTN_HOV if AI_BTN.collidepoint(mx, my) else AI_BTN_COL
            ai_txt = font.render("🤖  AI TRAIN", True, BTN_TXT)
        pygame.draw.rect(surf, ai_c, AI_BTN, border_radius=6)
        surf.blit(ai_txt, (AI_BTN.centerx - ai_txt.get_width() // 2,
                           AI_BTN.centery - ai_txt.get_height() // 2))
    else:
        pygame.draw.rect(surf, (50, 50, 65), AI_BTN, border_radius=6)
        ai_dis = font.render("🤖  AI TRAIN", True, (80, 80, 100))
        surf.blit(ai_dis, (AI_BTN.centerx - ai_dis.get_width() // 2,
                           AI_BTN.centery - ai_dis.get_height() // 2))

    # status below buttons
    status_y = AI_BTN.bottom + 6
    if status:
        sc = OK_COL if "Loaded" in status or "Spawned" in status else ERR_COL
        surf.blit(small_font.render(status, True, sc), (x0 + 14, status_y))

    # ── Car telemetry (shown when car is active) ──
    if car_active:
        ty = status_y + 22
        surf.blit(font.render("── CAR ──", True, ACCENT_COL), (x0 + 14, ty))
        ty += 18
        if ai_active:
            surf.blit(small_font.render("AI is driving…", True, AI_BTN_ACT), (x0 + 14, ty))
            ty += 14
            surf.blit(small_font.render("Click AI TRAIN to stop", True, HINT_COL), (x0 + 14, ty))
        else:
            surf.blit(small_font.render("WASD=drive  SPC=brake", True, HINT_COL), (x0 + 14, ty))
            ty += 14
            surf.blit(small_font.render("R = respawn car", True, HINT_COL), (x0 + 14, ty))
    else:
        ty = status_y + 22
        hints = ["CONTROLS", "", "O / Ctrl+O  open map",
                 "R           clear", "ESC         quit"]
        for h in hints:
            col  = TEXT_COL if h == "CONTROLS" else HINT_COL
            size = font if h == "CONTROLS" else small_font
            surf.blit(size.render(h, True, col), (x0 + 14, ty))
            ty += 20

    # version tag
    surf.blit(small_font.render("Map Reader v2.0", True, (70, 70, 90)),
              (x0 + 14, WIN_H - 24))


def draw_car_telemetry_overlay(surf, font, small_font, car):
    """Small HUD overlay on the canvas showing car speed / steer."""
    items = [
        f"Speed : {abs(car.speed):5.1f} px/s",
        f"Steer : {math.degrees(car.steer):+.1f}°",
        f"Hdg   : {math.degrees(car.heading) % 360:.1f}°",
    ]
    pad, lh = 8, 16
    bw = 170
    bh = pad * 2 + lh * len(items)
    bx, by = 10, 10
    overlay = pygame.Surface((bw, bh), pygame.SRCALPHA)
    overlay.fill((10, 10, 20, 180))
    surf.blit(overlay, (bx, by))
    for i, txt in enumerate(items):
        surf.blit(small_font.render(txt, True, (200, 200, 210)),
                  (bx + pad, by + pad + i * lh))


# ── Main ───────────────────────────────────────────────────────

def main():
    global _trainer, _ai_active, _training_state_path, _last_save_episode, \
           _ai_spawn, _stall_x, _stall_y, _stall_timer

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Autonomous Car Simulator – Map Viewer")
    clock  = pygame.time.Clock()

    font       = pygame.font.SysFont("consolas", 13, bold=True)
    small_font = pygame.font.SysFont("consolas", 12)

    canvas = pygame.Surface((CANVAS_W, WIN_H))

    map_data     = None
    filename     = ""
    status       = ""
    status_timer = 0
    car          = None   # MapCar instance or None

    def do_open_map(path):
        nonlocal map_data, filename, status, status_timer, car
        map_data, err = load_map(path)
        if map_data:
            filename = os.path.basename(path)
            status   = f"Loaded: {filename}"
            car      = None   # remove old car when new map loaded
        else:
            status = err or "Failed to load"
        status_timer = 240

    def do_spawn_car():
        nonlocal car, status, status_timer
        global _trainer, _ai_active ,_ai_spawn
        if map_data:
            car = spawn_car_on_road(map_data)
            if car:
                # Lock in this position as the fixed AI respawn point for this map
                _ai_spawn = (car.x, car.y, car.heading)
                status = "Spawned car on road!"
                _launch_car_gui()
                _trainer    = None
                _ai_active  = False
            else:
                status = "No center data!"
            status_timer = 240

    # ── CLI argument / immediate prompt ──
    if len(sys.argv) > 1:
        do_open_map(sys.argv[1])
    else:
        path = open_file_dialog()
        if path:
            do_open_map(path)

    left_pts  = []
    right_pts = []

    while True:
        dt = min(clock.tick(FPS) / 1000.0, 0.05)

        # read held keys for smooth car control
        keys = pygame.key.get_pressed()
        if car:
            thr_held   = (1.0 if keys[pygame.K_w] else
                         -1.0 if keys[pygame.K_s] else 0.0)
            str_held   = (-1.0 if keys[pygame.K_a] else
                           1.0 if keys[pygame.K_d] else 0.0)
            brake_held = bool(keys[pygame.K_SPACE])
        else:
            thr_held = str_held = 0.0
            brake_held = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                _kill_car_gui()
                pygame.quit(); sys.exit()

            elif event.type == pygame.KEYDOWN:
                ctrl = pygame.key.get_mods() & pygame.KMOD_CTRL

                if event.key == pygame.K_ESCAPE:
                    _kill_car_gui()
                    pygame.quit(); sys.exit()

                elif event.key == pygame.K_r:
                    if car:
                        do_spawn_car()   # respawn car
                    else:
                        map_data = None; filename = ""; status = ""; car = None

                elif event.key == pygame.K_o or (ctrl and event.key == pygame.K_o):
                    path = open_file_dialog()
                    if path:
                        do_open_map(path)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if OPEN_BTN.collidepoint(event.pos):
                        path = open_file_dialog()
                        if path:
                            do_open_map(path)
                    elif CAR_BTN.collidepoint(event.pos) and map_data:
                        do_spawn_car()
                    elif AI_BTN.collidepoint(event.pos) and car and HAS_MODEL:
                        if _ai_active:
                            # stop training
                            _ai_active = False
                            if _trainer:
                                _trainer.stats["running"] = False
                        else:
                            # ask model version now (only when starting AI)
                            agent = _ask_model_version()
                            if agent is not None:
                                global _training_state_path, _ai_spawn
                                fd, _training_state_path = tempfile.mkstemp(
                                    suffix=".json", prefix="train_state_")
                                os.close(fd)
                                with open(_training_state_path, "w") as _tf:
                                    json.dump({"running": False, "episode": 0,
                                               "ep_reward": 0.0, "best_reward": 0.0,
                                               "total_steps": 0, "reward_history": []}, _tf)
                                _trainer = PPOTrainer(agent)
                                _trainer.stats["running"] = True
                                _ai_active = True
                                # Lock the current car position as the fixed respawn point
                                _ai_spawn = (car.x, car.y, car.heading)
                                _stall_x, _stall_y, _stall_timer = car.x, car.y, 0.0
                                _launch_dashboard(_training_state_path)

            elif event.type == pygame.DROPFILE:
                do_open_map(event.file)

        # ── update car ──
        rays         = []
        crashed      = False
        signal_crash = False   # one-step-delayed crash flag for the trainer
        if car:
            if map_data:
                left_pts  = [tuple(p) for p in map_data.get("left",  [])]
                right_pts = [tuple(p) for p in map_data.get("right", [])]

            substeps = AI_SUBSTEPS if _ai_active else 1
            sub_dt   = dt / substeps

            for _ in range(substeps):
                # Recompute rays every substep so the policy sees current wall
                # distances.  Without this, rays stays [] for the whole frame
                # and the agent is completely blind to road geometry.
                if map_data and left_pts and right_pts:
                    rays = compute_rays(car, left_pts, right_pts)

                if _ai_active and _trainer:
                    thr_held, brake_held, str_held = _trainer.step(
                        rays, car.speed, car.steer, signal_crash, sub_dt,
                        training_state_path=_training_state_path,
                    )
                    signal_crash = False  # consumed — reset for next iteration

                    # ── Periodic checkpoint save + random respawn point ──
                    ep = _trainer.agent.episode
                    if ep > 0 and ep % SAVE_EVERY_EPISODES == 0 and ep != _last_save_episode:
                        _last_save_episode = ep
                        _trainer.agent.save_npz()
                        # Pick a new random spawn and lock it as the fixed point.
                        # Without updating _ai_spawn here, every crash would
                        # still go back to the ORIGINAL spawn, defeating the
                        # purpose of rotation.
                        rotated = spawn_car_on_road(map_data)
                        if rotated:
                            _ai_spawn = (rotated.x, rotated.y, rotated.heading)
                            rotated.speed = AI_MAX_SPEED * 0.25
                            car = rotated
                            _stall_x, _stall_y, _stall_timer = car.x, car.y, 0.0

                car.update(thr_held, brake_held, str_held, sub_dt,
                           ai_mode=_ai_active)
                old_x, old_y = car.x, car.y
                enforce_road_boundary(car, left_pts, right_pts, margin=6)
                crashed = (car.x != old_x or car.y != old_y)

                # ── Stall detection ──
                # If the car hasn't made meaningful net progress from its
                # last-checked position in STALL_TIMEOUT seconds, treat it
                # like a crash so the episode ends (gets the same -15 via
                # compute_reward's crashed flag) instead of idling forever
                # and wasting buffer steps on a policy that's just stuck.
                if _ai_active:
                    if _stall_x is None:
                        _stall_x, _stall_y, _stall_timer = car.x, car.y, 0.0
                    else:
                        moved = math.hypot(car.x - _stall_x, car.y - _stall_y)
                        if moved >= STALL_DIST_EPS:
                            _stall_x, _stall_y, _stall_timer = car.x, car.y, 0.0
                        else:
                            _stall_timer += sub_dt
                            if _stall_timer >= STALL_TIMEOUT:
                                crashed = True
                                _stall_timer = 0.0   # reset; new position set below on respawn

                if crashed and _ai_active and _trainer:
                    signal_crash = True
                    if _ai_spawn is not None:
                        sx, sy, sh = _ai_spawn
                        new_car = MapCar(sx, sy, sh)
                    else:
                        new_car = spawn_car_on_road(map_data)
                    if new_car:
                        new_car.speed = AI_MAX_SPEED * 0.25
                        car = new_car
                        _stall_x, _stall_y, _stall_timer = car.x, car.y, 0.0
                    crashed = False


            # ── poll save request OUTSIDE the substep loop ──
            # Must be here so PPOTrainer.step()'s file-write can't erase the
            # save_request that dashboard.py injected.
            if _ai_active and _trainer and _training_state_path:
                try:
                    with open(_training_state_path) as _sf:
                        _st = json.load(_sf)
                    req = _st.pop("save_request", None)
                    if req:
                        saved_path = _trainer.agent.save_npz(req)
                        # Put a confirmation back so dashboard sees it
                        _st["last_saved"] = req
                        with open(_training_state_path, "w") as _sf:
                            json.dump(_st, _sf)
                except Exception:
                    pass

            rays = compute_rays(car, left_pts, right_pts)
            _write_shared_state(car, rays)

        # ── status timer ──
        if status_timer > 0:
            status_timer -= 1
        else:
            status = ""

        # ── render ──
        canvas.fill(BG)
        draw_grid(canvas)

        if map_data:
            draw_road(canvas, map_data)
        else:
            msg = font.render("No map loaded — press O or click Open Map", True, HINT_COL)
            canvas.blit(msg, (CANVAS_W // 2 - msg.get_width() // 2,
                               WIN_H  // 2 - msg.get_height() // 2))

        if car:
            if rays:
                draw_rays_map(canvas, car, rays)
            car.draw(canvas)
            draw_car_telemetry_overlay(canvas, font, small_font, car)

        screen.blit(canvas, (0, 0))
        draw_sidebar(screen, font, small_font, map_data, filename,
                     status, car is not None, ai_active=_ai_active)

        pygame.display.flip()


if __name__ == "__main__":
    try:
        main()
    finally:
        _kill_dashboard()