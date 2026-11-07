"""
test_model.py  –  Autonomous Car Simulator  |  Model Tester
════════════════════════════════════════════════════════════
Load a trained PPO checkpoint and watch it drive on any map.
No training happens — pure inference / deterministic mode.

Usage
─────
  python test_model.py [map.json]

Controls
────────
  O / Ctrl+O   open another map
  R            respawn car (start new test episode)
  ESC          quit
  ── When model is active ──
  Click  "Open Map"    – load .json map file
  Click  "Spawn Car"   – place car on the road
  Click  "Load Model"  – pick a checkpoint to drive
  Click  "Restart"     – new episode, same spawn
"""

import pygame
import sys
import os
import json
import math
import random
import tkinter as tk
from tkinter import filedialog, messagebox

# ── Shared simulation utilities from map_reader ─────────────────
from map_reader import (
    load_map, MapCar, spawn_car_on_road, compute_rays,
    enforce_road_boundary, draw_road, draw_rays_map, draw_grid, track_state,
    open_file_dialog,
    WIN_W, WIN_H, CANVAS_W, SIDEBAR_W, FPS, AI_SUBSTEPS, AI_MAX_SPEED,
    BG, ROAD_SURF, EDGE_COL, CENTER_COL,
    SIDEBAR_BG, TEXT_COL, HINT_COL, ACCENT_COL,
    BTN_COL, BTN_HOV, BTN_TXT, OK_COL, ERR_COL,
    GRID_COL, WARN_COL,
    STALL_TIMEOUT, STALL_DIST_EPS, SAFETY_OVERRIDE,
)

# ── PPO model ───────────────────────────────────────────────────
try:
    from model import PPOAgent, MODELS_DIR, MANIFEST_PATH
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False
    MODELS_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    MANIFEST_PATH = os.path.join(MODELS_DIR, "manifest.json")

# ── Sidebar layout ──────────────────────────────────────────────
TESTER_BTN_COL = (140, 60, 200)   # purple – distinct from trainer
TESTER_BTN_HOV = (175, 90, 240)
MODEL_ACT_COL  = (30, 180, 120)   # green when running

# ── Helpers ─────────────────────────────────────────────────────

def _load_manifest():
    try:
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def _pick_model_dialog():
    """Tkinter popup that lists available checkpoints and returns a path."""
    manifest = _load_manifest()
    if not manifest:
        messagebox.showwarning("No models", "No saved checkpoints found in models/.")
        return None

    # Sort newest first
    manifest = sorted(manifest, key=lambda m: m["timestamp"], reverse=True)

    root = tk.Tk()
    root.withdraw()
    dlg  = tk.Toplevel(root)
    dlg.title("Select Model Checkpoint")
    dlg.resizable(False, False)
    dlg.configure(bg="#1a1a22")

    tk.Label(dlg, text="Select a checkpoint to test:",
             bg="#1a1a22", fg="#e0e0f0",
             font=("Consolas", 11, "bold")).pack(pady=(12, 4), padx=16)

    frame = tk.Frame(dlg, bg="#1a1a22")
    frame.pack(fill="both", expand=True, padx=12, pady=4)

    sb  = tk.Scrollbar(frame, orient="vertical")
    lb  = tk.Listbox(frame, yscrollcommand=sb.set,
                     bg="#12121a", fg="#c8c8d8",
                     selectbackground="#3282dc", selectforeground="#ffffff",
                     font=("Consolas", 10), width=68, height=18,
                     borderwidth=0, highlightthickness=1,
                     highlightcolor="#3282dc")
    sb.config(command=lb.yview)
    sb.pack(side="right", fill="y")
    lb.pack(side="left", fill="both", expand=True)

    for m in manifest:
        lb.insert("end",
                  f"ep={m['episode']:<6} "
                  f"best={m['best_reward']:<10.2f} "
                  f"{m['timestamp']}  {m['file']}")

    chosen = [None]

    def _ok():
        sel = lb.curselection()
        if sel:
            chosen[0] = os.path.join(MODELS_DIR, manifest[sel[0]]["file"])
        dlg.destroy()

    def _browse():
        p = filedialog.askopenfilename(
            title="Open checkpoint", initialdir=MODELS_DIR,
            filetypes=[("NumPy archive", "*.npz")])
        if p:
            chosen[0] = p
        dlg.destroy()

    btn_frame = tk.Frame(dlg, bg="#1a1a22")
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="Load Selected", command=_ok,
              bg="#3282dc", fg="white", font=("Consolas", 10, "bold"),
              relief="flat", padx=12, pady=4).pack(side="left", padx=6)
    tk.Button(btn_frame, text="Browse…", command=_browse,
              bg="#505060", fg="white", font=("Consolas", 10),
              relief="flat", padx=12, pady=4).pack(side="left", padx=6)
    tk.Button(btn_frame, text="Cancel",
              command=dlg.destroy,
              bg="#303040", fg="#b0b0c0", font=("Consolas", 10),
              relief="flat", padx=12, pady=4).pack(side="left", padx=6)

    root.wait_window(dlg)
    root.destroy()
    return chosen[0]


# ── Sidebar drawing ─────────────────────────────────────────────

def _draw_btn(surf, rect, label, font, hovered=False, active=False,
              col=None, hov=None, act_col=None):
    if col  is None: col     = BTN_COL
    if hov  is None: hov     = BTN_HOV
    if act_col is None: act_col = col
    c = act_col if active else (hov if hovered else col)
    pygame.draw.rect(surf, c, rect, border_radius=5)
    txt = font.render(label, True, BTN_TXT)
    surf.blit(txt, (rect.centerx - txt.get_width() // 2,
                    rect.centery - txt.get_height() // 2))
    return rect


def _section(surf, font, x, y, w, title, items, title_col=None):
    """Draw a titled block of key-value rows."""
    if title_col is None: title_col = ACCENT_COL
    lh  = 18
    pad = 8
    bh  = pad + lh + pad * 0.5 + len(items) * lh + pad
    bx, by = x, y

    overlay = pygame.Surface((w, int(bh)), pygame.SRCALPHA)
    overlay.fill((14, 14, 22, 200))
    surf.blit(overlay, (bx, by))
    pygame.draw.rect(surf, title_col, (bx, by, w, 2), border_radius=2)

    tf = pygame.font.SysFont("consolas", 12, bold=True)
    surf.blit(tf.render(title, True, title_col),
              (bx + pad, by + pad))

    sf = pygame.font.SysFont("consolas", 11)
    for i, (k, v, vc) in enumerate(items):
        iy = int(by + pad + lh + lh * i)
        surf.blit(sf.render(f"{k}", True, HINT_COL),      (bx + pad, iy))
        surf.blit(sf.render(str(v),  True, vc or TEXT_COL),
                  (bx + pad + 90, iy))

    return int(bh)


def draw_sidebar_test(screen, font, small_font,
                      map_loaded, filename, model_name,
                      status, car_active, model_active,
                      ep_num, ep_steps, ep_reward, ep_dist,
                      best_steps, best_reward, history,
                      mouse_pos, OPEN_BTN, SPAWN_BTN,
                      MODEL_BTN, RESTART_BTN):
    x0  = CANVAS_W
    w   = SIDEBAR_W
    h   = WIN_H
    pad = 10

    pygame.draw.rect(screen, SIDEBAR_BG, (x0, 0, w, h))
    pygame.draw.line(screen, (40, 40, 55), (x0, 0), (x0, h), 2)

    y = pad

    # Title
    title = font.render("MODEL TESTER", True, (200, 160, 255))
    screen.blit(title, (x0 + w // 2 - title.get_width() // 2, y))
    y += title.get_height() + 8
    pygame.draw.line(screen, (60, 40, 90), (x0 + pad, y), (x0 + w - pad, y))
    y += 8

    # ── Buttons ──
    mx, my = mouse_pos

    OPEN_BTN.x  = x0 + pad;  OPEN_BTN.y  = y
    OPEN_BTN.width = w - pad*2; OPEN_BTN.height = 26
    _draw_btn(screen, OPEN_BTN,  "Open Map",  font,
              hovered=OPEN_BTN.collidepoint(mx, my))
    y += 32

    SPAWN_BTN.x = x0 + pad; SPAWN_BTN.y = y
    SPAWN_BTN.width = w - pad*2; SPAWN_BTN.height = 26
    _draw_btn(screen, SPAWN_BTN, "Spawn Car", font,
              hovered=SPAWN_BTN.collidepoint(mx, my),
              col=(160, 60, 220), hov=(195, 90, 255))
    y += 32

    MODEL_BTN.x = x0 + pad; MODEL_BTN.y = y
    MODEL_BTN.width = w - pad*2; MODEL_BTN.height = 26
    _draw_btn(screen, MODEL_BTN, "Load Model", font,
              hovered=MODEL_BTN.collidepoint(mx, my),
              col=(50, 140, 90), hov=(70, 180, 110),
              active=model_active, act_col=(220, 150, 30))
    y += 32

    RESTART_BTN.x = x0 + pad; RESTART_BTN.y = y
    RESTART_BTN.width = w - pad*2; RESTART_BTN.height = 26
    _draw_btn(screen, RESTART_BTN, "Restart",   font,
              hovered=RESTART_BTN.collidepoint(mx, my),
              col=(180, 50, 50), hov=(220, 70, 70))
    y += 38

    pygame.draw.line(screen, (40, 40, 55), (x0 + pad, y), (x0 + w - pad, y))
    y += 8

    # ── Map / model info ──
    mf = pygame.font.SysFont("consolas", 11)
    def info_row(label, val, col=TEXT_COL):
        nonlocal y
        screen.blit(mf.render(label, True, HINT_COL), (x0 + pad, y))
        screen.blit(mf.render(str(val),  True, col),  (x0 + pad + 68, y))
        y += 16

    info_row("Map:",   (filename or "—")[:16])
    info_row("Model:", (model_name or "—")[:16],
             col=(OK_COL if model_active else TEXT_COL))
    y += 4

    pygame.draw.line(screen, (40, 40, 55), (x0 + pad, y), (x0 + w - pad, y))
    y += 8

    # ── Episode stats ──
    r_col = OK_COL if ep_reward >= 0 else ERR_COL
    items = [
        ("Episode",   ep_num,           None),
        ("Steps",     ep_steps,         None),
        ("Reward",    f"{ep_reward:.1f}", r_col),
        ("Distance",  f"{ep_dist:.0f}px", None),
        ("Best steps",best_steps,       OK_COL),
        ("Best rew",  f"{best_reward:.1f}", OK_COL),
    ]
    bh = _section(screen, font, x0 + pad, y, w - pad*2,
                  "CURRENT EPISODE", items, title_col=(100, 200, 255))
    y += bh + 8

    # ── Reward history sparkline ──
    if history:
        section_h = 52
        bx, by = x0 + pad, y
        bw = w - pad*2
        overlay = pygame.Surface((bw, section_h), pygame.SRCALPHA)
        overlay.fill((14, 14, 22, 200))
        screen.blit(overlay, (bx, by))
        pygame.draw.rect(screen, (80, 80, 100), (bx, by, bw, 2), border_radius=2)
        hdr = pygame.font.SysFont("consolas", 12, bold=True)
        screen.blit(hdr.render("REWARD HISTORY", True, (80, 80, 100)), (bx + 4, by + 4))

        n = min(len(history), bw - 4)
        sub = history[-n:]
        mn, mx_v = min(sub), max(sub)
        rng = (mx_v - mn) or 1.0
        spark_h = section_h - 22
        pts = []
        for i, rv in enumerate(sub):
            sx = bx + 2 + int(i * (bw - 4) / max(len(sub)-1, 1))
            sy = by + 20 + spark_h - int((rv - mn) / rng * (spark_h - 2))
            pts.append((sx, sy))
        if len(pts) >= 2:
            pygame.draw.lines(screen, (100, 200, 255), False, pts, 1)

        y += section_h + 8

    # ── Status bar ──
    if status:
        sf2 = pygame.font.SysFont("consolas", 11, bold=True)
        sc  = OK_COL if not status.startswith("!") else ERR_COL
        stxt = sf2.render(status, True, sc)
        screen.blit(stxt, (x0 + w // 2 - stxt.get_width() // 2, WIN_H - 20))


# ── Main ─────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Autonomous Car Simulator – Model Tester")
    clock  = pygame.time.Clock()

    font       = pygame.font.SysFont("consolas", 13, bold=True)
    small_font = pygame.font.SysFont("consolas", 12)
    canvas     = pygame.Surface((CANVAS_W, WIN_H))

    # ── State ──
    map_data   = None
    filename   = ""
    car        = None
    agent      = None
    model_name = ""
    model_active = False

    left_pts   = []
    right_pts  = []
    center_pts = []
    half_w     = 30.0

    # Episode tracking
    ep_num     = 0
    ep_steps   = 0
    ep_reward  = 0.0
    ep_dist    = 0.0
    ep_start_x = 0.0
    ep_start_y = 0.0
    best_steps = 0
    best_reward = -1e9
    reward_history = []

    # Stall detection
    stall_x = stall_y = stall_timer = 0.0

    spawn_pos  = None          # fixed spawn for this test session
    status     = ""
    status_timer = 0

    # Button rects (positioned by draw_sidebar_test each frame)
    OPEN_BTN    = pygame.Rect(0, 0, 1, 1)
    SPAWN_BTN   = pygame.Rect(0, 0, 1, 1)
    MODEL_BTN   = pygame.Rect(0, 0, 1, 1)
    RESTART_BTN = pygame.Rect(0, 0, 1, 1)

    rays = []
    hold_ctrl = None

    # ── Helpers ──
    def do_open_map(path):
        nonlocal map_data, filename, car, left_pts, right_pts
        nonlocal agent, model_active, model_name, status, status_timer
        nonlocal spawn_pos, center_pts, half_w
        map_data, err = load_map(path)
        if map_data:
            filename  = os.path.basename(path)
            left_pts  = [tuple(p) for p in map_data.get("left",  [])]
            right_pts = [tuple(p) for p in map_data.get("right", [])]
            center_pts = [tuple(p) for p in map_data.get("center", [])]
            half_w    = float(map_data.get("lane_width", 60)) / 2
            car       = None
            spawn_pos = None
            status    = f"Loaded: {filename}"
        else:
            status = err or "! Failed to load"
        status_timer = 240

    def do_spawn_car():
        nonlocal car, spawn_pos, status, status_timer
        nonlocal ep_num, ep_steps, ep_reward, ep_dist, stall_x, stall_y, stall_timer
        nonlocal ep_start_x, ep_start_y
        if not map_data:
            return
        c = spawn_car_on_road(map_data)
        if c:
            c.speed   = AI_MAX_SPEED * 0.25
            car       = c
            spawn_pos = (c.x, c.y, c.heading)
            # Reset episode
            ep_num    = 0
            ep_steps  = 0
            ep_reward = 0.0
            ep_dist   = 0.0
            ep_start_x, ep_start_y = c.x, c.y
            stall_x, stall_y, stall_timer = c.x, c.y, 0.0
            status       = "Car spawned!"
            status_timer = 180
        else:
            status       = "! No center data"
            status_timer = 180

    def do_load_model():
        nonlocal agent, model_name, model_active, status, status_timer
        path = _pick_model_dialog()
        if path:
            try:
                agent        = PPOAgent.load_npz(path)
                model_name   = os.path.basename(path).replace("ppo_", "").replace(".npz", "")
                model_active = True
                status       = f"Model loaded: {model_name[:20]}"
                status_timer = 240
            except Exception as e:
                status       = f"! Load failed: {e}"
                status_timer = 300
                agent        = None
                model_active = False

    def do_restart():
        nonlocal car, ep_num, ep_steps, ep_reward, ep_dist
        nonlocal stall_x, stall_y, stall_timer, ep_start_x, ep_start_y
        if spawn_pos is None:
            return
        sx, sy, sh = spawn_pos
        c         = MapCar(sx, sy, sh)
        c.speed   = AI_MAX_SPEED * 0.25
        nonlocal car
        car       = c
        ep_steps  = 0
        ep_reward = 0.0
        ep_dist   = 0.0
        ep_start_x, ep_start_y = c.x, c.y
        stall_x, stall_y, stall_timer = c.x, c.y, 0.0

    # ── CLI map ──
    if len(sys.argv) > 1:
        do_open_map(sys.argv[1])

    # ── Main loop ──
    while True:
        dt = min(clock.tick(FPS) / 1000.0, 0.05)

        mouse = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            elif event.type == pygame.KEYDOWN:
                ctrl = pygame.key.get_mods() & pygame.KMOD_CTRL
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                elif event.key == pygame.K_r:
                    do_restart()
                elif event.key == pygame.K_o or (ctrl and event.key == pygame.K_o):
                    p = open_file_dialog()
                    if p: do_open_map(p)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if OPEN_BTN.collidepoint(event.pos):
                    p = open_file_dialog()
                    if p: do_open_map(p)
                elif SPAWN_BTN.collidepoint(event.pos):
                    do_spawn_car()
                elif MODEL_BTN.collidepoint(event.pos) and HAS_MODEL:
                    do_load_model()
                elif RESTART_BTN.collidepoint(event.pos):
                    do_restart()

            elif event.type == pygame.DROPFILE:
                ext = os.path.splitext(event.file)[1].lower()
                if ext == ".json":
                    do_open_map(event.file)
                elif ext == ".npz" and HAS_MODEL:
                    try:
                        agent        = PPOAgent.load_npz(event.file)
                        model_name   = os.path.basename(event.file)
                        model_active = True
                        status       = f"Loaded: {model_name[:24]}"
                        status_timer = 240
                    except Exception as e:
                        status = f"! {e}"
                        status_timer = 240

        # ── Simulation step ──
        if car and left_pts and right_pts:
            sub_dt = dt / AI_SUBSTEPS
            for _sub in range(AI_SUBSTEPS):
                rays = compute_rays(car, left_pts, right_pts)

                # Frame-skip: decide once per frame (as in training), hold between.
                if _sub == 0 or hold_ctrl is None:
                    if agent and model_active:
                        obs    = agent.build_obs(rays, car.speed, car.steer)
                        # ── Greedy / deterministic inference ──
                        import numpy as np
                        logits = agent.actor.forward(obs)
                        action = int(np.argmax(logits))     # always best action
                        hold_ctrl = agent.action_to_controls(action)
                    else:
                        hold_ctrl = (0.0, False, 0.0)
                thr, brake, steer_cmd = hold_ctrl

                # Counter-steer safety override (same as trainer)
                if SAFETY_OVERRIDE and rays and steer_cmd != 0.0:
                    WALL_CRITICAL = 0.12
                    front = rays[:10]   # front arc only (RAY_N), same as trainer
                    if steer_cmd < 0:
                        side_fracs = [r['frac'] for r in front if r.get('rel_angle', 0.0) < 0]
                    else:
                        side_fracs = [r['frac'] for r in front if r.get('rel_angle', 0.0) > 0]
                    if side_fracs and min(side_fracs) < WALL_CRITICAL:
                        steer_cmd = -steer_cmd

                car.update(thr, brake, steer_cmd, sub_dt, ai_mode=True)
                old_x, old_y = car.x, car.y
                enforce_road_boundary(car, left_pts, right_pts, margin=6)
                crashed = (car.x != old_x or car.y != old_y)

                # Accumulate episode stats
                ep_steps += 1
                ep_dist  += math.hypot(car.x - ep_start_x,
                                       car.y - ep_start_y)
                ep_start_x, ep_start_y = car.x, car.y

                # Reward display: same formula as training (per decision = 1/AI_SUBSTEPS per substep)
                if len(center_pts) > 1:
                    lat_n, fwd = track_state(car, center_pts, half_w)
                    ms = agent.max_speed if agent else AI_MAX_SPEED
                    r_step = -15.0 if crashed else 3.0 * car.speed * fwd / ms - 2.0 * lat_n ** 2
                    ep_reward += r_step / AI_SUBSTEPS

                # Stall detection
                moved = math.hypot(car.x - stall_x, car.y - stall_y)
                if moved >= STALL_DIST_EPS:
                    stall_x, stall_y, stall_timer = car.x, car.y, 0.0
                else:
                    stall_timer += sub_dt
                    if stall_timer >= STALL_TIMEOUT:
                        crashed = True
                        stall_timer = 0.0

                if crashed:
                    # Record episode and reset
                    ep_num    += 1
                    reward_history.append(round(ep_reward, 1))
                    if ep_steps  > best_steps:  best_steps  = ep_steps
                    if ep_reward > best_reward: best_reward = ep_reward

                    if spawn_pos:
                        sx, sy, sh = spawn_pos
                        car        = MapCar(sx, sy, sh)
                        car.speed  = AI_MAX_SPEED * 0.25
                    else:
                        car = spawn_car_on_road(map_data) or car
                        car.speed = AI_MAX_SPEED * 0.25

                    ep_steps  = 0
                    ep_reward = 0.0
                    ep_dist   = 0.0
                    ep_start_x, ep_start_y = car.x, car.y
                    stall_x, stall_y, stall_timer = car.x, car.y, 0.0
                    rays = []
                    break   # restart substep loop with fresh car

        # ── Status timer ──
        if status_timer > 0:
            status_timer -= 1
        else:
            status = ""

        # ── Render ──
        canvas.fill(BG)
        draw_grid(canvas)

        if map_data:
            draw_road(canvas, map_data)
        else:
            msg = font.render("No map loaded — press O or drag a .json file here",
                              True, HINT_COL)
            canvas.blit(msg, (CANVAS_W // 2 - msg.get_width()  // 2,
                               WIN_H  // 2 - msg.get_height() // 2))

        if car:
            if rays:
                draw_rays_map(canvas, car, rays)
            car.draw(canvas)

            # Speed / steer overlay (top-left of canvas)
            of = pygame.font.SysFont("consolas", 11)
            spd_c = OK_COL if car.speed > 0 else HINT_COL
            for li, txt in enumerate([
                f"spd  {car.speed:>7.1f} px/s",
                f"str  {math.degrees(car.steer):>7.1f}°",
                f"step {ep_steps:>7}",
            ]):
                canvas.blit(of.render(txt, True, spd_c if li == 0 else TEXT_COL),
                            (8, 8 + li * 14))

        screen.blit(canvas, (0, 0))

        draw_sidebar_test(
            screen, font, small_font,
            map_loaded   = map_data is not None,
            filename     = filename,
            model_name   = model_name,
            status       = status,
            car_active   = car is not None,
            model_active = model_active,
            ep_num       = ep_num,
            ep_steps     = ep_steps,
            ep_reward    = ep_reward,
            ep_dist      = ep_dist,
            best_steps   = best_steps,
            best_reward  = best_reward if best_reward > -1e9 else 0.0,
            history      = reward_history,
            mouse_pos    = mouse,
            OPEN_BTN     = OPEN_BTN,
            SPAWN_BTN    = SPAWN_BTN,
            MODEL_BTN    = MODEL_BTN,
            RESTART_BTN  = RESTART_BTN,
        )

        pygame.display.flip()


if __name__ == "__main__":
    main()
