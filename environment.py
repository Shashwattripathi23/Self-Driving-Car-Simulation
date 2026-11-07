import pygame
import numpy as np
import sys
import json
import tkinter as tk
from tkinter import filedialog

# ── Window config ──────────────────────────────────────────────
WIN_W, WIN_H   = 960, 650
SIDEBAR_W      = 220
CANVAS_W       = WIN_W - SIDEBAR_W   # 740
LANE_WIDTH     = 60                  # total road width (px)
MIN_POINTS     = 20                  # minimum points to build a road
CLOSE_DIST     = 45                  # px: finishing the stroke within this of the start closes the loop
MIN_LOOP_LEN   = 250                 # px: shortest stroke that can count as a loop
RESAMPLE_STEP  = 6.0                 # px between points of the final smooth centre line
SMOOTH_PX      = 18.0                # px: size of the hand-jitter that gets smoothed away (shape is kept)

# ── Colours ────────────────────────────────────────────────────
BG          = (18,  18,  24)
ROAD_SURF   = (45,  45,  55)
EDGE_COL    = (220, 220, 220)
CENTER_COL  = (255, 220,  60)
RAW_COL     = (150, 150, 180)
CLOSE_COL   = (80,  200, 120)
SIDEBAR_BG  = (26,  26,  34)
TEXT_COL    = (180, 180, 190)
HINT_COL    = (100, 100, 115)
BTN_COL     = (50,  130, 220)
BTN_HOV     = (70,  160, 255)
BTN_DIS     = (50,   50,  65)
BTN_TXT     = (255, 255, 255)
SAVE_OK_COL = (80,  200, 120)
SAVE_ERR_COL= (220,  80,  80)


# ── Geometry helpers ───────────────────────────────────────────

def resample_closed(pts, step):
    """Resample a closed loop at (nearly) uniform arc length."""
    P = np.asarray(pts, dtype=float)
    Q = np.vstack([P, P[:1]])
    seg = np.hypot(np.diff(Q[:, 0]), np.diff(Q[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(round(s[-1] / step)), 16)
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.stack([np.interp(t, s, Q[:, 0]), np.interp(t, s, Q[:, 1])], axis=1)

def smooth_closed(P, sigma):
    """Circular Gaussian smoothing (sigma in samples). Wraps around the seam, so the
    start/finish joint gets blended like any other part of the loop."""
    n = len(P)
    r = int(3 * sigma) + 1
    k = np.arange(-r, r + 1)
    w = np.exp(-k ** 2 / (2.0 * sigma ** 2)); w /= w.sum()
    idx = (np.arange(n)[:, None] + k[None, :]) % n
    return (P[idx] * w[None, :, None]).sum(axis=1)

def radius_closed(P, k=3):
    """Radius of curvature at every sample of a closed, evenly sampled loop."""
    a, b, c = np.roll(P, k, axis=0), P, np.roll(P, -k, axis=0)
    cross = np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
    lens = (np.hypot(*(b - a).T) * np.hypot(*(c - b).T) * np.hypot(*(c - a).T))
    return lens / (2.0 * np.maximum(cross, 1e-9))

def smooth_road(raw, half_w):
    """Raw hand-drawn loop -> smooth, evenly sampled closed centre line with the same overall shape.
    1) resample evenly and low-pass away the hand jitter (periodic, so the seam is blended too);
    2) relax only the bends that are too tight for the road width, where the inner edge would
       fold into sharp shards; 3) resample again so the edges come out evenly spaced."""
    P = smooth_closed(resample_closed(raw, RESAMPLE_STEP), SMOOTH_PX / RESAMPLE_STEP)
    r_min = half_w + 4.0
    for _ in range(400):
        bad = radius_closed(P) < r_min
        if not bad.any():
            break
        mask = np.clip(smooth_closed(bad.astype(float)[:, None], 2.0)[:, 0] * 4.0, 0.0, 1.0)
        P = P + mask[:, None] * (smooth_closed(P, 2.0) - P)      # soft, local: shape elsewhere untouched
    return resample_closed(P, RESAMPLE_STEP)

def build_edges(P, half_w):
    """Left/right road edges of a closed centre line (offset along the local normal)."""
    d = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    ln = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)
    nx, ny = -d[:, 1] / ln, d[:, 0] / ln
    left  = np.stack([P[:, 0] + nx * half_w, P[:, 1] + ny * half_w], axis=1)
    right = np.stack([P[:, 0] - nx * half_w, P[:, 1] - ny * half_w], axis=1)
    return left, right

def draw_dashed_line(surf, color, pts, dash=14, gap=8, width=2):
    """Draw a polyline as a dashed line."""
    if len(pts) < 2:
        return
    accumulated = 0.0
    drawing = True
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        seg_len = np.hypot(bx - ax, by - ay)
        if seg_len == 0:
            continue
        dx, dy = (bx - ax) / seg_len, (by - ay) / seg_len
        pos = 0.0
        while pos < seg_len:
            remaining = seg_len - pos
            chunk = (dash if drawing else gap)
            step = min(chunk - accumulated, remaining)
            if drawing:
                sx = ax + dx * pos
                sy = ay + dy * pos
                ex = ax + dx * (pos + step)
                ey = ay + dy * (pos + step)
                pygame.draw.line(surf, color,
                                 (int(sx), int(sy)),
                                 (int(ex), int(ey)), width)
            accumulated += step
            pos += step
            if accumulated >= chunk:
                accumulated = 0.0
                drawing = not drawing


# ── Road class ─────────────────────────────────────────────────

class Road:
    def __init__(self):
        self.reset()

    def reset(self):
        self.raw_pts   = []
        self.center    = []
        self.left      = []
        self.right     = []
        self.finalized = False

    def add_point(self, pt):
        # avoid duplicate points
        if self.raw_pts and np.hypot(pt[0] - self.raw_pts[-1][0],
                                      pt[1] - self.raw_pts[-1][1]) < 3:
            return
        self.raw_pts.append(pt)

    def closure_index(self):
        """Index of the stroke point where the loop closes (None if the stroke does not
        come back to its start).  Looking at the tail lets an overshoot past the start
        count as closed too; the overshoot is trimmed off when the road is built."""
        n = len(self.raw_pts)
        if n < MIN_POINTS:
            return None
        P = np.asarray(self.raw_pts, dtype=float)
        p0 = P[0]
        length = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1])).sum()
        if length < MIN_LOOP_LEN or np.hypot(*(P - p0).T).max() < 2 * CLOSE_DIST:
            return None
        t0 = int(n * 0.65)
        d = np.hypot(*(P[t0:] - p0).T)
        return t0 + int(np.argmin(d)) if d.min() <= CLOSE_DIST else None

    def build(self):
        j = self.closure_index()
        if j is None:
            return False
        half_w = LANE_WIDTH / 2
        P = smooth_road(np.asarray(self.raw_pts[:j + 1], dtype=float), half_w)
        left, right = build_edges(P, half_w)
        close = lambda a: [tuple(p) for p in np.vstack([a, a[:1]])]   # explicit closed loop
        self.center, self.left, self.right = close(P), close(left), close(right)
        return True

    def finalize(self):
        if self.build():
            self.finalized = True
            return True
        return False

    def draw(self, surf):
        # ── road surface (filled polygon between edges) ──
        if len(self.left) >= 2 and len(self.right) >= 2:
            poly = self.left + list(reversed(self.right))
            poly_int = [(int(x), int(y)) for x, y in poly]
            pygame.draw.polygon(surf, ROAD_SURF, poly_int)

        # ── edges ──
        if len(self.left) >= 2:
            pygame.draw.lines(surf, EDGE_COL, False,
                              [(int(x), int(y)) for x, y in self.left], 2)
        if len(self.right) >= 2:
            pygame.draw.lines(surf, EDGE_COL, False,
                              [(int(x), int(y)) for x, y in self.right], 2)

        # ── dashed centerline ──
        if len(self.center) >= 2:
            draw_dashed_line(surf, CENTER_COL, self.center)

    def draw_preview(self, surf):
        """While drawing: only the raw stroke and a start ring (no road yet).
        The ring turns green once releasing the mouse would close the loop."""
        if len(self.raw_pts) >= 2:
            pygame.draw.lines(surf, RAW_COL, False,
                              [(int(x), int(y)) for x, y in self.raw_pts], 2)
        if self.raw_pts:
            col = CLOSE_COL if self.closure_index() is not None else RAW_COL
            x0, y0 = self.raw_pts[0]
            pygame.draw.circle(surf, col, (int(x0), int(y0)), CLOSE_DIST, 1)
            pygame.draw.circle(surf, col, (int(x0), int(y0)), 4)


# ── Save / Load helpers ────────────────────────────────────────

def save_map(road):
    """Serialize the finalized road to a JSON file via save dialog."""
    if not road.finalized:
        return False, "not finalized"

    data = {
        "version": 1,
        "closed": True,
        "lane_width": LANE_WIDTH,
        "canvas": {"width": CANVAS_W, "height": WIN_H},
        "center": [[round(x, 2), round(y, 2)] for x, y in road.center],
        "left":   [[round(x, 2), round(y, 2)] for x, y in road.left],
        "right":  [[round(x, 2), round(y, 2)] for x, y in road.right],
        "raw":    [[round(x, 2), round(y, 2)] for x, y in road.raw_pts],
    }

    root = tk.Tk()
    root.withdraw()          # hide the tiny Tk window
    root.attributes("-topmost", True)
    filepath = filedialog.asksaveasfilename(
        title="Save Map",
        defaultextension=".json",
        filetypes=[("Map JSON", "*.json"), ("All files", "*.*")],
    )
    root.destroy()

    if not filepath:
        return False, "cancelled"

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    return True, filepath


# ── Sidebar ────────────────────────────────────────────────────

# Button rect (relative to screen, not canvas)
SAVE_BTN = pygame.Rect(CANVAS_W + 14, 310, SIDEBAR_W - 28, 34)

def draw_sidebar(surf, font, small_font, road, state_msg, save_status=""):
    x0 = CANVAS_W
    pygame.draw.rect(surf, SIDEBAR_BG, (x0, 0, SIDEBAR_W, WIN_H))
    pygame.draw.line(surf, (60, 60, 75), (x0, 0), (x0, WIN_H), 1)

    # title
    title = font.render("SIM CONTROL", True, TEXT_COL)
    surf.blit(title, (x0 + 14, 20))

    # status
    status_color = (80, 200, 120) if road.finalized else (200, 160, 60)
    status_text  = "ROAD READY" if road.finalized else "DRAW A CLOSED LOOP"
    st = small_font.render(status_text, True, status_color)
    surf.blit(st, (x0 + 14, 52))

    # stats
    y = 90
    lines = [
        ("Points",    str(len(road.raw_pts))),
        ("Segments",  str(max(0, len(road.center) - 1))),
        ("Lane W",    f"{LANE_WIDTH}px"),
        ("State",     state_msg),
    ]
    for label, val in lines:
        lbl = small_font.render(label, True, HINT_COL)
        v   = small_font.render(val,   True, TEXT_COL)
        surf.blit(lbl, (x0 + 14, y))
        surf.blit(v,   (x0 + 14, y + 16))
        y += 44

    # ── Save Map button ──────────────────────────────────────────
    mx, my = pygame.mouse.get_pos()
    hovering = SAVE_BTN.collidepoint(mx, my)
    can_save = road.finalized

    if not can_save:
        btn_color = BTN_DIS
    elif hovering:
        btn_color = BTN_HOV
    else:
        btn_color = BTN_COL

    pygame.draw.rect(surf, btn_color, SAVE_BTN, border_radius=6)
    btn_label = font.render("💾  SAVE MAP", True, BTN_TXT)
    label_x = SAVE_BTN.centerx - btn_label.get_width() // 2
    label_y = SAVE_BTN.centery - btn_label.get_height() // 2
    surf.blit(btn_label, (label_x, label_y))

    # save status message
    if save_status:
        if save_status.startswith("Saved"):
            sc = SAVE_OK_COL
        elif save_status == "cancelled":
            sc = HINT_COL
        else:
            sc = SAVE_ERR_COL
        sm = small_font.render(save_status, True, sc)
        surf.blit(sm, (x0 + 14, SAVE_BTN.bottom + 6))

    # controls hint
    hints = [
        "CONTROLS",
        "",
        "Hold LMB  draw loop",
        "green ring closes it",
        "S         save map",
        "R         reset",
        "ESC       quit",
    ]
    y = WIN_H - 170
    for h in hints:
        col  = TEXT_COL if h == "CONTROLS" else HINT_COL
        size = font if h == "CONTROLS" else small_font
        surf.blit(size.render(h, True, col), (x0 + 14, y))
        y += 20

    # phase label
    ph = small_font.render("Phase 1 / 4 — Road", True, (70, 70, 90))
    surf.blit(ph, (x0 + 14, WIN_H - 24))


# ── Main loop ──────────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Autonomous Car Simulator")
    clock  = pygame.time.Clock()

    font       = pygame.font.SysFont("consolas", 13, bold=True)
    small_font = pygame.font.SysFont("consolas", 12)

    road        = Road()
    drawing     = False
    state_msg   = "idle"
    save_status = ""          # feedback message shown below the button
    save_status_timer = 0     # frames left to show the message

    # static canvas surface (road is drawn here so we don't redraw from scratch)
    canvas = pygame.Surface((CANVAS_W, WIN_H))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()

                elif event.key == pygame.K_r:
                    road.reset()
                    drawing      = False
                    state_msg    = "idle"
                    save_status  = ""

                elif event.key == pygame.K_SPACE:
                    if not road.finalized and not drawing and road.raw_pts:
                        state_msg = "loop closed" if road.finalize() else "not closed"

                elif event.key == pygame.K_s:
                    ok, msg = save_map(road)
                    if ok:
                        import os
                        save_status = "Saved: " + os.path.basename(msg)
                    else:
                        save_status = msg
                    save_status_timer = 180   # show for ~3 s @ 60 fps

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = event.pos
                    # ── Save button click ──
                    if SAVE_BTN.collidepoint(mx, my):
                        ok, msg = save_map(road)
                        if ok:
                            import os
                            save_status = "Saved: " + os.path.basename(msg)
                        else:
                            save_status = msg
                        save_status_timer = 180
                    elif mx < CANVAS_W and not road.finalized:
                        drawing = True
                        road.reset()
                        road.add_point((mx, my))
                        state_msg = "drawing"

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and drawing:
                    drawing = False
                    # the road is only built (and shown) once the loop is closed
                    if len(road.raw_pts) < MIN_POINTS:
                        state_msg = "too short"
                    elif road.finalize():
                        state_msg = "loop closed"
                    else:
                        state_msg = "not closed - redraw"

            elif event.type == pygame.MOUSEMOTION:
                if drawing:
                    mx, my = event.pos
                    if mx < CANVAS_W:
                        road.add_point((mx, my))

        # ── timer tick ──
        if save_status_timer > 0:
            save_status_timer -= 1
        else:
            save_status = ""

        # ── render ──
        canvas.fill(BG)

        if road.finalized:
            road.draw(canvas)
        elif drawing or len(road.raw_pts) > 0:
            road.draw_preview(canvas)

        # clip canvas to screen
        screen.blit(canvas, (0, 0))
        draw_sidebar(screen, font, small_font, road, state_msg, save_status)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()