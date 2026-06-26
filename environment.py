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
SMOOTH_WINDOW  = 1                  # moving-average kernel size
SUBSAMPLE      = 4                   # keep every Nth raw point
MIN_POINTS     = 20                  # minimum points to build a road

# ── Colours ────────────────────────────────────────────────────
BG          = (18,  18,  24)
ROAD_SURF   = (45,  45,  55)
EDGE_COL    = (220, 220, 220)
CENTER_COL  = (255, 220,  60)
RAW_COL     = (80,  80, 100)
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

def subsample(pts, n):
    return pts[::n]

def smooth(pts, window):
    """Simple moving-average smoothing over x and y independently."""
    if len(pts) < window:
        return pts
    arr = np.array(pts, dtype=float)
    kernel = np.ones(window) / window
    sx = np.convolve(arr[:, 0], kernel, mode='valid')
    sy = np.convolve(arr[:, 1], kernel, mode='valid')
    return list(zip(sx, sy))

def perpendicular(a, b):
    """Unit perpendicular to segment a→b, pointing left."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = max(np.hypot(dx, dy), 1e-6)
    return (-dy / length, dx / length)

def build_edges(center, half_w):
    """
    Given a smoothed centerline, return (left_edge, right_edge)
    as lists of (x, y) tuples.
    """
    left, right = [], []
    n = len(center)
    for i in range(n):
        # use neighbours for a stable perpendicular
        a = center[max(i - 1, 0)]
        b = center[min(i + 1, n - 1)]
        px, py = perpendicular(a, b)
        cx, cy = center[i]
        left.append( (cx + px * half_w, cy + py * half_w) )
        right.append((cx - px * half_w, cy - py * half_w) )
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

    def build(self):
        if len(self.raw_pts) < MIN_POINTS:
            return False
        sub    = subsample(self.raw_pts, SUBSAMPLE)
        center = smooth(sub, SMOOTH_WINDOW)
        if len(center) < 2:
            return False
        self.center = center
        self.left, self.right = build_edges(center, LANE_WIDTH // 2)
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
        """Draw the raw stroke while user is still drawing."""
        if len(self.raw_pts) >= 2:
            pygame.draw.lines(surf, RAW_COL, False,
                              [(int(x), int(y)) for x, y in self.raw_pts], 2)
        # live smooth preview
        if self.build() and len(self.center) >= 2:
            self.draw(surf)


# ── Save / Load helpers ────────────────────────────────────────

def save_map(road):
    """Serialize the finalized road to a JSON file via save dialog."""
    if not road.finalized:
        return False, "not finalized"

    data = {
        "version": 1,
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
    status_text  = "ROAD READY" if road.finalized else "DRAWING..."
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
        "Hold LMB  draw road",
        "SPACE     finalize",
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
                    if not road.finalized and len(road.raw_pts) >= MIN_POINTS:
                        ok = road.finalize()
                        state_msg = "finalized" if ok else "too short"
                    elif road.finalized:
                        state_msg = "finalized"

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
                    if len(road.raw_pts) >= MIN_POINTS:
                        state_msg = "press SPACE"
                    else:
                        state_msg = "too short"

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