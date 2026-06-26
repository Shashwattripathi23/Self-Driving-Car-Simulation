"""
car.py  –  F1 Top-View Car Simulator (Scaled Up)
─────────────────────────────────────
Left panel : top-down F1 car (steers front wheels, drives all 4)
Right panel: per-wheel telemetry + scrolling action log

Controls:  W=throttle  S=reverse  SPACE=brake  A=steer-left  D=steer-right  R=reset  ESC=quit
"""

import pygame, math, sys, time, json
from collections import deque

# ── Window ──────────────────────────────────────────────────────────────────
WIN_W, WIN_H = 1200, 700
VIEW_W       = 760          # car canvas
LOG_W        = WIN_W - VIEW_W

# ── Colours ─────────────────────────────────────────────────────────────────
BG_VIEW   = (13,  13,  20)
BG_LOG    = (18,  18,  28)
SEP       = (45,  45,  60)
TEXT      = (175, 175, 188)
HINT      = (90,  90,  108)
ACCENT    = (255, 200,  50)
GREEN     = (70,  210, 110)
RED_C     = (220,  55,  55)
ORANGE    = (240, 140,  30)
BLUE_C    = (50,  140, 220)
WHITE     = (235, 235, 245)
CAR_RED   = (210,   0,  35)
CAR_HI    = (230,  25,  55)
WING_C    = (18,   18,  28)
WHL_C     = (32,   32,  42)
WHL_RIM   = (75,   75,  90)
HALO_C    = (200, 175,  50)
COCKPIT_C = (25,   25,  38)
GRID_DOT  = (22,   22,  33)

# ── Physics constants ────────────────────────────────────────────────────────
FPS        = 60

# We scale the entire physics & rendering by 4.5x so it takes up ~80% of the window
SCALE      = 4.5  

WHEELBASE  = 85.0  * SCALE
MAX_SPEED  = 250.0 * SCALE
MAX_STEER  = math.radians(28)
STEER_RATE = math.radians(90)
STEER_RTN  = math.radians(150)
ACCEL      = 160.0 * SCALE
BRAKE_F    = 400.0 * SCALE
DRAG_K     = 1.8 


# ── Car ──────────────────────────────────────────────────────────────────────
class Car:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.heading  = -math.pi / 2   # pointing upward
        self.speed    = 0.0
        self.steer    = 0.0
        # telemetry
        self.thr = 0.0   # -1..1
        self.str_in = 0.0
        self.brake = False

    # ── per-wheel derived values ────────────────────────────────────────────
    @property
    def fl_steer(self): return math.degrees(self.steer)
    @property
    def fr_steer(self): return math.degrees(self.steer)
    @property
    def wheel_drive_N(self):
        """Drive force per wheel (arbitrary unit)."""
        if self.brake:
            return -BRAKE_F * 0.25
        return self.thr * ACCEL * 0.25

    def update(self, thr, brake, steer_dir, dt):
        self.thr    = thr
        self.brake  = brake
        self.str_in = steer_dir

        # ── Steering Physics ──
        # Wheels can only turn if the car has some speed (dry steering limitation)
        speed_factor = min(1.0, abs(self.speed) / (20.0 * SCALE)) 
        target = steer_dir * MAX_STEER
        diff   = target - self.steer
        
        if abs(steer_dir) > 0.01:
            rate = STEER_RATE * speed_factor
        else:
            # Slower return to center if stopped, normal if moving
            rate = STEER_RTN * max(0.1, speed_factor) 
            
        self.steer += math.copysign(min(abs(diff), rate * dt), diff)

        # ── Speed / Acceleration Physics ──
        if self.brake:
            # Hard braking using spacebar
            if self.speed > 0:
                self.speed = max(0, self.speed - BRAKE_F * dt)
            elif self.speed < 0:
                self.speed = min(0, self.speed + BRAKE_F * dt)
        else:
            # Engine throttle
            if thr > 0.01:
                self.speed += thr * ACCEL * dt
            elif thr < -0.01:
                self.speed -= abs(thr) * ACCEL * 0.5 * dt   # reverse / engine brake

            # Rolling friction & Aero Drag
            drag = DRAG_K * (self.speed**2) / MAX_SPEED
            friction = 15.0 * SCALE
            total_drag = (drag + friction) * dt
            
            if self.speed > 0:
                self.speed = max(0, self.speed - total_drag)
            elif self.speed < 0:
                self.speed = min(0, self.speed + total_drag)

        self.speed  = max(-MAX_SPEED * 0.35, min(MAX_SPEED, self.speed))

        # ── Kinematic bicycle model ──
        if abs(self.speed) > 0.1 and abs(self.steer) > 0.0005:
            R     = WHEELBASE / math.tan(self.steer)
            omega = self.speed / R
            self.heading += omega * dt
            
        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt

    def draw(self, surf, cx, cy):
        h = self.heading

        def rot(lx, ly):
            # Scale coordinates relative to car center
            lx *= SCALE
            ly *= SCALE
            c, s = math.cos(h), math.sin(h)
            return (cx + lx*c - ly*s, cy + lx*s + ly*c)

        def poly(pts, col, w=0):
            ip = [(int(x), int(y)) for x, y in pts]
            if w:  pygame.draw.polygon(surf, col, ip, w)
            else:  pygame.draw.polygon(surf, col, ip)

        # Base dimensions (unscaled, will be scaled inside rot())
        FL, RL, HW = 52, 40, 21  

        # rear wing
        rx = -(RL + 9)
        poly([rot(rx,-33), rot(rx,33), rot(rx+8,29), rot(rx+8,-29)], WING_C)
        poly([rot(rx-2,-33), rot(rx+10,-33), rot(rx+10,-29), rot(rx-2,-29)], WING_C)
        poly([rot(rx-2,29),  rot(rx+10,29),  rot(rx+10,33),  rot(rx-2,33)],  WING_C)

        # body
        body = [
            rot(FL, -8), rot(FL, 8),
            rot(FL-18, HW+4), rot(FL-36, HW+6),
            rot(0, HW),
            rot(-RL+18, HW+8), rot(-RL+4, HW+10), rot(-RL, HW+2),
            rot(-RL, -HW-2), rot(-RL+4, -HW-10), rot(-RL+18, -HW-8),
            rot(0, -HW),
            rot(FL-36, -HW-6), rot(FL-18, -HW-4),
        ]
        poly(body, CAR_RED)
        poly([rot(FL-2,-6), rot(FL-2,6), rot(FL-24,13), rot(-RL+10,13),
              rot(-RL+10,-13), rot(FL-24,-13)], CAR_HI)

        # front wing
        fx = FL + 7
        poly([rot(fx-2,-39), rot(fx-2,39), rot(fx+6,35), rot(fx+6,-35)], WING_C)
        poly([rot(fx-4,-39), rot(fx+8,-39), rot(fx+8,-35), rot(fx-4,-35)], WING_C)
        poly([rot(fx-4,35),  rot(fx+8,35),  rot(fx+8,39),  rot(fx-4,39)],  WING_C)

        # cockpit + halo
        poly([rot(22,-10), rot(22,10), rot(-4,8), rot(-4,-8)], COCKPIT_C)
        poly([rot(18,-12), rot(18,12), rot(5,10), rot(5,-10)], HALO_C)
        poly([rot(18,-2.5), rot(18,2.5), rot(2,5), rot(-2,5)], HALO_C)

        # wheels
        WL, WW = 16 * SCALE, 7 * SCALE 
        FA, RA, AO = 33, 28, 33 
        wheels = [
            (FA, -AO, self.steer, "FL"), (FA,  AO, self.steer, "FR"),
            (-RA,-AO, 0.0,        "RL"), (-RA, AO, 0.0,        "RR"),
        ]
        for lx, ly, sang, _ in wheels:
            wc  = rot(lx, ly)
            wa  = h + sang
            cw, sw = math.cos(wa), math.sin(wa)
            pts = [(wc[0] + px*cw - py*sw, wc[1] + px*sw + py*cw)
                   for px, py in [(-WL,-WW),(WL,-WW),(WL,WW),(-WL,WW)]]
            poly(pts, WHL_C)
            poly(pts, WHL_RIM, max(1, int(1 * SCALE))) # Thicker rims for scale

        # velocity arrow
        if abs(self.speed) > 5 * SCALE:
            s   = 1 if self.speed > 0 else -1
            al  = min(abs(self.speed) * 0.28, 55 * SCALE)
            ax  = cx + math.cos(h) * s * al
            ay  = cy + math.sin(h) * s * al
            pygame.draw.line(surf, ACCENT, (int(cx), int(cy)), (int(ax), int(ay)), int(2 * SCALE * 0.5))


# ── Helpers ───────────────────────────────────────────────────────────────────
def draw_grid(surf, cx, cy, spacing=int(40 * SCALE)):
    ox = int(cx % spacing)
    oy = int(cy % spacing)
    for gx in range(-spacing + ox, VIEW_W + spacing, spacing):
        for gy in range(-spacing + oy, WIN_H + spacing, spacing):
            pygame.draw.circle(surf, GRID_DOT, (gx, gy), 1)


def _shared_ray_color(frac):
    """Green(far) → Orange(medium) → Red(close)."""
    if frac < 0.30:
        return (220,  50,  50)
    elif frac < 0.60:
        t = (frac - 0.30) / 0.30
        return (220, int(50 + 110*t), 40)
    else:
        t = (frac - 0.60) / 0.40
        return (int(220 - 160*t), int(160 + 40*t), 40)


def draw_shared_rays(surf, cx, cy, heading, ray_data, max_visual=180):
    """
    Draw rays from (cx, cy) using rel_angle+heading and frac.
    ray_data = list of [rel_angle, frac] from shared state.
    """
    for rel_angle, frac in ray_data:
        angle  = heading + rel_angle
        length = frac * max_visual
        ex = cx + math.cos(angle) * length
        ey = cy + math.sin(angle) * length
        col = _shared_ray_color(frac)
        pygame.draw.line(surf, col, (int(cx), int(cy)), (int(ex), int(ey)), 1)
        pygame.draw.circle(surf, col, (int(ex), int(ey)), 3)


def bar(surf, x, y, w, h, frac, fg, bg=(35, 35, 50)):
    pygame.draw.rect(surf, bg,  (x, y, w, h), border_radius=3)
    fw = int(w * max(0, min(1, abs(frac))))
    if fw:
        pygame.draw.rect(surf, fg, (x, y, fw, h), border_radius=3)


def draw_wheel_box(surf, font, sf, x, y, label, steer_deg, force, is_front):
    BW, BH = 168, 110
    pygame.draw.rect(surf, (28, 28, 42), (x, y, BW, BH), border_radius=8)
    pygame.draw.rect(surf, SEP,          (x, y, BW, BH), 1, border_radius=8)

    surf.blit(font.render(label, True, ACCENT), (x+10, y+8))

    # steer
    if is_front:
        sc   = GREEN if abs(steer_deg) < 1 else ORANGE
        stxt = f"Steer  {steer_deg:+.1f}°"
        surf.blit(sf.render(stxt, True, sc), (x+10, y+32))
        bar(surf, x+10, y+50, BW-20, 8, steer_deg / 28, sc)
    else:
        surf.blit(sf.render("Steer  —", True, HINT), (x+10, y+32))
        bar(surf, x+10, y+50, BW-20, 8, 0, HINT)

    # drive force
    fc   = GREEN if force >= 0 else RED_C
    ftxt = f"Force  {force:+.1f} u"
    surf.blit(sf.render(ftxt, True, fc), (x+10, y+68))
    bar(surf, x+10, y+86, BW-20, 8, force / (ACCEL * 0.25), fc)


def draw_log_panel(surf, font, sf, car, log_lines, clock):
    x0 = VIEW_W
    surf.fill(BG_LOG, (x0, 0, LOG_W, WIN_H))
    pygame.draw.line(surf, SEP, (x0, 0), (x0, WIN_H), 1)

    # ── Title ──
    surf.blit(font.render("TELEMETRY", True, TEXT), (x0+14, 14))
    surf.blit(sf.render(f"FPS {int(clock.get_fps())}", True, HINT), (x0+LOG_W-60, 18))

    # ── Wheel boxes  2×2 grid ──
    WX = [x0+14, x0+14+178]
    WY = [46, 168]
    labels  = ["FL","FR","RL","RR"]
    front   = [True, True, False, False]
    steers  = [car.fl_steer, car.fr_steer, 0.0, 0.0]
    forces  = [car.wheel_drive_N] * 4

    for i, (lbl, is_f, sd, fd) in enumerate(zip(labels, front, steers, forces)):
        wx = WX[i % 2]
        wy = WY[i // 2]
        draw_wheel_box(surf, font, sf, wx, wy, lbl, sd, fd, is_f)

    # ── Car state ──
    sy = 296
    surf.blit(font.render("CAR STATE", True, TEXT), (x0+14, sy))
    sy += 22

    spd_col = GREEN if abs(car.speed) < MAX_SPEED*0.7 else RED_C
    items = [
        ("Speed",    f"{car.speed:+.1f} px/s",      spd_col),
        ("Heading",  f"{math.degrees(car.heading):.1f}°", WHITE),
        ("Throttle", f"{car.thr:+.2f}",             GREEN if car.thr > 0 else RED_C),
        ("Brakes",   "ENGAGED" if car.brake else "OFF", RED_C if car.brake else HINT),
        ("Steer",    f"{math.degrees(car.steer):+.1f}°", ORANGE),
    ]
    for label, val, col in items:
        surf.blit(sf.render(label, True, HINT), (x0+14, sy))
        surf.blit(sf.render(val,   True, col),  (x0+90, sy))
        sy += 18

    # speed bar
    bar(surf, x0+14, sy+4, LOG_W-28, 10,
        car.speed / MAX_SPEED,
        GREEN if car.speed >= 0 else RED_C)
    sy += 22

    # ── Action log ──
    surf.blit(font.render("ACTION log", True, TEXT), (x0+14, sy+4))
    sy += 24
    pygame.draw.line(surf, SEP, (x0+14, sy), (x0+LOG_W-14, sy), 1)
    sy += 6
    for line in list(log_lines)[-14:]:
        ts_col = HINT
        parts = line.split("|", 1)
        if len(parts) == 2:
            surf.blit(sf.render(parts[0], True, ts_col),  (x0+14, sy))
            surf.blit(sf.render(parts[1], True, WHITE),   (x0+70, sy))
        else:
            surf.blit(sf.render(line, True, WHITE), (x0+14, sy))
        sy += 16
        if sy > WIN_H - 10:
            break


# ── Main ──────────────────────────────────────────────────────────────────────────────────
def main():
    # ── Shared-display mode detection ────────────────────────────
    # When launched by map_reader.py: python car.py --shared <path>
    # In this mode physics are replaced by reading the shared JSON file.
    shared_file = None
    if "--shared" in sys.argv:
        idx = sys.argv.index("--shared")
        if idx + 1 < len(sys.argv):
            shared_file = sys.argv[idx + 1]
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    title  = "F1 Car Simulator – Linked to Map Viewer" if shared_file else "F1 Car Simulator"
    pygame.display.set_caption(title)
    clock  = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 13, bold=True)
    sf   = pygame.font.SysFont("consolas", 12)

    car    = Car(VIEW_W // 2, WIN_H // 2)
    canvas = pygame.Surface((VIEW_W, WIN_H))

    log      = deque(maxlen=200)
    t_start  = time.time()

    # track held keys for smooth control
    thr_held = 0.0
    str_held = 0.0
    brake_held = False

    # logging state
    prev_thr = 0.0
    prev_str = 0.0
    prev_brake = False
    log_timer = 0

    def ts():
        return f"{time.time()-t_start:6.1f}s"

    log.append(f"{ts()}|Car initialised. Use WASD & SPACE.")

    while True:
        dt = clock.tick(FPS) / 1000.0

        keys = pygame.key.get_pressed()
        thr_held = (1.0 if keys[pygame.K_w] else
                   -1.0 if keys[pygame.K_s] else 0.0)
        str_held = (-1.0 if keys[pygame.K_a] else
                     1.0  if keys[pygame.K_d] else 0.0)
        brake_held = keys[pygame.K_SPACE]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_r:
                    car = Car(VIEW_W // 2, WIN_H // 2)
                    log.append(f"{ts()}|Reset.")

        car.update(thr_held, brake_held, str_held, dt)

        # ── In shared mode: overwrite physics state from the shared file ──
        shared_rays = []
        if shared_file:
            try:
                with open(shared_file, "r") as _f:
                    _st = json.load(_f)
                car.heading  = _st.get("heading", car.heading)
                car.speed    = _st.get("speed",   car.speed)
                car.steer    = _st.get("steer",   car.steer)
                car.thr      = _st.get("thr",     car.thr)
                car.brake    = bool(_st.get("brake", car.brake))
                shared_rays  = _st.get("rays",    [])
            except (OSError, json.JSONDecodeError, KeyError):
                pass  # keep last good values

        # ── logging (every ~0.3 s or on state change) ──────────────────────
        log_timer += 1
        thr_changed = abs(thr_held - prev_thr) > 0.5
        str_changed = abs(str_held - prev_str) > 0.5
        brake_changed = brake_held != prev_brake

        if brake_changed:
            if brake_held: log.append(f"{ts()}|Brakes ENGAGED")
            else:          log.append(f"{ts()}|Brakes RELEASED")
        if thr_changed:
            if thr_held > 0:   log.append(f"{ts()}|Throttle ON")
            elif thr_held < 0: log.append(f"{ts()}|Engine Reverse")
            else:              log.append(f"{ts()}|Coast")
        if str_changed:
            if str_held > 0:   log.append(f"{ts()}|Steer RIGHT")
            elif str_held < 0: log.append(f"{ts()}|Steer LEFT")
            else:              log.append(f"{ts()}|Straight")
        if log_timer % (FPS * 1) == 0:  # every second
            log.append(f"{ts()}|{abs(car.speed):.1f} px/s  str={math.degrees(car.steer):+.1f}°")

        prev_thr = thr_held
        prev_str = str_held
        prev_brake = brake_held

        # ── camera: keep car centered on canvas ────────────────────────────
        cam_x = car.x - VIEW_W // 2
        cam_y = car.y - WIN_H  // 2

        # ── render canvas ──────────────────────────────────────────────────
        canvas.fill(BG_VIEW)
        draw_grid(canvas, -cam_x, -cam_y)
        if shared_rays:
            draw_shared_rays(canvas, VIEW_W // 2, WIN_H // 2,
                             car.heading, shared_rays)
        car.draw(canvas, VIEW_W // 2, WIN_H // 2)

        screen.blit(canvas, (0, 0))
        draw_log_panel(screen, font, sf, car, log, clock)

        pygame.display.flip()


if __name__ == "__main__":
    main()