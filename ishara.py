#!/usr/bin/env python3
"""
ইশারা (Ishara) — ওয়েবক্যাম দিয়ে হাতের ইশারায় ম্যাক চালান।

চালানো:
    python ishara.py              → মেনু বারে চলবে (✋ আইকন)
    python ishara.py --preview    → ক্যামেরা প্রিভিউ উইন্ডো সহ (টিউন করার জন্য)
    python ishara.py --camera 1   → নির্দিষ্ট ক্যামেরা

ইশারা:
    🖐  খোলা হাত (চার আঙুল উপরে)      → কার্সর চালু, তালুর সাথে চলে
    ✊  মুঠি / ফ্রেম থেকে হাত সরানো    → কার্সর থেমে যায় (park)
    👆  তর্জনী নামানো (মাউস চাপার মতো) → ক্লিক; নামিয়ে রেখে নাড়ালে ড্র্যাগ
    👆👆 একই জায়গায় দ্রুত আবার         → ডাবল / ট্রিপল ক্লিক
    🤙  কনিষ্ঠা নামানো                  → রাইট ক্লিক (ধরে রাখলে রাইট-ড্র্যাগ)
    🤘  মধ্যমা+অনামিকা ভাঁজ, হাত ওঠানামা → স্ক্রল
    👍  মুঠি + বুড়ো আঙুল উপর/নিচ/ডান/বাম → কাস্টম অ্যাকশন (config.json)
    🤙  শাকা (বুড়ো + কনিষ্ঠা)             → কাস্টম অ্যাকশন
    🙌  দুই হাত খুলে দুবার ক্রস করুন     → ট্র্যাকিং বন্ধ / চালু
"""

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import threading
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(APP_DIR, "config.json")
MODEL_PATH = os.path.join(APP_DIR, "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")

# ═══════════════════════════════ সেটিংস ═══════════════════════════════
DEFAULTS = {
    "camera": 0,
    "cursor_anchor": "palm",          # "palm" (স্থির) বা "fingertip" (তর্জনীর ডগা)
    "reach_mode": "auto",             # "auto": হাতের দূরত্ব বুঝে এলাকা ঠিক করে; "manual"
    "auto_reach_factor": 2.3,         # auto মোডে বড় = বেশি হাত নাড়াতে হবে
    "manual_reach": 0.5,              # manual মোডে ফ্রেমের কত অংশ = পুরো স্ক্রিন
    "smoothing": 0.5,                 # 0 = দ্রুত, 1 = খুব মসৃণ
    "click_sensitivity": 0.5,         # 0..1, বেশি = সহজে ক্লিক
    "drag_window_ms": 180,            # এই সময়ের মধ্যে ছোট নড়াচড়া ড্র্যাগ হবে না
    "drag_start_px": 14,
    "multi_click_ms": 450,
    "multi_click_radius_px": 24,
    "right_click": True,
    "scroll_speed": 1.6,
    "scroll_natural": True,           # True = হাতের সাথে পেজ চলে (টাচের মতো)
    "horizontal_scroll": False,
    "dwell_click": False,
    "dwell_ms": 900,
    "dwell_radius_px": 16,
    "two_hand_stop": True,
    "two_hand_crossings": 2,
    "sounds": True,
    "gestures": {
        "thumb_up":    {"type": "shortcut", "value": "ctrl+up"},     # Mission Control
        "thumb_down":  {"type": "shortcut", "value": "ctrl+down"},   # App Exposé
        "thumb_left":  {"type": "shortcut", "value": "ctrl+left"},   # আগের ডেস্কটপ
        "thumb_right": {"type": "shortcut", "value": "ctrl+right"},  # পরের ডেস্কটপ
        "shaka":       {"type": "none", "value": ""}
    },
    "per_app": {
        "Spotify": {"shaka": {"type": "shortcut", "value": "space"}},
        "Music":   {"shaka": {"type": "shortcut", "value": "space"}}
    }
}

EXT = 1.55      # এর বেশি reach = আঙুল সোজা
FOLD = 1.35     # এর কম reach = আঙুল ভাঁজ
THUMB_EXT = 0.5


def load_config():
    cfg = copy.deepcopy(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                user = json.load(f)
            for k, v in user.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
        except Exception as e:
            print(f"config.json পড়া যায়নি ({e}), ডিফল্ট সেটিংস ব্যবহার হচ্ছে।")
    else:
        save_config(cfg)
    return cfg


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def play(name, cfg):
    if cfg.get("sounds"):
        path = f"/System/Library/Sounds/{name}.aiff"
        if os.path.exists(path):
            subprocess.Popen(["afplay", "-v", "0.4", path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ═══════════════════════════ মডেল ও ক্যামেরা ═══════════════════════════
def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return True
    print("হাত চেনার মডেল ডাউনলোড হচ্ছে...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception:
        subprocess.run(["curl", "-L", "-o", MODEL_PATH, MODEL_URL], check=False)
    ok = os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000
    if not ok:
        print("মডেল ডাউনলোড হয়নি। এই লিংক থেকে নামিয়ে ishara.py-এর পাশে রাখুন:\n" + MODEL_URL)
    return ok


def open_camera(preferred):
    indices = [preferred] + [i for i in range(4) if i != preferred]
    fallback = None
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # পুরোনো ফ্রেম জমে ল্যাগ না হয়
        bright = 0.0
        for _ in range(45):
            ok, f = cap.read()
            if ok and f is not None:
                bright = float(f.mean())
                if bright > 8:
                    break
        if bright > 8:
            print(f"ক্যামেরা {idx} ব্যবহার হচ্ছে।")
            if fallback is not None:
                fallback.release()
            return cap
        if fallback is None:
            fallback = cap
        else:
            cap.release()
    if fallback is not None:
        print("ক্যামেরা থেকে ছবি আসছে না (কালো)। অনুমতি দিয়ে Terminal ⌘Q করে আবার খুলুন।")
    return fallback


# ═══════════════════════════ মসৃণতার ফিল্টার ═══════════════════════════
class OneEuro:
    """One Euro Filter: ধীর নড়াচড়ায় কাঁপুনি দূর করে, দ্রুত নড়াচড়ায় ল্যাগ রাখে না।"""

    def __init__(self, min_cutoff=1.2, beta=0.008, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x = None
        self.dx = 0.0
        self.t = None

    @staticmethod
    def _a(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.t = x, t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        ad = self._a(self.d_cutoff, dt)
        self.dx = ad * dx + (1 - ad) * self.dx
        a = self._a(self.min_cutoff + self.beta * abs(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x


# ═══════════════════════════ মাউস ও কীবোর্ড (Quartz) ═══════════════════════════
KEYCODES = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9,
    "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19,
    "3": 20, "4": 21, "6": 22, "5": 23, "=": 24, "9": 25, "7": 26, "-": 27, "8": 28,
    "0": 29, "]": 30, "o": 31, "u": 32, "[": 33, "i": 34, "p": 35, "return": 36,
    "enter": 36, "l": 37, "j": 38, "'": 39, "k": 40, ";": 41, "\\": 42, ",": 43,
    "/": 44, "n": 45, "m": 46, ".": 47, "tab": 48, "space": 49, "`": 50,
    "delete": 51, "backspace": 51, "esc": 53, "escape": 53,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98,
    "f8": 100, "f9": 101, "f10": 109, "f11": 103, "f12": 111,
    "left": 123, "right": 124, "down": 125, "up": 126,
}


class Mouse:
    def __init__(self):
        import Quartz
        self.Q = Quartz
        self.lock = threading.RLock()
        b = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
        self.ox, self.oy = b.origin.x, b.origin.y
        self.w, self.h = b.size.width, b.size.height
        self.refresh_pos()
        self.down_btn = None
        self.clicks = 1

    def refresh_pos(self):
        loc = self.Q.CGEventGetLocation(self.Q.CGEventCreate(None))
        self.x, self.y = loc.x, loc.y

    def _post(self, etype, button):
        Q = self.Q
        ev = Q.CGEventCreateMouseEvent(None, etype, (self.x, self.y), button)
        Q.CGEventSetIntegerValueField(ev, Q.kCGMouseEventClickState, self.clicks)
        Q.CGEventPost(Q.kCGHIDEventTap, ev)

    def move(self, x, y):
        Q = self.Q
        with self.lock:
            self.x = min(max(x, self.ox), self.ox + self.w - 1)
            self.y = min(max(y, self.oy), self.oy + self.h - 1)
            if self.down_btn == "left":
                self._post(Q.kCGEventLeftMouseDragged, Q.kCGMouseButtonLeft)
            elif self.down_btn == "right":
                self._post(Q.kCGEventRightMouseDragged, Q.kCGMouseButtonRight)
            else:
                ev = Q.CGEventCreateMouseEvent(None, Q.kCGEventMouseMoved, (self.x, self.y),
                                               Q.kCGMouseButtonLeft)
                Q.CGEventPost(Q.kCGHIDEventTap, ev)

    def press(self, btn, clicks=1):
        Q = self.Q
        with self.lock:
            if self.down_btn:
                return
            self.down_btn, self.clicks = btn, clicks
            if btn == "left":
                self._post(Q.kCGEventLeftMouseDown, Q.kCGMouseButtonLeft)
            else:
                self._post(Q.kCGEventRightMouseDown, Q.kCGMouseButtonRight)

    def release(self):
        Q = self.Q
        with self.lock:
            if not self.down_btn:
                return
            if self.down_btn == "left":
                self._post(Q.kCGEventLeftMouseUp, Q.kCGMouseButtonLeft)
            else:
                self._post(Q.kCGEventRightMouseUp, Q.kCGMouseButtonRight)
            self.down_btn = None

    def click(self, btn="left"):
        self.press(btn, 1)
        self.release()

    def scroll(self, dy, dx=0):
        Q = self.Q
        try:
            ev = Q.CGEventCreateScrollWheelEvent(None, Q.kCGScrollEventUnitPixel, 2, int(dy), int(dx))
        except Exception:
            ev = Q.CGEventCreateScrollWheelEvent(None, Q.kCGScrollEventUnitPixel, 1, int(dy))
        Q.CGEventPost(Q.kCGHIDEventTap, ev)

    def key_combo(self, combo):
        Q = self.Q
        mods = {
            "cmd": Q.kCGEventFlagMaskCommand, "command": Q.kCGEventFlagMaskCommand,
            "shift": Q.kCGEventFlagMaskShift, "alt": Q.kCGEventFlagMaskAlternate,
            "option": Q.kCGEventFlagMaskAlternate, "opt": Q.kCGEventFlagMaskAlternate,
            "ctrl": Q.kCGEventFlagMaskControl, "control": Q.kCGEventFlagMaskControl,
        }
        flags, code = 0, None
        for part in combo.lower().replace(" ", "").split("+"):
            if part in mods:
                flags |= mods[part]
            elif part in KEYCODES:
                code = KEYCODES[part]
        if code is None:
            print(f"অজানা শর্টকাট: {combo}")
            return
        if code in (123, 124, 125, 126):  # অ্যারো কী-তে macOS এই ফ্ল্যাগ চায়
            flags |= Q.kCGEventFlagMaskSecondaryFn | Q.kCGEventFlagMaskNumericPad
        for down in (True, False):
            ev = Q.CGEventCreateKeyboardEvent(None, code, down)
            Q.CGEventSetFlags(ev, flags)
            Q.CGEventPost(Q.kCGHIDEventTap, ev)


class CursorDriver(threading.Thread):
    """ক্যামেরা ৩০ FPS হলেও কার্সর ১২০ Hz-এ মসৃণভাবে লক্ষ্যের দিকে যায় — তাই মাখনের মতো চলে।"""

    def __init__(self, mouse, hz=120):
        super().__init__(daemon=True)
        self.mouse, self.hz = mouse, hz
        self.lock = threading.Lock()
        self.target = None
        self.pos = (mouse.x, mouse.y)
        self.pending = [0.0, 0.0]
        self.carry = [0.0, 0.0]
        self.tau = 0.03
        self.running = True

    def set_target(self, x, y):
        with self.lock:
            self.target = (x, y)

    def hold(self):
        with self.lock:
            self.target = None

    def sync(self):
        self.mouse.refresh_pos()
        with self.lock:
            self.pos = (self.mouse.x, self.mouse.y)
            self.target = None

    def jump(self, x, y):
        with self.lock:
            self.target = None
            self.pos = (x, y)
        self.mouse.move(x, y)

    def add_scroll(self, dx, dy):
        with self.lock:
            self.pending[0] += dx
            self.pending[1] += dy

    def run(self):
        last = time.perf_counter()
        period = 1.0 / self.hz
        while self.running:
            now = time.perf_counter()
            dt = max(now - last, 1e-4)
            last = now
            with self.lock:
                tgt = self.target
                k = 1 - math.exp(-dt / self.tau)
                ks = 1 - math.exp(-dt / 0.07)
                sx, sy = self.pending[0] * ks, self.pending[1] * ks
                self.pending[0] -= sx
                self.pending[1] -= sy
            if tgt is not None:
                nx = self.pos[0] + (tgt[0] - self.pos[0]) * k
                ny = self.pos[1] + (tgt[1] - self.pos[1]) * k
                if abs(nx - self.pos[0]) + abs(ny - self.pos[1]) > 0.1:
                    self.pos = (nx, ny)
                    self.mouse.move(nx, ny)
            self.carry[0] += sx
            self.carry[1] += sy
            ix, iy = int(self.carry[0]), int(self.carry[1])
            if ix or iy:
                self.carry[0] -= ix
                self.carry[1] -= iy
                self.mouse.scroll(iy, ix)
            time.sleep(max(0.0, period - (time.perf_counter() - now)))


# ═══════════════════════════ হাতের ভঙ্গি বোঝা ═══════════════════════════
TIPS, MCPS = (8, 12, 16, 20), (5, 9, 13, 17)
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
               (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
               (18, 19), (19, 20), (0, 17)]


def hand_features(img, world):
    """img: ছবির স্থানাঙ্ক (aspect ঠিক করা), world: 3D মিটার স্থানাঙ্ক।"""
    w0 = world[0]
    reach = []
    for tip, mcp in zip(TIPS, MCPS):
        reach.append(np.linalg.norm(world[tip] - w0) / max(np.linalg.norm(world[mcp] - w0), 1e-6))
    palm_len = max(np.linalg.norm(world[9] - w0), 1e-6)
    # মুঠির মধ্যে বুড়ো আঙুল গুঁজে থাকলে সেটা তর্জনী/মধ্যমার মাঝের গাঁটের কাছে থাকে
    thumb_ext = min(np.linalg.norm(world[4] - world[6]),
                    np.linalg.norm(world[4] - world[10])) / palm_len
    v = img[4] - img[2]
    if abs(v[0]) > abs(v[1]):
        thumb_dir = "right" if v[0] > 0 else "left"
    else:
        thumb_dir = "down" if v[1] > 0 else "up"
    def ang(a0, a1, b0, b1):
        u, v = world[a1] - world[a0], world[b1] - world[b0]
        c = np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-9)
        return math.degrees(math.acos(min(max(c, -1.0), 1.0)))

    return {
        # তর্জনী মধ্যমার তুলনায় কত ডিগ্রি নামলো (পুরো হাত কাত হলেও এটা বদলায় না)
        "index_dip": ang(5, 8, 9, 12),
        "pinky_dip": ang(17, 20, 13, 16),
        "reach": np.array(reach),
        "thumb_ext": thumb_ext,
        "thumb_dir": thumb_dir,
        "scale": float(np.linalg.norm(img[0] - img[9])),
        "palm": img[[0, 5, 9, 13, 17]].mean(axis=0),
        "tip": img[8],
    }


def frontmost_app():
    try:
        from AppKit import NSWorkspace
        return str(NSWorkspace.sharedWorkspace().frontmostApplication().localizedName())
    except Exception:
        return ""


# ═══════════════════════════ মূল ট্র্যাকার ═══════════════════════════
class Tracker:
    def __init__(self, cfg, mouse, driver, preview=False, camera=None):
        self.cfg, self.mouse, self.driver = cfg, mouse, driver
        self.preview = preview
        self.camera = cfg["camera"] if camera is None else camera
        self.running = True
        self.enabled = True
        self.state = "PARKED"
        self.info = ""
        self.fps = 0.0
        self.apply_smoothing()

        self.primary = None
        self.last_seen = 0.0
        self.open_frames = 0
        self.base = None
        self.scale_ema = None
        self.pose, self.pose_t, self.pose_fired = None, 0.0, False
        self.btn = None
        self.press_t = 0.0
        self.press_anchor = (0, 0)
        self.dragging = False
        self.dip_count = {"left": 0, "right": 0}
        self.up_count = 0
        self.freeze_until = 0.0
        self.last_click = (0.0, (0, 0), 0)
        self.scroll_prev = None
        self.dwell_anchor, self.dwell_t, self.dwell_armed = None, 0.0, True
        self.dwell_progress = 0.0
        self.cross_sign, self.cross_times = None, []
        self.metric = 0.0
        self.area = None
        self.screen_pt = None

    def apply_smoothing(self):
        s = float(self.cfg.get("smoothing", 0.5))
        self.fx = OneEuro(min_cutoff=2.0 * (1 - s) + 0.3, beta=0.012 * (1 - s) + 0.002)
        self.fy = OneEuro(min_cutoff=2.0 * (1 - s) + 0.3, beta=0.012 * (1 - s) + 0.002)

    # ── অবস্থা বদল ──
    def set_enabled(self, on):
        if self.enabled == on:
            return
        self.enabled = on
        self.park()
        self.state = "PARKED" if on else "OFF"
        play("Pop" if on else "Bottle", self.cfg)

    def park(self):
        self.end_press(cancel=True)
        self.driver.hold()
        self.scroll_prev = None
        self.open_frames = 0
        if self.state not in ("OFF",):
            self.state = "PARKED"

    def end_press(self, cancel=False):
        if self.btn:
            self.mouse.release()
            self.btn = None
            self.dragging = False

    # ── স্ক্রিনে ম্যাপ ──
    def map_to_screen(self, p, scale, aspect):
        W, H = self.mouse.w, self.mouse.h
        if self.cfg["reach_mode"] == "auto":
            self.scale_ema = scale if self.scale_ema is None else self.scale_ema * 0.97 + scale * 0.03
            ah = min(max(self.scale_ema * self.cfg["auto_reach_factor"], 0.22), 0.9)
        else:
            ah = float(self.cfg["manual_reach"])
        aw = min(ah * W / H, aspect * 0.95)
        cx = aspect / 2
        cy = 0.55 if self.cfg["cursor_anchor"] == "palm" else 0.45
        self.area = (cx - aw / 2, cy - ah / 2, aw, ah)
        u = min(max((p[0] - cx) / aw + 0.5, 0.0), 1.0)
        v = min(max((p[1] - cy) / ah + 0.5, 0.0), 1.0)
        return self.mouse.ox + u * (W - 1), self.mouse.oy + v * (H - 1)

    # ── কাস্টম ইশারা ──
    def fire_gesture(self, name):
        app = frontmost_app()
        action = self.cfg.get("per_app", {}).get(app, {}).get(name) or self.cfg["gestures"].get(name)
        if not action or action.get("type", "none") == "none":
            return
        t, v = action["type"], action.get("value", "")
        self.info = f"{name} → {v}"
        play("Tink", self.cfg)
        if t == "shortcut":
            self.mouse.key_combo(v)
        elif t == "app":
            subprocess.Popen(["open", "-a", v])
        elif t == "command":
            subprocess.Popen(v, shell=True)

    # ── দুই হাত ক্রস করে বন্ধ/চালু ──
    def two_hands(self, hands, now):
        if not self.cfg.get("two_hand_stop", True):
            return
        a, b = sorted(hands[:2], key=lambda h: h["label"])
        if all((h["f"]["reach"] > EXT * 0.95).all() for h in (a, b)):
            sign = 1 if a["f"]["palm"][0] > b["f"]["palm"][0] else -1
            if self.cross_sign is not None and sign != self.cross_sign:
                self.cross_times = [t for t in self.cross_times if now - t < 3.0] + [now]
                if len(self.cross_times) >= int(self.cfg.get("two_hand_crossings", 2)):
                    self.cross_times = []
                    self.set_enabled(not self.enabled)
            self.cross_sign = sign

    # ── প্রতি ফ্রেমের মূল লজিক ──
    def process(self, res, now, aspect):
        hands = []
        for i, lms in enumerate(res.hand_landmarks or []):
            img = np.array([[l.x * aspect, l.y] for l in lms])
            if res.hand_world_landmarks and i < len(res.hand_world_landmarks):
                world = np.array([[l.x, l.y, l.z] for l in res.hand_world_landmarks[i]])
            else:
                world = np.array([[l.x * aspect, l.y, l.z] for l in lms])
            label = res.handedness[i][0].category_name if res.handedness else str(i)
            hands.append({"img": img, "world": world, "label": label,
                          "f": hand_features(img, world)})
        self.hands = hands

        if len(hands) >= 2:
            self.last_seen = now
            self.two_hands(hands, now)
            if self.enabled:
                self.park()
            return
        self.cross_sign = None

        if not self.enabled:
            self.state = "OFF"
            return

        if not hands:
            if now - self.last_seen > 0.25 and self.state != "PARKED":
                self.park()
                self.fx.reset()
                self.fy.reset()
            return
        self.last_seen = now
        f = hands[0]["f"]
        reach = f["reach"]
        open_hand = bool((reach > EXT).all())
        folded = reach < FOLD

        # নিজের হাতের মাপ শিখে নেওয়া (খোলা হাতের বেসলাইন)
        if open_hand and not self.btn:
            self.base = reach.copy() if self.base is None else self.base * 0.95 + reach * 0.05
        base = self.base if self.base is not None else np.full(4, 1.9)
        rel = reach / base

        # ── মুঠি-ভিত্তিক ইশারা ──
        pose = None
        if folded.all():
            pose = ("thumb_" + f["thumb_dir"]) if f["thumb_ext"] > THUMB_EXT else "fist"
        elif f["thumb_ext"] > THUMB_EXT and folded[0] and folded[1] and folded[2] and reach[3] > EXT * 0.9:
            pose = "shaka"
        if pose != self.pose:
            self.pose, self.pose_t, self.pose_fired = pose, now, False
        if pose:
            if pose == "fist":
                if now - self.pose_t > 0.2:
                    self.park()
            else:
                self.park()
                if not self.pose_fired and now - self.pose_t > 0.4:
                    self.pose_fired = True
                    self.fire_gesture(pose)
            return

        # ── park থাকলে খোলা হাতের অপেক্ষা ──
        if self.state == "PARKED":
            self.open_frames = self.open_frames + 1 if open_hand else 0
            if self.open_frames >= 3:
                self.state = "ACTIVE"
                self.driver.sync()
                self.fx.reset()
                self.fy.reset()
                self.dwell_anchor = None
            else:
                return

        anchor = f["palm"] if self.cfg["cursor_anchor"] == "palm" else f["tip"]
        rx, ry = self.map_to_screen(anchor, f["scale"], aspect)
        sx, sy = self.fx(rx, now), self.fy(ry, now)
        self.screen_pt = (sx, sy)

        # ── স্ক্রল পোজ: তর্জনী+কনিষ্ঠা উপরে, মধ্যমা+অনামিকা ভাঁজ ──
        if reach[0] > EXT * 0.92 and reach[3] > EXT * 0.88 and folded[1] and folded[2]:
            self.end_press()
            self.driver.hold()
            self.state = "SCROLL"
            if self.scroll_prev is not None:
                dx, dy = sx - self.scroll_prev[0], sy - self.scroll_prev[1]
                sp = float(self.cfg["scroll_speed"])
                sgn = -1 if self.cfg["scroll_natural"] else 1
                if abs(dy) < 0.8:
                    dy = 0
                if abs(dx) < 0.8 or not self.cfg["horizontal_scroll"]:
                    dx = 0
                if dx or dy:
                    self.driver.add_scroll(sgn * -dx * sp, sgn * -dy * sp)
            self.scroll_prev = (sx, sy)
            return
        self.scroll_prev = None
        if self.state == "SCROLL":
            self.state = "ACTIVE"
            self.driver.sync()

        # ── ক্লিক: তর্জনী মধ্যমার তুলনায় নামলো কি না (হাত কাত হলে ভুল ক্লিক হবে না) ──
        sens = float(self.cfg["click_sensitivity"])
        dip_on = 36 - 16 * sens               # ডিগ্রি; sens 0.5 → 28°
        dip_off = dip_on - 10
        left_m, right_m = f["index_dip"], f["pinky_dip"]
        self.metric = left_m
        steady = rel[1] > 0.8 and rel[2] > 0.8   # মধ্যমা+অনামিকা সোজা থাকতে হবে

        if not self.btn:
            want = None
            if steady and left_m > dip_on:
                want = "left"
            elif steady and self.cfg["right_click"] and right_m > dip_on and left_m < dip_off:
                want = "right"
            for k in ("left", "right"):
                self.dip_count[k] = self.dip_count[k] + 1 if want == k else 0
            if want and self.dip_count[want] >= 2:
                self.start_press(want, now, sx, sy)
                return
            if want:
                self.driver.hold()   # আঙুল নামানো শুরু হলেই কার্সর থামে, ক্লিক ঠিক জায়গায় পড়ে
                return
        else:
            m = left_m if self.btn == "left" else right_m
            self.up_count = self.up_count + 1 if m < dip_off else 0
            if self.up_count >= 2:
                self.finish_press(now)
            elif not self.dragging:
                moved = math.hypot(sx - self.press_anchor[0], sy - self.press_anchor[1])
                el = (now - self.press_t) * 1000
                dp = float(self.cfg["drag_start_px"])
                if (el > self.cfg["drag_window_ms"] and moved > dp) or moved > dp * 3:
                    self.dragging = True
                    self.state = "DRAG"
            if self.dragging:
                self.driver.set_target(sx, sy)
            return

        # ── সাধারণ নড়াচড়া ──
        self.state = "ACTIVE"
        if now < self.freeze_until:
            return
        if steady or open_hand:
            self.driver.set_target(sx, sy)
        else:
            self.driver.hold()

        # ── Dwell click ──
        self.dwell_progress = 0.0
        if self.cfg["dwell_click"]:
            r = float(self.cfg["dwell_radius_px"])
            if self.dwell_anchor is None or math.hypot(sx - self.dwell_anchor[0], sy - self.dwell_anchor[1]) > r:
                self.dwell_anchor, self.dwell_t = (sx, sy), now
                self.dwell_armed = True
            elif self.dwell_armed:
                self.dwell_progress = min(1.0, (now - self.dwell_t) * 1000 / self.cfg["dwell_ms"])
                if self.dwell_progress >= 1.0:
                    self.dwell_armed = False
                    self.driver.hold()
                    self.mouse.click("left")
                    play("Tink", self.cfg)

    def start_press(self, btn, now, sx, sy):
        self.driver.hold()
        t0, pos0, n0 = self.last_click
        clicks = 1
        cur = (self.mouse.x, self.mouse.y)
        if btn == "left" and (now - t0) * 1000 < self.cfg["multi_click_ms"] and \
                math.hypot(cur[0] - pos0[0], cur[1] - pos0[1]) < self.cfg["multi_click_radius_px"]:
            clicks = min(n0 + 1, 3)
            self.driver.jump(*pos0)   # দ্বিতীয়/তৃতীয় ক্লিক প্রথমটার জায়গাতেই পড়বে
        self.mouse.press(btn, clicks)
        self.btn = btn
        self.press_t = now
        self.press_anchor = (sx, sy)
        self.dragging = False
        self.up_count = 0
        self.dip_count = {"left": 0, "right": 0}
        self.state = "CLICK" if btn == "left" else "RIGHT"

    def finish_press(self, now):
        was_drag = self.dragging
        clicks = self.mouse.clicks
        self.driver.hold()
        self.mouse.release()
        if self.btn == "left" and not was_drag:
            self.last_click = (now, (self.mouse.x, self.mouse.y), clicks)
        else:
            self.last_click = (0.0, (0, 0), 0)
        self.btn = None
        self.dragging = False
        self.up_count = 0
        self.freeze_until = now + 0.12   # আঙুল তোলার সময় হাতের কাঁপুনি বাদ
        self.state = "ACTIVE"

    # ── প্রিভিউ আঁকা ──
    def draw(self, frame, aspect):
        h, w = frame.shape[:2]
        sx = w / aspect
        for hand in getattr(self, "hands", []):
            pts = [(int(x * sx), int(y * h)) for x, y in hand["img"]]
            for a, b in CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
            for p in pts:
                cv2.circle(frame, p, 4, (255, 120, 60), cv2.FILLED)
            pc = hand["f"]["palm"]
            c = (int(pc[0] * sx), int(pc[1] * h))
            color = {"CLICK": (0, 0, 255), "DRAG": (0, 0, 255), "RIGHT": (255, 0, 255),
                     "SCROLL": (0, 200, 255)}.get(self.state, (0, 220, 0))
            cv2.circle(frame, c, 14, color, 3)
            if self.dwell_progress > 0:
                cv2.ellipse(frame, c, (22, 22), -90, 0, int(360 * self.dwell_progress), (0, 255, 255), 3)
        if self.area:
            x0, y0, aw, ah = self.area
            cv2.rectangle(frame, (int(x0 * sx), int(y0 * h)), (int((x0 + aw) * sx), int((y0 + ah) * h)),
                          (255, 0, 255), 1)
        th = self.hands[0]["f"]["thumb_ext"] if getattr(self, "hands", None) else 0
        cv2.putText(frame, f"{self.state}  {self.fps:.0f} FPS  dip:{self.metric:.0f}deg  thumb:{th:.2f}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if self.info:
            cv2.putText(frame, self.info, (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if frame.mean() < 3:
            cv2.putText(frame, "CAMERA IS BLACK: restart Terminal (Cmd+Q)", (10, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ── ক্যামেরা লুপ ──
    def run(self):
        if not ensure_model():
            self.running = False
            return
        landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
        ))
        cap = open_camera(self.camera)
        if cap is None:
            print("ক্যামেরা খোলা যায়নি। Camera অনুমতি দিন, তারপর Terminal ⌘Q করে আবার খুলুন।")
            self.running = False
            return
        t0 = time.monotonic()
        last_ts, last_t = -1, time.monotonic()
        print("ইশারা চালু ✋  খোলা হাত দেখান।")
        while self.running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            aspect = w / h
            now = time.monotonic()
            ts = max(int((now - t0) * 1000), last_ts + 1)
            last_ts = ts
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            try:
                self.process(res, now, aspect)
            except Exception as e:
                print("ত্রুটি:", e)
            dt = now - last_t
            last_t = now
            if dt > 0:
                self.fps = self.fps * 0.9 + 0.1 / dt
            if self.preview:
                self.draw(frame, aspect)
                cv2.imshow("Ishara (q = quit)", frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    self.running = False
        self.park()
        landmarker.close()
        cap.release()
        if self.preview:
            cv2.destroyAllWindows()


# ═══════════════════════════ মেনু বার অ্যাপ ═══════════════════════════
ICONS = {"ACTIVE": "🖐", "PARKED": "✊", "OFF": "⏸", "SCROLL": "↕︎", "CLICK": "👆",
         "DRAG": "✊➜", "RIGHT": "🤙"}


def run_menubar(tracker, cfg):
    import rumps

    class IsharaApp(rumps.App):
        def __init__(self):
            super().__init__("Ishara", title="✊", quit_button=None)
            self.t_item = rumps.MenuItem("ট্র্যাকিং চালু", callback=self.toggle)
            self.d_item = rumps.MenuItem("Dwell click (স্থির রাখলে ক্লিক)", callback=self.toggle_dwell)
            self.a_item = rumps.MenuItem("কার্সর আঙুলের ডগায়", callback=self.toggle_anchor)
            self.s_item = rumps.MenuItem("শব্দ", callback=self.toggle_sound)
            self.menu = [self.t_item, None, self.d_item, self.a_item, self.s_item, None,
                         rumps.MenuItem("Settings ফাইল খুলুন", callback=self.open_settings),
                         rumps.MenuItem("Settings রিলোড", callback=self.reload),
                         None, rumps.MenuItem("বন্ধ করুন", callback=self.quit)]
            self.sync_menu()
            rumps.Timer(self.refresh, 0.2).start()

        def sync_menu(self):
            self.t_item.state = int(tracker.enabled)
            self.d_item.state = int(bool(cfg["dwell_click"]))
            self.a_item.state = int(cfg["cursor_anchor"] == "fingertip")
            self.s_item.state = int(bool(cfg["sounds"]))

        def refresh(self, _):
            self.title = ICONS.get(tracker.state, "✋")
            self.t_item.state = int(tracker.enabled)
            if not tracker.running:
                rumps.quit_application()

        def toggle(self, _):
            tracker.set_enabled(not tracker.enabled)
            self.sync_menu()

        def toggle_dwell(self, _):
            cfg["dwell_click"] = not cfg["dwell_click"]
            save_config(cfg)
            self.sync_menu()

        def toggle_anchor(self, _):
            cfg["cursor_anchor"] = "palm" if cfg["cursor_anchor"] == "fingertip" else "fingertip"
            save_config(cfg)
            self.sync_menu()

        def toggle_sound(self, _):
            cfg["sounds"] = not cfg["sounds"]
            save_config(cfg)
            self.sync_menu()

        def open_settings(self, _):
            subprocess.Popen(["open", "-t", CONFIG_PATH])

        def reload(self, _):
            cfg.update(load_config())
            tracker.apply_smoothing()
            self.sync_menu()
            play("Pop", cfg)

        def quit(self, _):
            tracker.running = False
            time.sleep(0.2)
            rumps.quit_application()

    IsharaApp().run()


def main():
    ap = argparse.ArgumentParser(description="ইশারা — হাতের ইশারায় ম্যাক চালান")
    ap.add_argument("--preview", action="store_true", help="ক্যামেরা প্রিভিউ উইন্ডো দেখান")
    ap.add_argument("--camera", type=int, default=None, help="ক্যামেরা নম্বর")
    args = ap.parse_args()

    cfg = load_config()
    mouse = Mouse()
    driver = CursorDriver(mouse)
    driver.start()

    use_menubar = not args.preview
    if use_menubar:
        try:
            import rumps  # noqa: F401
        except ImportError:
            print("rumps ইনস্টল নেই, তাই প্রিভিউ মোডে চলছে।  (pip install rumps)")
            use_menubar = False

    tracker = Tracker(cfg, mouse, driver, preview=not use_menubar, camera=args.camera)
    try:
        if use_menubar:
            threading.Thread(target=tracker.run, daemon=True).start()
            run_menubar(tracker, cfg)
        else:
            tracker.run()
    except KeyboardInterrupt:
        pass
    finally:
        tracker.running = False
        driver.running = False
        mouse.release()


if __name__ == "__main__":
    main()
