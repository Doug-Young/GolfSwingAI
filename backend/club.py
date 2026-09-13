"""Club shaft detection from swing frames.

Uses Canny + probabilistic Hough line transform in a search region anchored to
the wrists. Candidate lines are scored by length, proximity to the grip, and
angular consistency with the previous frame's shaft. Falls back to extending
the forearm vector when no reliable line is found.

Pose-only approach — no ML, no external services. Works best on typical range
footage (contrast between shaft and background). Every frame reports a
'method' and 'confidence' so downstream code can weight accordingly.
"""
import math
import cv2
import numpy as np

from pose import LEFT_WRIST, RIGHT_WRIST, LEFT_ELBOW, RIGHT_ELBOW, LEFT_SHOULDER, RIGHT_SHOULDER


def _dist(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _angle_deg(a, b) -> float:
    """Angle of vector a→b in degrees, in [-180, 180]."""
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def _angle_diff_norm(a: float, b: float) -> float:
    """Smallest angular difference between two angles, in [0, 180]."""
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def _wrist_grip_and_forearm(lm, w: int, h: int):
    """Return (grip_px, elbow_mid_px, forearm_len_px, shoulder_width_px)."""
    lw = (lm[LEFT_WRIST].x * w, lm[LEFT_WRIST].y * h)
    rw = (lm[RIGHT_WRIST].x * w, lm[RIGHT_WRIST].y * h)
    le = (lm[LEFT_ELBOW].x * w, lm[LEFT_ELBOW].y * h)
    re = (lm[RIGHT_ELBOW].x * w, lm[RIGHT_ELBOW].y * h)
    ls = (lm[LEFT_SHOULDER].x * w, lm[LEFT_SHOULDER].y * h)
    rs = (lm[RIGHT_SHOULDER].x * w, lm[RIGHT_SHOULDER].y * h)

    grip = ((lw[0] + rw[0]) / 2, (lw[1] + rw[1]) / 2)
    elbow_mid = ((le[0] + re[0]) / 2, (le[1] + re[1]) / 2)
    forearm_len = (_dist(le, lw) + _dist(re, rw)) / 2
    shoulder_width = _dist(ls, rs)
    return grip, elbow_mid, forearm_len, shoulder_width


def _extrapolated_shaft(grip, elbow_mid, forearm_len) -> dict | None:
    """Fallback: extend the elbow-midpoint → grip vector by ~1.4× forearm length.

    Reasonable at address and follow-through when the arms and club are aligned.
    Unreliable at the top of backswing where the wrists cock.
    """
    dx = grip[0] - elbow_mid[0]
    dy = grip[1] - elbow_mid[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1:
        return None
    scale = forearm_len * 1.4 / length
    tip = (grip[0] + dx * scale, grip[1] + dy * scale)
    return {
        "grip": (int(grip[0]), int(grip[1])),
        "tip": (int(tip[0]), int(tip[1])),
        "angle": _angle_deg(grip, tip),
        "confidence": 0.30,
        "method": "extrapolated",
    }


def detect_shaft(image_bgr: np.ndarray, lm, prev_angle: float | None = None) -> dict | None:
    """Detect the golf shaft in one frame.

    Returns { grip, tip, angle, confidence, method } or None if pose isn't usable.
    """
    if lm is None or image_bgr is None:
        return None

    h, w = image_bgr.shape[:2]
    grip, elbow_mid, forearm_len, shoulder_width = _wrist_grip_and_forearm(lm, w, h)

    # Expected shaft length ~ 2× forearm or 1.2× torso-width, whichever is larger
    expected_len = max(forearm_len * 2.0, shoulder_width * 1.5)
    min_line_len = int(expected_len * 0.35)

    # Search region — square around the grip large enough for any club orientation
    r = int(expected_len * 1.3)
    x0 = max(0, int(grip[0] - r))
    y0 = max(0, int(grip[1] - r))
    x1 = min(w, int(grip[0] + r))
    y1 = min(h, int(grip[1] + r))
    if x1 - x0 < 40 or y1 - y0 < 40:
        return _extrapolated_shaft(grip, elbow_mid, forearm_len)

    roi = image_bgr[y0:y1, x0:x1]

    # Edge detection
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    med = np.median(gray)
    lower = int(max(0, 0.66 * med))
    upper = int(min(255, 1.33 * med))
    edges = cv2.Canny(gray, lower, upper, apertureSize=3)

    # Probabilistic Hough — returns line segments (endpoints), not infinite lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=min_line_len,
        maxLineGap=20,
    )
    if lines is None or len(lines) == 0:
        return _extrapolated_shaft(grip, elbow_mid, forearm_len)

    # OpenCV's HoughLinesP shape varies by build: sometimes (N, 1, 4), sometimes
    # (N, 4). Normalize to a (N, 4) 2-D array so unpacking always works.
    lines_arr = np.asarray(lines).reshape(-1, 4)

    grip_full = grip  # in full-image coords
    max_grip_dist = forearm_len * 0.9

    best = None
    best_score = -1.0
    for line in lines_arr:
        xa, ya, xb, yb = int(line[0]), int(line[1]), int(line[2]), int(line[3])
        p1 = (xa + x0, ya + y0)
        p2 = (xb + x0, yb + y0)

        length = _dist(p1, p2)
        if length < min_line_len:
            continue

        d1 = _dist(p1, grip_full)
        d2 = _dist(p2, grip_full)
        near_grip = min(d1, d2)
        if near_grip > max_grip_dist:
            continue

        # Orient: p1 = grip end, p2 = tip end
        if d1 > d2:
            p1, p2 = p2, p1

        # Reject lines that fold BACK through the body toward the hips — the shaft
        # should generally extend AWAY from the elbow-mid. Compute alignment of
        # (grip→tip) vs (elbow_mid→grip); highly negative alignment = folded back.
        arm_dx = grip_full[0] - elbow_mid[0]
        arm_dy = grip_full[1] - elbow_mid[1]
        shaft_dx = p2[0] - p1[0]
        shaft_dy = p2[1] - p1[1]
        arm_len = math.sqrt(arm_dx ** 2 + arm_dy ** 2)
        shaft_len_v = math.sqrt(shaft_dx ** 2 + shaft_dy ** 2)
        alignment = 0.0
        if arm_len > 0 and shaft_len_v > 0:
            alignment = (arm_dx * shaft_dx + arm_dy * shaft_dy) / (arm_len * shaft_len_v)
        # alignment in [-1, 1]. Reject strongly opposite (< -0.5 = shaft pointing back at elbow)
        if alignment < -0.5:
            continue

        # ---- Scoring ----
        # Length: favor lines close to expected shaft length
        length_score = min(length / expected_len, 1.25)
        # Grip proximity: nearer is better
        proximity_score = max(0.0, 1.0 - near_grip / max_grip_dist)
        # Alignment with the forearm direction (0.5 baseline, +up to 0.5 for good alignment)
        alignment_score = 0.5 + max(alignment, 0) * 0.5
        # Temporal consistency
        if prev_angle is not None:
            ang = _angle_deg(p1, p2)
            diff = _angle_diff_norm(ang, prev_angle)
            temporal_score = max(0.0, 1.0 - diff / 90.0)
        else:
            temporal_score = 0.5

        total = (0.35 * length_score
                 + 0.25 * proximity_score
                 + 0.20 * alignment_score
                 + 0.20 * temporal_score)

        if total > best_score:
            best_score = total
            best = {
                "grip": (int(p1[0]), int(p1[1])),
                "tip": (int(p2[0]), int(p2[1])),
                "angle": _angle_deg(p1, p2),
                "confidence": min(total, 1.0),
                "method": "hough",
            }

    if best is None:
        return _extrapolated_shaft(grip, elbow_mid, forearm_len)

    return best


def draw_shaft(image: np.ndarray, shaft: dict, color=(80, 220, 255)):
    """Overlay a detected shaft onto an image (BGR).

    Only draws when the shaft was actually detected via Hough (real edge lines).
    Extrapolated shafts are guesses from the arm direction and often mislead
    visually — better to draw nothing than a wrong line.
    """
    if shaft is None or shaft.get("method") != "hough":
        return
    g = shaft["grip"]
    t = shaft["tip"]
    cv2.line(image, g, t, (0, 0, 0), 5, cv2.LINE_AA)      # outline for readability
    cv2.line(image, g, t, color, 3, cv2.LINE_AA)
    cv2.circle(image, t, 6, color, -1, cv2.LINE_AA)
    cv2.circle(image, t, 6, (0, 0, 0), 1, cv2.LINE_AA)


def analyze_shaft_plane(club_history: list[dict], events: dict) -> dict | None:
    """Analyze the shaft trajectory across the swing.

    club_history entries are aligned with sample indices; None where undetected.
    Returns dict with:
      backswing_plane_angle: average shaft angle mid-backswing
      downswing_plane_angle: average shaft angle mid-downswing
      plane_delta: signed difference (positive = downswing above backswing = over-the-top)
      verdict: string description
    """
    top_i = events.get("top")
    address_i = events.get("address")
    impact_i = events.get("impact")
    if top_i is None or address_i is None or impact_i is None:
        return None

    # Mid-backswing: between address and top
    back_range = range(address_i + max(1, (top_i - address_i) // 3),
                       max(address_i + 1, top_i - max(1, (top_i - address_i) // 5)))
    down_range = range(top_i + max(1, (impact_i - top_i) // 5),
                       max(top_i + 1, impact_i - max(1, (impact_i - top_i) // 5)))

    def _avg_angle(idxs):
        vals = [club_history[i]["angle"] for i in idxs
                if 0 <= i < len(club_history) and club_history[i] is not None
                and club_history[i].get("method") == "hough"]  # only trust real detections
        if not vals:
            return None
        return sum(vals) / len(vals)

    back_angle = _avg_angle(back_range)
    down_angle = _avg_angle(down_range)
    if back_angle is None or down_angle is None:
        return None

    plane_delta = _angle_diff_norm(back_angle, down_angle)
    # Determine direction — is downswing steeper (over-the-top) or shallower (under)?
    # In image coords with Y down, a "steeper" shaft has a more vertical angle.
    # This is genuinely ambiguous without knowing camera view, so we hedge.
    if plane_delta < 10:
        verdict = f"Backswing and downswing trace roughly the same plane ({plane_delta:.0f}° difference) — good consistency."
        rating = "good"
    elif plane_delta < 20:
        verdict = f"Downswing plane differs from backswing plane by ~{plane_delta:.0f}°. Some deviation, common in most amateur swings."
        rating = "watch"
    else:
        verdict = f"Large plane change between backswing and downswing (~{plane_delta:.0f}°). Could indicate an over-the-top or steep transition — worth checking on video."
        rating = "issue"

    return {
        "backswing_plane_angle": round(back_angle, 1),
        "downswing_plane_angle": round(down_angle, 1),
        "plane_delta": round(plane_delta, 1),
        "verdict": verdict,
        "rating": rating,
    }
