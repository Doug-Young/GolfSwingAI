"""MediaPipe pose detection + biomechanical metrics for golf swing analysis."""
import math
import os
import urllib.request
from typing import Optional

import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "pose_landmarker_full.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"


def _ensure_model():
    """Download the pose model if it isn't already on disk."""
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"Downloading pose model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Pose model downloaded.")

# MediaPipe pose landmark indices
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

# Skeleton connections for drawing (subset that matters for golf)
CONNECTIONS = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
]

_detector = None


def get_detector():
    """Lazy-load the MediaPipe pose detector."""
    global _detector
    if _detector is None:
        _ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
        )
        _detector = mp_vision.PoseLandmarker.create_from_options(options)
    return _detector


def detect_landmarks(image_bgr: np.ndarray) -> Optional[list]:
    """Legacy shim — returns only the 2D landmarks."""
    result = detect_pose(image_bgr)
    return result["landmarks"] if result else None


def detect_pose(image_bgr: np.ndarray) -> Optional[dict]:
    """Full pose detection returning both 2D image landmarks and 3D world landmarks.

    2D landmarks: normalized to image coords (x, y in [0,1]).
    3D world landmarks: metric coordinates relative to hip center — camera-independent.
    Each landmark has a `visibility` attribute in [0,1] that we can use to filter noise.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = get_detector().detect(mp_image)
    if not result.pose_landmarks:
        return None
    return {
        "landmarks": result.pose_landmarks[0],
        "world_landmarks": result.pose_world_landmarks[0] if result.pose_world_landmarks else None,
    }


def landmark_confidence(lm) -> float:
    """Average visibility of the golf-relevant joints — a proxy for detection quality."""
    key = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
           LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]
    vals = []
    for i in key:
        v = getattr(lm[i], "visibility", None)
        if v is not None:
            vals.append(v)
    if not vals:
        return 1.0
    return sum(vals) / len(vals)


def _angle_deg(v1: tuple, v2: tuple) -> float:
    """Angle in degrees between two 2D vectors."""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag = math.sqrt(v1[0] ** 2 + v1[1] ** 2) * math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if mag == 0:
        return 0.0
    cos_a = max(-1.0, min(1.0, dot / mag))
    return math.degrees(math.acos(cos_a))


def _joint_angle(a, b, c) -> float:
    """Angle at joint b formed by a-b-c."""
    v1 = (a.x - b.x, a.y - b.y)
    v2 = (c.x - b.x, c.y - b.y)
    return _angle_deg(v1, v2)


def _midpoint(p1, p2):
    return type("P", (), {"x": (p1.x + p2.x) / 2, "y": (p1.y + p2.y) / 2})()


def _compute_3d_metrics(world_lm) -> dict:
    """Camera-independent angles from 3D world landmarks (in meters, hip-centered).

    World coord system per MediaPipe: X right, Y down, Z forward-from-camera (roughly).
    """
    if world_lm is None:
        return {}

    def v(i):
        return (world_lm[i].x, world_lm[i].y, world_lm[i].z)

    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def norm(a):
        return math.sqrt(a[0]**2 + a[1]**2 + a[2]**2)

    def dot(a, b):
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

    def angle_between(a, b):
        na, nb = norm(a), norm(b)
        if na == 0 or nb == 0:
            return 0.0
        cos_a = max(-1.0, min(1.0, dot(a, b) / (na * nb)))
        return math.degrees(math.acos(cos_a))

    ls, rs = v(LEFT_SHOULDER), v(RIGHT_SHOULDER)
    lh, rh = v(LEFT_HIP), v(RIGHT_HIP)
    mid_shoulder = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2, (ls[2]+rs[2])/2)
    mid_hip = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2, (lh[2]+rh[2])/2)

    # 3D spine tilt: angle between spine vector (hip → shoulder) and world vertical (0, -1, 0)
    spine_vec = sub(mid_shoulder, mid_hip)
    spine_tilt_3d = angle_between(spine_vec, (0, -1, 0))

    # 3D rotation angles about the vertical axis: project shoulder line and hip line onto XZ plane
    def rotation_about_vertical(a, b):
        # Vector from a to b in XZ plane; angle vs. X axis
        dx = b[0] - a[0]
        dz = b[2] - a[2]
        return math.degrees(math.atan2(dz, dx))

    shoulder_rot_3d = rotation_about_vertical(ls, rs)
    hip_rot_3d = rotation_about_vertical(lh, rh)

    # X-factor: absolute difference between shoulder and hip rotation, normalized to 0-180
    raw_diff = abs(shoulder_rot_3d - hip_rot_3d)
    x_factor_3d = min(raw_diff, 360 - raw_diff) if raw_diff > 180 else raw_diff

    return {
        "spine_tilt_3d": round(spine_tilt_3d, 1),
        "shoulder_rot_3d": round(shoulder_rot_3d, 1),
        "hip_rot_3d": round(hip_rot_3d, 1),
        "x_factor_3d": round(x_factor_3d, 1),
    }


def compute_metrics(lm, world_lm=None) -> dict:
    """Compute biomechanical metrics from pose landmarks.

    2D metrics from image landmarks (camera-angle dependent),
    plus 3D metrics from world landmarks (camera-independent) when available.
    """
    mid_shoulder = _midpoint(lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER])
    mid_hip = _midpoint(lm[LEFT_HIP], lm[RIGHT_HIP])
    mid_ankle = _midpoint(lm[LEFT_ANKLE], lm[RIGHT_ANKLE])

    # Spine angle: angle of spine vector vs vertical (0 = perfectly upright)
    spine_vec = (mid_shoulder.x - mid_hip.x, mid_shoulder.y - mid_hip.y)
    spine_tilt = _angle_deg(spine_vec, (0, -1))  # -1 because y-down in image coords

    # Shoulder line angle (horizontal = 0, rotated = angled)
    shoulder_line = (lm[RIGHT_SHOULDER].x - lm[LEFT_SHOULDER].x,
                     lm[RIGHT_SHOULDER].y - lm[LEFT_SHOULDER].y)
    shoulder_tilt = math.degrees(math.atan2(shoulder_line[1], shoulder_line[0]))

    # Hip line angle
    hip_line = (lm[RIGHT_HIP].x - lm[LEFT_HIP].x,
                lm[RIGHT_HIP].y - lm[LEFT_HIP].y)
    hip_tilt = math.degrees(math.atan2(hip_line[1], hip_line[0]))

    # X-factor proxy: difference between shoulder and hip tilt in the image plane
    x_factor = abs(shoulder_tilt - hip_tilt)

    # Knee flex (angles at each knee, 180 = straight)
    left_knee_angle = _joint_angle(lm[LEFT_HIP], lm[LEFT_KNEE], lm[LEFT_ANKLE])
    right_knee_angle = _joint_angle(lm[RIGHT_HIP], lm[RIGHT_KNEE], lm[RIGHT_ANKLE])

    # Wrist heights (min y = highest on screen; use whichever wrist is higher)
    wrist_y = min(lm[LEFT_WRIST].y, lm[RIGHT_WRIST].y)
    wrist_x = (lm[LEFT_WRIST].x + lm[RIGHT_WRIST].x) / 2

    # Head position
    head_x = lm[NOSE].x
    head_y = lm[NOSE].y

    # Weight balance proxy: horizontal offset of hip-center from ankle-center
    # positive = weight forward (toward target for right-handed swing = left)
    weight_shift = mid_hip.x - mid_ankle.x

    # Stance width (ankle-to-ankle) and shoulder width (shoulder-to-shoulder), normalized
    stance_width = abs(lm[LEFT_ANKLE].x - lm[RIGHT_ANKLE].x)
    shoulder_width = abs(lm[LEFT_SHOULDER].x - lm[RIGHT_SHOULDER].x)

    # Elbow angles (180 = straight arm)
    left_elbow_angle = _joint_angle(lm[LEFT_SHOULDER], lm[LEFT_ELBOW], lm[LEFT_WRIST])
    right_elbow_angle = _joint_angle(lm[RIGHT_SHOULDER], lm[RIGHT_ELBOW], lm[RIGHT_WRIST])

    # Hip center x (for tracking hip sway)
    hip_center_x = mid_hip.x

    # Confidence
    confidence = landmark_confidence(lm)

    # 3D metrics (camera-independent) from world landmarks
    metrics_3d = _compute_3d_metrics(world_lm)

    return {
        "spine_tilt": round(spine_tilt, 1),
        "shoulder_tilt": round(shoulder_tilt, 1),
        "hip_tilt": round(hip_tilt, 1),
        "x_factor": round(x_factor, 1),
        "left_knee_angle": round(left_knee_angle, 1),
        "right_knee_angle": round(right_knee_angle, 1),
        "left_elbow_angle": round(left_elbow_angle, 1),
        "right_elbow_angle": round(right_elbow_angle, 1),
        "stance_width": round(stance_width, 3),
        "shoulder_width": round(shoulder_width, 3),
        "hip_center_x": round(hip_center_x, 3),
        "wrist_y": round(wrist_y, 3),
        "wrist_x": round(wrist_x, 3),
        "head_x": round(head_x, 3),
        "head_y": round(head_y, 3),
        "confidence": round(confidence, 3),
        **metrics_3d,
        "weight_shift": round(weight_shift, 3),
    }


def draw_skeleton(image_bgr: np.ndarray, lm, annotate: bool = True, metrics: dict = None, reference_lines: bool = True) -> np.ndarray:
    """Draw skeleton overlay with optional reference lines (target line, ground line, extended spine)."""
    img = image_bgr.copy()
    h, w = img.shape[:2]

    def px(p):
        return int(p.x * w), int(p.y * h)

    # --- Reference lines (drawn first so skeleton sits on top) ---
    if reference_lines:
        # Ground line at ankle level
        ankle_y = int(((lm[LEFT_ANKLE].y + lm[RIGHT_ANKLE].y) / 2) * h)
        cv2.line(img, (0, ankle_y), (w, ankle_y), (80, 80, 80), 1, cv2.LINE_AA)
        # Ball position vertical hint (mid-stance)
        mid_ankle_x = int(((lm[LEFT_ANKLE].x + lm[RIGHT_ANKLE].x) / 2) * w)
        cv2.line(img, (mid_ankle_x, ankle_y - 8), (mid_ankle_x, ankle_y + 8), (80, 80, 80), 1, cv2.LINE_AA)

    # Skeleton connections (green lines)
    for a, b in CONNECTIONS:
        cv2.line(img, px(lm[a]), px(lm[b]), (60, 220, 90), 2, cv2.LINE_AA)

    # Spine line (bright gold — the money line for golf posture)
    mid_shoulder = ((lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x) / 2,
                    (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2)
    mid_hip = ((lm[LEFT_HIP].x + lm[RIGHT_HIP].x) / 2,
               (lm[LEFT_HIP].y + lm[RIGHT_HIP].y) / 2)
    ms_px = (int(mid_shoulder[0] * w), int(mid_shoulder[1] * h))
    mh_px = (int(mid_hip[0] * w), int(mid_hip[1] * h))

    # Extended spine line — short extension just above the head
    if reference_lines:
        dx = ms_px[0] - mh_px[0]
        dy = ms_px[1] - mh_px[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length > 0:
            ext_x = int(ms_px[0] + (dx / length) * 30)
            ext_y = int(ms_px[1] + (dy / length) * 30)
            cv2.line(img, ms_px, (ext_x, ext_y), (255, 200, 60), 1, cv2.LINE_AA)

    cv2.line(img, mh_px, ms_px, (255, 200, 60), 3, cv2.LINE_AA)

    # Joints (yellow dots)
    key_points = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
                  LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
                  LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]
    for i in key_points:
        cv2.circle(img, px(lm[i]), 4, (60, 200, 240), -1, cv2.LINE_AA)

    # Shoulder + hip reference lines drawn ABOVE the skeleton so they show clearly.
    # Thicker + dark outline for readability against the green skeleton bones.
    if reference_lines:
        ls_px = (int(lm[LEFT_SHOULDER].x * w), int(lm[LEFT_SHOULDER].y * h))
        rs_px = (int(lm[RIGHT_SHOULDER].x * w), int(lm[RIGHT_SHOULDER].y * h))
        cv2.line(img, ls_px, rs_px, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(img, ls_px, rs_px, (240, 180, 100), 3, cv2.LINE_AA)

        lh_px = (int(lm[LEFT_HIP].x * w), int(lm[LEFT_HIP].y * h))
        rh_px = (int(lm[RIGHT_HIP].x * w), int(lm[RIGHT_HIP].y * h))
        cv2.line(img, lh_px, rh_px, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(img, lh_px, rh_px, (60, 140, 240), 3, cv2.LINE_AA)

    # Angle annotations
    if annotate and metrics:
        # Prefer 3D spine tilt when available
        spine = metrics.get("spine_tilt_3d") or metrics.get("spine_tilt")
        if spine is not None:
            # Angle arc at mid-hip showing spine tilt from vertical
            _draw_angle_arc(img, mh_px, spine, w, h)
            label = f"Spine {spine:.0f}deg"
            tx = (ms_px[0] + mh_px[0]) // 2 + 10
            ty = (ms_px[1] + mh_px[1]) // 2
            _draw_label(img, label, (tx, ty), (255, 200, 60))

        # X-factor at top of frame (small badge, only when significant)
        xf = metrics.get("x_factor_3d") or metrics.get("x_factor")
        if xf is not None and xf > 10:
            _draw_label(img, f"X-Factor {xf:.0f}deg", (10, 22), (200, 240, 200))

    return img


def _draw_angle_arc(img: np.ndarray, center: tuple[int, int], angle_deg: float, w: int, h: int):
    """Draw a small arc showing the angle between vertical and the spine at the hip."""
    radius = 22
    # Vertical reference (pointing up): -90 in OpenCV angle system (0 = right, 90 = down)
    start_angle = -90
    end_angle = -90 - angle_deg  # tilt clockwise/counterclockwise
    cv2.ellipse(img, center, (radius, radius), 0, start_angle, end_angle, (255, 200, 60), 2, cv2.LINE_AA)


def _draw_label(img, text: str, pt: tuple[int, int], color: tuple[int, int, int]):
    """Draw a label with a dark backdrop for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = pt
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
