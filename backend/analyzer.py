"""Full-video pose analysis: sample densely, detect real swing events, pick key frames."""
import base64
import os
import time
import uuid
import cv2
import numpy as np

from pose import detect_pose, compute_metrics, draw_skeleton
from swing import detect_phases_from_events, generate_feedback, describe_frame
from club import detect_shaft, draw_shaft, analyze_shaft_plane

RENDERS_DIR = os.path.join(os.path.dirname(__file__), "renders")
os.makedirs(RENDERS_DIR, exist_ok=True)

# We sample the video densely to detect actual swing events (top of backswing, impact),
# then pick a set of key frames spanning the real swing rather than the whole video length.
MAX_SAMPLES = 60         # cap so we don't spend forever on long videos
NUM_KEY_FRAMES = 9       # frames returned to the frontend


def _sample_indices(total_frames: int, max_samples: int) -> list[int]:
    if total_frames <= max_samples:
        return list(range(total_frames))
    step = total_frames / max_samples
    return [int(i * step) for i in range(max_samples)]


def _read_frame(cap, idx: int, max_width: int = 640) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        return None
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))
    return frame


def _wrist_speeds(samples: list[dict]) -> list[float]:
    """Per-sample wrist speed (Euclidean distance from previous valid sample's wrist)."""
    speeds = [0.0] * len(samples)
    last = None
    for i, s in enumerate(samples):
        m = s["metrics"]
        if m is None:
            speeds[i] = 0.0
            continue
        cur = (m["wrist_x"], m["wrist_y"])
        if last is not None:
            dx = cur[0] - last[0]
            dy = cur[1] - last[1]
            speeds[i] = (dx * dx + dy * dy) ** 0.5
        last = cur
    return speeds


def _smooth(values: list[float], window: int = 3) -> list[float]:
    """Simple boxcar smoothing."""
    n = len(values)
    out = [0.0] * n
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = sum(values[lo:hi]) / (hi - lo)
    return out


def _detect_swing_window(samples: list[dict]) -> tuple[int, int]:
    """Find the sample indices where the actual swing starts and ends.

    Strategy: compute wrist speed across all samples, find the peak, then walk
    outward until speed drops below a fraction of the peak. That's the swing.
    """
    n = len(samples)
    if n == 0:
        return 0, 0

    raw_speeds = _wrist_speeds(samples)
    speeds = _smooth(raw_speeds, window=3)

    peak_speed = max(speeds) if speeds else 0
    if peak_speed == 0:
        # No motion detected — treat the whole thing as the swing
        return 0, n - 1

    peak_i = speeds.index(peak_speed)
    threshold = peak_speed * 0.15  # ~15% of peak speed = "moving"

    # Walk backward from peak to find start
    start = peak_i
    for i in range(peak_i - 1, -1, -1):
        if speeds[i] < threshold:
            start = i
            break
        start = i

    # Walk forward from peak to find end
    end = peak_i
    for i in range(peak_i + 1, n):
        if speeds[i] < threshold:
            end = i
            break
        end = i

    # Pad slightly so we catch the true address and finish frames
    pad = max(1, n // 30)
    start = max(0, start - pad)
    end = min(n - 1, end + pad)

    return start, end


def _detect_events(samples: list[dict]) -> dict:
    """Locate address, top-of-backswing, impact, and finish within the swing window."""
    n = len(samples)
    valid = [(i, s) for i, s in enumerate(samples) if s["metrics"] is not None]
    if not valid:
        return {
            "address": 0,
            "top": n // 3,
            "impact": 2 * n // 3,
            "finish": max(n - 1, 0),
            "window": (0, max(n - 1, 0)),
        }

    swing_start, swing_end = _detect_swing_window(samples)

    # Restrict "valid" to samples within the swing window
    window_valid = [(i, s) for i, s in valid if swing_start <= i <= swing_end]
    if not window_valid:
        window_valid = valid  # fallback

    # Top of backswing within the window
    top_i = min(window_valid, key=lambda x: x[1]["metrics"]["wrist_y"])[0]

    # Address = swing_start (or first valid sample at/after swing_start)
    address_i = swing_start
    for i, _ in window_valid:
        if i >= swing_start:
            address_i = i
            break

    # Impact = post-top frame where wrist_y is closest to address wrist_y
    address_wrist_y = samples[address_i]["metrics"]["wrist_y"]
    impact_i = None
    best_score = -1
    for i, s in window_valid:
        if i <= top_i:
            continue
        proximity = 1 - abs(s["metrics"]["wrist_y"] - address_wrist_y)
        if proximity > best_score:
            best_score = proximity
            impact_i = i

    # Finish = swing_end (or last valid sample within window)
    finish_i = swing_end
    for i, _ in reversed(window_valid):
        if i <= swing_end:
            finish_i = i
            break

    if impact_i is None:
        impact_i = min(top_i + (finish_i - top_i) // 2, finish_i)

    return {
        "address": address_i,
        "top": top_i,
        "impact": impact_i,
        "finish": finish_i,
        "window": (swing_start, swing_end),
    }


def _pick_key_indices(events: dict, total_samples: int) -> list[tuple[int, str]]:
    """Pick 9 sample indices spanning the actual swing, with a phase label for each."""
    a, t, imp, f = events["address"], events["top"], events["impact"], events["finish"]

    key = [
        (a, "Address / Setup"),
        (a + (t - a) // 3, "Takeaway"),
        (a + 2 * (t - a) // 3, "Backswing"),
        (t, "Top of Backswing"),
        (t + (imp - t) // 3, "Downswing"),
        (t + 2 * (imp - t) // 3, "Downswing"),
        (imp, "Impact"),
        (imp + (f - imp) // 2, "Follow-Through"),
        (f, "Finish"),
    ]

    # Deduplicate identical indices (short videos) while preserving order and phase priority
    seen = set()
    result = []
    for idx, phase in key:
        idx = max(0, min(idx, total_samples - 1))
        if idx not in seen:
            seen.add(idx)
            result.append((idx, phase))
    return result


def analyze_swing(video_path: str) -> dict:
    """Pipeline:
    1. Sample the video densely
    2. Run pose on every sample
    3. Detect real swing events (top, impact) from wrist trajectory
    4. Pick 9 key frames anchored to real events, return with metrics + feedback
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total == 0:
        raise ValueError("Video has no frames")

    sample_indices = _sample_indices(total, MAX_SAMPLES)

    # Densely sample: extract frame + pose + metrics + shaft
    samples = []
    prev_shaft_angle = None
    for idx in sample_indices:
        frame = _read_frame(cap, idx)
        if frame is None:
            samples.append({"frame": None, "landmarks": None, "metrics": None, "shaft": None, "video_idx": idx})
            continue
        pose_result = detect_pose(frame)
        if pose_result is None:
            samples.append({"frame": frame, "landmarks": None, "metrics": None, "shaft": None, "video_idx": idx})
            continue
        lm = pose_result["landmarks"]
        world_lm = pose_result["world_landmarks"]
        metrics = compute_metrics(lm, world_lm)
        # Drop low-confidence detections — occluded body poisons the metrics
        if metrics.get("confidence", 1.0) < 0.55:
            samples.append({"frame": frame, "landmarks": None, "metrics": None, "shaft": None, "video_idx": idx})
            continue

        # Club shaft — anchored to the wrists, temporal-consistency-aware
        shaft = detect_shaft(frame, lm, prev_angle=prev_shaft_angle)
        if shaft is not None:
            prev_shaft_angle = shaft["angle"]

        samples.append({"frame": frame, "landmarks": lm, "metrics": metrics, "shaft": shaft, "video_idx": idx})

    cap.release()

    # Temporal smoothing on numeric metrics — reduces frame-to-frame jitter
    _smooth_metrics(samples)

    # Detect actual swing events
    events = _detect_events(samples)
    key_selection = _pick_key_indices(events, len(samples))

    # Build per-frame entries from the selected key samples
    key_metrics_list = [samples[i]["metrics"] for i, _ in key_selection]
    key_phases = [phase for _, phase in key_selection]

    # If we detected no poses at all, run the fallback path
    poses_detected_total = sum(1 for s in samples if s["metrics"] is not None)
    if poses_detected_total == 0:
        # Return a helpful "couldn't analyze" response with raw frames
        frame_entries = []
        for i, (idx, phase) in enumerate(key_selection):
            s = samples[idx]
            frame_bgr = s["frame"] if s["frame"] is not None else np.zeros((240, 320, 3), dtype=np.uint8)
            frame_entries.append({
                "frame_index": i,
                "phase": "Unknown",
                "observation": "Could not detect body pose in this frame. Ensure the golfer is fully visible.",
                "metrics": None,
                "thumbnail": _encode_jpeg_b64(frame_bgr),
                "pose_detected": False,
            })
        feedback = generate_feedback(key_metrics_list, key_phases)
        return {
            "overall_assessment": feedback["overall_assessment"],
            "frames": frame_entries,
            "strengths": feedback["strengths"],
            "improvements": feedback["improvements"],
            "key_focus": feedback["key_focus"],
            "metrics_summary": feedback["metrics_summary"],
            "poses_detected": 0,
            "total_frames": len(key_selection),
            "camera_note": "Could not detect a person in the video. Camera-angle detection requires a visible full body.",
            "tempo": None,
            "swing_plane_image": None,
            "phase_scores": [],
            "handedness": "unknown",
            "overlay_video_url": None,
            "grade": None,
            "grade_band": None,
            "practice_plan": [],
            "shaft_plane": None,
            "shaft_detection_rate": 0.0,
        }

    frame_entries = []
    for i, (idx, phase) in enumerate(key_selection):
        s = samples[idx]
        frame_bgr = s["frame"]
        if s["landmarks"] is not None:
            annotated = draw_skeleton(frame_bgr, s["landmarks"], annotate=True, metrics=s["metrics"], reference_lines=True)
            if s.get("shaft") is not None:
                draw_shaft(annotated, s["shaft"])
        else:
            annotated = frame_bgr

        frame_entries.append({
            "frame_index": i,
            "phase": phase,
            "observation": describe_frame(phase, s["metrics"]),
            "metrics": s["metrics"],
            "thumbnail": _encode_jpeg_b64(annotated),
            "pose_detected": s["metrics"] is not None,
        })

    # Feedback uses only samples within the detected swing window — this filters
    # out pre-swing waggle and post-finish hold time, so metrics like head-movement
    # reflect the actual swing rather than the whole video.
    win_start, win_end = events["window"]
    window_samples = samples[win_start:win_end + 1]
    window_metrics = [s["metrics"] for s in window_samples]
    all_phases_full = detect_phases_from_events(len(samples), events)
    window_phases = all_phases_full[win_start:win_end + 1]
    feedback = generate_feedback(window_metrics, window_phases)

    # Swing tempo — real durations derived from FPS and sample indices
    tempo = _compute_tempo(samples, events, fps)

    # Shaft plane analysis from club-history across all samples
    club_history = [s.get("shaft") for s in samples]
    shaft_plane = analyze_shaft_plane(club_history, events)
    shaft_detection_rate = sum(1 for s in club_history if s is not None and s.get("method") == "hough") / max(len(club_history), 1)

    # Swing plane trace — wrist trajectory summary + club-tip trajectory
    plane_image_b64 = _make_swing_plane_image(samples, events)

    # Overlay video — replay the swing itself with skeleton + shaft drawn on every frame
    overlay_video_url = _generate_overlay_video(video_path, samples, events, fps)

    return {
        "overall_assessment": feedback["overall_assessment"],
        "frames": frame_entries,
        "strengths": feedback["strengths"],
        "improvements": feedback["improvements"],
        "key_focus": feedback["key_focus"],
        "metrics_summary": feedback["metrics_summary"],
        "poses_detected": poses_detected_total,
        "total_frames": len(samples),
        "camera_note": feedback.get("camera_note"),
        "tempo": tempo,
        "swing_plane_image": plane_image_b64,
        "phase_scores": feedback.get("phase_scores", []),
        "handedness": feedback.get("handedness", "unknown"),
        "overlay_video_url": overlay_video_url,
        "grade": feedback.get("grade"),
        "grade_band": feedback.get("grade_band"),
        "practice_plan": feedback.get("practice_plan", []),
        "shaft_plane": shaft_plane,
        "shaft_detection_rate": round(shaft_detection_rate * 100, 1),
    }


def _smooth_metrics(samples: list[dict], window: int = 3):
    """In-place moving-average smoothing on numeric metrics across valid samples.

    Only smooths continuous quantities where jitter matters (angles, positions);
    leaves confidence and discrete fields alone.
    """
    numeric_keys = [
        "spine_tilt", "shoulder_tilt", "hip_tilt", "x_factor",
        "left_knee_angle", "right_knee_angle",
        "left_elbow_angle", "right_elbow_angle",
        "stance_width", "shoulder_width", "hip_center_x",
        "wrist_y", "wrist_x", "head_x", "head_y",
        "spine_tilt_3d", "x_factor_3d", "shoulder_rot_3d", "hip_rot_3d",
    ]
    half = window // 2
    n = len(samples)
    original = [s.get("metrics") for s in samples]

    for i in range(n):
        m = original[i]
        if m is None:
            continue
        smoothed = dict(m)
        for k in numeric_keys:
            if k not in m:
                continue
            vals = []
            for j in range(max(0, i - half), min(n, i + half + 1)):
                if original[j] is not None and k in original[j]:
                    vals.append(original[j][k])
            if vals:
                smoothed[k] = round(sum(vals) / len(vals), 3 if abs(smoothed[k]) < 5 else 1)
        samples[i]["metrics"] = smoothed


def _encode_jpeg_b64(image_bgr: np.ndarray, quality: int = 85) -> str:
    _, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.standard_b64encode(buffer.tobytes()).decode("utf-8")


def _compute_tempo(samples: list[dict], events: dict, fps: float) -> dict:
    """Compute backswing/downswing durations and the classic tempo ratio.

    Tour pros average ~3:1 (backswing takes ~3x as long as downswing).
    """
    a_idx = events["address"]
    t_idx = events["top"]
    imp_idx = events["impact"]

    if a_idx is None or t_idx is None or imp_idx is None:
        return None

    a_video = samples[a_idx]["video_idx"]
    t_video = samples[t_idx]["video_idx"]
    imp_video = samples[imp_idx]["video_idx"]

    if fps <= 0:
        return None

    backswing_s = max(0.0, (t_video - a_video) / fps)
    downswing_s = max(0.001, (imp_video - t_video) / fps)  # avoid div/0
    total_s = backswing_s + downswing_s
    ratio = backswing_s / downswing_s

    # Verdict on ratio
    if ratio >= 2.8 and ratio <= 3.4:
        verdict = "Excellent tempo — right in the tour-pro 3:1 zone."
    elif ratio >= 2.3 and ratio < 2.8:
        verdict = "Solid tempo — a touch quicker than the classic 3:1, but well within a normal range."
    elif ratio > 3.4 and ratio <= 4.0:
        verdict = "Slightly slow tempo — backswing is on the longer side relative to the downswing."
    elif ratio < 2.3:
        verdict = "Quick transition — downswing is fast relative to the backswing. Can work, but often costs consistency."
    else:
        verdict = "Unusual tempo ratio — worth double-checking the event detection on this video."

    return {
        "backswing_s": round(backswing_s, 2),
        "downswing_s": round(downswing_s, 2),
        "total_s": round(total_s, 2),
        "ratio": round(ratio, 2),
        "verdict": verdict,
    }


def _generate_overlay_video(video_path: str, samples: list[dict], events: dict, fps: float) -> str | None:
    """Render an MP4 of the swing window with skeleton drawn on every frame.

    Stored under backend/renders/<uuid>.mp4. Returns the URL path (served by the
    static mount in main.py). Old renders are cleaned up on each call.
    """
    win_start, win_end = events["window"]
    if win_start is None or win_end is None:
        return None

    start_video = samples[win_start]["video_idx"]
    end_video = samples[min(win_end, len(samples) - 1)]["video_idx"]
    if end_video <= start_video:
        return None

    # Cleanup: delete render files older than 1 hour (any extension)
    now = time.time()
    for name in os.listdir(RENDERS_DIR):
        path = os.path.join(RENDERS_DIR, name)
        try:
            if now - os.path.getmtime(path) > 3600:
                os.unlink(path)
        except OSError:
            pass

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_video)
    ret, first = cap.read()
    if not ret:
        cap.release()
        return None

    # Downscale to a reasonable output size
    h, w = first.shape[:2]
    max_w = 720
    if w > max_w:
        scale = max_w / w
        out_w, out_h = max_w, int(h * scale)
    else:
        scale = 1.0
        out_w, out_h = w, h

    render_id = uuid.uuid4().hex
    # WebM/VP8 — H.264 encoding in this OpenCV build requires the OpenH264 DLL
    # which isn't installed. All modern browsers play WebM natively.
    out_path = os.path.join(RENDERS_DIR, f"{render_id}.webm")

    playback_fps = max(fps * 0.5, 15)
    fourcc = cv2.VideoWriter_fourcc(*"VP80")
    writer = cv2.VideoWriter(out_path, fourcc, playback_fps, (out_w, out_h))
    if not writer.isOpened():
        # Fallback to mp4v — won't play in browsers but at least the file exists
        writer.release()
        out_path = os.path.join(RENDERS_DIR, f"{render_id}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, playback_fps, (out_w, out_h))
        if not writer.isOpened():
            cap.release()
            return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_video)
    frame_idx = start_video
    # Mutable holder for temporal-consistency angle across the loop
    overlay_prev_angle = {"v": None}
    while frame_idx <= end_video:
        ret, frame = cap.read()
        if not ret:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h))

        pose_result = detect_pose(frame)
        if pose_result is not None:
            lm = pose_result["landmarks"]
            world_lm = pose_result["world_landmarks"]
            m = compute_metrics(lm, world_lm)
            annotated = draw_skeleton(frame, lm, annotate=True, metrics=m, reference_lines=True)
            # Detect the shaft on every rendered frame — expensive but the whole
            # point of the annotated video is to actually see the club
            shaft = detect_shaft(frame, lm, prev_angle=overlay_prev_angle["v"])
            if shaft is not None:
                overlay_prev_angle["v"] = shaft["angle"]
                draw_shaft(annotated, shaft)
        else:
            annotated = frame

        writer.write(annotated)
        frame_idx += 1

    writer.release()
    cap.release()

    # Return the URL matching whatever extension we actually wrote
    return f"/renders/{os.path.basename(out_path)}"


def _make_swing_plane_image(samples: list[dict], events: dict, width: int = 480, height: int = 320) -> str | None:
    """Draw the wrist trajectory across the swing on a blank canvas.

    Backswing shown in blue, downswing in orange, so you can see if they
    trace the same plane (on-plane) or diverge (off-plane).
    """
    top_i = events["top"]
    if top_i is None:
        return None

    win_start, win_end = events["window"]
    canvas = np.full((height, width, 3), 20, dtype=np.uint8)

    # Extract wrist trajectory (grip) AND club-tip trajectory within the swing window
    back_pts = []      # grip: address → top
    down_pts = []      # grip: top → finish
    back_tips = []     # clubhead: address → top
    down_tips = []     # clubhead: top → finish
    for i in range(win_start, win_end + 1):
        m = samples[i]["metrics"]
        if m is None:
            continue
        x, y = int(m["wrist_x"] * width), int(m["wrist_y"] * height)
        if i <= top_i:
            back_pts.append((x, y))
        else:
            down_pts.append((x, y))

        # Club tip in image-normalized coords: we stored pixels, so normalize by original frame
        shaft = samples[i].get("shaft")
        if shaft is not None and samples[i]["frame"] is not None:
            fh, fw = samples[i]["frame"].shape[:2]
            tx = int(shaft["tip"][0] / max(fw, 1) * width)
            ty = int(shaft["tip"][1] / max(fh, 1) * height)
            if i <= top_i:
                back_tips.append((tx, ty))
            else:
                down_tips.append((tx, ty))

    if len(back_pts) < 2 and len(down_pts) < 2:
        return None

    # Grid
    for gx in range(0, width, 40):
        cv2.line(canvas, (gx, 0), (gx, height), (40, 40, 40), 1)
    for gy in range(0, height, 40):
        cv2.line(canvas, (0, gy), (width, gy), (40, 40, 40), 1)

    # Backswing path (soft blue)
    for j in range(1, len(back_pts)):
        cv2.line(canvas, back_pts[j - 1], back_pts[j], (240, 180, 100), 3, cv2.LINE_AA)

    # Downswing path (warm orange)
    for j in range(1, len(down_pts)):
        cv2.line(canvas, down_pts[j - 1], down_pts[j], (60, 140, 240), 3, cv2.LINE_AA)

    # Dots at each sample position
    for (x, y) in back_pts:
        cv2.circle(canvas, (x, y), 4, (240, 180, 100), -1, cv2.LINE_AA)
    for (x, y) in down_pts:
        cv2.circle(canvas, (x, y), 4, (60, 140, 240), -1, cv2.LINE_AA)

    # Club-tip trajectory (thinner and brighter) drawn on top
    for j in range(1, len(back_tips)):
        cv2.line(canvas, back_tips[j - 1], back_tips[j], (200, 220, 255), 1, cv2.LINE_AA)
    for j in range(1, len(down_tips)):
        cv2.line(canvas, down_tips[j - 1], down_tips[j], (80, 220, 255), 1, cv2.LINE_AA)
    for pt in back_tips + down_tips:
        cv2.circle(canvas, pt, 2, (200, 220, 255), -1, cv2.LINE_AA)

    # Legend
    cv2.rectangle(canvas, (10, 10), (24, 22), (240, 180, 100), -1)
    cv2.putText(canvas, "Grip - back", (30, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (10, 30), (24, 42), (60, 140, 240), -1)
    cv2.putText(canvas, "Grip - down", (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
    if back_tips or down_tips:
        cv2.line(canvas, (10, 55), (24, 55), (80, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Clubhead path", (30, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

    return _encode_jpeg_b64(canvas, quality=90)
