"""Swing phase detection, comprehensive rule-based feedback, and phase scorecard.

Feedback here is intentionally hedged where the 2D pose signal is noisy, but we
still cover many more checkpoints than the original engine so the user gets a
real breakdown of what's working and what isn't across every phase of the swing.
"""
import math


# ---------- Phase detection ----------

def detect_phases_from_events(num_samples: int, events: dict) -> list[str]:
    a, t, imp, f = events["address"], events["top"], events["impact"], events["finish"]
    phases = []
    for i in range(num_samples):
        if i < a:
            phases.append("Address / Setup")
        elif i < a + max(1, (t - a) // 3):
            phases.append("Address / Setup")
        elif i < a + 2 * max(1, (t - a) // 3):
            phases.append("Takeaway")
        elif i < t:
            phases.append("Backswing")
        elif i == t:
            phases.append("Top of Backswing")
        elif i < imp:
            phases.append("Downswing")
        elif i == imp:
            phases.append("Impact")
        elif i < imp + max(1, (f - imp) // 2):
            phases.append("Follow-Through")
        else:
            phases.append("Finish")
    return phases


# ---------- Utility ----------

def _detect_camera_view(metrics_list: list[dict]) -> str:
    tilts = [abs(m["shoulder_tilt"]) for m in metrics_list if m is not None]
    if len(tilts) < 5:
        return "unknown"
    tilt_range = max(tilts) - min(tilts)
    if tilt_range > 80:
        return "down-the-line"
    if tilt_range < 40:
        return "face-on"
    return "unknown"


def _detect_handedness(metrics_list: list[dict], phases: list[str]) -> str:
    """Guess right- vs left-handed from the direction the wrists travel during the backswing."""
    address = _phase_avg(metrics_list, phases, "Address / Setup", "wrist_x")
    top = _phase_avg(metrics_list, phases, "Top of Backswing", "wrist_x")
    if address is None or top is None:
        return "unknown"
    delta = top - address
    # Right-handed golfer: at top the wrists are on the trail (right) side, which is
    # the viewer's left in face-on view → lower x. Down-the-line view is more variable
    # so this heuristic works best face-on.
    if delta < -0.05:
        return "right"
    if delta > 0.05:
        return "left"
    return "unknown"


def _phase_avg(metrics_list, phases, phase_name, key):
    vals = [m[key] for m, p in zip(metrics_list, phases) if m is not None and p == phase_name and key in m]
    return sum(vals) / len(vals) if vals else None


def _phase_range(metrics_list, phases, phase_name, key):
    vals = [m[key] for m, p in zip(metrics_list, phases) if m is not None and p == phase_name and key in m]
    if not vals:
        return None
    return max(vals) - min(vals)


def describe_frame(phase: str, metrics: dict) -> str:
    if metrics is None:
        return "Could not detect body pose in this frame."
    spine = metrics["spine_tilt"]
    knee_avg = (metrics["left_knee_angle"] + metrics["right_knee_angle"]) / 2
    if phase == "Address / Setup":
        return f"Setup posture — {spine:.0f}° spine tilt, knees flexed at ~{knee_avg:.0f}° average."
    if phase == "Takeaway":
        return f"Club moving away from the ball. Spine at {spine:.0f}°."
    if phase == "Backswing":
        return f"Body rotating into the backswing. Spine {spine:.0f}°."
    if phase == "Top of Backswing":
        return f"Top position — body is fully coiled. Spine {spine:.0f}°."
    if phase == "Downswing":
        return f"Transitioning down toward the ball. Spine {spine:.0f}°."
    if phase == "Impact":
        return f"Near impact — spine {spine:.0f}°, knees at ~{knee_avg:.0f}°."
    if phase == "Follow-Through":
        return f"Body continuing to rotate through the ball. Spine {spine:.0f}°."
    if phase == "Finish":
        return f"Finish position — knees at ~{knee_avg:.0f}° average."
    return f"Spine {spine:.0f}°."


# ---------- Phase scorecard: each phase gets a status + notes ----------
# Status can be: 'good', 'watch', 'issue', 'unknown'

def _rate_setup(metrics_list, phases, handedness):
    """Setup checkpoints: stance width, spine tilt, knee flex, weight balance."""
    findings = []
    status = "unknown"

    address_spine = _phase_avg(metrics_list, phases, "Address / Setup", "spine_tilt_3d") \
        or _phase_avg(metrics_list, phases, "Address / Setup", "spine_tilt")
    address_knees_l = _phase_avg(metrics_list, phases, "Address / Setup", "left_knee_angle")
    address_knees_r = _phase_avg(metrics_list, phases, "Address / Setup", "right_knee_angle")
    stance = _phase_avg(metrics_list, phases, "Address / Setup", "stance_width")
    shoulders = _phase_avg(metrics_list, phases, "Address / Setup", "shoulder_width")

    if address_spine is None:
        return "unknown", "Could not measure setup posture."

    # Spine tilt — tightened acceptance window (was 20-40; 20 is quite upright)
    if 25 <= address_spine <= 38:
        findings.append({"kind": "good", "text": f"Athletic forward spine tilt of {address_spine:.0f}°."})
    elif 20 <= address_spine < 25 or 38 < address_spine <= 45:
        findings.append({"kind": "watch", "text": f"Spine tilt of {address_spine:.0f}° is workable but a touch off the athletic ~30° target."})
    elif address_spine < 20:
        findings.append({"kind": "issue", "text": f"Spine looks upright ({address_spine:.0f}°) — could be low bend from the hips.",
                         "drill": "Hinge from the hips, not the waist, until your hands hang under your shoulders. Practice in a mirror."})
    else:
        findings.append({"kind": "issue", "text": f"Spine is very tilted forward ({address_spine:.0f}°) — possibly hunched.",
                         "drill": "Stand tall, then hinge from hips only. Keep chest up and shoulder blades pulled back."})

    # Knee flex (average of both) — tightened: 165° was too generous
    if address_knees_l is not None and address_knees_r is not None:
        knee_avg = (address_knees_l + address_knees_r) / 2
        if 150 <= knee_avg <= 162:
            findings.append({"kind": "good", "text": f"Knees are athletically flexed (~{knee_avg:.0f}°)."})
        elif knee_avg < 150:
            findings.append({"kind": "watch", "text": f"Knees look deeply bent ({knee_avg:.0f}°) — might be a low-camera-angle artifact or genuine crouch."})
        elif knee_avg > 175:
            findings.append({"kind": "issue", "text": f"Knees are nearly locked at address ({knee_avg:.0f}°).",
                             "drill": "Soften the knees so you feel a slight sit — should be around 160-165°. Bounce lightly in your stance to find the position."})

    # Stance vs shoulder width
    if stance is not None and shoulders is not None and shoulders > 0.05:
        ratio = stance / shoulders
        if ratio < 0.8:
            findings.append({"kind": "watch", "text": f"Stance is narrower than shoulder width (~{ratio:.1f}x). Can hurt balance on longer clubs.",
                             "drill": "For irons, feet just outside shoulder width. Widen slightly for driver."})
        elif ratio > 1.6:
            findings.append({"kind": "watch", "text": f"Stance is much wider than shoulders (~{ratio:.1f}x). Can limit hip turn.",
                             "drill": "Narrow slightly so your ankles are roughly under your shoulders on iron shots."})

    # Determine overall status
    kinds = [f["kind"] for f in findings]
    if "issue" in kinds:
        status = "issue"
    elif "watch" in kinds:
        status = "watch"
    else:
        status = "good"

    return status, findings


def _rate_backswing(metrics_list, phases, camera_view, handedness):
    """Backswing checkpoints: head sway, hip sway, trail leg stability."""
    findings = []

    # Head sway between address and top — tightened: 3% is a real "good"
    addr_head_x = _phase_avg(metrics_list, phases, "Address / Setup", "head_x")
    top_head_x = _phase_avg(metrics_list, phases, "Top of Backswing", "head_x")
    if addr_head_x is not None and top_head_x is not None:
        sway = abs(top_head_x - addr_head_x)
        if sway < 0.025:
            findings.append({"kind": "good", "text": f"Head stays very stable during the backswing (moves <2.5% of frame)."})
        elif sway > 0.08:
            findings.append({"kind": "issue", "text": f"Head slides ~{sway*100:.0f}% of frame width during the backswing — a swaying move.",
                             "drill": "Feel your head stay centered over the ball while you turn. Use a mirror or have someone hold a club vertically behind you."})

    # Hip sway
    addr_hip_x = _phase_avg(metrics_list, phases, "Address / Setup", "hip_center_x")
    top_hip_x = _phase_avg(metrics_list, phases, "Top of Backswing", "hip_center_x")
    if addr_hip_x is not None and top_hip_x is not None:
        hip_sway = abs(top_hip_x - addr_hip_x)
        if hip_sway > 0.10:
            findings.append({"kind": "watch", "text": f"Hips slide sideways ~{hip_sway*100:.0f}% during the backswing rather than rotating.",
                             "drill": "Trail-side glute drill — feel your trail cheek load into a wall or chair as you turn. Rotate, don't slide."})

    # Trail knee stability — for a right-handed golfer, right knee should stay flexed
    trail_side = "right" if handedness == "right" else "left" if handedness == "left" else None
    if trail_side:
        key = f"{trail_side}_knee_angle"
        addr_knee = _phase_avg(metrics_list, phases, "Address / Setup", key)
        top_knee = _phase_avg(metrics_list, phases, "Top of Backswing", key)
        if addr_knee is not None and top_knee is not None:
            straightening = top_knee - addr_knee
            if straightening > 15:
                findings.append({"kind": "issue", "text": f"Trail leg straightens noticeably during the backswing (+{straightening:.0f}°).",
                                 "drill": "Keep the flex in your trail knee from address to the top. Feel it stay 'stuck' to the ground."})
            elif abs(straightening) < 5:
                findings.append({"kind": "good", "text": "Trail leg stays flexed and stable during the backswing."})

    if not findings:
        return "unknown", "Not enough data on the backswing to evaluate."

    kinds = [f["kind"] for f in findings]
    if "issue" in kinds:
        status = "issue"
    elif "watch" in kinds:
        status = "watch"
    else:
        status = "good"
    return status, findings


def _rate_top(metrics_list, phases, camera_view, handedness):
    """Top of backswing: x-factor, lead arm straightness, over-swing check."""
    findings = []

    # X-factor — hedge on non-face-on views
    top_x = _phase_avg(metrics_list, phases, "Top of Backswing", "x_factor_3d") \
        or _phase_avg(metrics_list, phases, "Top of Backswing", "x_factor")
    if top_x is not None and camera_view == "face-on":
        if top_x >= 35:
            findings.append({"kind": "good", "text": f"Excellent shoulder-hip separation of {top_x:.0f}° — great coil."})
        elif top_x >= 20:
            findings.append({"kind": "good", "text": f"Solid coil ({top_x:.0f}° shoulder-hip separation)."})
        elif top_x >= 10:
            findings.append({"kind": "watch", "text": f"Modest coil ({top_x:.0f}°) — more separation between shoulders and hips would add power.",
                             "drill": "Feel your shoulders turn while your hips resist. Rehearse with a mid-iron and a slower tempo."})

    # Lead arm straightness at top
    lead_side = "left" if handedness == "right" else "right" if handedness == "left" else None
    if lead_side:
        key = f"{lead_side}_elbow_angle"
        top_lead_elbow = _phase_avg(metrics_list, phases, "Top of Backswing", key)
        if top_lead_elbow is not None:
            if top_lead_elbow >= 155:
                findings.append({"kind": "good", "text": f"Lead arm stays extended at the top ({top_lead_elbow:.0f}° at elbow)."})
            elif top_lead_elbow < 130:
                findings.append({"kind": "issue", "text": f"Lead arm is bent quite a bit at the top ({top_lead_elbow:.0f}° at elbow) — losing swing width.",
                                 "drill": "Feel the lead arm extend as you complete the backswing. A towel under the lead armpit drill can help."})

    # Head position at top
    addr_head_y = _phase_avg(metrics_list, phases, "Address / Setup", "head_y")
    top_head_y = _phase_avg(metrics_list, phases, "Top of Backswing", "head_y")
    if addr_head_y is not None and top_head_y is not None:
        head_drop = addr_head_y - top_head_y  # negative if head rises
        if head_drop < -0.04:
            findings.append({"kind": "watch", "text": f"Head appears to rise ~{abs(head_drop)*100:.0f}% during the backswing.",
                             "drill": "Feel your head stay at its address height throughout the turn. Standing up early is a common power leak."})

    if not findings:
        return "unknown", "Not enough data at the top of the backswing."
    kinds = [f["kind"] for f in findings]
    status = "issue" if "issue" in kinds else "watch" if "watch" in kinds else "good"
    return status, findings


def _rate_downswing_impact(metrics_list, phases, camera_view, handedness):
    """Downswing/impact: spine retention, hip lead, weight shift, head stability."""
    findings = []

    addr_spine = _phase_avg(metrics_list, phases, "Address / Setup", "spine_tilt")
    impact_spine = _phase_avg(metrics_list, phases, "Impact", "spine_tilt_3d") \
        or _phase_avg(metrics_list, phases, "Impact", "spine_tilt")

    # Spine retention (only trustworthy face-on)
    if camera_view == "face-on" and addr_spine is not None and impact_spine is not None:
        change = addr_spine - impact_spine
        if change <= 5:
            findings.append({"kind": "good", "text": f"Spine angle stays consistent to impact (~{change:.0f}° change)."})
        elif change <= 12:
            findings.append({"kind": "watch", "text": f"Spine straightens ~{change:.0f}° through impact — early-extension tendency.",
                             "drill": "Rear-against-wall drill — set up with your seat lightly touching a wall and rehearse swings keeping contact."})
        else:
            findings.append({"kind": "issue", "text": f"Significant early extension — spine drops ~{change:.0f}° from setup to impact.",
                             "drill": "Foam roller between lower back and wall, keep contact through the swing. Do this daily until it feels natural."})

    # Weight shift to lead side by impact
    addr_hip_x = _phase_avg(metrics_list, phases, "Address / Setup", "hip_center_x")
    imp_hip_x = _phase_avg(metrics_list, phases, "Impact", "hip_center_x")
    if handedness in ("right", "left") and addr_hip_x is not None and imp_hip_x is not None:
        # For right-handed, target is to viewer's right (higher x). Lead-side hip shift = positive delta.
        expected_direction = 1 if handedness == "right" else -1
        shift = (imp_hip_x - addr_hip_x) * expected_direction
        if shift >= 0.04:
            findings.append({"kind": "good", "text": "Nice pressure shift toward the lead side by impact."})
        elif shift <= -0.04:
            findings.append({"kind": "issue", "text": "Weight looks stuck on the trail side at impact — reverse pivot signs.",
                             "drill": "'Step-through' drill: start feet together, step toward the target with your lead foot as you begin the downswing."})

    # Head position at impact should still be near address
    addr_head_x = _phase_avg(metrics_list, phases, "Address / Setup", "head_x")
    imp_head_x = _phase_avg(metrics_list, phases, "Impact", "head_x")
    if addr_head_x is not None and imp_head_x is not None:
        drift = abs(imp_head_x - addr_head_x)
        if drift < 0.03:
            findings.append({"kind": "good", "text": "Head stays over the ball into impact."})
        elif drift > 0.09:
            findings.append({"kind": "watch", "text": f"Head drifts ~{drift*100:.0f}% by impact — try to stay behind the ball.",
                             "drill": "Slow-motion swings with a friend holding a club vertically at your head — swing without moving into it."})

    if not findings:
        return "unknown", "Not enough data at the impact phase."
    kinds = [f["kind"] for f in findings]
    status = "issue" if "issue" in kinds else "watch" if "watch" in kinds else "good"
    return status, findings


def _rate_finish(metrics_list, phases, handedness):
    """Finish: full rotation, lead-leg extension, balance."""
    findings = []

    lead_side = "left" if handedness == "right" else "right" if handedness == "left" else None

    if lead_side:
        knee_key = f"{lead_side}_knee_angle"
        finish_lead_knee = _phase_avg(metrics_list, phases, "Finish", knee_key)
        if finish_lead_knee is not None:
            if finish_lead_knee >= 165:
                findings.append({"kind": "good", "text": "Lead leg extends fully at the finish — proper post-impact posture."})
            elif finish_lead_knee < 145:
                findings.append({"kind": "watch", "text": f"Lead leg is still bent at the finish ({finish_lead_knee:.0f}°).",
                                 "drill": "Hold your finish for 3 seconds after every shot — belt buckle to target, weight ~95% on the lead foot."})

    # Finish head vs finish ankle center — proxy for balance
    finish_metrics = [m for m, p in zip(metrics_list, phases) if m is not None and p == "Finish"]
    if finish_metrics:
        # simple: head_x should be close to the lead-ankle x
        pass  # skip — noisy signal

    # Full rotation via hip_tilt sign change (rough)
    addr_hip_tilt = _phase_avg(metrics_list, phases, "Address / Setup", "hip_tilt")
    finish_hip_tilt = _phase_avg(metrics_list, phases, "Finish", "hip_tilt")
    if addr_hip_tilt is not None and finish_hip_tilt is not None:
        rotation = abs(finish_hip_tilt - addr_hip_tilt)
        if rotation > 30:
            findings.append({"kind": "good", "text": "Full body rotation into the finish — hips have turned through."})
        elif rotation < 10:
            findings.append({"kind": "watch", "text": "Body doesn't appear to fully rotate to the target at finish.",
                             "drill": "Finish drill: after each swing, touch your right heel with your trail hand (right-handed) to force full hip turn."})

    if not findings:
        return "unknown", "Not enough data at the finish."
    kinds = [f["kind"] for f in findings]
    status = "issue" if "issue" in kinds else "watch" if "watch" in kinds else "good"
    return status, findings


# ---------- Main entry point ----------

def generate_feedback(metrics_list: list[dict], phases: list[str]) -> dict:
    valid = [(m, p) for m, p in zip(metrics_list, phases) if m is not None]
    if not valid:
        return {
            "overall_assessment": "Could not detect a person in the video. Make sure the golfer is fully visible in frame, well-lit, and not obstructed.",
            "strengths": [],
            "improvements": [],
            "key_focus": "Re-record with the full body in frame, ideally from a face-on or down-the-line angle at ~10 feet away.",
            "metrics_summary": {},
            "camera_note": None,
            "phase_scores": [],
            "handedness": "unknown",
        }

    camera_view = _detect_camera_view([m for m, _ in valid])
    handedness = _detect_handedness(metrics_list, phases)

    # Run every phase evaluator
    scorecard = [
        ("Setup", *_rate_setup(metrics_list, phases, handedness)),
        ("Backswing", *_rate_backswing(metrics_list, phases, camera_view, handedness)),
        ("Top", *_rate_top(metrics_list, phases, camera_view, handedness)),
        ("Downswing / Impact", *_rate_downswing_impact(metrics_list, phases, camera_view, handedness)),
        ("Finish", *_rate_finish(metrics_list, phases, handedness)),
    ]

    # Collect all strengths and improvements from all phases
    strengths = []
    improvements = []
    phase_scores = []
    for phase_name, status, findings in scorecard:
        if isinstance(findings, str):
            phase_scores.append({"phase": phase_name, "status": status, "notes": findings, "findings": []})
            continue
        finding_summaries = []
        for f in findings:
            finding_summaries.append({"kind": f["kind"], "text": f["text"], "drill": f.get("drill")})
            if f["kind"] == "good":
                strengths.append(f["text"])
            else:
                # 'watch' and 'issue' both become improvements, but issues rank higher
                improvements.append({
                    "issue": f["text"],
                    "drill": f.get("drill", ""),
                    "severity": "high" if f["kind"] == "issue" else "medium",
                    "phase": phase_name,
                })
        phase_scores.append({
            "phase": phase_name,
            "status": status,
            "notes": None,
            "findings": finding_summaries,
        })

    # Sort improvements: issues first, then watches
    improvements.sort(key=lambda x: 0 if x["severity"] == "high" else 1)

    # Head movement (kept as separate high-level metric)
    head_xs = [m["head_x"] for m, _ in valid]
    head_ys = [m["head_y"] for m, _ in valid]
    head_movement = math.sqrt((max(head_xs) - min(head_xs)) ** 2 + (max(head_ys) - min(head_ys)) ** 2)

    # Fundamentals score — measures visible body mechanics only.
    # Explicitly NOT a swing quality grade: a whiff can still show sound
    # body positions. Weights are deliberately conservative — even a fully
    # "good" scorecard tops out around 85, and any "unknown" phases drag toward
    # the middle rather than counting as pass.
    status_pts = {"good": 15, "watch": 9, "issue": 3, "unknown": 8}
    total = 0
    weighted_max = 0
    for _, s, _ in scorecard:
        total += status_pts.get(s, 8)
        weighted_max += 18  # cap of 18 not 20 → "good on everything" ≈ 83
    fundamentals_score = round((total / weighted_max) * 100) if weighted_max > 0 else None

    # Score band labels (renamed to reflect what the score actually measures)
    if fundamentals_score is None:
        grade_band = None
    elif fundamentals_score >= 78:
        grade_band = "Sound"
    elif fundamentals_score >= 62:
        grade_band = "Developing"
    elif fundamentals_score >= 45:
        grade_band = "Rough"
    else:
        grade_band = "Needs Work"

    grade = fundamentals_score  # keep the key name for backward compat

    # Prioritized practice plan — group top issues into a ranked sequence
    plan = []
    if improvements:
        for i, imp in enumerate(improvements[:3]):
            plan.append({
                "priority": i + 1,
                "focus": imp["issue"],
                "drill": imp["drill"] or "No specific drill — awareness alone can help.",
                "phase": imp.get("phase", ""),
                "severity": imp.get("severity", "medium"),
            })
    else:
        plan.append({
            "priority": 1,
            "focus": "Nothing obvious to fix — grooving what you're doing.",
            "drill": "Keep filming your swing periodically to catch drift early.",
            "phase": "",
            "severity": "low",
        })

    # High-level metrics summary
    address_spine = _phase_avg(metrics_list, phases, "Address / Setup", "spine_tilt_3d") \
        or _phase_avg(metrics_list, phases, "Address / Setup", "spine_tilt")
    impact_spine = _phase_avg(metrics_list, phases, "Impact", "spine_tilt_3d") \
        or _phase_avg(metrics_list, phases, "Impact", "spine_tilt")
    top_x_factor = _phase_avg(metrics_list, phases, "Top of Backswing", "x_factor_3d") \
        or _phase_avg(metrics_list, phases, "Top of Backswing", "x_factor")

    # Overall assessment
    issue_count = sum(1 for _, s, _ in scorecard if s == "issue")
    watch_count = sum(1 for _, s, _ in scorecard if s == "watch")
    good_count = sum(1 for _, s, _ in scorecard if s == "good")

    if good_count >= 4 and issue_count == 0:
        overall = "Strong swing across the board — most checkpoints look sound. The remaining refinements are past what a phone-video pose reading can reliably see."
    elif issue_count >= 2:
        overall = "The swing has good elements but multiple checkpoints are flagging real mechanical patterns worth working on. Start with the highest-severity items below."
    elif issue_count == 1:
        overall = "The swing looks generally solid with one specific pattern worth addressing. See the phase scorecard below for exactly where."
    elif watch_count >= 2:
        overall = "The swing has decent fundamentals with a few things to keep an eye on. None are deal-breakers — a bit of focused practice would tighten things up."
    else:
        overall = "The swing looks mechanically reasonable in the frames we could analyze. Nothing obvious is jumping out from the pose data."

    key_focus = improvements[0]["issue"] + " " + improvements[0]["drill"] if improvements else \
        "Keep grooving what you're doing — the pose data doesn't show any obvious flaws."

    camera_note = None
    if camera_view == "face-on":
        camera_note = "Detected face-on camera view. Spine tilt and head-movement readings are reliable; x-factor and shoulder rotation are approximate."
    elif camera_view == "down-the-line":
        camera_note = "Detected down-the-line camera view. Swing plane and posture retention readings are more meaningful than shoulder-hip separation numbers."
    else:
        camera_note = "Camera view is ambiguous — readings should be treated as rough."

    return {
        "overall_assessment": overall,
        "strengths": strengths[:6],
        "improvements": improvements[:6],
        "key_focus": key_focus,
        "metrics_summary": {
            "address_spine_tilt_deg": round(address_spine, 1) if address_spine is not None else None,
            "impact_spine_tilt_deg": round(impact_spine, 1) if impact_spine is not None else None,
            "top_x_factor_deg": round(top_x_factor, 1) if top_x_factor is not None else None,
            "head_movement_pct": round(head_movement * 100, 1),
        },
        "camera_note": camera_note,
        "phase_scores": phase_scores,
        "handedness": handedness,
        "grade": grade,
        "grade_band": grade_band,
        "practice_plan": plan,
    }
