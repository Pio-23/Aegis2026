"""AEGIS Gemma navigation shadow. No motor or UART access."""

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

_BUSY = threading.Lock()
ENDPOINT = "http://127.0.0.1:9379/v1/chat/completions"

PROMPT = """
You are Gemma, the LOCAL SHADOW navigation AI for AEGIS.
You observe the environment and propose an independent action.
You NEVER execute actions or control motors.

AEGIS is a six-wheel skid-steer rover.
Possible actions: MOVE, TURN, SCAN, CAMERA, NO_OP.

Exploration priorities:
- Explore new space efficiently.
- Prefer continued straight travel along observed open corridors.
- Reuse existing 360-degree LiDAR geometry across turns.
- Do not request a new scan merely because the rover turned.
- Consider side openings when front and back are constrained.
- Avoid repeated left-right oscillation without progress.
- Prefer longer forward segments when actual clearance supports them.
- Consider the rover footprint, nearby furniture and stopping margin.
- Do not confuse front-left/right ultrasonics with side clearance.
- Never interpret missing LiDAR returns as free space.
- If measurements are insufficient, identify the specific uncertainty.

LiDAR sensor frame: +Y FRONT, +X RIGHT, -X LEFT, -Y BACK.
The navigation map uses +X FRONT, +Y LEFT.
Pose and IMU heading are estimates.

Do not assume an old scan is current at a new physical position.
Your answer is a PROPOSAL ONLY, not a command.

MOVEMENT DECISION QUALITY:
- MOVE must specify a nonzero speed and duration.
- Use speed 0.20-0.35 and up to 8 seconds ONLY when
  measured corridor clearance supports that movement.
- A desire to explore is not proof that movement is safe.
- A dense near-field LiDAR return may be an obstacle;
  examine its position before proposing forward travel.
- Do not use an IMU yaw angle as a global map direction
  unless the coordinate alignment has been verified.
- If no viable path has been established, explain the
  missing evidence rather than inventing clearance.
- For TURN, specify LEFT or RIGHT and an angle from
  15, 30, 45, 60, 75 or 90 degrees.
- Never return MOVE with speed 0 or duration 0.


Return JSON only:
{
  "action": "MOVE|TURN|SCAN|CAMERA|NO_OP",
  "direction": "FORWARD|BACK|LEFT|RIGHT|NONE",
  "speed": 0.0,
  "duration_s": 0,
  "turn_degrees": 0,
  "reason": "brief explanation",
  "confidence": 0.0
}
"""

# AEGIS_GEMMA_COACHING_V1
def _coached_prompt():
    lessons = Path(__file__).with_name(
        "gemma_navigation_lessons.txt"
    )
    try:
        extra = lessons.read_text(
            encoding="utf-8"
        )[:3000]
    except OSError:
        extra = ""

    return PROMPT + "\n\n" + extra


def _scene_from_trip(folder):
    package = folder / "lidar_package"
    scene = {}

    files = list(package.glob("*_luna_vision.txt"))
    if files:
        newest = max(files, key=lambda p: p.stat().st_mtime)
        scene["lidar_description"] = newest.read_text(
            encoding="utf-8"
        )[:3000]

    geometry = package / "geometry_regions.json"
    if geometry.exists():
        data = json.loads(geometry.read_text(encoding="utf-8"))
        coverage = data.get("directional_point_coverage_v1", {})
        scene["directional_measurements"] = json.dumps(
            coverage, separators=(",", ":")
        )[:3500]

    return scene

# GEMMA_ACTION_VALIDATION_V1
def _validate_shadow_proposal(proposal):
    errors = []

    if not isinstance(proposal, dict):
        return ["Response is not a JSON object"]

    action = proposal.get("action")
    direction = proposal.get("direction")

    if action not in {
        "MOVE", "TURN", "SCAN", "CAMERA", "NO_OP"
    }:
        errors.append("Unknown action")

    if action == "MOVE":
        if direction not in {"FORWARD", "BACK"}:
            errors.append("MOVE requires FORWARD or BACK")

        speed = proposal.get("speed")
        duration = proposal.get("duration_s")

        if not (
            isinstance(speed, (int, float))
            and not isinstance(speed, bool)
            and 0 < speed <= 0.35
        ):
            errors.append("MOVE requires speed > 0 and <= 0.35")

        if not (
            isinstance(duration, (int, float))
            and not isinstance(duration, bool)
            and 0 < duration <= 8
        ):
            errors.append("MOVE requires duration > 0 and <= 8s")

    if action == "TURN":
        if direction not in {"LEFT", "RIGHT"}:
            errors.append("TURN requires LEFT or RIGHT")

        if proposal.get("turn_degrees") not in {
            15, 30, 45, 60, 75, 90
        }:
            errors.append("Unsupported turn angle")

    if not isinstance(proposal.get("reason"), str):
        errors.append("Missing explanation")

    confidence = proposal.get("confidence")
    if not (
        isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and 0 <= confidence <= 1
    ):
        errors.append("Invalid confidence")

    return errors


def evaluate(trip_folder, nav_context, luna_action):
    folder = Path(trip_folder)
    record = {
        "timestamp": datetime.now().isoformat(),
        "mode": "SHADOW_ONLY",
        "model": "gemma4-e2b",
        "luna_action": luna_action,
        "navigation_context": nav_context,
        "review_label": "unreviewed",
    }

    try:
        scene = _scene_from_trip(folder)
        record["scene"] = scene

        # GEMMA_COMPACT_CONTEXT_V1
        # Preserve directional geometry, reduce repeated context.
        if isinstance(nav_context, dict):
            if "latest_telemetry" in nav_context:
                compact_nav = {
                    "mode": nav_context.get("mode"),
                    "latest_telemetry": nav_context.get(
                        "latest_telemetry"
                    ),
                    "recent_history": (
                        nav_context.get("recent_sensor_history") or []
                    )[-2:],
                    "sensor_mounting": nav_context.get(
                        "sensor_mounting"
                    ),
                    "movement_note": nav_context.get(
                        "movement_note"
                    )
                }
            else:
                compact_nav = {
                    "sensor_summary": nav_context.get(
                        "sensor_summary"
                    ),
                    "navigation_state": nav_context.get(
                        "navigation_state"
                    ),
                    "navigation_map": nav_context.get(
                        "navigation_map"
                    ),
                    "mode": nav_context.get("mode")
                }
        else:
            compact_nav = {"context": str(nav_context)}

        nav_text = json.dumps(
            compact_nav, default=str, separators=(",", ":")
        )

        if len(nav_text) > 1200:
            nav_text = json.dumps({
                "excerpt": nav_text[:1050],
                "truncated": True
            })

        coverage = str(
            scene.get("directional_measurements", "")
        )

        if len(coverage) > 3300:
            coverage = coverage[:3300] + " [TRUNCATED]"

        description = str(
            scene.get("lidar_description", "")
        )[:400]

        text = (
            "Propose one independent SHADOW navigation action. "
            "No motor execution. State uncertainty.\\n"
            "NAVIGATION:\\n" + nav_text
            + "\\n360-DEGREE LIDAR GEOMETRY:\\n" + coverage
            + "\\nSCAN DESCRIPTION EXCERPT:\\n"
            + description
        )

        print(
            "[GEMMA SHADOW] Compact prompt:",
            len(text), "characters"
        )

        payload = {
            "model": os.getenv(
                "AEGIS_GEMMA_SHADOW_MODEL", "gemma4-e2b"
            ),
            "messages": [
                {"role": "system", "content": _coached_prompt()},
                {"role": "user", "content": text},
            ],
            "max_tokens": 250,
        }

        request = Request(
            ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(request, timeout=90) as response:
            result = json.load(response)

        answer = result["choices"][0]["message"]["content"]
        record["gemma_raw"] = answer

        start = answer.find("{")
        end = answer.rfind("}") + 1
        record["gemma_action"] = json.loads(answer[start:end])
        errors = _validate_shadow_proposal(
            record["gemma_action"]
        )
        record["validation_errors"] = errors
        record["status"] = (
            "invalid_proposal" if errors else "completed"
        )

    except Exception as exc:
        record["status"] = "error"
        record["error"] = str(exc)[:500]

    try:
        folder.mkdir(parents=True, exist_ok=True)
        logfile = folder / "gemma_shadow_decisions.jsonl"
        with logfile.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        print(
            "[GEMMA SHADOW]",
            record["status"],
            record.get("gemma_action", {}).get("action", ""),
            "| log:", logfile
        )
    except Exception as exc:
        print("[GEMMA SHADOW] Logging error:", exc)

    return record

def submit_shadow(trip_folder, nav_context, luna_action):
    if os.getenv("AEGIS_GEMMA_SHADOW", "0") != "1":
        return

    if not _BUSY.acquire(blocking=False):
        print("[GEMMA SHADOW] Busy; skipping overlapping request.")
        return

    def worker():
        try:
            evaluate(trip_folder, nav_context, luna_action)
        finally:
            _BUSY.release()

    threading.Thread(target=worker, daemon=True).start()

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python3 -m rover.local_ai TRIP_FOLDER"
        )

    result = evaluate(
        sys.argv[1],
        {
            "mode": "offline_replay",
            "note": "No live telemetry. Analyze saved geometry only."
        },
        {"action": "REPLAY_TEST"}
    )
    print(json.dumps({
        "status": result["status"],
        "gemma_action": result.get("gemma_action"),
        "error": result.get("error")
    }, indent=2))
