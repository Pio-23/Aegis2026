"""Optional local Gemma shadow observer. Never controls motors."""

import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from rover.aegis_corridor_geometry import query_forward_corridor


def run_shadow(geometry_path):
    """Analyze saved geometry and log a response without rover actions."""

    path = Path(geometry_path)
    output = path.with_name("gemma_geometry_shadow.json")

    record = {
        "scan": str(path),
        "mode": "SHADOW_ONLY_NO_MOVEMENT"
    }

    try:
        scene = query_forward_corridor(str(path))

        compact = {
            "frame": scene["coordinate_frame"],
            "floor_z_estimate_m": scene["floor_z_estimate_m"],
            "half_width_m_uncalibrated": (
                scene["corridor_half_width_m_uncalibrated"]
            ),
            "grids": scene["grid_front_to_far"],
            "band_counts": [
                {
                    "forward_m": band["forward_m"],
                    "layers": {
                        name: {
                            side: item["points"]
                            for side, item in lanes.items()
                        }
                        for name, lanes in band["layers"].items()
                    }
                }
                for band in scene["bands"]
            ],
            "warning": scene["warning"]
        }

        record["observation"] = compact

        prompt = (
            "Analyze this HISTORICAL AEGIS LiDAR measurement summary. "
            "X means at least ten measured returns, "
            "x means 1-9, ? means UNKNOWN, never free. "
            "Forward is +X, left is +Y. "
            "No vehicle dimensions or current pose verified. "
            "Describe obstacles and unknowns. "
            "NEVER recommend motion or claim clearance. "
            "Return concise JSON with keys obstacles, "
            "uncertainties, and notes.\n"
            + json.dumps(compact, separators=(",", ":"))
        )

        request = Request(
            "http://127.0.0.1:9379/v1/chat/completions",
            data=json.dumps({
                "model": os.getenv(
                    "AEGIS_GEMMA_SHADOW_MODEL",
                    "gemma4-e2b"
                ),
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 300
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urlopen(request, timeout=100) as answer:
            record["response"] = json.loads(
                answer.read().decode("utf-8")
            )

        record["status"] = "completed"

    except Exception as exc:
        record["status"] = "unavailable_or_error"
        record["error"] = str(exc)[:500]

    try:
        output.write_text(
            json.dumps(record, indent=2),
            encoding="utf-8"
        )
        print(
            f"[GEMMA SHADOW] Log only: {output} "
            f"({record['status']})"
        )
    except OSError as exc:
        print(f"[GEMMA SHADOW] Could not log: {exc}")
