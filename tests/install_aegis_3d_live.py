
"""Install AEGIS 3D geometry tools without running rover hardware.

Run from ~/aegis:
    ~/venv/bin/python3 tests/install_aegis_3d_live.py
"""

from pathlib import Path
from datetime import datetime
import ast
import re
import shutil

ROOT = Path.cwd()
AUTO = ROOT / "rover/autopilot.py"
VIS = ROOT / "rover/lidar_visualizer.py"
CORRIDOR = ROOT / "rover/aegis_corridor_geometry.py"
GEMMA = ROOT / "rover/aegis_gemma_geometry_shadow.py"

# Check the existing AEGIS installation before changing anything.
for path in (AUTO, VIS, CORRIDOR):
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Run from ~/aegis.")

auto = AUTO.read_text()
visual = VIS.read_text()
corridor = CORRIDOR.read_text()

required = [
    "ENABLE_LIDAR_QUERY_TOOL",
    "def _query_lidar_geometry(",
    "latest_lidar_geometry_file",
    "def _request_model_action(",
    "def decide_actions(",
    "scene_full_geometry",
]

missing = [name for name in required if name not in auto]

if missing:
    raise SystemExit(
        f"Existing 3D integration missing: {missing}. No changes made."
    )

if "ENABLE_3D_PERCEPTION" not in visual:
    raise SystemExit("3D visualizer flag missing. No changes made.")

if "HEIGHT_BANDS" not in corridor:
    raise SystemExit(
        "Corridor must use shared HEIGHT_BANDS. No changes made."
    )

if "def _query_forward_corridor(" in auto:
    raise SystemExit("Corridor tool already installed. No changes made.")

if GEMMA.exists():
    raise SystemExit(
        f"{GEMMA} already exists. No changes made."
    )

# Ensure scans already store their new geometry file path.
handoff_pattern = (
    r"(?m)^([ \t]*)self\.latest_lidar_geometry_file\s*=\s*"
    r"lidar_views\.get\(\s*['\"]scene_full_geometry['\"]\s*\)"
)

if not re.search(handoff_pattern, auto):
    raise SystemExit(
        "Latest scan geometry handoff not found. No changes made."
    )

if '                elif name == "move_rover":' not in auto:
    raise SystemExit("Autopilot dispatch anchor missing. No changes made.")

if not re.search(
    r"^ENABLE_LIDAR_QUERY_TOOL\s*=\s*(False|True)",
    auto,
    re.M,
):
    raise SystemExit("Geometry flag missing. No changes made.")

# 1. Register Luna's new read-only corridor tool.
tool_schema = '''    aegis_tools.append({
        "type": "function",
        "function": {
            "name": "query_forward_corridor",
            "description": (
                "Read measured low/mid-height XYZ returns in the experimental "
                "forward 0.5-2m, left/center/right corridor, with a 10cm grid. "
                "No-return cells are UNKNOWN, not traversable. "
                "No motion permission."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    })

'''

if "    def __init__(self)" not in auto:
    raise SystemExit("Autopilot initializer missing. No changes made.")

auto = auto.replace(
    "    def __init__(self)",
    tool_schema + "    def __init__(self)",
    1,
)

# 2. Add a read-only corridor query using the latest processed scan.
new_methods = '''    def _query_forward_corridor(self):
        """Read measurements from the most recent processed LiDAR scan."""
        import os
        import time
        from rover.aegis_corridor_geometry import query_forward_corridor

        path = self.latest_lidar_geometry_file

        if not path or not os.path.isfile(path):
            return {
                "status": "unavailable",
                "reason": "No processed 3D scan yet"
            }

        last = self.navigation_state.get("last_lidar_time")
        age = (
            round(time.time() - last, 1)
            if isinstance(last, (int, float))
            else None
        )

        heading = self.navigation_state.get(
            "heading_change_since_lidar_deg"
        )

        try:
            result = query_forward_corridor(path)
        except (OSError, ValueError, KeyError) as exc:
            return {
                "status": "error",
                "reason": str(exc)
            }

        return {
            "status": "ok",
            "frame": "rover-local at scan time",
            "scan_age_s": age,
            "heading_change_since_scan_deg": heading,
            "measurement": result,
            "warning": (
                "Historical scan frame, not clearance "
                "or motion authorization."
            )
        }

    def _handle_forward_corridor_query(self, call):
        """Answer a geometry tool call without accessing rover motors."""
        try:
            arguments = json.loads(
                call.function.arguments or "{}"
            )

            if arguments != {}:
                raise ValueError(
                    "query_forward_corridor takes no arguments"
                )

            result = self._query_forward_corridor()

        except (ValueError, TypeError) as exc:
            result = {
                "status": "invalid_query",
                "reason": str(exc)
            }

        self.update_memory({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result)
        })

'''

anchor = "    def _request_model_action(self):"

auto = auto.replace(
    anchor,
    new_methods + anchor,
    1,
)

# 3. Keep both geometry tools behind the existing feature flag.
start = auto.index("    def _request_model_action(self):")
end = auto.index("    def decide_actions(", start)

new_request = '''    def _request_model_action(self):
        """Request model action with optional read-only 3D tools."""

        geometry_names = {
            "query_lidar_geometry",
            "query_forward_corridor"
        }

        tools = [
            tool for tool in Autopilot.aegis_tools
            if ENABLE_LIDAR_QUERY_TOOL
            or tool["function"]["name"] not in geometry_names
        ]

        return self.client.chat.completions.create(
            model=self.model_name,
            messages=self.memory,
            tools=tools,
            tool_choice="required",
            parallel_tool_calls=False,
            reasoning_effort="none"
        )

'''

auto = auto[:start] + new_request + auto[end:]

# 4. Dispatch the new tool through its read-only handler.
auto = auto.replace(
    '                elif name == "move_rover":',
    '''                elif name == "query_forward_corridor":
                    self._handle_forward_corridor_query(call)
                    continue

                elif name == "move_rover":''',
    1,
)

# 5. Leave the new model capabilities disabled for now.
auto = re.sub(
    r"(?m)^ENABLE_LIDAR_QUERY_TOOL\s*=\s*(False|True)",
    "ENABLE_LIDAR_QUERY_TOOL = False\n"
    "ENABLE_GEMMA_GEOMETRY_SHADOW = False",
    auto,
    count=1,
)

# 6. Prepare a passive Gemma shadow worker after each processed scan.
handoff = re.search(handoff_pattern, auto)
indent = handoff.group(1)

shadow_call = (
    f"\n{indent}if ENABLE_GEMMA_GEOMETRY_SHADOW "
    f"and self.latest_lidar_geometry_file:\n"
    f"{indent}    from threading import Thread\n"
    f"{indent}    from rover.aegis_gemma_geometry_shadow "
    f"import run_shadow\n"
    f"{indent}    Thread(\n"
    f"{indent}        target=run_shadow,\n"
    f"{indent}        args=(self.latest_lidar_geometry_file,),\n"
    f"{indent}        daemon=True\n"
    f"{indent}    ).start()\n"
)

auto = (
    auto[:handoff.end()]
    + shadow_call
    + auto[handoff.end():]
)

# Scene processing is enabled for future scans,
# but AI geometry tools remain disabled until validation.
visual, changed = re.subn(
    r"(?m)^ENABLE_3D_PERCEPTION\s*=\s*(False|True)",
    "ENABLE_3D_PERCEPTION = True",
    visual,
    count=1,
)

if changed != 1:
    raise SystemExit(
        "Cannot set scene generation flag. No changes made."
    )

# 7. Create the local-only Gemma shadow bridge.
GEMMA_SOURCE = '''"""Optional local Gemma shadow observer. Never controls motors."""

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
            "uncertainties, and notes.\\n"
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
'''

# Verify syntax BEFORE writing any modified files.
for name, source in (
    (str(AUTO), auto),
    (str(VIS), visual),
    (str(GEMMA), GEMMA_SOURCE),
):
    ast.parse(source, filename=name)

# 8. Create timestamped backups and install.
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for path in (AUTO, VIS):
    backup = path.with_name(
        path.name + ".before_3d_tools_" + stamp
    )
    shutil.copy2(path, backup)
    print(f"[BACKUP] {backup}")

AUTO.write_text(auto)
VIS.write_text(visual)
GEMMA.write_text(GEMMA_SOURCE)

print("[INSTALLED] Luna corridor tool registered, but OFF.")
print("[INSTALLED] New 3D scene generation ON for future scans.")
print("[INSTALLED] Gemma shadow bridge installed, but OFF.")
print("[SAFE] No rover hardware initialized or API requests made.")
