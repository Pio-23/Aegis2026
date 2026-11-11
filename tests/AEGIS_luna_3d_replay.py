
"""AEGIS offline Luna LiDAR replay - READ-ONLY, NO ROVER CONTROL.

Run from ~/aegis using a Python interpreter with the OpenAI SDK and NumPy.
This does NOT import rover.autopilot (which may initialize hardware).
It extracts only the already-tested _query_lidar_geometry method by AST.
"""
import ast
import base64
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from openai import OpenAI

from rover.aegis_corridor_geometry import query_forward_corridor

# STEP 1: Historical, saved test files (not live sensor data).
IMAGE = Path('/tmp/aegis_scene_v2/perception_layers.png')
GEOMETRY = Path('/tmp/aegis_scene_v2/full_point_geometry.npz')
SOURCE = Path('rover/autopilot.py')
MODEL = os.environ.get('AEGIS_LUNA_MODEL', 'gpt-5.6-luna')
MAX_TOOL_QUERIES = 3

for path in (IMAGE, GEOMETRY, SOURCE):
    if not path.is_file():
        raise SystemExit(f'Missing required test file: {path}')
if not os.environ.get('OPENAI_API_KEY'):
    raise SystemExit('OPENAI_API_KEY is not set in this shell.')

# STEP 2: Extract JUST the proven read-only query function.
tree = ast.parse(SOURCE.read_text())
cls = next(node for node in tree.body
           if isinstance(node, ast.ClassDef) and node.name == 'Autopilot')
method = next(node for node in cls.body
              if isinstance(node, ast.FunctionDef)
              and node.name == '_query_lidar_geometry')
namespace = {'os': os, 'time': time}
exec(compile(ast.Module(body=[method], type_ignores=[]),
             str(SOURCE), 'exec'), namespace)

pilot = SimpleNamespace(
    latest_lidar_geometry_file=str(GEOMETRY),
    # Historical November 4 scan: NEVER pretend it is fresh.
    navigation_state={'last_lidar_time': None,
                      'heading_change_since_lidar_deg': None},
)
query_geometry = namespace['_query_lidar_geometry']

# STEP 3: The ONLY tool available to Luna. No movement or scan tool.
TOOLS = [{
    'type': 'function',
    'function': {
        'name': 'query_lidar_geometry',
        'description': ('Read actual XYZ/intensity measurements from a saved '
                        'LiDAR scan in a narrow angular slice and height layer. '
                        'This is NOT verified motion clearance.'),
        'parameters': {
            'type': 'object',
            'properties': {
                'angle_deg': {'type': 'number', 'minimum': -180,
                              'maximum': 180},
                'half_width_deg': {'type': 'number',
                                   'exclusiveMinimum': 0,
                                   'maximum': 20},
                'layer': {'type': 'string', 'enum': [
                    'floor_candidate', 'low', 'mid', 'high', 'overhead']},
            },
            'required': ['angle_deg', 'half_width_deg', 'layer'],
            'additionalProperties': False,
        },
    },
}]


# Second read-only tool: forward 3D corridor.
# Luna decides whether to request this report.

TOOLS.append({
    "type": "function",
    "function": {
        "name": "query_forward_corridor",
        "description": (
            "Inspect measured 3D LiDAR returns from 0.5 to "
            "2.0 meters ahead, across an experimental "
            "0.92-meter-wide forward region. Returns "
            "left/center/right point counts, height-layer "
            "measurements, and a 10 cm spatial grid. "
            "Missing returns mean UNKNOWN, not free. "
            "This does not establish collision clearance."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
})


encoded = base64.b64encode(IMAGE.read_bytes()).decode('ascii')
messages = [
    {'role': 'system', 'content': (
        'You are analyzing a HISTORICAL saved AEGIS rover LiDAR scan from '
        'November 4, 2026. This is a stationary OFFLINE analysis experiment, '
        'not a live rover session. Coordinate frame at scan time: +X front, '
        '+Y left, +Z up; angles 0 front, +90 left, -90 right. '
        'Floor height is an uncalibrated estimate. '
        'Visual colors are measured returns, not clearance labels. '
        'An absence of returns means UNKNOWN, never verified FREE. '
        'No current camera, ultrasonic, rover pose, or motion history is supplied. '
        'Use the measurement tool for concrete 3D evidence. '
        'Do not claim that any movement is safe or command the rover. '
        'Conclude with a short description of obstacles, plausible openings, '
        'important uncertainties, and what additional information would be '
        'needed before recommending a motion. '
        'Do not fabricate measurements.'
    )},
    {'role': 'user', 'content': [
        {'type': 'text', 'text': (
            'Inspect this 3D LiDAR scene from the saved 4:46 PM scan. '
            'You may make up to three read-only tool calls total. '
            'Choose between inspecting the forward corridor and querying '
            'specific angles and heights, depending on what you need. '
            'Compare observed geometry. Explain what the evidence does '
            'and does not establish; do not issue movement commands.'
        )},
        {'type': 'image_url', 'image_url': {
            'url': 'data:image/png;base64,' + encoded, 'detail': 'high'
        }},
    ]},
]

# STEP 4: Cloud AI loop; dispatch ONLY the read-only geometry query.
client = OpenAI(timeout=120.0)
queries_used = 0
start = time.monotonic()

for round_number in range(MAX_TOOL_QUERIES + 2):
    choice = ('required' if round_number == 0 else
              'auto' if queries_used < MAX_TOOL_QUERIES else 'none')
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice=choice,
        parallel_tool_calls=False,
        reasoning_effort='none',
    )
    msg = response.choices[0].message
    if response.usage:
        print(f'[USAGE] tokens this request: {response.usage.total_tokens}')

    if not msg.tool_calls:
        print('\n=== LUNA OFFLINE SCENE INTERPRETATION ===')
        print(msg.content or '(No text returned)')
        break

    # Preserve assistant request so the API can match tool_call_id replies.
    messages.append({
        'role': 'assistant', 'content': msg.content,
        'tool_calls': [
            {'id': call.id, 'type': 'function',
             'function': {'name': call.function.name,
                          'arguments': call.function.arguments}}
            for call in msg.tool_calls
        ],
    })

    
    for call in msg.tool_calls:
        if call.function.name not in (
            "query_lidar_geometry",
            "query_forward_corridor"
        ):
            result = {
                "status": "rejected",
                "reason": "Tool not allowed"
            }

        elif queries_used >= MAX_TOOL_QUERIES:
            result = {
                "status": "limit_reached",
                "reason": "Maximum 3 tool queries"
            }

        else:
            queries_used += 1

            try:
                args = json.loads(
                    call.function.arguments or "{}"
                )

                if not isinstance(args, dict):
                    raise ValueError(
                        "Tool arguments must be a JSON object"
                    )

                if call.function.name == "query_forward_corridor":
                    if args:
                        raise ValueError(
                            "Corridor tool takes no arguments"
                        )

                    # Read the saved 3D point cloud.
                    result = query_forward_corridor(
                        str(GEOMETRY)
                    )

                else:
                    # Original exact-angle geometry tool.
                    result = query_geometry(
                        pilot,
                        angle_deg=args.get("angle_deg"),
                        half_width_deg=args.get(
                            "half_width_deg", 2
                        ),
                        layer=args.get("layer", "mid"),
                    )

            except (TypeError, ValueError, OSError, KeyError) as exc:
                result = {
                    "status": "invalid_query",
                    "reason": str(exc)
                }

        print(
            "\n[READ-ONLY TOOL]",
            call.function.name,
            call.function.arguments
        )

        print(json.dumps(result, indent=2))

        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result)
        })

else:
    print('Stopped after reaching the maximum API rounds.')

print(f'\nOffline replay finished in {time.monotonic() - start:.1f}s; '
      f'{queries_used} geometry queries; zero rover actions.')
