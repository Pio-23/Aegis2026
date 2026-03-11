# LLM Pilot class for rover. Provides high-level autopilot functions and holds

# AEGIS Senior Design, Created 10/22/2025

import os
import json
import time
from typing import Optional

from google import genai
from google.genai import types

# your lidar module (you already had this)
from lidar import scan as lidar_scan

class Autopilot:

    system_context = """
        You are controlling a rover. Follow these rules:
        - Always prioritize safety of the motors and sensors.
        - Decisions must be based on incoming telemetry.
        - Telemetry is provided as an array with the following mapping:
            [uptime, remaining storage in GB, scan resolution, camera connected, 
            camera recording, 
            front left mot. amps, mid left mot. amps, rear left mot. amps, 
            front right mot. amps, mid right mot. amps, rear right mot. amps, 
            front left ultrasonic dist (cm), front center US. dist, front right US. 
            dist, lidar US. dist, rear US. dist, IMU yaw (deg)]
        - Respond only using tools. If unsafe or unclear, call `no_op` with a safe 
        fallback and a short reason.
        - Rings per scan is inversely proportional to angular resolution. Better resolution
        occurs with higher ring scans. Use only multiples of 200 rings when configuring
        rings per scan.
        - Target 400 ring scans unless you believe a higher resolution scan is warranted
        and take a scan once for every 5 times you move. Ring count will latch.
        """

    # Define available tools for the rover autopilot
    aegis_tools = [
        {
            "type": "function",
            "function": {
                "name": "scan_environment",
                "description": (
                    "Captures a high-res LiDAR scan of the rover's surroundings."
                    "Use this to gather detailed environmental data whenever"
                    "telemetry suggests anything worth scanning, or if it has been"
                    "a while since the last scan."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "move_rover",
                "description": (
                    "Issues movement commands to the rover."
                    "Use this to move or turn the rover."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "enum": ["TURN", "MOVE"],
                            "description": (
                                "Type of movement."
                                "Either TURN (rotational) or MOVE (linear)."
                            )
                        },
                        "spd": {
                            "type": "number",
                            "description": (
                                "Speed factor in range [-1, 1]."
                                "Required for MOVE and TURN."
                                "Positive only for TURN."
                            )
                        },
                        "turn_dir": {
                            "type": "string",
                            "enum": ["LEFT", "RIGHT"],
                            "description": (
                                "Turn direction. Only used when op is TURN."
                            )
                        }
                    },
                    "required": ["op", "spd"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "no_op",
                "description": (
                    "Signal that no safe/clear action can be taken now."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type":"string", "description":"Why action is deferred."},
                        "confidence": {
                            "type":"number", "minimum":0, "maximum":1,
                            "description":"Model confidence that doing nothing is correct."
                        },
                        "needs": {
                            "type":"array",
                            "items":{"type":"string"},
                            "description":"Specific data/information needed to proceed."
                        }
                    },
                    "required": ["reason"]
                }
            }
        }
    ]

    def __init__(self) -> None:
        self.memory_depth = 21  # Number of past interactions to remember

        # Keep system_context string as-is (used as system_instruction below)
        self.model_name = "models/gemini-2.0-flash"  # or whichever model you prefer

        # Connect API account to client
        self.client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )

        # setup tools for gemini (convert aegis_tools if necessary)
        self.tools = []
        if hasattr(self, "aegis_tools") and isinstance(self.aegis_tools, list):
            for tool in self.aegis_tools:
                # handle both shapes: either the tool is already a FunctionDeclaration/SDK object
                if isinstance(tool, types.FunctionDeclaration):
                    self.tools.append(tool)
                    continue

                # If your tool was a dict like {"type":"function","function":{...}}, dig in:
                func = tool.get("function") if isinstance(tool, dict) else None
                if func and isinstance(func, dict):
                    name = func.get("name")
                    desc = func.get("description", "")
                    params = func.get("parameters", {"type":"object","properties":{}, "required":[]})
                    # Create an SDK FunctionDeclaration
                    fd = types.FunctionDeclaration(name=name, description=desc, parameters=params)
                    self.tools.append(fd)
                else:
                    # Fallback: if tool looks like a bare function-decl dict with keys name/description/parameters
                    if isinstance(tool, dict) and "name" in tool:
                        fd = types.FunctionDeclaration(
                            name=tool.get("name"),
                            description=tool.get("description", ""),
                            parameters=tool.get("parameters", {"type":"object","properties":{}, "required":[]})
                        )
                        self.tools.append(fd)
                    else:
                        # ignore unknown entries but warn
                        print(f"[WARN] Skipping invalid tool entry in aegis_tools: {tool}")

        # If conversion left self.tools empty, fallback to a minimal set (optional)
        if not self.tools:
            self.tools = [
                types.FunctionDeclaration(
                    name="scan_environment",
                    description="Captures a high-res LiDAR scan.",
                    parameters={"type":"object","properties":{}, "required":[]}
                ),
                types.FunctionDeclaration(
                    name="move_rover",
                    description="Issues movement commands to the rover.",
                    parameters={
                        "type":"object",
                        "properties":{
                            "op":{"type":"string","enum":["TURN","MOVE"]},
                            "spd":{"type":"number"},
                            "turn_dir":{"type":"string","enum":["LEFT","RIGHT"]}
                        },
                        "required":["op","spd"]
                    }
                ),
                types.FunctionDeclaration(
                    name="no_op",
                    description="No safe action available right now.",
                    parameters={
                        "type":"object",
                        "properties":{
                            "reason":{"type":"string"},
                            "confidence":{"type":"number","minimum":0,"maximum":1},
                            "needs":{"type":"array","items":{"type":"string"}}
                        },
                        "required":["reason"]
                    }
                )
            ]

        self.model = self.client.models

        self.memory = [
        types.Content(
        role="system",
        parts=[types.Part(text=self.system_context)]
        )
        ]

    def decide_actions(self, telemetry: list) -> Optional[list]:
        """
        Decide on the next action based on telemetry input.
        Returns a dict with action details.
        """
        t0 = time.perf_counter()

        telemetry_text = json.dumps(telemetry)
        user_content = types.Content(role="user", parts=[types.Part(text=f"Telemetry: {telemetry_text}")])

        # Use update_memory (you implement update_memory) to store the typed content
        self.update_memory({"role": "user", "content": f"Telemetry: {telemetry_text}"})

        # Use generate_content with the tools; SDK will decide to return function_call parts
        try:
            resp = self.client.models.generate_content(
            model=self.model_name,
            contents=[user_content],
            config=types.GenerateContentConfig(
            tools=self.tools,
            system_instruction=self.system_context,
         ),
)
        except Exception as e:
            print(f"[ERROR] model.generate_content failed: {e}")
            # publish_stop() is expected elsewhere; don't alter behavior
            return []

        # response may have function_calls; try to extract them robustly
        function_calls = []

        # new SDK sometimes exposes .function_calls or .candidates[].content.parts
        try:
            # preferred: resp.function_calls (list of FunctionCall objects)
            fc = getattr(resp, "function_calls", None)
            if fc:
                function_calls = fc
            else:
                # inspect candidates' content parts for function_call elements
                candidates = getattr(resp, "candidates", []) or []
                for cand in candidates:
                    content = getattr(cand, "content", None)
                    if content and getattr(content, "parts", None):
                        for part in content.parts:
                            if getattr(part, "function_call", None):
                                function_calls.append(part.function_call)
        except Exception as e:
            print(f"[WARN] parsing function calls: {e}")

        # Print debug text if available
        try:
            print("[DEBUG] model text:", resp.text)
        except Exception:
            pass

        elapsed = time.perf_counter() - t0
        print(f"[INFO] Gemini call took {elapsed:.3f}s, got {len(function_calls)} function call(s).")
        return function_calls

    def validate_action(self, toolcall) -> bool:
        """
        Validate the proposed action for safety and feasibility.
        Returns True if valid, False otherwise.
        """

        status = True
        name = toolcall.function.name                           # type: ignore
        args = json.loads(toolcall.function.arguments or "{}")  # type: ignore
        match name:
            case "scan_environment":
                print("Calling scan_environment().")

            case "move_rover":
                print(f"Calling move_rover with args {args}.")

            case "no_op":
                print(f"Calling no_op with args {args}.")
                status = False  # No-op is not an action

            case _: # Default
                print(f"Called {name} with args {args}.")

        return status

    def update_memory(self, msg : dict) -> None:
        """
        Update the internal memory with a new message.
        """
        if (len(self.memory) > self.memory_depth):
            self.memory.pop(1)  # Remove oldest, keep system context
        self.memory.append(msg)