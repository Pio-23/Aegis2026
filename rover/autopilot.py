# LLM Pilot class for rover. Provides high-level autopilot functions and holds

# AEGIS Senior Design, Created 10/22/2025

import os
import google.generativeai as genai

import json
import time

from lidar import scan


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
    context_msg: dict[str, str] = {"role": "system", "content": system_context}

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
        self.memory = [self.context_msg]
        self.model_name = "gemini-1.5-flash"  # gemini model to use

        # Connect API account to client
        api_key = os.environ.get("GEMINI_API_KEY")
        genai.configure(api_key=api_key)

        #setup tools for gemini
     
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            tools=self.tools,
            system_instruction=self.system_context
        )
        self.chat = self.model.start_chat(history=[])

    def decide_actions(self, telemetry):
        """
        Decide on the next action based on telemetry input.
        Returns a dict with action details.
        """
        start_time = time.perf_counter()

        user_input = {"role": "user", "parts": [json.dumps(telemetry)]}
        self.update_memory(user_input)

        response = self.model.generate_content(contents=self.memory)

        # Prompt the model with tools and telemetry
        res_content = response.candidates[0].content
        tool_calls = [part.function_call for part in res_content.parts if part.function_call]

        self.update_memory(res_content)

        # 1. Debug prints
        print("[DEBUG] Autopilot.decide_action: Received response from GEMINI.")
        # In Gemini, the response is a full object, use response.text if you want just text
        print(f"[DEBUG] Full response: {response}")
        self.update_memory({"role": "assistant", "content": msg.content or ""})

        print(f"Took {round(time.perf_counter() - start_time, 3)} seconds.")

        return msg.tool_calls

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
