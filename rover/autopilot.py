# LLM Pilot class for rover. Provides high-level autopilot functions and holds

# AEGIS Senior Design, Created 10/22/2025

import os
import openai

import json
import time

from lidar import scan


class Autopilot:

    system_context = """
    You are controlling a rover. Follow these rules:
     - Always prioritize safety of the motors and sensors.
     - Decisions must be based only on the incoming telemetry JSON.
     - Telemetry is a nested JSON object with these sections:
      - rpi
      - arduino
      - lidar
      - camera
      - motors
     - ultrasonics
      - imu
      - ugv
    - Battery information is located at:
         telemetry["ugv"]["battery"]["capacity_pct"]
         telemetry["ugv"]["battery"]["voltage_v"]
         telemetry["ugv"]["battery"]["current_a"]
    - Motor information is under:
         telemetry["motors"]["front_left"], ["mid_left"], ["rear_left"],
        ["front_right"], ["mid_right"], ["rear_right"]
    - Ultrasonic information is under:
         telemetry["ultrasonics"]["lidar_cm"], ["left_cm"], ["center_cm"],
         ["right_cm"], ["rear_cm"]
    - IMU information is under telemetry["imu"].
    - Respond only using tools.
    - If unsafe or unclear, call `no_op` with a short reason.
    - Use scan_environment only when additional environmental sensing is needed.
    - TEST MODE: If battery voltage is above 15.0V and battery percent is above 30%, command a small TURN RIGHT with spd=0.15.
    - In TEST MODE, do not call scan_environment first.
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
                    "a while since the last scan. Make sure to make it as cheap as possible"
                    "by making the scan only 200 rings"
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
                              "minimum": -1,
                              "maximum": 1,
                               "description": (
                                    "Speed factor in range [-1, 1]. "
                                    "Required for MOVE and TURN. "
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
        self.model_name = "gpt-5-nano"  # LLM model to use

        # Connect API account to client
        self.client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
        )

    def decide_actions(self, telemetry):
        """
        Decide on the next action based on telemetry input.
        Returns a dict with action details.
        """

        start_time = time.perf_counter()

        self.update_memory({"role": "user", "content": json.dumps(telemetry)})

        # Prompt the model with tools and telemetry
        try:    
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.memory,                   # type: ignore
                tools=Autopilot.aegis_tools,                  # type: ignore
                tool_choice={
                    "type": "function",
                 "function": {"name": "move_rover"}
                 },
                temperature=1   # Token probability differential (creativity) [0,2]
        )
        except Exception as e:
            print(f"[ERROR] Autopilot.decide_actions: {e}")
            return []


        msg = response.choices[0].message
        print("[DEBUG] Autopilot.decide_action: Received response from LLM.")
        print(f"[DEBUG] Full response: {response}")
        self.update_memory({
            "role": "assistant",
             "content": msg.content or ""
             })

        print(f"Took {round(time.perf_counter() - start_time, 3)} seconds.")

        return msg.tool_calls or []

    def validate_action(self, toolcall, telemetry=None) -> bool:
        """
        Validate the proposed action for safety and feasibility.
        Returns True if valid, False otherwise.
        """

        
        name = toolcall.function.name                           # type: ignore
        args = json.loads(toolcall.function.arguments or "{}")  # type: ignore
        
        if name =="no_op":
            print(f"calling no_op with args {args}.")
            return  False
        
        if name =="scan_enviroment":
            print("Calling scan_enviroments().")
            return True
        
        if name != "move_rover":
            print(f"[SAFE] Unknown tool call: {name}")
            return False
        
        print(f"Calling move_rover with args {args}.")

        op = args.get("op")
        spd = args.get("spd")
        turn_dir = args.get("turn_dir")

        if op not in {"MOVE", "TURN"}:
            print("[SAFE] Invalid op.")
            return false

        if not isinstance(spd, (int, float)):
            print("[SAFE] Invalid op.")
            return false
        
        if spd <-1 or spd >1:
            print("[SAFE] Speed outside [-1,1]")
            return False
        
        if abs(spd) > 0.25:
            print("[SAFE] Speed too high for test.")
            return False

        if op == "TURN":
            if turn_dir not in ["LEFT", "RIGHT"]:
             print("[SAFE] TURN missing valid turn_dir.")
             return False
            if spd <= 0:
             print("[SAFE] TURN speed must be positive.")
             return False
            
        if op == "MOVE":
        # Optional: block forward movement if front obstacle is close
            if telemetry is not None and spd > 0:
              us = telemetry.get("ultrasonics", {})
              front_vals = [
                  us.get("left_cm"),
                   us.get("center_cm"),
                   us.get("right_cm"),
                   us.get("lidar_cm")
                ]
            valid_front = [x for x in front_vals if isinstance(x, (int, float))]

            if valid_front and min(valid_front) < 35:
                print("[SAFE] Obstacle too close. Blocking forward MOVE.")
                return False

        return True

    def update_memory(self, msg : dict) -> None:
        """
        Update the internal memory with a new message.
        """
        if (len(self.memory) > self.memory_depth):
            self.memory.pop(1)  # Remove oldest, keep system context
        self.memory.append(msg)