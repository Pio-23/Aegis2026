"""
Motor driver for AEGIS senior design. Version 2
Compatible with A4988 Stepper motor driver and NEMA17 stepper motor.
"""

#################################
# MS1 | MS2 | MS3 | Resolution  #
#-----|-----|-----|-------------#
#  0  |  0  |  0  | Full Step   #
#  1  |  0  |  0  | Half        #
#  0  |  1  |  0  | Quarter     #
#  1  |  1  |  0  | Eighth      #
#  1  |  1  |  1  | Sixteenth   #
#################################

import gpiozero as gpz		# gpz.OutputDevice, gpz.CompositeOutputDevice
from time import sleep

from utils import pin_utils as pins


class Motor:
    """
    The NEMA17 stepper motor class.

    Attributes:
        microstep_resolutions (dict[str, dict[str, tuple[int,int,int] | int]]): 
            A dictionary containing the names, pin values, and denominator 
            values for each of the five possible resolutions (full, half, 
            quarter, eighth, and sixteenth-step).
        ms_res_pins (gpiozero.CompositeOutputDevice): A GPZ device that ties the
            MS1, MS2, and MS3 GPIO pins together, allowing the resolution to be 
            configured.
        ms_res (str): The name of the current microstep resolution.
        ms_res_denom (int): The denominator of the current microstep resolution.
        dir (str): The motor's turn direction. Either clockwise ("CW") or 
            counterclockwise ("CCW"). 
        step (gpiozero.OutputDevice): A GPZ device that provides access to the 
            motor driver's step pin via the step.on() and step.off() methods.
        speed (float): The speed at which the motor turns in Hz (0, 4.5].
        start_angle (float): The starting angle of the motor in degrees.
        curr_angle (float): The current angle of the motor in degrees.
    """

    microstep_resolutions: dict[str, dict[str, tuple[int,int,int] | int]] = {
        "full":     { "pin_vals": (0,0,0), "denom": 1 },
        "half":     { "pin_vals": (1,0,0), "denom": 2 },
        "quarter":  { "pin_vals": (0,1,0), "denom": 4 },
        "eighth":   { "pin_vals": (1,1,0), "denom": 8 },
        "sixteenth":{ "pin_vals": (1,1,1), "denom": 16 }
    }

    def __init__(self, res_name: str, start_angle: float, speed: float) -> None:
        """
        Initializes a motor object by attaching R.Pi GPIO pins to various fields
        and declaring the speed and starting angle.
        
        Args:
            res_name (str): The name of the microstep resolution of the motor.
            speed (float): The speed at which the motor turns in Hz (0, 4.5].
            start_angle (float): The starting angle of the motor in degrees.
        """
        
        self.ms_res_pins: gpz.CompositeOutputDevice = gpz.CompositeOutputDevice(
            MS1 = gpz.OutputDevice(pin=pins.MS1_PIN),
            MS2 = gpz.OutputDevice(pin=pins.MS2_PIN),
            MS3 = gpz.OutputDevice(pin=pins.MS3_PIN))
        self.ms_res: str = "full"                            # Always overriden
        self.ms_res_denom: int = 1                           # Always overriden
        self.set_microstep_resolution(res_name)

        self.dir: gpz.OutputDevice = gpz.OutputDevice(pin=pins.DIR_PIN)
        self.step: gpz.OutputDevice = gpz.OutputDevice(pin=pins.STEP_PIN)
        self.speed: float = speed
        self.start_angle: float = start_angle
        self.curr_angle: float = start_angle

    def set_microstep_resolution(self, res_name: str) -> None:
        """
        Sets the microstep resolution by looking up the argument key in the 
        microstep_resolution dictionary and writing corresponding values to the 
        MS1, MS2, and MS3 pins. Also updates resolution info in class fields.
        
        Args:
            res_name (str): The name of the desired resolution (e.g. "half").
        Raises:
            ValueError: If the res_name string is not a key in the 
                microstep_resolutions dictionary.
        """

        res: dict|None = self.microstep_resolutions.get(res_name, None)

        if res is None:
            raise ValueError(
                f"[ERR] motor.py: Invalid resolution! ('{res_name}')"
            )
        
        self.ms_res = res_name
        self.ms_res_pins.value = res["pin_vals"]
        self.ms_res_denom = res["denom"]

    def set_speed(self, speed : float) -> None:
        """
        Sets the speed of the motor.

        Args:
            speed (float): The speed at which the motor turns in Hz (0, 4.5].
        Raises:
            ValueError: If speed is outside of the valid range.
        """

        if speed <= 0 or speed > 4.5:
            raise ValueError(
                f"[ERR] motor.py: Invalid speed (0, 4.5]! ({speed})")
        
        self.speed = speed

    def set_dir(self, dir : str) -> None:
        """
        Sets the direction of the motor.
        
        Args:
            dir (str): The motor's turn direction. Either clockwise ("CW") or 
                counterclockwise ("CCW"). 
        Raises:
            ValueError: If dir does not match either "CW" or "CCW".
        """

        if   dir == "CW":  self.dir.on()
        elif dir == "CCW": self.dir.off()
        else: raise ValueError(
            f"[ERR] motor.py: Invalid direction! ('{dir})")

    def set_start_angle(self, angle: float) -> None:
        """
        Sets the starting angle of the motor in software. This currently has no
        relationship to physical angle and is only used in coordinate transform.

        Args:
            angle (float): The starting angle of the motor in degrees.
        """

        self.curr_angle: float = angle

    def turn(self, direction: str, steps: int, verbose: bool = False) -> None:
        """
        Turns the stepper motor a specified number of steps. Also updates
        curr_angle in accordance with the current microstep resolution. 

        Args:
            steps (int): The number of steps to turn.
            direction (str): The direction of turn ("CW" or "CCW").
            verbose (bool): Whether or not to print debug info. Defaults to
                False.
        Raises:
            ValueError: If the steps argument is less than 1.
        """

        if steps < 1:
            raise ValueError(f"Steps argument must be positive! ({steps})")

        degrees: float = float(steps) / (200 * self.ms_res_denom) * 360
        step_delay: float = 1 / (200 * self.ms_res_denom * self.speed)
        
        if verbose:
            print(f"Current Angle: {round(self.curr_angle, 3)}")

        for i in range(0, steps):
            self.step.on()
            sleep(step_delay/2)
            self.step.off()
            sleep(step_delay/2)

        if direction == "CCW":
            self.curr_angle += degrees
        else:
            self.curr_angle -= degrees

        if self.curr_angle > 360:
            self.curr_angle %= 360


def test_motor() -> None:
    """
    Tests the motor by instantiating a motor object and allowing the user to
    repeatedly turn it a set number of steps in either direction.
    """

    M1 = Motor(
        res_name="sixteenth",
        start_angle=0,
        speed=1.0
    )

    print(f"[INI] motor.py: Created motor ({M1.ms_res}-step, {M1.speed}Hz)...")

    while True:
        dir_in: str = input("Enter direction (0 for CW, anything else for CCW):")
        print(f"Direction: {'CW' if int(dir_in) == 0 else 'CCW'}")
        M1.set_dir("CW") if int(dir_in) == 0 else M1.set_dir("CCW")

        steps_in = int(input("Enter the number of steps to turn:"))
        print(f"Steps: {steps_in}")

        if steps_in > 100 * M1.ms_res_denom:
            print("Nice try, you're gonna break the cable!")
            continue
            
        chunk_in = int(input("Enter chunk size (must cleanly divide steps):"))
        print(f"Chunk size: {chunk_in}")

        for i in range(0, int(steps_in / chunk_in)):
            M1.turn(direction=dir_in, steps=chunk_in, verbose=True)
            sleep(0.1)

        input("Enter anything to continue, CTRL+C to exit:")