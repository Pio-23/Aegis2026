# Raspberry Pi to Arduino Communication Driver
# AEGIS Senior Design, Created on 5/23/2025
# Designed for Arduino MEGA 2560 rev3 and R.Pi 5
# Uses RP1 chip's uart2 on GPIO4/5 (pins 7/29 (TX/RX))

from serial import Serial
from threading import Thread
import os
import time
import json


from utils import serial_utils      # UGV_BAUDRATE
from utils import file_utils        # make_telemetry_JSON(), update_telemetry_JSON(), TRIPS_FOLDER
from utils.led_utils import *       # map_ultrasonic_to_pixel()
from utils import pin_utils as pins
from rover import controller
from rover import camera            # ugv_cam
from lidar import scan

INPUT_BUFFER_SECONDS = 0.1
STICK_MOVE_THRESHOLD = 0.05         # Fixes stick drift

LLM_DRIVE_ENABLED = True     # If True, disables manual control for LLM autopilot
HUMAN_APPROVAL = False #temporal for the rover to take aproval from user before executing any action

LEFT_TURN_SPEED = 0.96
RIGHT_TURN_SPEED = 0.90

LEFT_SECONDS_PER_15_DEG = 0.50
RIGHT_SECONDS_PER_15_DEG = 3.5 / 6.0   # ~= 0.5833

emergency_stop = False

tripping: bool = False
scanner = scan.Scanner()
#ugv_cam = camera.Camera()

def get_cpu_util() -> float:
    """
    Helper function to capture last second of CPU utilization by comparing
    jiffies. I swear, look it up.
    Returns:
        util (float): The CPU utilization percentage since the last call.
    """
    with open(file='/proc/stat') as f:
        line: str | None = next((ln for ln in f if ln.startswith('cpu ')), None)
    if not line: return 0.0

    # Kernel Clock Ticks: user, nice, system, idle, iowait, irq, softirq, steal
    vals: list[float] = list(map(float, line.split()[1:9]))
    idle_all: float = vals[3] + vals[4]
    total: float = sum(vals)

    prev = getattr(get_cpu_util, "_prev", None)
    get_cpu_util._prev = (total, idle_all)                      #type: ignore
    if not prev: return 0.0

    dt_total = total - prev[0]
    if dt_total <= 0: return 0.0

    util = (dt_total - (idle_all - prev[1])) / dt_total
    return round(util * 100, 2)

def open_serial_connection() -> Serial:
    """
    Configures and opens a serial connection on UART2 (ttyAMA2) (GPIO 4/5).

    Returns:
        s (Serial): The configured and open serial port.
    """
    
    s = Serial()		    # Create serial connection
    s.port = pins.ARDUINO_PORT	# UART port on GPIO 4 and 5
    s.baudrate = serial_utils.UGV_BAUDRATE
    s.bytesize = 8			# 8 bits per byte
    s.parity = 'N'			# No parity bit
    s.stopbits = 1			# One stop bit per byte
    s.timeout = None		# Wait forever if necessary
    s.exclusive = True      # Restrict cross-module access

    # Open configured serial connection and allow time for connection to form
    s.open()
    time.sleep(0.05)
    print(f"[INI] UART.py: UGV serial port opened at {s.name}.")
    return s

def read_data(serial_conn : Serial) -> bytes | None:
    """
    Reads in as many bytes as are available from the Arduino via the serial
    connection.

    Args:
        serial_conn (Serial): The serial connection between the Arduino and the
            Raspberry Pi.
    Returns:
        data (bytes | None): Either the bytes waiting to be read from the buffer
            or None if the buffer is empty.
    Raises:
        RuntimeError: If the read_data function was called on a closed serial 
            port.
    """

    if serial_conn.is_open:
        if bytes_waiting := serial_conn.in_waiting > 0:
            data = serial_conn.read(bytes_waiting)
        else:
            data = None
        return data
    else:
        raise RuntimeError("Attempted to read from closed serial port!")
    
def listen_to_UGV(serial_conn: Serial, trip_json : str, dump_folder: str, controller_thread : Thread, ugv_cam) -> None:
    """
    Captures telemetry data from the Arduino and writes it to the trip's 
    telemetry JSON file. If the data is malformed, skips the frame.

    Args:
        serial_conn (Serial): The serial connection between the Arduino and the
            Raspberry Pi.
        start_time (str): The timestamp representing the start of the current 
            trip.
        dump_folder (str): The path to the telemetry JSON file's folder.
    """

    while controller_thread.is_alive():

        set_pixel(ARD_ADDR, PX_WHITE) # MIGHT BE TOO QUICK TO OBSERVE

        ugv_data: bytes = serial_conn.read_until(expected=b'\n')

        try:
            tel_dict = process_telemetry(ugv_data, ugv_cam)

            filename: str = file_utils.update_telemetry_JSON(
                filepath=dump_folder, filename=trip_json, telemetry=tel_dict)

        except RuntimeError:
            print("[ERR] UART.py: INVALID ARDUINO TELEMETRY (BADLEN)\n")
            print(f"[DEBUG] Raw Arduino frame: {ugv_data!r}")
        
        set_pixel(ARD_ADDR, PX_OFF) # MIGHT BE TOO QUICK TO OBSERVE
   
def process_telemetry(data: bytes, ugv_cam=None) -> dict:
    """
    Converts the bytestream from the Arduino into key-value pairs inside of a 
    telemetry dictionary. This is terrible and I'm sorry.

    Args:
        data (bytes): The unprocessed telemetry byte array from the Arduino.
        ugv_cam: The camera object for the UGV.
    Returns:
        telemetry (dict): The dictionary of telemetry key-value pairs.
    Raises:
        RuntimeError: If the Arduino's byte array does not contain the correct
            number of elements.
    """

    try:
        t_str: str = data.decode('utf-8')

    except UnicodeDecodeError as e:
        print("[ERR] UART.py: UTF8 Decode Error\n")
        print(e)
        t_str: str = ''

    ard_vals: list[str] = t_str.split('|')
    del ard_vals[-1]    # Pesky CRLF ending
    
    val_prefixes: list[str] = [
        "TIME=", 
        "LFV=", "LFA=", "LFR=", "LMV=", "LMA=", "LMR=", "LRV=", "LRA=", "LRR=",
        "RFV=", "RFA=", "RFR=", "RMV=", "RMA=", "RMR=", "RRV=", "RRA=", "RRR=",
        "USLI=", "USLF=", "USCT=", "USRT=", "USRR=",
        "R=", "P=", "Y=", "AX=", "AY=", "AZ=",
        "TEMP=", "RHUM=", "LVIS=", "LINF=",
        "BV=", "BA=", "BPCT="
    ]

    if len(ard_vals) != len(val_prefixes):
        raise RuntimeError

    ard_up_s = float(ard_vals[ 0].replace(val_prefixes[ 0], ''))

    mot_lf_v = float(ard_vals[ 1].replace(val_prefixes[ 1], ''))
    mot_lf_a = float(ard_vals[ 2].replace(val_prefixes[ 2], ''))
    mot_lf_r = float(ard_vals[ 3].replace(val_prefixes[ 3], ''))
    map_mot_current_to_pixel(LF_ADDR, mot_lf_a)
    mot_lm_v = float(ard_vals[ 4].replace(val_prefixes[ 4], ''))
    mot_lm_a = float(ard_vals[ 5].replace(val_prefixes[ 5], ''))
    mot_lm_r = float(ard_vals[ 6].replace(val_prefixes[ 6], ''))
    map_mot_current_to_pixel(LM_ADDR, mot_lf_a)
    mot_lr_v = float(ard_vals[ 7].replace(val_prefixes[ 7], ''))
    mot_lr_a = float(ard_vals[ 8].replace(val_prefixes[ 8], ''))
    mot_lr_r = float(ard_vals[ 9].replace(val_prefixes[ 9], ''))
    map_mot_current_to_pixel(LR_ADDR, mot_lf_a)
    mot_rf_v = float(ard_vals[10].replace(val_prefixes[10], ''))
    mot_rf_a = float(ard_vals[11].replace(val_prefixes[11], ''))
    mot_rf_r = float(ard_vals[12].replace(val_prefixes[12], ''))
    map_mot_current_to_pixel(RF_ADDR, mot_lf_a)
    mot_rm_v = float(ard_vals[13].replace(val_prefixes[13], ''))
    mot_rm_a = float(ard_vals[14].replace(val_prefixes[14], ''))
    mot_rm_r = float(ard_vals[15].replace(val_prefixes[15], ''))
    map_mot_current_to_pixel(RM_ADDR, mot_lf_a)
    mot_rr_v = float(ard_vals[16].replace(val_prefixes[16], ''))
    mot_rr_a = float(ard_vals[17].replace(val_prefixes[17], ''))
    mot_rr_r = float(ard_vals[18].replace(val_prefixes[18], ''))
    map_mot_current_to_pixel(RR_ADDR, mot_lf_a)
    rpm_avg: float = (mot_lf_r+mot_lm_r+mot_lr_r+mot_rf_r+mot_rm_r+mot_rr_r) / 6
    map_rpm_to_pixel(RPM_ADDR, rpm=rpm_avg)

    us_li_cm = float(ard_vals[19].replace(val_prefixes[19], ''))
    us_lf_cm = float(ard_vals[20].replace(val_prefixes[20], ''))
    us_ct_cm = float(ard_vals[21].replace(val_prefixes[21], ''))
    us_rt_cm = float(ard_vals[22].replace(val_prefixes[22], ''))
    us_rr_cm = float(ard_vals[23].replace(val_prefixes[23], ''))
    map_ultrasonic_to_pixel(
        USFT_ADDR, min(us_lf_cm, us_ct_cm, us_rt_cm)
    )
    map_ultrasonic_to_pixel(USRR_ADDR, us_rr_cm)
    map_ultrasonic_to_pixel(USLI_ADDR, us_li_cm)

    imu_roll_deg  = float(ard_vals[24].replace(val_prefixes[24], ''))
    imu_pitch_deg = float(ard_vals[25].replace(val_prefixes[25], ''))
    imu_yaw_deg   = float(ard_vals[26].replace(val_prefixes[26], ''))
    imu_ax_mps2   = float(ard_vals[27].replace(val_prefixes[27], ''))
    imu_ay_mps2   = float(ard_vals[28].replace(val_prefixes[28], ''))
    imu_az_mps2   = float(ard_vals[29].replace(val_prefixes[29], ''))

    ambient_temp_c = float(ard_vals[30].replace(val_prefixes[30], ''))
    relative_hum_pct = float(ard_vals[31].replace(val_prefixes[31], ''))
    visible_light_l  = float(ard_vals[32].replace(val_prefixes[32], ''))
    infrared_light_l = float(ard_vals[33].replace(val_prefixes[33], ''))

    batt_v = float(ard_vals[34].replace(val_prefixes[34], ''))
    batt_a = float(ard_vals[35].replace(val_prefixes[35], ''))
    batt_pct = float(ard_vals[36].replace(val_prefixes[36], ''))

    # TEMP FAKE BATTERY VALUES FOR AUTOPILOT TESTING
    batt_v = 16.2
    batt_a = 0.5
    batt_pct = 80.0
    
    map_batt_to_pixel(BAT_ADDR, batt_pct) # Pass as val from 0 to 100.0

    # POPULATE RASPBERRY PI TELEMETRY
    cpu_util_pct: float = get_cpu_util()
    adc_vals: list[str] = os.popen('vcgencmd pmic_read_adc').read().split()
    fmt = lambda idx, pre, post: round(
        float(adc_vals[idx].replace(pre,'').replace(post,'')), 4)
    vdd_core_a: float = fmt(13, 'current(6)=', 'A')
    vdd_core_v: float = fmt(49, 'volt(24)=', 'V')

    mem: list[str] = os.popen('free').read().split()[7:9]
    mem_util_pct: float = round(100*int(mem[1])/int(mem[0]), 2)

    storage_avail_str: str = os.popen('df -h /').read().split()[10]
    if storage_avail_str.find('G') >= 0:
        storage_avail_gb = float(storage_avail_str[:-1])
    elif storage_avail_str.find('M') >= 0:
        storage_avail_gb = float(storage_avail_str[:-1]) / 1000
    else:
        storage_avail_gb = 0.0

    uptime: str = os.popen("awk '{print $1}' /proc/uptime").read(
                            ).replace('\n', '')
    uptime_s: float = round(float(uptime))

    soc_temp_c: float = float(os.popen('vcgencmd measure_temp').read()
                            .replace('temp=','').replace("'C\n", ''))

    telemetry = {
        "rpi": {
            "uptime_s":         uptime_s,
            "cpu_util_pct":     cpu_util_pct,
            "mem_util_pct":     mem_util_pct,
            "storage_avail_gb": storage_avail_gb,
            "temp_c":           soc_temp_c,
            "vdd_core_a":       vdd_core_a,
            "vdd_core_v":       vdd_core_v
        },
        "arduino": {
            "uptime_s": ard_up_s
        },
        "lidar": {
            "scanning":      scanner.is_scanning,
            "scan_pct":      scanner.scan_pct,
            "trimming":      scanner.is_trimming,
            "converting":    scanner.is_converting,
            "saving":        scanner.is_saving,
            "motor_pos_deg": scanner.motor.curr_angle
        },
        "camera": {
            "connected": (
                ugv_cam is not None and ugv_cam.connected
            ),
            "recording": (
                ugv_cam is not None and ugv_cam.recording
            ),
        },
        "motors": {
            "front_left": {
                "voltage_v": mot_lf_v, "current_a": mot_lf_a, "rpm": mot_lf_r
            },
            "mid_left": {
                "voltage_v": mot_lm_v, "current_a": mot_lm_a, "rpm": mot_lm_r
            },
            "rear_left": {
                "voltage_v": mot_lr_v, "current_a": mot_lr_a, "rpm": mot_lr_r
            },
            "front_right": {
                "voltage_v": mot_rf_v, "current_a": mot_rf_a, "rpm": mot_rf_r
            },
            "mid_right": {
                "voltage_v": mot_rm_v, "current_a": mot_rm_a, "rpm": mot_rm_r
            },
            "rear_right": {
                "voltage_v": mot_rr_v, "current_a": mot_rr_a, "rpm": mot_rr_r
            }
        },
        "ultrasonics": {
            "lidar_cm": us_li_cm,
            "left_cm": us_lf_cm,
            "center_cm": us_ct_cm,
            "right_cm": us_rt_cm,
            "rear_cm": us_rr_cm
        },
        "imu": {
            "roll_deg": imu_roll_deg,
            "pitch_deg": imu_pitch_deg,
            "yaw_deg": imu_yaw_deg,
            "accel_x_mps2": imu_ax_mps2,
            "accel_y_mps2": imu_ay_mps2,
            "accel_z_mps2": imu_az_mps2
        },
        "ugv": {
            "battery": {
                "capacity_pct": batt_pct,
                "voltage_v": batt_v,
                "current_a": batt_a
            },
            "headlights": False, 
            "ambient_temp_c": ambient_temp_c,
            "relative_hum_pct": relative_hum_pct,
            "ambient_light_l": visible_light_l,
            "ambient_infrared_l": infrared_light_l
        }
    }
    
    return telemetry

def control_UGV(serial_conn : Serial, dump_folder: str, tripping: bool) -> None:
    """
    Enables the UGV to be piloted by a controller, such as an XBOX controller.
    Reads controller input states from a background listener thread. Includes 
    measures to prevent the overflowing of the Arduino command buffer.

    Args:
        serial_conn (Serial): The serial connection between the Arduino and the
            Raspberry Pi.
    """
    listener_thread = Thread(target=controller.listen, daemon=True)
    listener_thread.start()
    time_since_last_command = time.time()

    while tripping:

        current_time: float = time.time()
        global emergency_stop

        if LLM_DRIVE_ENABLED:
            if controller.input_states['BTN_START'] and controller.input_states['BTN_SELECT']:
                print("[SAFE] Emergency stop triggered from controller.")
                emergency_stop = True
                serial_conn.write(generate_command(op="MOVE", spd=0.0))
                tripping = False
                break
            # B button = stop motors but keep program running
            if controller.input_states['BTN_B']:
                print("[SAFE] Controller stop override.")
                emergency_stop = True
                serial_conn.write(generate_command(op="MOVE", spd=0.0))

            # A button = clear stop override
            if controller.input_states['BTN_A']:
                print("[SAFE] Controller override cleared.")
                emergency_stop = False
            time.sleep(0.05)
            continue    
        
        if (current_time - time_since_last_command) > INPUT_BUFFER_SECONDS:

            # Hold zero speed
            if (not controller.input_states['BTN_RZ']
            and not controller.input_states['BTN_Z']):
                #print("[RUN] UART.py: Holding zero speed.")
                serial_conn.write(
                    generate_command(
                        op = "MOVE",
                        spd = 0.0
                    )   # type: ignore
                )
            
            # RIGHT TRIGGER: move forward
            if (controller.input_states['BTN_RZ'] 
            and not controller.input_states['BTN_Z']
            and not controller.input_states['BTN_TL']
            and not controller.input_states['BTN_TR']):
                serial_conn.write(
                    generate_command(
                        op = "MOVE", 
                        spd = controller.input_states['BTN_RZ']
                    )   # type: ignore
                )
            # LEFT TRIGGER: move backward
            if (controller.input_states['BTN_Z'] 
            and not controller.input_states['BTN_RZ'] 
            and not controller.input_states['BTN_TL']
            and not controller.input_states['BTN_TR']):
                serial_conn.write(
                    generate_command(
                        op = "MOVE", 
                        spd = controller.input_states['BTN_Z'] * -1
                    )   # type: ignore
                )
            # LEFT BUMPER + RIGHT TRIGGER: spin turn left
            if (controller.input_states['BTN_TL']
            and controller.input_states['BTN_RZ']
            and not controller.input_states['BTN_START']):
                serial_conn.write(
                    generate_command(
                        op = "TURN", turn_dir = "LEFT",
                        spd = controller.input_states['BTN_RZ']
                    )   # type: ignore
                )
            # RIGHT BUMPER + RIGHT TRIGGER: spin turn right
            if (controller.input_states['BTN_TR']
            and controller.input_states['BTN_RZ']
            and not controller.input_states['BTN_START']):
                serial_conn.write(
                    generate_command(
                        op = "TURN", turn_dir = "RIGHT",
                        spd = controller.input_states['BTN_RZ']
                    )   # type: ignore
                )
                
            # if ugv_cam is not None:
            #     # START + A: start recording
            #     if (controller.input_states['BTN_START'] 
            #     and controller.input_states['BTN_A'] 
            #     and not ugv_cam.recording):
            #         video_filename = ugv_cam.my_start_recording()
            #         #sleep(0.01)
            #         #print(f"UART.py: Recording video to '{video_filename}'...")

            #     # START + B: stop recording
            #     if (controller.input_states['BTN_START'] 
            #     and controller.input_states['BTN_B'] 
            #     and ugv_cam.recording):
            #         #ugv_cam.my_stop_recording()
            #         #sleep(0.01)
            #         print(f"UART.py: Recording saved to '{video_filename}'.")   # type: ignore
                    
            # else:
            #     if ((controller.input_states['BTN_START']
            #          and controller.input_states['BTN_A']) 
            #     or  (controller.input_states['BTN_START']
            #          and controller.input_states['BTN_B'])):
            #         print("[RUN] UART.py: No camera connected!")

            # START + Y: take LiDAR scan
            if (controller.input_states['BTN_START']
            and controller.input_states['BTN_Y']):
                scanner.scan(filepath=dump_folder)

            # START + RIGHT BUMPER: increase scan resolution
            if (controller.input_states['BTN_START'] 
            and controller.input_states['BTN_TR']):
                if (scanner.rings_per_cloud < 1600):
                    scanner.set_rings_per_cloud(
                        num_rings=scanner.rings_per_cloud * 2)
                    print(f"[RUN] UART.py: Increased scan resolution to "
                          f"{scanner.rings_per_cloud} rings.")
                    
            # START + LEFT BUMPER: decrease scan resolution
            if (controller.input_states['BTN_START']
            and controller.input_states['BTN_TL']):
                if (scanner.rings_per_cloud > 50):
                    scanner.set_rings_per_cloud(
                        num_rings=int(scanner.rings_per_cloud / 2))
                    print(f"[RUN] UART.py: Decreased scan resolution to "
                          f"{scanner.rings_per_cloud} rings.")
                    
            # START + SELECT: end trip
            if (controller.input_states['BTN_START']
            and controller.input_states['BTN_SELECT']
            and tripping):
                print("[RUN] UART.py: Trip terminated.")
                tripping = False # Breaks out of control loop

            time_since_last_command: float = current_time
    
def generate_command(op : str, **kwargs) -> bytes | None:
    """
    Generates instructions to be sent to the Arduino MEGA in accordance with a 
    simple one-byte command structure. MSB is op, second MSB is dir, the rest 
    encode speed from 0 to 220.4 rpm with a resolution of about 3.5 rpm.

    Args:
        op (str): The category of command to transmit. Currently either "TURN"
            or "MOVE".
        **kwargs: A series of key-value pairs that depend on the op selected.
            Currently TURN uses 'turn_dir' = 0 or 1, and both use 'spd' = 
            [-1, 1].
    Returns:
        command (bytes): A one-byte command to be sent to the Arduino.
    Raises:
        OverflowError: If the command overflows the one byte container.
    """

    # MSB is op, second MSB is dir, the rest encode speed
    command: int = 256  # Will raise OverflowError if not overwritten
    
    if op == "MOVE":
        command = 0 << 7                        # 0000_0000
        for key, val in kwargs.items():
            if key == "spd":
                command += (val < 0) << 6	    # Sets dir to bit seven
                command += int(abs(val) * 63)   # Speed [0,63]

    if op == "TURN":
        command = 1 << 7	                    # 1000_0000
        for key, val in kwargs.items():
            if key == "spd":
                command += int(abs(val) * 63)   # Speed [0,63]
            if key == "turn_dir":
                command += (val == "RIGHT") << 6
    
    try:
        return command.to_bytes(1)
    except OverflowError:
        print("[ERR] UART.py: Invalid command generated!")


#this is for constant movement if it doesnt work remove it and change other stuff
def execute_supervised_movement(serial_conn, command, args, get_current_telemetry, duration_s=1.5 ):
    """
    Continuously resend an AI movement command for a short period
    while monitoring ultrasonic telemetry.

    Stops early if an obstacle becomes too close or telemetry is lost.
    """

    global emergency_stop

    # Temporary safety distances.
    # We can tune these after checking the ultrasonic sensors.
    FRONT_STOP_CM = 35.0
    REAR_STOP_CM = 35.0

    # Get movement information once
    op = args.get("op")
    spd = args.get("spd", 0.0)

    # duration_s is calculated by the caller.
    print(f"[AI] Movement duration: {duration_s:.3f}s")

    print(f"[AI] Movement duration: {duration_s}s")

    start_time = time.time()

    while (time.time() - start_time) < duration_s:

        # Controller emergency stop always wins
        if emergency_stop:
            print("[SAFE] Emergency stop during AI movement.")
            break

        telemetry = get_current_telemetry()

        if not isinstance(telemetry, dict):
            print("[SAFE] Telemetry lost during movement.")
            break

        ultrasonics = telemetry.get("ultrasonics", {})

        left = ultrasonics.get("left_cm")
        center = ultrasonics.get("center_cm")
        right = ultrasonics.get("right_cm")
        rear = ultrasonics.get("rear_cm")

        # ------------------------------------------
        # FORWARD SAFETY
        # ------------------------------------------

        if op == "MOVE" and spd > 0:

            front_readings = [
                value for value in [left, center, right]
                if value is not None and value > 0
            ]

            if (
                front_readings
                and min(front_readings) < FRONT_STOP_CM
            ):
                print(
                    "[SAFE] Front ultrasonic obstacle detected. "
                    "Stopping movement."
                )
                break

        # ------------------------------------------
        # REVERSE SAFETY
        # ------------------------------------------

        if op == "MOVE" and spd < 0:

            if (
                rear is not None
                and rear > 0
                and rear < REAR_STOP_CM
            ):
                print(
                    "[SAFE] Rear ultrasonic obstacle detected. "
                    "Stopping movement."
                )
                break

        # Keep sending the same movement command
        serial_conn.write(command)

        # Roughly 10 Hz, similar to manual control behavior
        time.sleep(0.1)

    # ALWAYS stop at the end of the segment
    serial_conn.write(
        generate_command(
            op="MOVE",
            spd=0.0
        )
    )

    print("[AI] Supervised movement segment completed.")


def give_controls_to_autopilot(serial_conn: Serial, trip_json: str, dump_folder: str, tripping: bool, ugv_cam ) -> None:

    print("[INI] UART.py: LLM Autopilot Enabled.")

    from rover.autopilot import Autopilot
    spartan = Autopilot()

    def get_current_telemetry():
        return file_utils.get_latest_telemetry(
            filepath=dump_folder,
            filename=trip_json
        )

    while tripping:

        telemetry = file_utils.get_latest_telemetry(
            filepath=dump_folder,
            filename=trip_json
        )

        if telemetry is None:
            print("[ERR] UART.py: No telemetry available for autopilot.")

            serial_conn.write(
                generate_command(
                    op="MOVE",
                    spd=0.0
                )   
            )
            time.sleep(0.2)
            continue

        actions = spartan.decide_actions(
            telemetry,
            scanner,
            ugv_cam,
            dump_folder,
            get_current_telemetry
        )

        for action in actions:

            name = action.function.name
            args = json.loads(
                action.function.arguments or "{}"
            )

            if name == "move_rover":

                print("[AI] GPT requested rover movement:")
                print(args)

                if emergency_stop:
                    print("[SAFE] AI movement blocked by emergency stop.")

                    serial_conn.write(
                        generate_command(
                            op="MOVE",
                            spd=0.0
                        )
                    )

                    spartan.add_tool_result(
                        action.id,
                        {
                            "status": "blocked",
                            "reason": "Emergency stop active"
                        }
                    )

                    continue

                op = args.get("op")
                spd = args.get("spd", 0.0)
                turn_dir = args.get("turn_dir")

                try:

                    if op == "MOVE":
                        
                        duration_s = 1.5

                        command = generate_command(
                            op="MOVE",
                            spd=spd
                        )

                    elif op == "TURN":
                        turn_dir = args.get("turn_dir", "LEFT")
                        turn_degrees = args.get("turn_degrees", 15)

                        # Only allow calibrated 15-degree increments.
                        allowed_turns = [15, 30, 45, 60, 75, 90]

                        if turn_degrees not in allowed_turns:
                            turn_degrees = min(
                                allowed_turns,
                                key=lambda x: abs(x - turn_degrees)
                            )

                        if turn_dir == "LEFT":
                            spd = LEFT_TURN_SPEED
                            duration_s = (
                                turn_degrees / 15.0
                            ) * LEFT_SECONDS_PER_15_DEG

                        else:
                            spd = RIGHT_TURN_SPEED
                            duration_s = (
                                turn_degrees / 15.0
                            ) * RIGHT_SECONDS_PER_15_DEG

                        args["spd"] = spd
                        args["turn_degrees"] = turn_degrees

                        command = generate_command(
                            op="TURN",
                            spd=spd,
                            turn_dir=turn_dir
                        )

                        print(
                            f"[AI] TURN {turn_dir}: "
                            f"{turn_degrees} deg | "
                            f"speed={spd:.2f} | "
                            f"time={duration_s:.2f}s"
                        )
                        

                    if command is not None:
                        #this stuff too is for the constant movement if it doesnt work remove it and change other stuff
                        print(
                            f"[AI] Executing supervised movement: {args}"
                        )

                        print(
                            f"[DEBUG] Sending command byte: "
                            f"{int.from_bytes(command, 'big')} "
                            f"binary={int.from_bytes(command, 'big'):08b}"
                        )

                        execute_supervised_movement(
                            serial_conn=serial_conn,
                            command=command,
                            args=args,
                            get_current_telemetry=get_current_telemetry,
                            duration_s=duration_s
                        )

                        spartan.add_tool_result(
                            action.id,
                            {
                                "status": "completed",
                                "command": args,
                                "movement_duration_s": duration_s
                            }
                        )

                        # ------------------------------------------
                        # FRESH CAMERA IMAGE AFTER MOVEMENT
                        # ------------------------------------------

                        if ugv_cam is not None and ugv_cam.connected:

                            print("[AI] Capturing fresh camera image after movement.")

                            # Give motor current / electrical noise a moment to settle
                            time.sleep(0.75)

                            camera_filename = ugv_cam.capture_image(
                                filepath=dump_folder
                            )

                            if camera_filename is not None:

                                print(
                                    f"[AI] Post-movement camera image saved: "
                                    f"{camera_filename}"
                                )

                                spartan.add_camera_observation(
                                    camera_filename
                                )

                                print(
                                    "[AI] Fresh post-movement camera image "
                                    "added to navigation memory."
                                )

                            else:
                                print(
                                    "[AI] Post-movement camera capture failed."
                                )

                    else:
                        spartan.add_tool_result(
                            action.id,
                            {
                                "status": "failed",
                                "reason": "generate_command returned None"
                            }
                        )

                except Exception as e:

                    print(
                        f"[ERR] AI movement failed: {e}"
                    )

                    spartan.add_tool_result(
                        action.id,
                        {
                            "status": "failed",
                            "reason": str(e)
                        }
                    )

            elif name == "no_op":

                print("[AI] GPT decided not to move.")
                print(args)

                serial_conn.write(
                    generate_command(
                        op="MOVE",
                        spd=0.0
                    )
                )

                spartan.add_tool_result(
                    action.id,
                    {
                        "status": "completed",
                        "result": "Rover remained stationary"
                    }
                )

        time.sleep(3)

def run_comms() -> None:
    """
    Establishes bidirectional UART communication between the Arduino and
    Raspberry Pi. Records the beginning of the trip, creates a trip folder,
    and spawns and joins two threads: one for the controls and one for the 
    telemetry retrieval.
    """
    
    tripping = True

    # Create camera inside UART subprocess
    ugv_cam = camera.Camera()

    if ugv_cam.connected:
        print("[INI] UART.py: Camera connected.")
    else:
        print("[ERR] UART.py: Camera unavailable.")

    trip_start_timestamp: str = file_utils.get_current_timestamp()
    trip_folder: str = file_utils.make_folder(
        file_utils.TRIPS_FOLDER, trip_start_timestamp)
    print(f"[INI] UART.py: Created trip folder at {trip_folder}.")

    trip_json: str = file_utils.make_telemetry_JSON(filepath=trip_folder)
    print(f"[INI] UART.py: Created trip telemetry JSON at {trip_json}.")

    serial_conn: Serial = open_serial_connection()

    tripping = True

    controller_thread = Thread(target=control_UGV,
                                args=[
                                    serial_conn,
                                    trip_folder,
                                    tripping
                                ],
        daemon=True
    )

    telemetry_thread = Thread(target=listen_to_UGV, 
                                args=[
                                    serial_conn, 
                                    trip_json, 
                                    trip_folder, 
                                    controller_thread,
                                    ugv_cam
                                ]
    )

    controller_thread.start()   # Background thread, exit signals tel exit

    telemetry_thread.start()

    time.sleep(0.5) #little delay

    #moves this to last to ahv ethe telemetry generated first and then start the LLM thread, so it has telemetry to work with
    if LLM_DRIVE_ENABLED: 
        autopilot_thread = Thread(target=give_controls_to_autopilot,
                                    args=[
                                        serial_conn,
                                        trip_json,
                                        trip_folder,
                                        tripping,
                                        ugv_cam
                                    ],
            daemon=True
        )
        autopilot_thread.start()

    telemetry_thread.join()