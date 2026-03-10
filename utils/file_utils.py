# File Utilities
# Created 6/26/2025

from datetime import datetime
import os
import json
from filelock import FileLock

TRIPS_FOLDER = "./stream/static/trips"

def make_folder(path: str, name: str) -> str:
    '''
    Don't include the trailing slash in path or leading slash in name
    '''
    dest = f"{path}/{name}"
    if not os.path.exists(path=dest):
        os.mkdir(dest)
        return dest

    raise ValueError(f"Path already exists to '{dest}'!")

def get_current_timestamp() -> str:
    return datetime.now().strftime("%2Y%m%d_%H%M%S")

def get_timestamped_filename(save_path: str, prefix: str, ext: str) -> str:
    """
    Returns a unique formatted filename for each second.\n
    **Include a period in file extensions, and do not include a trailing slash in save paths!** \n 
    Filename follows the format: *"{save_path}/{prefix}_YearMonthDay_HourMinuteSecond{ext}"*
    """
    if (ext.find('.') == -1):
        raise ValueError("File extension arguments must include a leading period.")
    if (save_path[-1] == '/'):
        raise ValueError("Save path must not include a trailing slash.")
    if (prefix.find('.') != -1 or prefix.find('/') != -1):
        raise ValueError("Filename prefix must not include periods or slashes.")

    timestamp: str = datetime.now().strftime("%2Y%m%d_%H%M%S")
    filename: str = f"{save_path}/{prefix}_{timestamp}{ext}"
    return filename

def write_pcd_file_header(filename: str, pts: int | None = None, header: list[str] | None = None) -> None:
    """
    Will write the following PCD header if header argument is not specified:\n
    header = [
        "VERSION .7",
        "FIELDS x y z rgb",
        "SIZE 4 4 4 4",
        "TYPE F F F F",
        "COUNT 1 1 1 1",
        "WIDTH pts",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        "POINTS pts",
        "DATA ascii"
    ]
    Args:
        filename: A string representing the filename destination for the header.
        pts: An integer representing the number of points that the file will contain.
        header: A list of strings representing the header to be written to the file.
    """
    if (filename.find('.pcd') != -1 and pts is None):
        raise ValueError("You must specify the number of points in the cloud!")

    if filename.find('.pcd') != -1 and header is None:
        header = [
            "VERSION .7\n",
            "FIELDS x y z rgb\n",
            "SIZE 4 4 4 4\n",
            "TYPE F F F F\n",
            "COUNT 1 1 1 1\n",
            f"WIDTH {pts}\n",
            "HEIGHT 1\n",
            "VIEWPOINT 0 0 0 1 0 0 0\n",
            f"POINTS {pts}\n",
            "DATA ascii\n"
        ]
    
    with open(filename, "a") as file:
        file.writelines(header) # type: ignore

def write_points_to_file(filename: str, points: list[list[float]]) -> None:
    '''
    Writes point data to a file. Currently no validation or error handling.

    Args:
        filename (str): The filename to write to.
        points (list[list[float]]): The points to be written.
    '''

    with open(filename, 'a') as file:
        for point in points:
            file.write(f"{' '.join([str(val) for val in point])}\n")

def make_telemetry_JSON(filepath = '') -> str:

    timestamp: str = get_current_timestamp()

    if not os.path.exists(filepath):
        filepath = f"{TRIPS_FOLDER}/{timestamp}"

    filename: str = f"{filepath}/tel_{timestamp}.json"

    # If the file exists, don't make another one, just go home
    if os.path.exists(path=filename):
        raise FileExistsError(f"[ERR] file_utils.py: JSON file already exists at '{filename}'!")

    telemetry = {
        "timestamp": timestamp,
        "duration_s": 0,
        "traits": {
            "battery_cap_mah": 6000,
            "models": {
                "lidar": "STL27L",
                "lidar_motor": "NEMA17 Stepper Motor",
                "motors": "Yellowjacket 5203",
                "ultrasonics": "HC-SR04",
                "camera": "Raspberry Pi Camera Module 3",
                "rpi": "Raspberry Pi 5 8GB",
                "arduino": "Arduino Mega 2560 Rev3",
                "battery": "Zeee 14.8V Lipo Battery (4S, 60C)"
            }
        },
        "videos": [],
        "scans": [],
        "telemetry": []
    }

    # Make JSON file
    with open(file=filename, mode='w') as tel_json:
        json.dump(obj=telemetry, fp=tel_json, indent=4)
    
    print(f"[RUN] file_utils.py: Created trip JSON at {filename}.")
    return filename

def update_telemetry_JSON(filepath = 'badwords', filename = 'badwords', **kwargs) -> str:
    '''
    Updates the telemetry JSON file with new data.
    Uses a lock to prevent read/write conflicts.
    '''

    if not os.path.exists(filepath):
        print("file_utils.py: NO!")

    if not os.path.isfile(path=filename):
        filename = make_telemetry_JSON(filepath)

    with open(filename) as tel_f:
        telemetry = json.load(tel_f)
    
    for key, value in kwargs.items():
        if key == "video":
            telemetry["videos"].append(value)
        if key == "scan":
            telemetry["scans"].append(value)
        if key == "telemetry":
            telemetry["telemetry"].append(value)
            telemetry["duration_s"] += 1

    # Export the dict to the json file
    with open(filename, 'w') as f:
        json.dump(telemetry, f, indent=4)

    return filename


def get_latest_telemetry(filepath: str, filename: str) -> dict:
    '''
    Opens the trip telemetry JSON and captures the last index of the telemetry 
    list.
    Args:
        filepath (str): The path to the trip folder.
        filename (str): The filename of the telemetry JSON.
    Returns:
        telemetry (dict): The latest telemetry data.
    '''
    if not os.path.isfile(path=filename):
        raise FileNotFoundError(f"[ERR] file_utils.py: No telemetry JSON found at '{filename}'!")

    with FileLock(f"{filename}.lock"):
        with open(filename, 'r') as f:
            telemetry_data = json.load(f)
            if len(telemetry_data["telemetry"]) == 0:
                raise ValueError("[ERR] file_utils.py: No telemetry data found in JSON!")
            latest_telemetry = telemetry_data["telemetry"][-1]

    return latest_telemetry