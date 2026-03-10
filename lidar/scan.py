'''
3D LiDAR scanner driver for AEGIS senior design. Version 2
Compatible with STL27L LiDAR scanner, A4988 motor driver, and NEMA17 stepper 
motor.
'''

import time                     # time()
import random                   # seed(), sample()

from lidar import lidar
from lidar import motor
from utils import file_utils    # get_timestamped_filename()
from utils import math_utils    # sph_to_cart_array()
from utils.led_utils import *   # set_pixel


class Scanner():
    """
    The AEGIS 3D LiDAR scanner class.
    
    This class combines the STL27L LiDAR scanner and NEMA17 stepper motor as a 
    single 360 degree, 3D scanner with configurable angular resolution. It 
    provides methods that configure the resolution of, capture, print, and save 
    point cloud scans.

    Attributes:
        lidar (Lidar): An object of the Lidar class which enables STL27L sensor
            functionality.
        motor (Motor): An object of the Motor class which enables stepper motor
            functionality.
        rings_per_cloud (int): The number of rings per point cloud.
        steps_per_ring (int): The number of steps taken by the stepper motor
            after capturing each ring.
        resolution (float): The angular resolution of scans on the XY plane
            (i.e. the angular distance between rings).
    """

    def __init__(self) -> None:
        """
        Initializes a scanner object by combining an instance of the Lidar class
        and an instance of the Motor class. Configures the number of rings per
        cloud (400 by default) and related values.
        """
        self.lidar: lidar.Lidar = lidar.Lidar()
        self.motor: motor.Motor = motor.Motor(res_name="sixteenth", start_angle=90, speed=1)
        self.rings_per_cloud: int = 400
        self.steps_per_ring: int = int(100 * self.motor.ms_res_denom / self.rings_per_cloud)
        self.resolution: float = 180 / self.rings_per_cloud
        self.is_scanning = False
        self.scan_pct = 0.0
        self.is_trimming = False
        self.is_converting = False
        self.is_saving = False

        set_pixel(LQ1_ADDR, PX_WHITE)
        set_pixel(LQ2_ADDR, PX_WHITE)
        set_pixel(LQ3_ADDR, PX_WHITE)
    
    def set_rings_per_cloud(self, num_rings: int) -> None:
        '''
        Configures the angular resolution of the scanner by adjusting the number
        of rings taken per scan.
        
        Args:
            num_rings (int): The number of rings per scan. 180 degrees divided
                by this value gives the angular distance between rings.
        Raises:
            ValueError: If the selected resolution is too high for the currently
                selected stepper motor microstep resolution.
        '''

        if (num_rings > self.motor.ms_res_denom * 100):
            raise ValueError(f"Resolution is too high!",
                "Must not exceed (steps/rev) / 2 at currently configured",
                "microstep resolution. ({resolution}, {self.motor.ms_res})")
        
        self.rings_per_cloud = num_rings
        self.steps_per_ring: int = int(100 * self.motor.ms_res_denom / self.rings_per_cloud)
        self.resolution: float = 180 / self.rings_per_cloud

    def capture_cloud(self) -> list[list[float]]:
        """
        Takes a 3D scan of the environment. This function is the powerhouse of 
        the cell.\n
        Performs the following steps:
            1. Sets direction of stepper motor to counterclockwise.
            2. Captures a ring's worth of points, extends a cloud array with the 
               points, and turns the motor a specified amount.
            3. Repeats step 2 until the motor angle is at least 180 degrees.
            4. Reverses direction of stepper motor and resets to starting 
               position.
            5. Prints information about size and duration of scan.
            6. Returns the captured cloud.

        Returns:
            cloud (list[list[float]]): A 3D point cloud array.
        """

        start_time_s: float = time.time()
        self.is_scanning = True
        print("[RUN] scan.py: Beginning cloud capture...")

        # Quarter turn to start position from forward facing rest
        self.motor.set_dir("CW")
        self.motor.turn("CW", self.motor.ms_res_denom * 50)
        self.motor.set_dir("CCW")

        cloud: list[list[float]] = []

        while self.motor.curr_angle < 180:
            if (self.motor.curr_angle >= 0 and self.motor.curr_angle < 60):
                set_pixel(LQ1_ADDR, PX_BLUE)
            if (self.motor.curr_angle >= 60 and self.motor.curr_angle < 120):
                set_pixel(LQ2_ADDR, PX_BLUE)     
            if (self.motor.curr_angle >= 120 and self.motor.curr_angle < 180):
                set_pixel(LQ3_ADDR, PX_BLUE)

            self.lidar.open_serial()    # See cylindrical distortion error in documentation

            ring: list[list[float]] = self.lidar.capture_ring(motor_angle=self.motor.curr_angle)
            cloud.extend(ring)
            self.motor.turn("CCW", self.steps_per_ring)

            self.lidar.close_serial()

        set_pixel(LQ1_ADDR, PX_WHITE)
        set_pixel(LQ2_ADDR, PX_WHITE)
        set_pixel(LQ3_ADDR, PX_WHITE)

        self.motor.set_dir("CW")
        self.motor.turn("CW", self.motor.ms_res_denom * 50)

        num_points: int = len(cloud)

        duration_s: float = time.time() - start_time_s
        duration_s = round(duration_s, 2)

        print(f"[RUN] scan.py: Cloud captured in {duration_s} seconds ({num_points} points).")

        self.is_scanning = False
        return cloud

    def trim_cloud(self, cloud: list[list[float]], nonfat_pct: float = 0.2) -> list[list[float]]:
        '''
        Downsamples a cloud (non-destructively) to reduce resolution, file size, 
        and load times. Uses python random library to uniformly downsample.

        Args:
            cloud (list[list[float]]): A cloud of points to be trimmed.
            nonfat_pct (float): A percentage of points to retain (non-fat).
        Returns:
            out (list[list[float]]): The trimmed point cloud.
        '''

        self.is_trimming = True
        start_time_s: float = time.time()

        random.seed()
        num_pts_untrimmed = len(cloud)
        num_pts_trimmed = int(num_pts_untrimmed * nonfat_pct)
        nonfat_cloud: list[list[float]] = random.sample(cloud, num_pts_trimmed)

        duration_s: float = time.time() - start_time_s
        duration_s = round(duration_s, 2)

        print(f"[RUN] UART.py: Cloud trimmed to {nonfat_pct * 100}%, "
              f"removed {num_pts_untrimmed - num_pts_untrimmed} points in "
              f"{duration_s} seconds.")

        self.is_trimming = False
        return nonfat_cloud
    
    def convert_cloud(self, cloud: list[list[float]]) -> list[list[float]]:
        """
        Converts a point cloud from spherical (default) coordinates to cartesian
        (X,Y,Z) coordinates.

        Args: 
            cloud (list[list[float]]): A cloud of points to be converted.
        Returns:
            cartesian_cloud (list[list[float]]): The converted cloud.
        """

        self.is_converting = True
        start_time_s = time.time()
        print("[RUN] scan.py: Converting scan to Cartesian coordinates...")

        set_pixel(LQ2_ADDR, PX_BLUE)
        cartesian_cloud: list[list[float]] = math_utils.sph_to_cart_array(cloud)
        set_pixel(LQ2_ADDR, PX_WHITE)

        duration_s: float = time.time() - start_time_s
        duration_s = round(duration_s, 2)

        print(f"[RUN] scan.py: Converted scan in {duration_s} seconds.")

        self.is_converting = False
        return cartesian_cloud

    def save_cloud(self,  
                  cloud: list[list[float]],
                  filepath: str = '.') -> str:
        """
        Saves the cloud to a timestamped text file.

        Args:
            cloud (list[list[float]]): A cloud of points to be saved.
            filepath (str | None): Where to save the file. Defaults to the 
                current directory ('.').
        Returns:
            filename (str): The name and path of the saved .txt file. For
                example, './path/to/cloud_19690420_080085.txt'.
        """

        self.is_saving = True

        start_time_s: float = time.time()

        set_pixel(LQ3_ADDR, PX_BLUE)

        filename = file_utils.get_timestamped_filename(
            save_path=filepath,
            prefix='cloud', ext='.txt')
            
        print(f"[RUN] scan.py: Saving cloud to {filename}...")
        file_utils.write_points_to_file( filename=filename, points=cloud)

        duration_s: float = time.time() - start_time_s
        duration_s = round(duration_s, 2)

        print(f"[RUN] scan.py: Cloud saved in {duration_s} seconds.")

        set_pixel(LQ3_ADDR, PX_WHITE)

        self.is_saving = False
        return filename
    
    def scan(self,
             trim=False,
             nonfat_pct=0.2,
             convert=True,
             save=True,
             filepath='.') -> str | None:
        """
        Captures, trims, converts, and saves a cloud. Arguments are optional.

        Args:
            trim (bool): Whether or not to trim the scan.
            nonfat_pct (float): How much of the scan to keep as a percentage of 
                points.
            convert (bool): Whether or not to convert the scan's coordinates to
                Cartesian.
            save (bool): Whether or not to save the scan to the Raspberry Pi.
        Returns:
            filename (str | None): The name and path of the saved .txt file. For
                example, './path/to/cloud_19690420_080085.txt'. Saves to root by
                default. Returns None if not saved.
        """

        start_time_s = time.time()
        cloud: list[list[float]] = self.capture_cloud()
        set_pixel(LQ1_ADDR, PX_GREEN)

        if trim and nonfat_pct:
            cloud = self.trim_cloud(cloud, nonfat_pct)
        if convert:
            cloud = self.convert_cloud(cloud)
            set_pixel(LQ2_ADDR, PX_GREEN)
        if save:
            filename: str = self.save_cloud(cloud=cloud, filepath=filepath)
            set_pixel(LQ3_ADDR, PX_GREEN)
            duration_s: float = time.time() - start_time_s
            duration_s = round(duration_s, 2)

            print(f"[RUN] scan.py: Scan completed in {duration_s} seconds.")
            return filename
        
        duration_s: float = time.time() - start_time_s
        duration_s = round(duration_s, 2)

        print(f"[RUN] scan.py: Scan completed in {duration_s} seconds.")
        return None


def test_scan() -> None:
    test_scanner = Scanner()
    test_scanner.set_rings_per_cloud(num_rings=200)
    test_scanner.scan()
