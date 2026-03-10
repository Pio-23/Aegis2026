'''
LiDAR module driver for AEGIS senior design. Version 2
Compatible with STL27L LiDAR sensor.
'''

import numpy    # abs()
import serial   # Serial()
import time     # sleep()
import gpiozero as gpz

from utils import serial_utils  # CRC_TABLE
from utils import file_utils    # get_timestamped_filename(), make_file(), write_pcd_header_to_file()
from utils import math_utils    # pol_to_cart_array()
from utils import pin_utils as pins    # LiDAR pins

class Lidar:
    '''
    The STL27L Planar LiDAR scanner class.

    This class provides methods that open and close the serial port attribute,
    capture point data from the serial port, validate the data using a CRC 
    checksum, and write captured points to specified files. It is used to take
    ring scans that are later merged into 3D point cloud scans.

    Attributes:
        name (str): The name of the LiDAR scanner ('STL27L').
        max_packets (int): The number of packets expected to be taken per ring 
            [245 = (921600b/s) / (10Hz) / (8b/B) / (47 B/packet)].
        packet_size (int): The number of bytes per packet in accordance with the
            STL27L communication protocol (47).
        start_byte (int):  The fixed start byte of each packet in accordance 
            with the STL27L communication protocol (0x54).
        hit_rate_threshold (float): The percentage of points per ring above 
            which a ring is considered acceptable.
        pwm_pin (OutputDevice): The PWM control pin, currently pulled low 
            (~10Hz). Note that this is currently on a digital (non-PWM) pin.
        serial (Serial): The pyserial connection between Raspberry Pi and LiDAR.
    '''

    def __init__(self) -> None:
        '''
        Initializes a Lidar object with application-specific parameters. Also
        creates (but does not open) a serial connection.
        '''
        
        self.name        = 'STL27L'
        self.max_packets = 245
        self.packet_size = 47
        self.start_byte  = 0x54
        self.hit_rate_threshold = 0.80

        self.pwm_pin = gpz.OutputDevice(pin=pins.LIDAR_PWM, initial_value=False)
        
        self.serial = serial.Serial()

    def open_serial(self) -> None:
        '''
        Opens a serial connection with parameters set in the class initializer.
        '''

        self.serial.port = pins.LIDAR_PORT	    # UART port on GPIO 14 and 15
        self.serial.baudrate = serial_utils.LIDAR_BAUDRATE  # STL27L Baudrate
        self.serial.bytesize = 8                # 8 bits per byte
        self.serial.parity = 'N'                # No parity bit
        self.serial.stopbits = 1                # One stop bit per byte
        self.serial.timeout = 0                 # Non-blocking mode

        self.serial.open()
        time.sleep(0.05)                    # Allow connection to form

    def close_serial(self) -> None:
        '''
        Closes the serial connection. If it is already closed, has no effect.
        '''

        self.serial.close()

    def capture_packets(self, packets: int, verbose: bool = False) -> list[bytes]:
        '''
        Obtains data packets from the LiDAR device over the serial connection.
        
        Args:  
            packets (int): The number of packets to retrieve.
            verbose (bool): Whether or not to print debug info. Defaults to
                False.
        Returns:
            data_arr (list[bytes]): A list containing the requested number of 
                packets.
        '''

        data_arr: list[bytes] = []

        for packet in range(0, packets):
            
            data: bytes = self.serial.read(self.packet_size)
            
            if verbose:
                print("\n#%3.3d (L=%2.2d): " %(packet, len(data)), end=' ')
                for byte in data: print("%2x" % byte, end=' ')
            
            # If the packet is not 47 bytes long, reject it
            if (len(data) != self.packet_size):
                if verbose: 
                    print("Incorrect Packet Size! N =", len(data), end='')
                packets += 1
                continue
            
            # If the packet does not begin with the start byte, reject it
            if (data[0] != self.start_byte):
                if verbose: 
                    print("Misaligned Packet!", end='')
                self.serial.read_until(bytes(self.start_byte))  # Realigns stream
                packets += 1
                continue
            
            # If the packet has an invalid CRC checksum, reject it
            if (data[self.packet_size-1] != self.validate_crc(data)):
                if verbose: 
                    print("Incorrect Checksum!", end='')
                self.serial.read_until(bytes(self.start_byte))  # Realigns stream
                packets += 1
                continue
            
            data_arr.append(data)

            time.sleep(0.0005)	# Allow time to scan, this cannot be zero!
            
        return data_arr 

    def process_packet(self, packet: bytes, verbose: bool = False, motor_angle: float | None = None) -> list[list[float]]:
        '''
        Extracts information from packet data in accordance with STL27L 
        communication protocol.

        Args:
            packet (bytes): A packet of points to be processed.
            verbose (bool): Whether or not to print debug info. Defaults to
                False.
            motor_angle (float | None): The current motor angle (or None for no 
                connected motor). Defaults to None.
        Returns:
            pts_arr (list[list[float]]): Array containing points. Points will 
                either have 3 dimensions (rho, phi, intensity) or 4 dimensions 
                (rho, phi, theta, intensity) depending on the presence of the 
                stepper motor.
        '''
        
        # data comes in as LSB then MSB
        speed: int        = packet[3]  * (2**8) + packet[2]		# deg / sec
        start_angle: int  = packet[5]  * (2**8) + packet[4]		# 0.01 deg
        end_angle: int    = packet[43] * (2**8) + packet[42]	# 0.01 deg
        timestamp_ms: int = packet[45] * (2**8) + packet[44]	# rolls over at 30000
        
        # If we've rolled over from 360 degrees to 0, add 1 revolution to correct delta
        if end_angle < start_angle:
            angle_delta: float = numpy.abs(end_angle - start_angle + 36000) / 11
        else:
            angle_delta: float = numpy.abs(end_angle - start_angle) / 11
        
        # Construct points array (12 points per packet)
        pts_arr: list[list[float]] = []
        for i in range(0,12):
            pt_dist: float    = (packet[7+3*i] * (2**8) + packet[6+3*i]) / 1000	# distance in mm, converted to meters
            pt_angle: float   = (start_angle + angle_delta*i) / 100		        # angle in 0.01 deg, converted to deg
            pt_intensity: float = packet[8+3*i] / 255					        # reflection intensity (normalized)
            
            # Enable 2D for tests without motor
            if motor_angle is not None: 
                pts_arr.append([pt_dist, pt_angle, motor_angle, pt_intensity])
            else:
                pts_arr.append([pt_dist, pt_angle, pt_intensity])
                
        if verbose:
            print("Packet Info: SB=%2.2x|VL=%2.2x|SP=%d|ST_A=%d|E_A=%d|TIME=%d"
                % (packet[0], packet[1], speed, start_angle, end_angle, timestamp_ms))
    
        return pts_arr

    def capture_ring(self, verbose: bool = False, motor_angle: float | None = None) -> list[list[float]]:
        '''
        Obtains a ring's worth of points from the LiDAR device.

        Args:
            verbose (bool): Whether or not to print debug info. Defaults to
                False.
            motor_angle (float | None): The current motor angle (or None for no 
                connected motor). Defaults to None.
        Returns:
            ring (list[list[float]]): Array containing points. Points will 
                either have 3 dimensions (rho, phi, intensity) or 4 dimensions 
                (rho, phi, theta, intensity) depending on the presence of the 
                stepper motor.
        '''

        while True:
            ring: list[list[float]] = []
            packets: list[bytes] = self.capture_packets(
                self.max_packets, verbose)
            for packet in packets:
                points: list[list[float]] = self.process_packet(
                    packet, verbose, motor_angle)
                ring.extend(points)

            if len(ring) >= self.max_packets * 12 * self.hit_rate_threshold:
                break

        if verbose: print(f"Recieved {len(ring)} points...")
        
        return ring

    def validate_crc(self, packet: bytes) -> int:
        '''
        Calculates the CRC8 checksum of a data packet in accordance with the 
        STL27L communications protocol.
        
        Args:
            packet (list[bytes]): The packet of data to be checked.
        Returns:
            crc (int): The checksum of the packet.
        '''

        crc = 0
        for i in range(0, self.packet_size - 1):
            crc = serial_utils.CRC_TABLE[(crc ^ packet[i]) & 0xff]
        return crc


def test_ring_capture(save: bool = False, verbose: bool = True) -> tuple[float, int]:
    '''
    Captures a planar ring of LiDAR data.

    Args:
        save (bool): Whether or not to save the ring scan to a file. Defaults to
            False.
        verbose (bool): Whether or not to print debug info. Defaults to True.
    Returns:
        out (tuple[float, int]): The duration of scan in seconds and number of 
            points captured.
    '''
    L1 = Lidar()
    if verbose: 
        print("[INI] lidar.py: Instantiated LiDAR object!")

    L1.open_serial()
    if verbose: 
        print("[INI] lidar.py: Opened serial connection!")

    start_time_s: float = time.time()

    polar_points: list[list[float]] = L1.capture_ring(verbose)

    end_time_s: float = time.time()
    duration_s: float = end_time_s - start_time_s

    cartesian_points: list[list[float]] = math_utils.pol_to_cart_array(polar_points)

    if verbose:
        print(f"[RUN] lidar.py: Point capture took {round(duration_s, 4)} seconds...") 
        print(f"[RUN] lidar.py: Count: {len(polar_points)}")
        print("[RUN] lidar.py: Points converted to Cartesian coordinates!")

    if save:
        test_file: str = file_utils.get_timestamped_filename(save_path='.', prefix='test', ext='.txt')
        file_utils.write_points_to_file(filename=test_file, points=cartesian_points)
        if verbose:
            print(f"[RUN] lidar.py: Points written to {test_file}!")

    return duration_s, len(cartesian_points)