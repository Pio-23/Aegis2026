# Pin Mapping Utilities
# Created 10/7/2025

# Motor, encoder, and sensor pins can be found in rover/rover_mega/config.h

# LIDAR PINS
LIDAR_PORT = '/dev/ttyAMA0' # UART port on GPIO 14 and 15
LIDAR_PWM = 22 # CURRENTLY UNROUTED ON MOTHERBOARD!

# NEMA17 STEPPER MOTOR PINS
DIR_PIN  = 26
STEP_PIN = 20
MS1_PIN  = 25 # From GPIO 13
MS2_PIN  = 24 # From GPIO 19
MS3_PIN  = 23 # From GPIO 6

# INTERBOARD UART
ARDUINO_PORT = '/dev/ttyAMA2' # UART port on GPIO 4 and 5

# STATUS BOARD LED PIN
import board
LED_CTL_PIN = board.D18 # Must be a hardware PWM pin!
