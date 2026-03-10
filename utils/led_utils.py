# Address Map Utilities
# Created 9/28/2025

'''
This file is used to address and operate the status LEDs on the rover.
It uses the RGB Smart NeoPixels from Adafruit: www.adafruit.com/product/1312

They are arranged in a 4x4 matrix on the status board.

Install the Adafruit CircuitPython NeoPixel libraries:
    pip3 install adafruit-circuitpython-neopixel

It should pull in the dependencies, which include:
    pip3 install adafruit-circuitpython
    pip3 install adafruit-circuitpython-pixelbuf

https://docs.circuitpython.org/projects/neopixel/en/latest/
    
Whoops. Needs PWM pin, and we didn't plan for that. Should use 
GPIO 10, 12, 18, or 21 for 800kHz PWM, can't bit bang that on linux :(
Maybe splice under boards? 

'''

import neopixel
from time import sleep
from time import sleep

from . import pin_utils as pins

PX_COUNT = 16
BRIGHTNESS = 0.2

#             RRGGBB
PX_OFF    = 0x000000
PX_WHITE  = 0xFFFFFF
PX_RED    = 0xFF0000
PX_GREEN  = 0x00FF00
PX_BLUE   = 0x0000FF
PX_PURPLE = 0xFF00FF
PX_YELLOW = 0xFFFF00
PX_TEAL   = 0x00FFFF

colors = [
    PX_WHITE, PX_RED, PX_GREEN, PX_BLUE, PX_PURPLE, PX_YELLOW, PX_TEAL
]

'''
15  14  13  12     LI1 USF  LF  RF
08  09  10  11     LI2 USR  LM  RM
07  06  05  04     LI3 USL  LR  RR
07  06  05  04     LI3 USL  LR  RR
00  01  02  03     RPM CAM ARD BAT
'''

LQ1_ADDR = 15
LQ2_ADDR = 8
LQ3_ADDR = 7
LQ3_ADDR = 7

USFT_ADDR = 14
USRR_ADDR = 9
USLI_ADDR = 6
USLI_ADDR = 6

LF_ADDR = 13
RF_ADDR = 12
LM_ADDR = 10
RM_ADDR = 11
LR_ADDR = 5
RR_ADDR = 4
LR_ADDR = 5
RR_ADDR = 4

RPM_ADDR = 0
CAM_ADDR = 1
ARD_ADDR = 2
BAT_ADDR = 3

pixels = None

def init_on_first_use() -> None:
    global pixels
    if pixels is None:
        pixels = neopixel.NeoPixel(
            pins.LED_CTL_PIN,
            PX_COUNT,
            brightness=BRIGHTNESS,
            auto_write=True     # Immediately update pixel state when written
        )

def set_pixel(addr : int, color) -> None:
    if addr < 0 or addr > PX_COUNT - 1:
        raise ValueError("Invalid pixel index!")
    
    init_on_first_use()
    pixels[addr] = color

def pulse_board(color, pulses, pulse_duration_s) -> None:
    steps = 100
    pixels.fill(color)
    for _ in range (0, pulses):
        for i in range(0, steps):
            if i < steps/2:
                pixels.brightness = i / steps/2 * BRIGHTNESS
            else:
                pixels.brightness = (steps - i) / steps/2 * BRIGHTNESS
            sleep(pulse_duration_s / steps)
    pixels.fill(PX_OFF)

def pulse_board(color, pulses, pulse_duration_s) -> None:
    steps = 100
    pixels.fill(color)
    for _ in range (0, pulses):
        for i in range(0, steps):
            if i < steps/2:
                pixels.brightness = i / steps/2 * BRIGHTNESS
            else:
                pixels.brightness = (steps - i) / steps/2 * BRIGHTNESS
            sleep(pulse_duration_s / steps)
    pixels.fill(PX_OFF)

def map_ultrasonic_to_pixel(addr, dist_cm):
    if dist_cm > 100:
        set_pixel(addr, PX_GREEN)
    elif dist_cm < 100 and dist_cm >= 30:
        set_pixel(addr, PX_YELLOW)
    elif dist_cm < 30 and dist_cm >= 0.01:
        set_pixel(addr, PX_RED)
    else:
        set_pixel(addr, PX_WHITE)

def map_rpm_to_pixel(addr, rpm):
    if rpm > 125:
        set_pixel(addr, PX_RED)
    elif rpm < 125 and rpm >= 50:
        set_pixel(addr, PX_YELLOW)
    elif rpm < 50 and rpm > 0.01:
        set_pixel(addr, PX_GREEN)
    else:
        set_pixel(addr, PX_WHITE)

def map_mot_current_to_pixel(addr, curr):
    if curr > 3:
        set_pixel(addr, PX_RED)
    elif curr < 3 and curr >= 1:
        set_pixel(addr, PX_YELLOW)
    elif curr < 1 and curr > 0.01:
        set_pixel(addr, PX_GREEN)
    else:
        set_pixel(addr, PX_WHITE)

def map_batt_to_pixel(addr, pct):
    if pct > 66:
        set_pixel(addr, PX_GREEN)
    elif pct < 66 and pct >= 33:
        set_pixel(addr, PX_YELLOW)
    elif pct < 33 and pct > 0.01:
        set_pixel(addr, PX_RED)
    else:
        set_pixel(addr, PX_WHITE)
