# XBOX Controller Input Capture
# AEGIS Senior Design, Created 6/13/2025
# Second try using xow --> evdev after an update killed xone --> inputs

# Uses inbuilt joystick api:
# https://www.kernel.org/doc/Documentation/input/joystick-api.txt
#
# And evdev library:
# https://python-evdev.readthedocs.io
#
# See EOF for specific controller characteristics

import asyncio
from evdev import InputDevice, list_devices
from time import sleep

USE_XBOX = False
USE_BITDO = True

XBOX_DEVICE_NAME = 'Xbox One Wireless Controller'
BITDO_DEVICE_NAME = '8BitDo Ultimate 2C Wireless Controller'

STICK_NORMALIZER_Y: float = -1 / 32768
STICK_NORMALIZER_X: float =  1 / 32768
TRIG_NORMALIZER: float    =  1 / 1023

# Believe it or not, same for XBOX and 8BitDo. Python is occasionally not awful
INPUT_CODES: dict[int, str] = {
    315: 'BTN_START', 314: 'BTN_SELECT',
    304: 'BTN_A',  305: 'BTN_B', 307: 'BTN_X', 308: 'BTN_Y',
    310: 'BTN_TL', 311: 'BTN_TR', 
    2: 'BTN_Z',    3: 'ABS_RX',  4: 'ABS_RY',  5: 'BTN_RZ'
}

# To get controller state from another file, access 'input_states' directly.
input_states: dict[str, int|float] = { 
    'BTN_START': 0, 'BTN_SELECT': 0, 
    'BTN_A': 0, 'BTN_B': 0, 'BTN_X': 0, 'BTN_Y': 0,
    'BTN_TL': 0.0, 'BTN_TR': 0.0,
    'BTN_Z': 0.0, 'ABS_RX': 0.0, 'ABS_RY': 0.0, 'BTN_RZ': 0.0
}

def get_event_path(dev_name: str) -> str:
    devices = [InputDevice(path) for path in list_devices()]
    dev_event_path = None
    for device in devices:
        #print(' ', device.path, device.name, device.phys)
        if device.name == dev_name:
            dev_event_path = device.path

    if dev_event_path is None:
        raise OSError('[ERR] Event object path not found!')
    
    return dev_event_path

def normalize_input(dev_name: str, input: str, value: int) -> float:

    BITDO_MULT: int = 4 if dev_name == BITDO_DEVICE_NAME else 1

    if input in ['ABS_X', 'ABS_RX']:
        return round(value * STICK_NORMALIZER_X, 3)
    if input in ['ABS_Y', 'ABS_RY']:
        return round(value * STICK_NORMALIZER_Y, 3)
    if input in ['BTN_Z', 'BTN_RZ']:
        return round(value * TRIG_NORMALIZER*BITDO_MULT, 3)
    return value

# Asynchronous approach
async def listener(dev, print_updates) -> None:

    async for event in dev.async_read_loop(): 
        if event.code in INPUT_CODES:
            event_name: str = INPUT_CODES[event.code]
            input_states[event_name] = normalize_input(dev.name, event_name, event.value)

            if print_updates: 
                print(input_states)
            
        #print(dev.path, evdev.categorize(event), sep=': ') # Debug print

def listen(print_updates=0) -> None:
    '''
    Attempts to connect to a controller infinitely. Upon connection, runs the asynchronous
    listener function, which updates the controller input_state dict. Upon disconnection,
    an OSError is caught and the loop waits 5 seconds before trying to connect again.
    '''
    first_error = False
    while True:
        try:
            if USE_XBOX:
                device_name = XBOX_DEVICE_NAME
            elif USE_BITDO:
                device_name = BITDO_DEVICE_NAME
            else:
                raise 

            event_path = get_event_path(dev_name=device_name)
            device = InputDevice(event_path)

            print(f"[RUN] controller.py: {device_name} connected!")
            asyncio.run(listener(device, print_updates))
            first_error = False  # Reset error flag on successful connection
        
        except OSError:
            if first_error == False:
                print(f"[ERR] controller.py: Controller not connected, searching every second...")
                first_error = True
        
        finally:
            sleep(1)


if __name__ == '__main__':
    listen(print_updates=1)


"""
device.capabilities(verbose=True)   # Returns the following (XBOX Controller):
{
    ('EV_SYN', 0): [
        ('SYN_REPORT', 0), 
        ('SYN_CONFIG', 1), 
        ('SYN_DROPPED', 3), 
        ('?', 21)], 
    ('EV_KEY', 1): [
        (('BTN_A', 'BTN_GAMEPAD', 'BTN_SOUTH'), 304),
        (('BTN_B', 'BTN_EAST'), 305),
        (('BTN_NORTH', 'BTN_X'), 307), 
        (('BTN_WEST', 'BTN_Y'), 308), 
        ('BTN_TL', 310), 
        ('BTN_TR', 311), 
        ('BTN_SELECT', 314), 
        ('BTN_START', 315), 
        ('BTN_MODE', 316), 
        ('BTN_THUMBL', 317), 
        ('BTN_THUMBR', 318)], 
    ('EV_ABS', 3): [
        (('ABS_X', 0),      AbsInfo(value=-2764,    min=-32768, max=32767,  fuzz=255,   flat=4095,  resolution=0)), 
        (('ABS_Y', 1),      AbsInfo(value=436,      min=-32768, max=32767,  fuzz=255,   flat=4095,  resolution=0)), 
        (('ABS_Z', 2),      AbsInfo(value=0,        min=0,      max=1023,   fuzz=3,     flat=63,    resolution=0)), 
        (('ABS_RX', 3),     AbsInfo(value=-591,     min=-32768, max=32767,  fuzz=255,   flat=4095,  resolution=0)), 
        (('ABS_RY', 4),     AbsInfo(value=-662,     min=-32768, max=32767,  fuzz=255,   flat=4095,  resolution=0)), 
        (('ABS_RZ', 5),     AbsInfo(value=0,        min=0,      max=1023,   fuzz=3,     flat=63,    resolution=0)), 
        (('ABS_HAT0X', 16), AbsInfo(value=0,        min=-1,     max=1,      fuzz=0,     flat=0,     resolution=0)),
        (('ABS_HAT0Y', 17), AbsInfo(value=0,        min=-1,     max=1,      fuzz=0,     flat=0,     resolution=0))], 
    ('EV_FF', 21): [
        (('FF_EFFECT_MIN', 'FF_RUMBLE'), 80)]
}

device.capabilities(verbose=True)   # Returns the following (8BitDo Controller):
{
    ('EV_SYN', 0): [
        ('SYN_REPORT', 0), 
        ('SYN_CONFIG', 1), 
        ('SYN_DROPPED', 3), 
        ('?', 21)
    ], 
    ('EV_KEY', 1): [
        (['BTN_A', 'BTN_GAMEPAD', 'BTN_SOUTH'], 304),
        (['BTN_B', 'BTN_EAST'], 305),
        (['BTN_NORTH', 'BTN_X'], 307),
        (['BTN_WEST', 'BTN_Y'], 308),
        ('BTN_TL', 310),
        ('BTN_TR', 311),
        ('BTN_SELECT', 314),
        ('BTN_START', 315),
        ('BTN_MODE', 316),
        ('BTN_THUMBL', 317),
        ('BTN_THUMBR', 318)
    ],
    ('EV_ABS', 3): [
        (('ABS_X', 0), AbsInfo(value=0, min=-32768, max=32767, fuzz=16, flat=128, resolution=0)),
        (('ABS_Y', 1), AbsInfo(value=-1, min=-32768, max=32767, fuzz=16, flat=128, resolution=0)),
        (('ABS_Z', 2), AbsInfo(value=0, min=0, max=255, fuzz=0, flat=0, resolution=0)),
        (('ABS_RX', 3), AbsInfo(value=0, min=-32768, max=32767, fuzz=16, flat=128, resolution=0)),
        (('ABS_RY', 4), AbsInfo(value=-1, min=-32768, max=32767, fuzz=16, flat=128, resolution=0)),
        (('ABS_RZ', 5), AbsInfo(value=0, min=0, max=255, fuzz=0, flat=0, resolution=0)),
        (('ABS_HAT0X', 16), AbsInfo(value=0, min=-1, max=1, fuzz=0, flat=0, resolution=0)),
        (('ABS_HAT0Y', 17), AbsInfo(value=0, min=-1, max=1, fuzz=0, flat=0, resolution=0))
    ],
    ('EV_FF', 21): [
        (['FF_EFFECT_MIN', 'FF_RUMBLE'], 80),
        ('FF_PERIODIC', 81),
        (['FF_SQUARE', 'FF_WAVEFORM_MIN'], 88),
        ('FF_TRIANGLE', 89), ('FF_SINE', 90),
        (['FF_GAIN', 'FF_MAX_EFFECTS'], 96)
    ]
}
"""