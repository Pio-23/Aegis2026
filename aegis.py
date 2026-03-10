# Main program for AEGIS rover software. 
# Starts UART communication and web viewer in separate processes.

from multiprocessing import Process
import atexit

from utils.led_utils import *
from rover import UART, camera
from stream import web_viewer as site

cam = camera.Camera()
#cam.my_start_recording()

@atexit.register
def exit_handler() -> None:
#    cam.my_stop_recording()
    print("[EXIT] aegis.py: Exiting program...")
    pulse_board(PX_RED, 5, 0.2)

uart_p = Process(target=UART.run_comms)
#site_p = Process(target=site.app.run, daemon=True)

print("[INI] aegis.py: Starting UART subprocess...")
uart_p.start()
#print("[INI] aegis.py: Starting website subprocess...")
#site_p.start()

print("[INI] aegis.py: Joining subprocesses...")
uart_p.join()
#site_p.join()
