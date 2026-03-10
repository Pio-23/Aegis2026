# LiDAR Testbench
# This file will run a configurable number of scans and aggregate the data captured.

import os

from lidar import scan

dir = "z_scan_test_dir"
os.makedirs(dir, exist_ok=True)

RINGS = 1600
ITERS = 5

UUT = scan.Scanner()
UUT.set_rings_per_cloud(RINGS)

for i in range(ITERS):
    print(f"SCAN {i+1}:")
    UUT.scan(filepath=dir)
