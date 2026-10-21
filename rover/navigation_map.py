import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


class NavigationMap:
    """
    Persistent local navigation map for AEGIS.

    World coordinate convention:

        +X = the rover's initial FRONT direction
        +Y = the rover's initial LEFT direction

    Internal heading convention:

        0 deg   = initial FRONT
        +90 deg = LEFT
        -90 deg = RIGHT

    This first version uses dead reckoning for movement.
    Later this can be corrected using LiDAR scan matching / SLAM.
    """

    # ----------------------------------------------------------
    # TEMPORARY MOVEMENT CALIBRATION
    # ----------------------------------------------------------

    # Dummy calibration requested for initial development:
    #
    # 3 seconds = 5 feet
    # 5 feet = 1.524 meters
    #
    # 1.524 / 3 = 0.508 m/s
    #
    # IMPORTANT:
    # This is intentionally temporary.
    MOVE_MPS_AT_050 = 0.396

    # ----------------------------------------------------------
    # OCCUPANCY MAP
    # ----------------------------------------------------------

    # Each map cell represents 10 cm x 10 cm.
    GRID_RESOLUTION_M = 0.10

    # Same approximate rover self-mask currently used
    # by the LiDAR summary.
    FRONT_LIMIT = 0.1524
    REAR_LIMIT = 0.6096
    LEFT_LIMIT = 0.4572
    RIGHT_LIMIT = 0.4572

    # Ignore floor returns.
    FLOOR_CUTOFF_Z = -0.27
    NAV_MAX_Z = 1.0

    def __init__(self):

        # ------------------------------------------------------
        # Rover pose in world coordinates
        # ------------------------------------------------------

        self.x_m = 0.0
        self.y_m = 0.0
        self.heading_deg = 0.0

        # ------------------------------------------------------
        # Path history
        # ------------------------------------------------------

        self.path = [
            {
                "x_m": 0.0,
                "y_m": 0.0,
                "heading_deg": 0.0,
                "event": "START"
            }
        ]

        # ------------------------------------------------------
        # Occupied map cells
        # ------------------------------------------------------

        # Each entry is:
        #
        #     (grid_x, grid_y)
        #
        # A set avoids storing millions of duplicate points.
        self.occupied_cells = set()

        # ------------------------------------------------------
        # LiDAR scan history
        # ------------------------------------------------------

        self.scan_records = []

        self.total_moves = 0
        self.total_turns = 0
        self.total_lidar_scans = 0


    # ==========================================================
    # ANGLE HELPERS
    # ==========================================================

    def normalize_heading(self, angle_deg):
        """
        Normalize heading into [-180, 180).
        """

        return (
            (angle_deg + 180.0) % 360.0
        ) - 180.0


    # ==========================================================
    # MOVEMENT UPDATE
    # ==========================================================

    def apply_move(self, duration_s, spd):
        """
        Update estimated rover X/Y position after a completed MOVE.

        Temporary model:

            distance = 0.508 m/s * movement duration

        Speed magnitude is intentionally ignored for now because
        the real drivetrain has not yet been calibrated.

        Positive spd = forward
        Negative spd = reverse
        """

        try:
            duration_s = float(duration_s)
            spd = float(spd)

        except (TypeError, ValueError):
            print(
                "[MAP] Invalid movement information. "
                "Pose not updated."
            )
            return

        if duration_s <= 0:
            return

        direction = 1.0 if spd >= 0 else -1.0


        speed_mps = (
            self.MOVE_MPS_AT_050
            * (abs(spd) / 0.50)
        )

        distance_m = (
            speed_mps
            * duration_s
            * direction
        )

        heading_rad = math.radians(
            self.heading_deg
        )

        delta_x = (
            distance_m
            * math.cos(heading_rad)
        )

        delta_y = (
            distance_m
            * math.sin(heading_rad)
        )

        self.x_m += delta_x
        self.y_m += delta_y

        self.total_moves += 1

        self.path.append({
            "x_m": round(self.x_m, 3),
            "y_m": round(self.y_m, 3),
            "heading_deg": round(
                self.heading_deg,
                1
            ),
            "event": "MOVE",
            "distance_m": round(
                distance_m,
                3
            ),
            "duration_s": round(
                duration_s,
                3
            )
        })

        print(
            "[MAP] MOVE "
            f"{distance_m:.2f} m | "
            f"pose=({self.x_m:.2f}, "
            f"{self.y_m:.2f}) | "
            f"heading={self.heading_deg:.1f} deg"
        )


    # ==========================================================
    # TURN UPDATE
    # ==========================================================

    def apply_turn(
        self,
        turn_dir,
        turn_degrees
    ):
        """
        Update estimated rover heading after a completed TURN.
        """

        try:
            turn_degrees = float(
                turn_degrees
            )

        except (TypeError, ValueError):
            print(
                "[MAP] Invalid turn angle. "
                "Heading not updated."
            )
            return

        turn_dir = str(
            turn_dir
        ).upper()

        if turn_dir == "LEFT":

            self.heading_deg +=  turn_degrees
            

        elif turn_dir == "RIGHT":

            self.heading_deg -=  turn_degrees
            

        else:
            print(
                f"[MAP] Unknown turn direction: "
                f"{turn_dir}"
            )
            return

        self.heading_deg = (
            self.normalize_heading(
                self.heading_deg
            )
        )

        self.total_turns += 1

        self.path.append({
            "x_m": round(self.x_m, 3),
            "y_m": round(self.y_m, 3),
            "heading_deg": round(
                self.heading_deg,
                1
            ),
            "event": "TURN",
            "turn_dir": turn_dir,
            "turn_degrees": round(
                turn_degrees,
                1
            )
        })

        print(
            "[MAP] TURN "
            f"{turn_dir} "
            f"{turn_degrees:.1f} deg | "
            f"heading={self.heading_deg:.1f} deg"
        )


    # ==========================================================
    # LIDAR -> WORLD MAP
    # ==========================================================

    def add_lidar_scan(
        self,
        filename,
        summary=None,
        visualization_files=None
    ):
        """
        Add a Cartesian LiDAR scan to the persistent world map.

        The LiDAR .txt file contains rover-relative XYZ points.

        These points are transformed into world coordinates using
        the rover's current estimated X, Y, and heading.

        Instead of storing ~1 million raw points forever, the map
        stores occupied 10 cm grid cells.
        """

        if not filename:
            return

        if not os.path.exists(filename):

            print(
                f"[MAP] LiDAR file not found: "
                f"{filename}"
            )

            return

        print(
            "[MAP] Adding LiDAR scan to "
            "persistent world map..."
        )

        start_time = time.time()

        heading_rad = math.radians(
            self.heading_deg
        )

        cos_h = math.cos(
            heading_rad
        )

        sin_h = math.sin(
            heading_rad
        )

        points_read = 0
        points_used = 0

        cells_before = len(
            self.occupied_cells
        )

        with open(filename, "r") as f:

            for line in f:

                values = line.strip().split()

                if len(values) < 3:
                    continue

                try:

                    local_x = float(
                        values[0]
                    )

                    local_y = float(
                        values[1]
                    )

                    local_z = float(
                        values[2]
                    )

                except ValueError:
                    continue

                points_read += 1

                if not (
                    math.isfinite(local_x)
                    and math.isfinite(local_y)
                    and math.isfinite(local_z)
                ):
                    continue

                # ----------------------------------------------
                # Remove rover body
                # ----------------------------------------------

                if (
                    -self.REAR_LIMIT
                    <= local_x
                    <= self.FRONT_LIMIT

                    and

                    -self.RIGHT_LIMIT
                    <= local_y
                    <= self.LEFT_LIMIT
                ):
                    continue

                # ----------------------------------------------
                # Remove floor
                # ----------------------------------------------

                if (
                    local_z
                    <= self.FLOOR_CUTOFF_Z
                    or
                    local_z
                    >= self.NAV_MAX_Z
                ):
                    continue

                # ----------------------------------------------
                # Rover coordinates -> world coordinates
                # ----------------------------------------------

                world_x = (
                    self.x_m
                    + local_x * cos_h
                    - local_y * sin_h
                )

                world_y = (
                    self.y_m
                    + local_x * sin_h
                    + local_y * cos_h
                )

                # ----------------------------------------------
                # Convert world point into map grid cell
                # ----------------------------------------------

                grid_x = round(
                    world_x
                    / self.GRID_RESOLUTION_M
                )

                grid_y = round(
                    world_y
                    / self.GRID_RESOLUTION_M
                )

                self.occupied_cells.add(
                    (
                        grid_x,
                        grid_y
                    )
                )

                points_used += 1


        cells_after = len(
            self.occupied_cells
        )

        new_cells = (
            cells_after
            - cells_before
        )

        self.total_lidar_scans += 1

        scan_record = {
            "scan_number":
                self.total_lidar_scans,

            "x_m":
                round(self.x_m, 3),

            "y_m":
                round(self.y_m, 3),

            "heading_deg":
                round(
                    self.heading_deg,
                    1
                ),

            "points_read":
                points_read,

            "points_used":
                points_used,

            "new_occupied_cells":
                new_cells,

            "summary":
                summary,

            "visualizations":
                visualization_files or {}
        }

        self.scan_records.append(
            scan_record
        )

        elapsed = (
            time.time()
            - start_time
        )

        print(
            "[MAP] LiDAR scan added | "
            f"read={points_read:,} | "
            f"used={points_used:,} | "
            f"new cells={new_cells:,} | "
            f"total cells={cells_after:,} | "
            f"time={elapsed:.2f}s"
        )


    # ==========================================================
    # MAP BOUNDS
    # ==========================================================

    def get_bounds(self):

        if not self.occupied_cells:

            return None

        xs = [
            cell[0]
            for cell in self.occupied_cells
        ]

        ys = [
            cell[1]
            for cell in self.occupied_cells
        ]

        min_x = (
            min(xs)
            * self.GRID_RESOLUTION_M
        )

        max_x = (
            max(xs)
            * self.GRID_RESOLUTION_M
        )

        min_y = (
            min(ys)
            * self.GRID_RESOLUTION_M
        )

        max_y = (
            max(ys)
            * self.GRID_RESOLUTION_M
        )

        return {
            "min_x_m": round(
                min_x,
                2
            ),
            "max_x_m": round(
                max_x,
                2
            ),
            "min_y_m": round(
                min_y,
                2
            ),
            "max_y_m": round(
                max_y,
                2
            )
        }


    # ==========================================================
    # COMPACT AI CONTEXT
    # ==========================================================

    def get_ai_context(self):
        """
        Return a compact representation for Luna.

        Luna does NOT receive the entire raw point map.
        """

        recent_path = (
            self.path[-20:]
        )

        recent_scans = []

        for scan_record in self.scan_records[-5:]:

            recent_scans.append({
                "scan_number":
                    scan_record[
                        "scan_number"
                    ],

                "x_m":
                    scan_record[
                        "x_m"
                    ],

                "y_m":
                    scan_record[
                        "y_m"
                    ],

                "heading_deg":
                    scan_record[
                        "heading_deg"
                    ],

                "summary":
                    scan_record[
                        "summary"
                    ]
            })

        return {
            "pose": {
                "x_m":
                    round(
                        self.x_m,
                        2
                    ),

                "y_m":
                    round(
                        self.y_m,
                        2
                    ),

                "heading_deg":
                    round(
                        self.heading_deg,
                        1
                    )
            },

            "map": {
                "grid_resolution_m":
                    self.GRID_RESOLUTION_M,

                "occupied_cells":
                    len(
                        self.occupied_cells
                    ),

                "bounds":
                    self.get_bounds(),

                "lidar_scans":
                    self.total_lidar_scans
            },

            "movement": {
                "total_moves":
                    self.total_moves,

                "total_turns":
                    self.total_turns,

                "recent_path":
                    recent_path
            },

            "recent_lidar_scans":
                recent_scans,

            "temporary_calibration": {
                "move_mps":
                    round(
                        self.MOVE_MPS_AT_050,
                        3
                    ),

                "note":
                    (
                        "Temporary dead-reckoning "
                        "calibration. Replace after "
                        "physical rover testing."
                    )
            }
        }

    def render_map_png(self, filename):
        """
        Render the persistent AEGIS navigation map.

        Shows:
            - all accumulated occupied LiDAR cells
            - rover travel path
            - current rover position
            - current rover heading

        World coordinates:
            +X = initial rover FRONT
            +Y = initial rover LEFT
        """

        if not filename:
            return None

        output_dir = os.path.dirname(filename)

        if output_dir:
            os.makedirs(
                output_dir,
                exist_ok=True
            )

        start_time = time.time()

        fig, ax = plt.subplots(
            figsize=(10, 10)
        )

        # ==========================================================
        # ACCUMULATED LIDAR MAP
        # ==========================================================

        if self.occupied_cells:

            map_x = []
            map_y = []

            for grid_x, grid_y in self.occupied_cells:

                world_x = (
                    grid_x
                    * self.GRID_RESOLUTION_M
                )

                world_y = (
                    grid_y
                    * self.GRID_RESOLUTION_M
                )

                # Plot:
                # horizontal = world Y
                # vertical   = world X

                map_x.append(
                    world_y
                )

                map_y.append(
                    world_x
                )

            ax.scatter(
                map_x,
                map_y,
                s=2,
                alpha=0.6,
                label="Mapped LiDAR geometry"
            )

            ax.invert_xaxis()

        # ==========================================================
        # ROVER PATH
        # ==========================================================

        if self.path:

            path_horizontal = [
                point["y_m"]
                for point in self.path
            ]

            path_vertical = [
                point["x_m"]
                for point in self.path
            ]

            ax.plot(
                path_horizontal,
                path_vertical,
                marker="o",
                markersize=3,
                linewidth=1.5,
                label="Rover path"
            )


        # ==========================================================
        # START POSITION
        # ==========================================================

        ax.scatter(
            [0],
            [0],
            s=100,
            marker="s",
            label="Start"
        )


        # ==========================================================
        # CURRENT ROVER POSITION
        # ==========================================================

        ax.scatter(
            [self.y_m],
            [self.x_m],
            s=140,
            marker="^",
            label="Current rover position"
        )


        # ==========================================================
        # CURRENT HEADING
        # ==========================================================

        heading_rad = math.radians(
            self.heading_deg
        )

        heading_length = 0.5

        heading_dx = (
            math.cos(heading_rad)
            * heading_length
        )

        heading_dy = (
            math.sin(heading_rad)
            * heading_length
        )

        ax.arrow(
            self.y_m,
            self.x_m,

            heading_dy,
            heading_dx,

            width=0.02,
            length_includes_head=True
        )


        # ==========================================================
        # MAP INFORMATION
        # ==========================================================

        ax.set_xlabel(
            "World Y (m)   +Y = initial LEFT"
        )

        ax.set_ylabel(
            "World X (m)   +X = initial FRONT"
        )

        ax.set_title(
            "AEGIS Persistent Navigation Map\n"
            f"Pose: X={self.x_m:.2f} m, "
            f"Y={self.y_m:.2f} m, "
            f"Heading={self.heading_deg:.1f}° | "
            f"Scans={self.total_lidar_scans}"
        )

        ax.set_aspect(
            "equal",
            adjustable="box"
        )

        ax.grid(True)
        ax.legend()

        plt.tight_layout()

        plt.savefig(
            filename,
            dpi=150
        )

        plt.close(fig)

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"[MAP] Persistent map PNG saved: "
            f"{filename} "
            f"({elapsed:.2f}s)"
        )

        return filename

        # ==========================================================
        # SAVE MAP STATE
        # ==========================================================

    def save_state(self, filename):
        """
        Save compact map state for debugging / recovery.
        """

        data = {
            "map_context":
                self.get_ai_context(),

            "occupied_cells": [
                [
                    cell[0],
                    cell[1]
                ]
                for cell
                in self.occupied_cells
            ]
        }

        with open(
            filename,
            "w"
        ) as f:

            json.dump(
                data,
                f,
                indent=2
            )

        print(
            f"[MAP] Navigation map saved: "
            f"{filename}"
        )
