# This program will remove empty trip folders from the 'trips' directory.
# An empty trip folder is defined as one that contains no files, or one that
# only contains a single json file.

import os
from os.path import join, isdir, isfile
from shutil import rmtree

def flush_null_trips(trips_folder: str) -> None:
    """
    Removes empty trip folders from the specified trips directory.
    
    Args:
        trips_folder (str): Path to the trips directory.
    """
    try:
        for trip in os.listdir(trips_folder):
            trip_path = join(trips_folder, trip)
            if isdir(trip_path):
                # Check if the directory is empty
                if not any(isfile(join(trip_path, f)) for f in os.listdir(trip_path)):
                    os.rmdir(trip_path)
                    print(f"Removed empty trip folder: {trip_path}")

                elif len(os.listdir(trip_path)) == 1:
                    only_file = os.listdir(trip_path)[0]
                    if only_file.endswith('.json'):
                        rmtree(trip_path)
                        print(f"Removed null trip folder (only JSON): {trip_path}")
    except Exception as e:
        print(f"Error while flushing null trips: {e}")

if __name__ == "__main__":
    trips_folder = join(os.path.dirname(__file__), 'stream', 'static', 'trips')
    flush_null_trips(trips_folder)