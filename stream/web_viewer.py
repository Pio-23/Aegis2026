# Unified web viewer backend
# AEGIS Senior Design, Created on 6/9/25

from flask import Flask, render_template, jsonify, request
import os   # listdir(), endswith(), path.join(), path.isdir(), path.isfile()
from os.path import join, isdir, isfile

app = Flask(__name__) # Creates Flask app instance

# Directory to be monitored for new trip folders
# Sets absolute directory path because python is stupid
trips_folder = join(app.static_folder, 'trips')     # type: ignore

# ROUTES -----------------------------------------------------------------------

@app.route('/')
def render_home_page():
    return render_template('home.html')

@app.route('/telemetry')
def render_telemetry_page():
    return render_template('telemetry.html')

@app.route('/about')
def render_about_page():
    return render_template('about.html')

@app.route('/queryFilenames')
def get_all_filenames():
    """
    Retrieves all files belonging to a specific category (LiDAR txts, video 
    MP4s, or telemetry JSONs) and sends them to the frontend as a JSON object.
    Can also retrieve all trip folder names from trips folder.
    """
    trip = request.args.get('trip','')
    category = request.args.get('cat','')

    try:
        names = []
        if category == 'Trips':
            for f in os.listdir(trips_folder):
                if isdir(join(trips_folder, f)):
                    names.append(f)
        else:
            ext = {"Video":".mp4", "LiDAR":".txt", "Graph":".json"}[category]
            for f in os.listdir(join(trips_folder, trip)):
                if isfile(join(trips_folder, trip, f)) and f.endswith(ext):
                    names.append(f)

        return jsonify(names)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    os.makedirs(trips_folder, exist_ok=True)
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True)