// Trips Page Tools
// Created: 7/30/2025

const tripsFolder = '/static/trips';
const maxCloudSize = 250000;
let tripName, tripTelemetry, scanNames, videoNames;
const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", 
                "Oct", "Nov", "Dec"];

// Plotly labels for each trace
const traceLabels = {
    ".rpi.uptime_s":                    "Raspberry Pi Uptime [s]",
    ".rpi.cpu_util_pct":                "Raspberry Pi CPU Utilization [%]",
    ".rpi.mem_util_pct":                "Raspberry Pi Memory Utilization [%]",
    ".rpi.storage_avail_gb":            "Raspberry Pi Available Storage [GB]",
    ".rpi.temp_c":                      "Raspberry Pi Temperature [C]",
    ".rpi.vdd_core_a":                  "Raspberry Pi Core Current [A]",
    ".rpi.vdd_core_v":                  "Raspberry Pi Core Voltage [V]",

    ".arduino.uptime_s":                "Arduino Uptime [s]",

    ".lidar.scanning":                  "LiDAR Scanning",
    ".lidar.scan_pct":                  "LiDAR Scan Completion [%]",
    ".lidar.trimming":                  "LiDAR Trimming Scan",
    ".lidar.converting":                "LiDAR Converting Scan",
    ".lidar.saving":                    "LiDAR Saving File",
    ".lidar.motor_pos_deg":             "LiDAR Motor Position [deg]",

    ".camera.connected":                "Camera Connected",
    ".camera.recording":                "Camera Recording",

    ".motors.front_left.voltage_v":     "Front Left Motor Voltage [V]",
    ".motors.front_left.current_a":     "Front Left Motor Current [A]",
    ".motors.front_left.rpm":           "Front Left Motor Speed [RPM]",
    ".motors.mid_left.voltage_v":       "Mid Left Motor Voltage [V]",
    ".motors.mid_left.current_a":       "Mid Left Motor Current [A]",
    ".motors.mid_left.rpm":             "Mid Left Motor Speed [RPM]",
    ".motors.rear_left.voltage_v":      "Rear Left Motor Voltage [V]",
    ".motors.rear_left.current_a":      "Rear Left Motor Current [A]",
    ".motors.rear_left.rpm":            "Rear Left Motor Speed [RPM]",
    ".motors.front_right.voltage_v":    "Front Right Motor Voltage [V]",
    ".motors.front_right.current_a":    "Front Right Motor Current [A]",
    ".motors.front_right.rpm":          "Front Right Motor Speed [RPM]",
    ".motors.mid_right.voltage_v":      "Mid Right Motor Voltage [V]",
    ".motors.mid_right.current_a":      "Mid Right Motor Current [A]",
    ".motors.mid_right.rpm":            "Mid Right Motor Speed [RPM]",
    ".motors.rear_right.voltage_v":     "Rear Right Motor Voltage [V]",
    ".motors.rear_right.current_a":     "Rear Right Motor Current [A]",
    ".motors.rear_right.rpm":           "Rear Right Motor Speed [RPM]",

    ".ultrasonics.lidar_cm":            "LiDAR Ultrasonic Distance [cm]",
    ".ultrasonics.left_cm":             "Left Ultrasonic Distance [cm]",
    ".ultrasonics.center_cm":           "Center Ultrasonic Distance [cm]",
    ".ultrasonics.right_cm":            "Right Ultrasonic Distance [cm]",
    ".ultrasonics.rear_cm":             "Rear Ultrasonic Distance [cm]",

    "imu.roll_deg":                     "IMU Roll [deg]",
    "imu.pitch_deg":                    "IMU Pitch [deg]",
    "imu.yaw_deg":                      "IMU Yaw [deg]",
    "imu.accel_x_mps2":                 "IMU X Acceleration [m/s^2]",
    "imu.accel_y_mps2":                 "IMU Y Acceleration [m/s^2]",
    "imu.accel_z_mps2":                 "IMU Z Acceleration [m/s^2]",

    ".ugv.battery.capacity_pct":        "UGV Battery Remaining [%]",
    ".ugv.battery.voltage_v":           "UGV Battery Voltage [V]",
    ".ugv.battery.current_a":           "UGV Battery Current [A]",
    ".ugv.headlights":                  "Headlights",
    ".ugv.ambient_temp_c":              "Ambient Temperature [C]",
    ".ugv.relative_hum_pct":            "Earth Relative Humidity [%]",
    ".ugv.ambient_light_l":             "Ambient Visible Light [lm]",
    ".ugv.ambient_infrared_l":          "Ambient Infrared Light [lm]"
};

// Map of telemetry plots and all traces that they will show. To make a new 
// option, add a title with an array containing all traces you want to plot.
const telPlotsMap = {
    "Raspberry Pi Util + Temp": [
        //".rpi.uptime_s",
        ".rpi.cpu_util_pct",
        ".rpi.mem_util_pct",
        ".rpi.storage_avail_gb",
        ".rpi.temp_c",
    ],
    "Raspberry Pi Voltage + Current": [
        ".rpi.vdd_core_a",
        ".rpi.vdd_core_v",
    ],
    "Ultrasonic Sensor Distances": [
        ".ultrasonics.lidar_cm",
        ".ultrasonics.left_cm",
        ".ultrasonics.center_cm",
        ".ultrasonics.right_cm",
        ".ultrasonics.rear_cm",
    ],
    "LiDAR Scan Information": [
        ".lidar.scanning",
        ".lidar.scan_pct",
        ".lidar.trimming",
        ".lidar.converting",
        ".lidar.saving",
        ".lidar.motor_pos_deg",
    ],
    "Intertial Measurements": [
        "imu.roll_deg",   
        "imu.pitch_deg",
        "imu.yaw_deg",
        "imu.accel_x_mps2",
        "imu.accel_y_mps2",
        "imu.accel_z_mps2",
    ],
    "Camera Information": [
        ".camera.connected",
        ".camera.recording",
    ],
    "Motor Voltages": [
        ".motors.front_left.voltage_v",
        ".motors.mid_left.voltage_v",
        ".motors.rear_left.voltage_v",
        ".motors.front_right.voltage_v",
        ".motors.mid_right.voltage_v",
        ".motors.rear_right.voltage_v",
    ],
    "Motor Currents": [
        ".motors.front_left.current_a",
        ".motors.mid_left.current_a",
        ".motors.rear_left.current_a",
        ".motors.front_right.current_a",
        ".motors.mid_right.current_a",
        ".motors.rear_right.current_a",
    ],
    "Motor Speeds": [
        ".motors.front_left.rpm",
        ".motors.mid_left.rpm",
        ".motors.rear_left.rpm",
        ".motors.front_right.rpm",
        ".motors.mid_right.rpm",
        ".motors.rear_right.rpm",
    ],
    "Battery Information": [
        ".ugv.battery.capacity_pct",
        ".ugv.battery.voltage_v",
        ".ugv.battery.current_a",
    ],
    "Ambient Temp + ERH": [
        ".ugv.ambient_temp_c",
        ".ugv.relative_hum_pct",
    ],
    "Ambient Light + Headlights": [
        ".ugv.headlights",
        ".ugv.ambient_light_l",
        ".ugv.ambient_infrared_l",
    ]
}

/**
 * Gets file or folder information from the Flask server on the Raspberry Pi. 
 * 
 * @description If the category is 'Trips', gets the trip folders. If the 
 * category is 'Graph', gets the telemetry object from JSON. If the category is 
 * 'Video' or 'LiDAR', gets all filenames of found videos or scans respectively.
 * 
 * @param {string} tripFolder - The folder of the currently selected trip.
 * @param {string} category - Menu type ("Trips", "Video", "LiDAR", or "Graph").
 * @returns Either telemetry object, filenames string array, or null on error.
 */
async function queryFilenames(tripFolder, category) {
    try {   // Try to get object containing files in folder from Flask route
        const fileResponse = await fetch( '/queryFilenames?' + 
            `trip=${encodeURIComponent(tripFolder)}` +
            `&cat=${encodeURIComponent(category)}`
        );
        if (!fileResponse.ok) {console.warn("Couldn't fetch files!"); return;}
        const filenames = await fileResponse.json();

        // Sort by length, then alphabetically
        filenames.sort((a, b) => {  
            if (a.length !== b.length) {
                return a.length - b.length;
            } else return a.localeCompare(b);
        });

        // If you're querying for telemetry, just return the telemetry directly
        if (category == "Graph") {
            const path = `${tripsFolder}/${tripFolder}/${filenames[0]}`;
            const telResponse = await fetch(path);
            if (!telResponse.ok) {console.warn('No telemetry found!'); return;}
            let telStr = await telResponse.text();

            try {   // Try to parse JSON as is
                return JSON.parse(telStr);
            } catch {
                try {   // Try to clean malformed JSON
                    // Usually due to file write interruption on R.Pi shutdown
                    // Should eventually be cleaned at write time, but trickier
                    while (telStr.length > 0) {
                        let idx = telStr.lastIndexOf("}") + 1;  // where to trim to
                        telStr = telStr.slice(0, idx) + "]}";   // trim and cap
                        try {
                            telData = JSON.parse(telStr);
                        } catch {
                            telStr = telStr.slice(0, -3);
                        }
                        return telData;
                    }
                } catch { throw new Error("Could not repair malformed JSON!"); }
            }
        } 

        return filenames;

    } catch (error) { console.warn('Data not fetched!' , error); return; }
}

/**
 * Populates a selector menu with options depending on menu category.
 * 
 * @param {string} category - Menu type ("Trips", "Video", "LiDAR", or "Graph").
 * @param {string} selectorId - ID of selector menu to populate.
 * @param {string[]} options - List of options to provide the user.
 * @returns {void}
 */
function populateSelector(category, selectorId, options) {
    const selector = document.getElementById(selectorId);
    selector.replaceChildren();     // Flush selector first

    switch (category) {
    case "Trips":
        
        // Populates trip selector with found trips
        if (Array.isArray(options) && options.length > 0) {
            options.forEach(tripName => {
                // If trip name follows "YYYYMMDD_HHMMSS", clean it up
                if (/\d{8}_\d{6}/.test(tripName)) {
                    const y  = tripName.substring(0, 4);
                    const mo = months[Number(tripName.substring(4,6))-1];
                    const d  = tripName.substring(6,8);
                    const h  = tripName.substring(9,11);
                    const mi = tripName.substring(11,13);
                    const s  = tripName.substring(13,15);
                    const tripLabel = (mo+" "+d+", "+y+" "+h+":"+mi+":"+s);
                    selector.appendChild(new Option(tripLabel, tripName));
                // Otherwise just use it as is
                } else selector.appendChild(new Option(tripName));
            });
        } else selector.appendChild(new Option("Error Loading Trips"));
        break;
    case "Video":
        if (Array.isArray(options) && options.length > 0) {
            options.forEach(label => {
                selector.appendChild(new Option(label));
            });
        } else selector.appendChild(new Option("No Videos to Display"));
        break;
    case "LiDAR":
        if (Array.isArray(options) && options.length > 0) {
            options.forEach(label => {
                selector.appendChild(new Option(label));
            });
        } else selector.appendChild(new Option("No Scans to Display"));
        break;
    case "Graph":
        // Populates graph selector with premade plot options from telPlotsMap
        if (tripTelemetry && Object.keys(tripTelemetry).length > 0) {
            Object.keys(telPlotsMap).forEach(label => {
                selector.appendChild(new Option(label));
            });
        } else selector.appendChild(new Option("No Telemetry to Display"));
        break;
    }
}

/**
 * Inserts a toggle button with listener to make a plot fullscreen when clicked.
 * 
 * @param {string} plotId - ID of the plot div that will recieve the button.
 * @returns {void}
 */
function addFullscreenButton(plotId) {
    const btn = document.createElement("button");
    btn.style.cssText = "position:absolute; left:0; top:10px; margin:5px;";
    const icon = document.createElement("i");
    icon.className = "fa-regular fa-square-plus fa-2xl";
    icon.style.filter = "invert(100%)";
    btn.appendChild(icon);

    const plot = document.getElementById(plotId);
    plot.append(btn);
    btn.addEventListener("click", () => {
        plot.classList.toggle('fullscreen_div');
        Plotly.relayout(plot, {autosize: true});
    });
}

/**
 * Creates and inserts a video into a plot.
 * 
 * @param {string} plotId - ID of the div that will display the video.
 * @param {string} videoName - File name of the video to display.
 * @returns {void}
 */
async function makeVideoPlot(plotId, videoName) {
    if (!videoNames.includes(videoName)) return;

    const video = Object.assign(document.createElement("video"), {
        src: `${tripsFolder}/${tripName}/${videoName}`,
        controls: true, muted: true, autoplay: true,
        style: {
            width: "100%", height: "100%",
            objectFit: "contain", background: "black"
        }
    });

    document.getElementById(plotId).append(video);
}

/**
 * Creates and inserts a LiDAR scan into a plot.
 * 
 * @description Fetches, unzips, downsamples, and builds a cloud, then 
 * configures visualizer parameters and generates a Plotly plot. Additionally 
 * displays the time it took to do this in the web console in milliseconds.
 * 
 * @param {string} plotId - ID of the div that will display the scan.
 * @param {string} scanName - File name of the scan to display.
 * @returns {void}
 */
async function makeScanPlot(plotId, scanName) {
    const start = performance.now();

    try {   // Try to get scan from Flask backend
        const result = await fetch(`${tripsFolder}/${tripName}/${scanName}`);
        if (!result.ok) throw new Error(`Error from server: ${result.status}`);
        var scanData = await result.text();
    } catch (err) { console.error("Fetch error: ", err); return; }
    let cloud = scanData.split('\n');

    // Downsample large clouds to some target size (in number of points)
    if (cloud.length > maxCloudSize) {
        cloud = cloud.filter(() => Math.random() < maxCloudSize/cloud.length);
    }

    const xVals = [], yVals = [], zVals = [], iVals = [];
    for (let i = 0; i < cloud.length; i++) {
        const pt = cloud[i];
        if (!pt) continue;
        const ptArr = pt.split(' ');
        xVals.push(ptArr[0]);
        yVals.push(ptArr[1]);
        zVals.push(ptArr[2]);
        iVals.push(ptArr[3]);
    }

    // Using Float32Arrays should provide optimization for WebGL --> Plotly
    const x = new Float32Array(xVals);
    const y = new Float32Array(yVals);
    const z = new Float32Array(zVals);
    const i = new Float32Array(iVals);

    const trace = {
        type: 'scatter3d',
        mode: 'markers',
        x, y, z,
        marker: { size: 1, color: i, colorscale: 'Jet', showscale: false },
        hoverinfo: 'none',
    };

    const layout = {
        paper_bgcolor: "black",
        margin: { l: 0, r: 0, t: 0, b: 0 },
        scene: {
            aspectratio: { x: 1, y: 1, z: 0.4},
            camera: {
                //eye: { x: 1, y: 1, z: 0.5 },
                center: { x: 0, y: 0, z: 0 },
                up: { x: 0, y: 0, z: 1 }
            },
            xaxis: {visible:false},
            yaxis: {visible:false},
            zaxis: {visible:false}
        }
    };

    Plotly.newPlot(plotId, [trace], layout, {responsive: true});
    addFullscreenButton(plotId);

    const end = performance.now();
    const elapsed = Math.round(end-start);
    console.log(`Displayed ${scanName} in ${elapsed} ms`);
}

/**
 * Creates and inserts a telemetry graph into a plot. 
 * 
 * @description Gets all grouped traces associated with the plot name in the 
 * telPlotsMap object and builds them, then generates Plotly plot for them.
 * 
 * @param {string} plotId - ID of the div that will display the plot.
 * @param {string} telPlotName - Name of the plot from the telPlotsMap
 * @returns {void}
 */
async function makeTelemetryPlot(plotId, telPlotName) {
    if (!tripTelemetry) return;

    // Helper function that validates JSON string as number
    const toNum = val => {
        if (val == null) return null;
        const n = Number(val);      // Strings, undefined become NaN
        return Number.isFinite(n) ? n : null;
    };

    // Helper function: fucking wizardry, don't ask me, ask GPT
    const getAtPath = (obj, dotPath) =>
        dotPath.replace(/^\./, '').split('.').reduce((o, k) => o[k], obj);

    // X [time] should be same for all traces, so just make it once
    let x = Array.from({length: tripTelemetry.telemetry.length}, (_, i) => i);
    
    const traces = [];
    telPlotsMap[telPlotName].forEach(field => {
        // Builds array of values to plot from telemetry object
        const y = tripTelemetry.telemetry.map(r => toNum(getAtPath(r, field)));

        let trace = {
            x, y,
            type: 'scatter', mode: 'lines', connectgaps: false,
            name: traceLabels[field],
        };
        // Fancy onHover formatting for specific telemetry graphs
        let traceLabel = traceLabels[field].split(" ");
        let hoverLabel = null;
        switch (telPlotName) {
        case "Raspberry Pi Util + Temp":
            hoverLabel = traceLabel.slice(2).join(" ");
            hoverLabel += ": %{y} <extra></extra>";
            trace.hovertemplate = hoverLabel;
            break;
        case "Ultrasonic Sensor Distances":
            hoverLabel = traceLabel[0];
            hoverLabel += ": %{y} cm <extra></extra>";
            trace.hovertemplate = hoverLabel;
            break;
        case "Motor Voltages":
        case "Motor Currents":
        case "Motor Speeds":
            hoverLabel = traceLabel.slice(0,2).join(" ");
            hoverLabel += ": %{y} <extra></extra>";
            trace.hovertemplate = hoverLabel;
            break;
        };
        traces.push(trace);
    });

    const layout = {
        plot_bgcolor: 'black',
        paper_bgcolor: 'black',
        font: {color: 'white'},
        margin: {t: 20, b: 0, l: 40, r: 5},
        hovermode: "x unified",
        hoverlabel: {bgcolor: 'black'},
        xaxis: {title: "Time [s]"},
        yaxis: {rangemode: 'tozero'},
        legend: {
            x: 0.5, y: -0.07, 
            xanchor: 'center', orientation: 'h', 
            font: {size: 10}
        }
    };

    Plotly.newPlot(plotId, traces, layout, {responsive: true});
    addFullscreenButton(plotId);
}

document.addEventListener("DOMContentLoaded", async function() {
    // Get trip folders, aggregate relevant HTML elements for each plot section
    let tripNames = await queryFilenames('', "Trips");
    const tripSelector = document.getElementById("trip_select");
    const plotTypeSelectors = [], plotDataSelectors = [];
    for (let i = 0; i < 3; i++) {
        plotTypeSelectors.push(document.getElementById(`plot${i+1}_type`));
        plotDataSelectors.push(document.getElementById(`plot${i+1}_data`));
    }

    // Fetch data and refresh plots when user selects a trip
    tripSelector.addEventListener("change", async function() {
        // Gets new trip's relevant filenames
        tripName =      tripSelector.value;
        videoNames =    await queryFilenames(tripName, "Video");
        scanNames =     await queryFilenames(tripName, "LiDAR");
        tripTelemetry = await queryFilenames(tripName, "Graph");

        // Resets plots on trip change
        for (let i = 0; i < 3; i++) {
            plotTypeSelectors[i].dispatchEvent(new Event("change"));
        }
    });

    // Register listeners to each menu (typeSelector) and submenu (dataSelector)
    for (let i = 0; i < plotTypeSelectors.length; i++) {
        const typeSelector = plotTypeSelectors[i];
        const dataSelector = plotDataSelectors[i];

        // When users select a new type of plot, clear plot and refresh selector
        typeSelector.addEventListener("change", function() {
            const selector = document.getElementById(`plot${i+1}_data`);
            document.getElementById(`plot${i+1}`).replaceChildren();
            selector.replaceChildren();
            
            let choices;
            if (typeSelector.value == "Video") choices = videoNames;
            if (typeSelector.value == "LiDAR") choices = scanNames;
            if (typeSelector.value == "Graph") choices = undefined;
            populateSelector(typeSelector.value, `plot${i+1}_data`, choices);

            dataSelector.dispatchEvent(new Event("change"));
        });

        // When users select new data to plot, refresh plot with new data
        dataSelector.addEventListener("change", function() {
            document.getElementById(`plot${i+1}`).replaceChildren();
            switch (typeSelector.value) {
            case "LiDAR":
                if (dataSelector.value != "No Scans to Display") 
                    makeScanPlot(`plot${i+1}`, dataSelector.value);
                break;
            case "Video":
                if (dataSelector.value != "No Videos to Display")
                    makeVideoPlot(`plot${i+1}`, dataSelector.value);
                break;
            case "Graph":
                if (dataSelector.value != "No Data to Display")
                    makeTelemetryPlot(`plot${i+1}`, dataSelector.value);
                break;
            }
        });
    }
    
    // Runs after DOMContentLoaded and loads in the first trip by default
    populateSelector("Trips", "trip_select", tripNames);
    tripSelector.dispatchEvent(new Event("change"));
});