# CS 690 - MiniProject 3 - CARLA Data collection and Processing.

## Overview

This project provides a framework for spawning a vehicle in the CARLA simulator and recording and projecting data from
multiple sensors (LiDAR and Depth) onto an RGB camera. It allows you to configure various sensor types (e.g., cameras,
LiDAR) and save the captured data for later analysis or use in machine learning applications.

[Sample Output (Google Drive)](https://drive.google.com/drive/folders/14sAM0Tg_8Qq1EABdM_fMnW05E99ldiHE?usp=drive_link)

**Important:** This project requires an active, running instance of the CARLA simulator.

## Requirements

*   CARLA simulator (version 0.9.15 or earlier)
*   Python 3.10 (use conda or venv)
*   Required Python packages:
    *   `carla` (same version as carla simulator)
    *   `numpy`
    *   `argparse`
    *   `cv2`
    *   `os`
    *   `subprocess`
    * `PIL`
    * `matplotlib`

    You can install the required packages using pip:

    ```bash
    pip install carla numpy opencv-python
    ```

## Project Structure

*   `sensors.py`: Contains the core logic for spawning the car, configuring sensors, and recording data.
* `my_sensor.json`: JSON file containing the default configuration for cameras in the script

## Usage

1.  **Start the CARLA simulator:** Make sure the CARLA simulator is running before you execute the script.

2.  **Run the `sensors.py` script (can be run without any arguments successfully):**

    ```bash
    python sensors.py [options]
    ```

    The script accepts several command-line arguments to customize the sensor setup and recording process.  Use `--help` to see all available arguments. Example:
    ```bash
    python sensors.py --save_dir /path/to/output/folder --runtime 10
    ```

3. **Configure Sensors:** The script uses JSON to define the sensors and their positions to be attached to the vehicle.
   The default configuration satisfies the following requirements (no `--sensor_config_file` or `sensor_configs`
   argument specified). See [Example Sensor Flag Input](#example-sensor-flag-input) for examples.
- `Lidar` sensor placed above car, centered
- `RGB` sensor placed near left headlight
- `Depth` sensor placed near right headlight

4. **Output:** The recorded sensor data (videos, LiDAR point clouds) will be saved in the specified output directory.
   The directory structure will be organized by sensor type and fused models.

## Command-Line Options

The `sensors.py` script uses `argparse` to handle command-line arguments. Here's a description of some of the available options and their defaults:

*   `--save_dir`: Specifies the directory where the recorded sensor data will be saved. (Will be created if it does not exist, runs non-destructively) **Default:**  If not provided, the default directory is *sensor_output* in the directory the script is run from.
*   `--town`: Specifies the town to use in the simulation. **Default:** If not provided, `Town04` will be selected, which comes default with Carla.
*   `--runtime`: Specifies runtime in seconds **Default:** If not provided, default is 10 seconds


## Example Sensor Flag Input
*Lidar*, *RGB*, and *Depth* Sensors with all coordinate values set to `10`

See included `my_sensors.json` for example input.

## Notes

*   Ensure that the CARLA server is running with the same Python version as your script.
*   Experiment with different sensor configurations to achieve the desired data capture setup.