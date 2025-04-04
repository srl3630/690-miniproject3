import argparse
import glob
import json
import os
import datetime
import time

import carla
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np

@dataclass
class SpawnCar:

    # Optional parameters, all are required in order to specify car location
    SPAWN_X = None  # X-coordinate of the spawn location
    SPAWN_Y = None  # Y-coordinate of the spawn location
    SPAWN_Z = None  # Z-coordinate of the spawn location (height)
    PITCH = None  # Pitch rotation of the vehicle
    YAW = None  # Yaw rotation of the vehicle
    ROLL = None  # Roll rotation of the vehicle

    # Required Parameter
    client: carla.Client

    # Instance specific parameters not set in init function
    world: carla.World = field(init=False) # Carla world from client
    blueprint_library: carla.BlueprintLibrary = field(init=False) # blueprint library from world
    vehicle_bp: carla.ActorBlueprint = field(init=False) # blueprint of vehicle
    vehicle: carla.Vehicle = field(init=False) # self driving vehicle
    sensors: List[carla.Sensor] = field(default_factory=list) # list of sensors attached to car

    frame_counters: dict = field(init=False, default_factory=dict)  # Keeps track of frame numbers for each sensor
    output_dirs: dict = field(init=False, default_factory=dict) # keeps track of cameras and their output directories

    def __post_init__(self):
        self.world = self.client.get_world()
        self.blueprint_library = self.get_blueprint_library()
        self.vehicle_bp = self.blueprint_library.filter('vehicle.*')[0]
        self.vehicle = self.spawn_vehicle()
        self.vehicle.set_autopilot(True)

    def get_blueprint_library(self) -> carla.BlueprintLibrary:
        """Retrieve the blueprint library for the current world."""
        return self.world.get_blueprint_library()

    def spawn_vehicle(self) -> carla.Vehicle:
        """Spawn a vehicle at the specified location and rotation."""
        if self.SPAWN_X is None or self.SPAWN_Y is None or self.SPAWN_Z is None or self.PITCH is None or self.YAW is None or self.ROLL is None:
            spawn_point = self.world.get_map().get_spawn_points()[0]
            spawn_location = spawn_point.location  # Extract the location
            spawn_rotation = spawn_point.rotation  # Extract the rotation
            spawn_transform = carla.Transform(spawn_location, spawn_rotation)
        else:
            spawn_location = carla.Location(x=self.SPAWN_X, y=self.SPAWN_Y, z=self.SPAWN_Z)
            spawn_rotation = carla.Rotation(pitch=self.PITCH, yaw=self.YAW, roll=self.ROLL)
            spawn_transform = carla.Transform(spawn_location, spawn_rotation)
        print(f"Spawning vehicle at {spawn_location} with rotation {spawn_rotation}")
        return self.world.spawn_actor(self.vehicle_bp, spawn_transform)

    def add_sensor(self, sensor_type: str, location_offset: carla.Transform, callback=None) -> carla.Sensor:
        """Add a sensor to the car."""
        sensor_bp = self.blueprint_library.find(sensor_type)
        if not sensor_bp:
            raise ValueError(f"Sensor type '{sensor_type}' not found in blueprint library.")

        sensor = self.world.spawn_actor(sensor_bp, location_offset, attach_to=self.vehicle)
        if callback:
            sensor.listen(callback)  # Attach the callback
        self.sensors.append(sensor)
        return sensor


    def save_frame_as_image(self, frame, camera_name):
            """
            Saves a carla.Image frame to disk as a JPG file using OpenCV.

            Args:
                frame: The carla.libcarla.Image object received from the sensor.
                camera_name: A string identifier for the camera (used for path/naming).
            """
            image_data = np.frombuffer(frame.raw_data, dtype=np.uint8)
            image_bgra = np.reshape(image_data, (frame.height, frame.width, 4))
            image_bgr = image_bgra[:, :, :3]
            os.makedirs(self.output_dirs[camera_name], exist_ok=True)
            image_filename = f"frame_{camera_name}_{self.frame_counters[camera_name]:05d}.jpg"
            image_path = os.path.join(self.output_dirs[camera_name], image_filename)
            cv2.imwrite(image_path, image_bgr)
            self.frame_counters[camera_name] += 1
            print(f"Saved image: {image_path}")

    def add_multiple_sensors(self, sensor_configs: list, save_dir: str):
        """Add multiple sensors based on a list of configurations.

        Args:
            sensor_configs: sensor configuration in JSON format.
            save_dir: directory to save sensor data.
        """
        # Get the current date for a unique directory name
        current_date = datetime.datetime.now().isoformat(timespec='seconds')

        for sensor_config in sensor_configs:
            sensor_type = sensor_config.get("type")
            x = sensor_config.get("x", 0.0)
            y = sensor_config.get("y", 0.0)
            z = sensor_config.get("z", 0.0)
            print(f'{sensor_type} placed at {x}, {y}, {z}')
            location_offset = carla.Transform(carla.Location(x=x, y=y, z=z))

            # Create distinct subdirectories for each sensor type
            if sensor_type == "sensor.camera.rgb":
                self.output_dirs["RGB"] = os.path.join(save_dir, f"rgb_{current_date}")
                self.frame_counters["RGB"] = 0
                self.add_sensor(sensor_type, location_offset,
                                callback=lambda data: self.save_frame_as_image(data, "RGB"))
            elif sensor_type == "sensor.camera.depth":
                self.output_dirs["Depth"] = os.path.join(save_dir, f"depth_{current_date}")
                self.frame_counters["Depth"] = 0
                self.add_sensor(sensor_type, location_offset,
                                callback=lambda data: self.save_frame_as_image(data, "Depth"))
            elif sensor_type == "sensor.lidar.ray_cast":
                lidar_output_dir = os.path.join(save_dir, f"lidar_{current_date}")
                self.add_sensor(sensor_type, location_offset,
                                callback=lambda data: self.save_lidar_data(data, lidar_output_dir))
            else:
                self.add_sensor(sensor_type, location_offset)

    def save_lidar_data(self, lidar_data, output_dir):
        """Callback to save LiDAR point cloud data as .ply files."""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"lidar_{lidar_data.frame:06d}.ply")
        print(f"Saving LiDAR point cloud to {filepath}")
        self.__write_lidar_to_ply(lidar_data, filepath)

    @staticmethod
    def __write_lidar_to_ply(lidar_data, filepath):
        """
        Writes LiDAR point cloud data to a PLY file, including positional coordinates, intensity,
        and timestamp information. This method ensures proper structuring of the PLY file format
        and restricts intensity values to the valid range between 0 and 255.

        Args:
            lidar_data (Iterable): A collection containing LiDAR detection data where each
                detection includes positional coordinates (x, y, z), an intensity value,
                and a timestamp.
            filepath (str): Path to the file where the PLY data will be written.

        Returns:
            None
        """
        num_points = len(lidar_data)
        header = (
            "ply\n"
            "format ascii 1.0\n"
            f"element vertex {num_points}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar intensity\n"  # Added intensity property
            "property double timestamp\n"  # Added timestamp property
            "end_header\n"
        )

        with open(filepath, 'w') as ply_file:
            ply_file.write(header)
            for detection in lidar_data:
                # Grab directional data from detection points
                x = detection.point.x
                y = detection.point.y
                z = detection.point.z
                intensity = detection.intensity  # Access intensity value
                timestamp = lidar_data.timestamp  # Access timestamp

                # Ensure intensity is an integer between 0 and 255
                intensity = int(max(0, min(intensity, 255)))

                ply_file.write(f"{x} {y} {z} {intensity} {timestamp}\n")  # Save intensity and timestamp

    def cleanup(self):
        """
        Cleanup the car and its sensors at the end of the simulation.
        Also, print the file locations of sensor data (videos/Ply files).
        """

        # Allow ongoing tasks to finish
        time.sleep(2)

        try:
            # Destroy sensors
            if hasattr(self, 'sensors') and self.sensors:
                for cur_sensor in self.sensors:
                    cur_sensor.destroy()
                print("All sensors destroyed successfully.")

            # Destroy the vehicle
            if hasattr(self, 'vehicle') and self.vehicle:
                self.vehicle.destroy()
                print("Vehicle destroyed successfully.")

            if hasattr(self, 'sensor_configs') and self.sensor_configs:  # For other sensors (e.g., LiDAR)
                for config in self.sensor_configs:
                    if 'file_path' in config:  # Assuming config contains file paths
                        print(f"LiDAR data saved at: {config['file_path']}")

        except Exception as e:
            print(f"An error occurred during cleanup: {e}")

    def create_video_from_images(self, fps=30):
        """
        Iterates through all output directories, creates videos from images in those directories,
        and then deletes the images.

        Args:
            output_dirs: A list of output directories containing the images.
            fps: Frames per second for the output video.
        """
        for output_dir in self.output_dirs.values():
            image_files = sorted(glob.glob(os.path.join(output_dir, "*.jpg")))  # Or *.jpg, depending on your image format
            if not image_files:
                print(f"No images found in {output_dir}. Skipping.")
                continue

            # Determine video dimensions from the first image
            img = cv2.imread(image_files[0])
            height, width, layers = img.shape
            size = (width, height)

            # Define the video filename
            video_filename = os.path.join(output_dir, "video.avi")  # Or .mp4

            # Create VideoWriter object
            out = cv2.VideoWriter(video_filename, cv2.VideoWriter.fourcc(*'X', 'V', 'I', 'D'), fps, size)  # Use XVID

            try:
                for image_file in image_files:
                    img = cv2.imread(image_file)
                    out.write(img)
                print(f"Video created successfully in {output_dir}: {video_filename}")
            except Exception as e:
                print(f"Error creating video in {output_dir}: {e}")
            finally:
                out.release()  # Release VideoWriter object

            # Delete the images
            for image_file in image_files:
                try:
                    os.remove(image_file)
                except OSError as e:
                    print(f"Error deleting {image_file}: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred while deleting {image_file}: {e}")

            print(f"Images deleted from {output_dir}")


if __name__ == "__main__":
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Spawn a car in CARLA, attach multiple sensors, and save sensor data.")
    parser.add_argument("--host", type=str, default="localhost", help="CARLA server host.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port.")
    parser.add_argument("--world", type=str, default="Town04", help="World to load in CARLA.")
    parser.add_argument("--autopilot", default=True, help="Enable autopilot for the spawned car.")
    parser.add_argument("--save_dir", type=str, default="sensor_output", help="Directory to save sensor data.")
    parser.add_argument("--reload_map", type=bool, default=False, help="Reload the map before spawning the car.")
    parser.add_argument("--runtime", type=float, default=10.0, help="Duration of the simulation in seconds.")
    parser.add_argument(
        "--sensors",
        type=str,
        default=json.dumps([  # Default sensors
            {"type": "sensor.lidar.ray_cast", "x": 0.0, "y": 0.0, "z": 2},
            {"type": "sensor.camera.rgb", "x": 0.7, "y": -0.9, "z": 1},
            {"type": "sensor.camera.depth", "x": 0.7, "y": 0.9, "z": 1},
        ]),
        help="""JSON string specifying the sensors. Default sensors are:
                '[{"type": "sensor.lidar.ray_cast", "x": 0.0, "y": 0.0, "z": 2}, 
                   {"type": "sensor.camera.rgb", "x": 0.7, "y": -0.9, "z": 1}, 
                   {"type": "sensor.camera.depth", "x": 0.7, "y": 0.9, "z": 1}]'"""
    )

    args = parser.parse_args()

    # Connect to CARLA server
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)  # Set a timeout in case of connection issues
    if args.reload_map:
        world = client.load_world(args.world)

    # Parse sensors configuration
    sensor_configs = json.loads(args.sensors)

    # Spawn the car
    car = SpawnCar(client=client)
    car.vehicle.set_autopilot(args.autopilot)
    print(
        f"Spawning vehicle at ({car.vehicle.get_location().x}, {car.vehicle.get_location().y}, {car.vehicle.get_location().z})"
    )

    # Add sensors to the car with appropriate callbacks
    car.add_multiple_sensors(sensor_configs, args.save_dir)

    print(f"Car spawned in world '{args.world}' with client connected to {args.host}:{args.port}")
    print(f"Sensors attached: {len(car.sensors)}")
    for idx, sensor in enumerate(car.sensors, 1):
        print(f" - Sensor {idx}: {sensor.type_id}")

    # sleep for runtime seconds
    time.sleep(args.runtime)
    car.cleanup()
    car.create_video_from_images()

    
    
    
