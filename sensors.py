import argparse
import glob
import math
import os
import sys
import time
import traceback
from queue import Empty, Queue

import carla
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image
from matplotlib import colormaps as cm

DEFAULT_SENSOR_CONFIG = [
    {
        'type': 'sensor.camera.rgb',
        'name': 'front_camera',
        'attributes': {
            'image_size_x': '1280',
            'image_size_y': '720',
            'fov': '90'
        },
        'x': 1.8,
        'y': -0.5,
        'z': 0.7
    },
    {
        'type': 'sensor.camera.depth',
        'name': 'front_depth_camera',
        'attributes': {
            'image_size_x': '1280',
            'image_size_y': '720',
            'fov': '90'
        },
        'x': 1.8,
        'y': 0.5,
        'z': 0.7
    },
    {
        'type': 'sensor.lidar.ray_cast',
        'name': 'lidar',
        'attributes': {
            'channels': '64',
            'points_per_second': '100000',
            'rotation_frequency': '10',
            'range': '50',
            'upper_fov': '15',
            'lower_fov': '-25'
        },
        'x': 0.0,
        'y': 0.0,
        'z': 2.0
    }
]


@dataclass
class SpawnCar:

    # Optional parameters, all are required in order to specify car location
    SPAWN_X = None  # X-coordinate of the spawn location
    SPAWN_Y = None  # Y-coordinate of the spawn location
    SPAWN_Z = None  # Z-coordinate of the spawn location (height)
    PITCH = None  # Pitch rotation of the vehicle
    YAW = None  # Yaw rotation of the vehicle
    ROLL = None  # Roll rotation of the vehicle

    VIRIDIS = np.array(cm.get_cmap('viridis').colors)
    VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

    # Required Parameter
    client: carla.Client

    # Instance specific parameters not set in init function
    world: carla.World = field(init=False) # Carla world from client
    blueprint_library: carla.BlueprintLibrary = field(init=False) # blueprint library from world
    vehicle_bp: carla.ActorBlueprint = field(init=False) # blueprint of vehicle
    vehicle: carla.Vehicle = field(init=False) # self driving vehicle
    sensors: dict = field(default_factory=dict) # list of sensors attached to car
    original_settings: object = field(init=False)

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

    def get_focal_length(self, sensor_bp):
        print(f"Getting focal length for sensor type {sensor_bp}")
        K = np.identity(3)
        if sensor_bp.id == "sensor.camera.rgb" or sensor_bp.id == "sensor.camera.depth":
            image_w = sensor_bp.get_attribute("image_size_x").as_int()
            image_h = sensor_bp.get_attribute("image_size_y").as_int()
            fov = sensor_bp.get_attribute("fov").as_float()
            focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))

        # In this case Fx and Fy are the same since the pixel aspect
        # ratio is 1
            K = np.identity(3)
            K[0, 0] = K[1, 1] = focal
            K[0, 2] = image_w / 2.0
            K[1, 2] = image_h / 2.0

        return K

    def _sensor_callback(self, data, sensor_name):
        """
        This simple callback just stores the data on a thread safe Python Queue
        to be retrieved from the "main thread".
        """
        self.frame_counters[sensor_name]['queue'].put(data)

    def add_sensor(self, sensor_type, sensor_name, config=None, position=None, rotation=None):
        """
        Add a sensor to the vehicle.

        Args:
            sensor_type: Type of the sensor (e.g., 'sensor.camera.rgb')
            sensor_name: Name to identify this sensor
            config: Dictionary of sensor configuration parameters
            position: Custom position (x, y, z) tuple or None for default
            rotation: Custom rotation (pitch, yaw, roll) tuple or None for default
        """
        # Get the blueprint
        bp = self.blueprint_library.find(sensor_type)

        # Set configuration attributes
        if config:
            for key, value in config.items():
                bp.set_attribute(key, str(value))

        # Use position from config if available, otherwise fall back to provided or default
        if position is None:
            position = (config['x'], config['y'], config['z']) if config and all(
                k in config for k in ['x', 'y', 'z']) else (1.6, 0.0, 1.6)

        # Use rotation from config if available, otherwise fall back to provided or default
        if rotation is None:
            rotation = (config['pitch'], config['yaw'], config['roll']) if config and all(
                k in config for k in ['pitch', 'yaw', 'roll']) else (0.0, 0.0, 0.0)

        # Create the transform
        transform = carla.Transform(
            carla.Location(x=position[0], y=position[1], z=position[2]),
            carla.Rotation(pitch=rotation[0], yaw=rotation[1], roll=rotation[2])
        )

        # Spawn the sensor
        sensor = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)

        # Store in sensors dictionary
        self.sensors[sensor_name] = sensor

        # Set up queues for data
        if 'lidar' in sensor_type:
            self.frame_counters[sensor_name] = {'queue': Queue(), 'counter': 0}
            sensor.listen(lambda data: self._sensor_callback(data, sensor_name))
        elif 'camera' in sensor_type:
            self.frame_counters[sensor_name] = {'queue': Queue(), 'counter': 0,
                                                'K': self.get_focal_length(bp)}
            sensor.listen(lambda data: self._sensor_callback(data, sensor_name))

        return sensor

    def project_lidar_to_all_cameras(self, lidar_data, lidar_name='lidar'):
        """
        Project lidar points to all available cameras and return the results.

        Args:
            lidar_data: Raw lidar data
            lidar_name: Name of the lidar sensor

        Returns:
            Dictionary of camera_name -> projection results
        """
        # Get lidar data as numpy array
        p_cloud_size = len(lidar_data)
        p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
        p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))

        # Extract intensity
        intensity = np.array(p_cloud[:, 3])

        # Point cloud in lidar sensor space
        local_lidar_points = np.array(p_cloud[:, :3]).T
        local_lidar_points = np.r_[
            local_lidar_points, [np.ones(local_lidar_points.shape[1])]]

        # Get lidar to world transformation
        lidar_sensor = self.sensors[lidar_name]
        lidar_2_world = lidar_sensor.get_transform().get_matrix()

        # Transform points from lidar to world space
        world_points = np.dot(lidar_2_world, local_lidar_points)

        # Dictionary to store projection results
        projection_results = {}

        # Project to each camera
        for camera_name, camera in self.sensors.items():
            # Skip if not a camera
            if 'camera' not in camera_name:
                continue

            # Get camera intrinsic matrix (K)
            K = self.frame_counters[camera_name]['K']

            # Get camera transform
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

            # Transform points from world to camera space
            sensor_points = np.dot(world_2_camera, world_points)

            # Convert to standard camera coordinates
            point_in_camera_coords = np.array([
                sensor_points[1],
                sensor_points[2] * -1,
                sensor_points[0]])

            # Project 3D points to 2D using camera's intrinsic matrix
            points_2d = np.dot(K, point_in_camera_coords)

            # Normalize coordinates
            points_2d = np.array([
                points_2d[0, :] / points_2d[2, :],
                points_2d[1, :] / points_2d[2, :],
                points_2d[2, :]])

            # Filter points that are within camera view
            image_w = int(self.sensors[camera_name].attributes['image_size_x'])
            image_h = int(self.sensors[camera_name].attributes['image_size_y'])

            points_2d = points_2d.T
            filtered_intensity = intensity.T
            points_in_canvas_mask = \
                (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < image_w) & \
                (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < image_h) & \
                (points_2d[:, 2] > 0.0)

            # Get final 2D coordinates and intensity values
            filtered_points = points_2d[points_in_canvas_mask]
            filtered_intensity = filtered_intensity[points_in_canvas_mask]

            # Store in the results dictionary
            projection_results[camera_name] = {
                'points_2d': filtered_points,
                'intensity': filtered_intensity
            }

        return projection_results

    def process_frame_with_multiple_cameras(self, lidar_name='lidar'):
        """
        Process a single frame with multiple cameras and lidar.
        """
        # Get lidar data
        try:
            lidar_data = self.frame_counters[lidar_name]['queue'].get(True, 1.0)
        except Empty:
            print("[Warning] Lidar data was missed")
            return None

        # Save the lidar data
        self.save_lidar_data(lidar_name, lidar_data, args.save_dir)

        # Project lidar points to all cameras
        projections = self.project_lidar_to_all_cameras(lidar_data, lidar_name)

        # Process each camera's data
        for camera_name in [name for name in self.sensors.keys() if 'camera' in name]:
            try:
                # Get camera data
                camera_data = self.frame_counters[camera_name]['queue'].get(True, 1.0)

                # Skip if projections not available
                if camera_name not in projections:
                    continue

                # Get projection data
                proj_data = projections[camera_name]
                points_2d = proj_data['points_2d']
                intensity = proj_data['intensity']

                # Create image from raw data
                im_array = np.copy(np.frombuffer(camera_data.raw_data, dtype=np.dtype("uint8")))
                im_array = np.reshape(im_array, (camera_data.height, camera_data.width, 4))
                im_array = im_array[:, :, :3][:, :, ::-1]  # BGRA to RGB

                # Extract screen coordinates
                u_coord = points_2d[:, 0].astype(np.int_)
                v_coord = points_2d[:, 1].astype(np.int_)

                # Create color mapping for points
                intensity = 4 * intensity - 3
                color_map = np.array([
                    np.interp(intensity, self.VID_RANGE, self.VIRIDIS[:, 0]) * 255.0,
                    np.interp(intensity, self.VID_RANGE, self.VIRIDIS[:, 1]) * 255.0,
                    np.interp(intensity, self.VID_RANGE, self.VIRIDIS[:, 2]) * 255.0
                ]).astype(np.int_).T

                # Create output directory if it doesn't exist
                if camera_name not in self.output_dirs:
                    output_dir = os.path.join(args.save_dir, camera_name)
                    os.makedirs(output_dir, exist_ok=True)
                    self.output_dirs[camera_name] = output_dir

                # Draw points on image (using the dot_extent parameter if available)
                dot_extent = 1  # Default dot size
                if hasattr(args, 'dot_extent'):
                    dot_extent = args.dot_extent

                if dot_extent <= 0:
                    # Draw the 2d points on the image as a single pixel
                    im_array[v_coord, u_coord] = color_map
                else:
                    # Draw the 2d points on the image as squares
                    for i in range(len(points_2d)):
                        # Calculate valid ranges for dot placement
                        v_min = max(0, v_coord[i] - dot_extent)
                        v_max = min(camera_data.height, v_coord[i] + dot_extent)
                        u_min = max(0, u_coord[i] - dot_extent)
                        u_max = min(camera_data.width, u_coord[i] + dot_extent)

                        # Draw the dot
                        im_array[v_min:v_max, u_min:u_max] = color_map[i]

                # Save the frame as an image
                frame_count = self.frame_counters[camera_name]['counter']
                file_name = f"{frame_count:06d}.png"
                file_path = os.path.join(self.output_dirs[camera_name], file_name)

                # Save image using PIL
                image = Image.fromarray(im_array)
                image.save(file_path)

                # Increment frame counter
                self.frame_counters[camera_name]['counter'] += 1

            except Empty:
                print(f"[Warning] Camera {camera_name} data was missed")
                continue

        # Increment lidar frame counter
        self.frame_counters[lidar_name]['counter'] += 1

    def perform_carla_sensor_fusion(self):
        """
        Use CARLA's built-in capabilities to perform sensor fusion between cameras
        """
        # Get all camera sensors
        camera_sensors = {name: sensor for name, sensor in self.sensors.items() if 'camera' in name}

        if len(camera_sensors) < 2:
            print("Need at least 2 cameras for fusion")
            return

        # Create output directory for fusion results
        fusion_dir = os.path.join(self.output_dirs.get('base', 'sensor_output'), 'fusion_camera')
        self.output_dirs['fusion_camera'] = fusion_dir
        os.makedirs(fusion_dir, exist_ok=True)

        # Get the frame number
        frame_num = max(
            counter_data['counter'] for sensor_name, counter_data in self.frame_counters.items()
            if 'counter' in counter_data
        ) - 1 if self.frame_counters else 0

        # Get main reference sensor (e.g., RGB camera)
        reference_sensor_name = next((name for name in camera_sensors if 'rgb' in name or 'front_camera' in name), None)
        if not reference_sensor_name:
            reference_sensor_name = next(iter(camera_sensors))

        reference_sensor = self.sensors[reference_sensor_name]
        reference_transform = reference_sensor.get_transform()

        # Get reference image
        ref_img_path = os.path.join(
            self.output_dirs.get(reference_sensor_name, reference_sensor_name),
            f"{self.frame_counters[reference_sensor_name]['counter'] - 1:06d}.png"
        )
        if not os.path.exists(ref_img_path):
            print(f"Reference image {ref_img_path} not found")
            return

        reference_img = cv2.imread(ref_img_path)

        # Get world coordinates of pixels using depth information
        depth_sensor_name = next((name for name in camera_sensors if 'depth' in name), None)
        if depth_sensor_name:
            depth_sensor = self.sensors[depth_sensor_name]
            depth_transform = depth_sensor.get_transform()

            # Get depth image
            depth_img_path = os.path.join(
                self.output_dirs.get(depth_sensor_name, depth_sensor_name),
                f"{self.frame_counters[depth_sensor_name]['counter'] - 1:06d}.png"
            )
            if not os.path.exists(depth_img_path):
                print(f"Depth image {depth_img_path} not found")
                return

            # Load and process depth image
            depth_img = np.array(Image.open(depth_img_path))

            # Print depth image shape and type for debugging
            print(f"Depth image shape: {depth_img.shape}, dtype: {depth_img.dtype}")

            # Make sure depth image is single-channel
            if len(depth_img.shape) > 2:
                # If RGB, convert to grayscale (use first channel or average)
                depth_img = depth_img[:, :, 0].astype(np.float32)
            else:
                depth_img = depth_img.astype(np.float32)

            # Convert depth image to meters (assuming CARLA's depth encoding)
            normalized_depth = depth_img / 255.0
            depth_in_meters = 1000 * normalized_depth  # Assuming max range is 1000m

            # Get camera attributes
            depth_fov = float(depth_sensor.attributes.get('fov', 90))
            depth_width = depth_img.shape[1]
            depth_height = depth_img.shape[0]

            # Create a colored point cloud from depth
            colored_point_cloud = []
            for y in range(0, depth_height, 5):  # Subsample for efficiency
                for x in range(0, depth_width, 5):
                    # Now depth_in_meters[y, x] should be a scalar
                    depth_value = depth_in_meters[y, x]

                    if depth_value > 0:
                        # Use CARLA's camera intrinsic calculation method
                        image_point = carla.Vector2D(x, y)

                        # Convert from image space to camera space using CARLA helper
                        # Convert from image space to camera space manually
                        fov_rad = math.radians(depth_fov)
                        focal_length = depth_width / (2.0 * math.tan(fov_rad / 2.0))

                        # Convert to camera coordinates
                        cx = depth_width / 2.0
                        cy = depth_height / 2.0
                        x_corr = (image_point.x - cx) / focal_length
                        y_corr = (image_point.y - cy) / focal_length

                        # Create 3D point in camera space
                        x_val = float(depth_value)
                        y_val = float(-depth_value * x_corr)
                        z_val = float(depth_value * y_corr)
                        camera_point = carla.Location(x_val, y_val, z_val)

                        # Convert from camera space to world space
                        world_point = depth_transform.transform(camera_point)

                        # Store the world point
                        colored_point_cloud.append(world_point)

            # Project all world points to the reference camera
            fused_img = reference_img.copy()

            rgb_fov = float(reference_sensor.attributes.get('fov', 90))
            rgb_width = reference_img.shape[1]  # Use the actual RGB image width
            rgb_height = reference_img.shape[0]  # Use the actual RGB image height

            # Calculate projection parameters using RGB camera's values
            fov_rad = math.radians(rgb_fov)
            focal_length = rgb_width / (2.0 * math.tan(fov_rad / 2.0))

            # Center points of the RGB image (not depth image)
            cx = rgb_width / 2.0
            cy = rgb_height / 2.0

            # Calculate camera matrices
            # This requires the camera intrinsic and extrinsic parameters

            # Get camera intrinsics
            K = np.zeros((3, 3))
            K[0, 0] = K[1, 1] = focal_length
            K[0, 2] = cx
            K[1, 2] = cy
            K[2, 2] = 1

            # Get camera extrinsics (reference_transform)
            # This is more complex and requires converting CARLA's transform to a matrix
            # For now, let's use a simpler approach

            for world_point in colored_point_cloud:
                # Alternative approach: Use a different projection method
                # Convert world_point to a numpy array
                point_np = np.array([world_point.x, world_point.y, world_point.z, 1.0])

                # Convert from world to camera coordinates
                # This requires creating a view matrix from the camera transform
                camera_location = reference_transform.location
                camera_rotation = reference_transform.rotation

                # Create a rotation matrix from camera rotation (simplified)
                # This is an approximation; a more accurate version would use quaternions
                roll_rad = math.radians(camera_rotation.roll)
                pitch_rad = math.radians(camera_rotation.pitch)
                yaw_rad = math.radians(camera_rotation.yaw)

                # Create rotation matrices for each axis
                # This is simplified; a full solution would use matrix multiplication
                cos_roll, sin_roll = math.cos(roll_rad), math.sin(roll_rad)
                cos_pitch, sin_pitch = math.cos(pitch_rad), math.sin(pitch_rad)
                cos_yaw, sin_yaw = math.cos(yaw_rad), math.sin(yaw_rad)

                # World to camera transformation (simplified)
                camera_relative_x = (world_point.x - camera_location.x) * cos_yaw + \
                                    (world_point.y - camera_location.y) * sin_yaw
                camera_relative_y = -(world_point.x - camera_location.x) * sin_yaw + \
                                    (world_point.y - camera_location.y) * cos_yaw
                camera_relative_z = world_point.z - camera_location.z

                # Project to image space
                if camera_relative_x > 0:  # Point is in front of camera
                    image_x = cx + focal_length * camera_relative_y / camera_relative_x
                    image_y = cy - focal_length * camera_relative_z / camera_relative_x

                    x, y = int(image_x), int(image_y)

                    if 0 <= x < rgb_width and 0 <= y < rgb_height:
                        # Color based on distance
                        distance = math.sqrt(camera_relative_x ** 2 + camera_relative_y ** 2 + camera_relative_z ** 2)
                        normalized_dist = min(distance / 50.0, 1.0)

                        color = (
                            int(255 * (1 - normalized_dist)),
                            int(255 * normalized_dist),
                            int(255 * (1 - normalized_dist))
                        )

                        cv2.circle(fused_img, (x, y), 2, color, -1)

            # Save the fused image
            fusion_path = os.path.join(fusion_dir, f"carla_fusion_{frame_num:06d}.png")
            cv2.imwrite(fusion_path, fused_img)
            print(f"\nCreated CARLA-based fusion image: {fusion_path}")

    def save_lidar_data(self, name, lidar_data, output_dir):
        """Callback to save LiDAR point cloud data as .ply files."""
        output_dir = os.path.join(output_dir, name)
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

    def add_multiple_sensors(self, sensor_configs):
        """
        Add multiple sensors to the vehicle based on the configurations.

        Args:
            sensor_configs: List of sensor configuration dictionaries
        """

        # Add other sensors from configs
        for idx, sensor_config in enumerate(sensor_configs):
            # Skip cameras if we're using stereo setup

            sensor_type = sensor_config['type']
            sensor_name = sensor_config.get('name', f"{sensor_type.split('.')[-1]}_{idx}")
            sensor_attributes = sensor_config.get('attributes', {})

            # Add the sensor
            self.add_sensor(sensor_type, sensor_name, config=sensor_attributes)

    def set_asynchronous_mode(self):
        """
        Reset the world back to asynchronous mode by updating the settings.
        """
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        print("World set back to asynchronous mode.")

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
                for cur_sensor in self.sensors.values():
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

        self.set_asynchronous_mode()


    def create_video_from_images(self, output_dir, name, fps=30):
        """
        Iterates through all output directories, creates videos from images in those directories,
        and then deletes the images.

        Args:
            :param fps:
            :param name:
            :param output_dir:
        """
        print(f"Creating video from images in {output_dir}...")
        image_files = sorted(glob.glob(os.path.join(output_dir, "*.png")))  # Or *.jpg, depending on your image format
        if not image_files:
            print(f"No images found in {output_dir}. Skipping.")

        print(f"Found {len(image_files)} images in {output_dir}.")
        # Determine video dimensions from the first image
        img = cv2.imread(image_files[0])
        height, width, layers = img.shape
        size = (width, height)

        # Define the video filename
        video_filename = os.path.join(output_dir, f"{name}.avi")  # Or .mp4

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
    parser.add_argument("--host", type=str, default="localhost", help="CARLA server host. Default is 'localhost'.")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port. Default is 2000.")
    parser.add_argument("--world", type=str, default="Town04", help="World to load in CARLA. Default is 'Town04'.")
    parser.add_argument("--autopilot", default=True, help="Enable autopilot for the spawned car. Default is True.")
    parser.add_argument("--save_dir", type=str, default="sensor_output",
                        help="Directory to save sensor data. Default is 'sensor_output'")
    parser.add_argument("--reload_map", type=bool, default=True, help="Reload the map before spawning the car. Default is True.")
    parser.add_argument("--frames", type=int, default=250, help="Number of frames to capture. Default is 100.")
    parser.add_argument("--create_videos", type=bool, default=True, help="Create a video from the captured sensor data. Default is True.")
    parser.add_argument(
        "--sensor_configs",
        type=str,
        default=DEFAULT_SENSOR_CONFIG,
        help="""JSON string specifying the sensors. Default sensors are:
                '[{"type": "sensor.lidar.ray_cast", "x": 0.0, "y": 0.0, "z": 2}, 
                   {"type": "sensor.camera.rgb", "x": 1.8, "y": -0.5, "z": 0.7}, 
                   {"type": "sensor.camera.depth", "x": 1.8, "y": 0.5, "z": 0.7}]'"""
    )

    args = parser.parse_args()

    # Set up the client
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    client.load_world(args.world)

    world = client.get_world()

    # Set up traffic manager in synchronous mode if needed
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    # Configure synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    
    world.apply_settings(settings)

    # Create the car with sensors
    car = SpawnCar(client)

    try:
        # Add sensors from configuration
        car.add_multiple_sensors(args.sensor_configs)

        # Run the simulation for the specified number of frames
        for frame in range(args.frames):
            # Tick the world
            world.tick()

            # Process frame with multiple cameras
            car.process_frame_with_multiple_cameras()

            car.perform_carla_sensor_fusion()

            # Print progress
            sys.stdout.write(f"\r({frame}/{args.frames}) Processing... ")
            sys.stdout.flush()

        print("\nDone!")

        # Create videos from captured frames if desired
        if args.create_videos:
            for camera_name in [name for name in car.sensors.keys() if 'camera' in name] + ['fusion_camera']:
                print(f'Is {camera_name} in {car.output_dirs.keys()}?')
                if camera_name in car.output_dirs.keys():
                    car.create_video_from_images(
                        output_dir=car.output_dirs[camera_name],
                        name=f"{camera_name}_output",
                        fps=20
                    )

    except Exception as e:
        # --- THIS IS THE ADDED EXCEPT BLOCK ---
        # Catches any exception derived from the base Exception class
        print(f"\n--- ERROR CAUGHT ---")
        print(f"An exception of type {type(e).__name__} occurred.")
        print(f"Error details: {e}")
        print("--- Traceback ---")
        # Print the standard exception traceback
        traceback.print_exc(file=sys.stderr)  # Print traceback to standard error
        print("-----------------")
        print("Re-raising the caught exception...")
        raise  # Re-raises the exception that was just caught

    finally:
        # Clean up
        car.set_asynchronous_mode()
        car.cleanup()
