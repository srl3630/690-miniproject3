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

        # Dictionary to hold camera images
        camera_images = {}

        # First process all camera data
        for camera_name in [name for name in self.sensors.keys() if 'camera' in name]:
            try:
                # Get camera data
                camera_data = self.frame_counters[camera_name]['queue'].get(True, 1.0)

                # Convert camera data to image
                img = np.array(camera_data.raw_data)
                img = img.reshape((camera_data.height, camera_data.width, 4))
                img = img[:, :, :3]  # Remove alpha channel

                # Store the image
                camera_images[camera_name] = img

                # Save to original location
                output_dir = self.output_dirs.get(camera_name)
                if not output_dir:
                    output_dir = os.path.join(args.save_dir, camera_name)
                    os.makedirs(output_dir, exist_ok=True)
                    self.output_dirs[camera_name] = output_dir

                # Save original camera image
                camera_frame_num = self.frame_counters[camera_name]['counter']
                output_path = os.path.join(output_dir, f"{camera_frame_num:06d}.png")
                cv2.imwrite(output_path, img)

                # Increment frame counter
                self.frame_counters[camera_name]['counter'] += 1

            except Empty:
                print(f"[Warning] Camera {camera_name} data was missed")

        # First perform camera fusion
        fused_images = self.perform_carla_sensor_fusion(camera_images)

        # Then project lidar points on both original and fused images with enhanced visibility
        self.project_lidar_to_camera_images(lidar_data, camera_images, lidar_name, fused_images)

        # Increment lidar frame counter
        self.frame_counters[lidar_name]['counter'] += 1

        return camera_images

    def project_lidar_to_camera_images(self, lidar_data, camera_images, lidar_name='lidar', fused_images=None):
        """
        Project lidar points to camera images with improved visualization.

        Args:
            lidar_data: Raw lidar data
            camera_images: Dictionary of camera_name -> camera image
            lidar_name: Name of the lidar sensor
            fused_images: Dictionary of fused camera images (optional)
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

        # Project to each camera that we have an image for
        for camera_name, camera_image in camera_images.items():
            camera = self.sensors[camera_name]

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

            # Store filtered depth values for point size calculation
            depth_values = point_in_camera_coords[2, points_in_canvas_mask]

            # Generate high contrast lidar visualization on both original and fused images
            self._apply_high_contrast_lidar_visualization(
                camera_image, filtered_points, depth_values, camera_name,
                is_fusion=False, frame_num=self.frame_counters[camera_name]['counter'] - 1
            )

            # For fused images - apply the same lidar visualization
            if fused_images and camera_name in fused_images:
                self._apply_high_contrast_lidar_visualization(
                    fused_images[camera_name], filtered_points, depth_values, camera_name,
                    is_fusion=True, frame_num=self.frame_counters[camera_name]['counter'] - 1
                )

    def _apply_high_contrast_lidar_visualization(self, image, points, depth_values, camera_name,
                                                 is_fusion=False, frame_num=0):
        """
        Apply high contrast lidar visualization on an image

        Args:
            image: The image to apply lidar points on
            points: The 2D points to draw
            depth_values: Depth values for each point
            camera_name: Name of the camera
            is_fusion: Whether this is a fusion image or original camera image
            frame_num: Frame number for saving
        """
        # Create a copy of the image for visualization
        visualization = image.copy()
        visible_points = 0

        # Draw lidar points with high contrast colors
        for i, (point, depth) in enumerate(zip(points, depth_values)):
            x, y = int(point[0]), int(point[1])

            # Use a point size that scales with distance (closer = larger)
            point_size = max(1, int(5 - min(depth / 10, 4)))

            # Use high-contrast colors with distance-based hue
            if depth < 10:
                # Magenta for close objects - highly visible
                color = (255, 0, 255)
            elif depth < 20:
                # Cyan for medium distance - also high contrast
                color = (255, 255, 0)
            elif depth < 30:
                # Bright green - visible on most backgrounds
                color = (0, 255, 0)
            else:
                # White for far objects
                color = (255, 255, 255)

            # Draw with anti-aliasing for better visibility
            cv2.circle(visualization, (x, y), point_size, color, -1)
            visible_points += 1

        # Add informative elements
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Add a semi-transparent black background for text to ensure readability
        bg_rect = np.zeros((140, 200, 3), dtype=np.uint8)
        visualization[10:150, 10:210] = cv2.addWeighted(
            visualization[10:150, 10:210], 0.3, bg_rect, 0.7, 0)

        cv2.putText(visualization, f"LiDAR Points: {visible_points}",
                    (15, 35), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Add a color legend with improved visibility
        legend_y = 70
        cv2.putText(visualization, "Distance:", (15, legend_y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.circle(visualization, (100, legend_y - 5), 5, (255, 0, 255), -1)  # Magenta
        cv2.putText(visualization, "<10m", (110, legend_y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        legend_y += 20
        cv2.circle(visualization, (100, legend_y - 5), 5, (255, 255, 0), -1)  # Cyan
        cv2.putText(visualization, "10-20m", (110, legend_y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        legend_y += 20
        cv2.circle(visualization, (100, legend_y - 5), 5, (0, 255, 0), -1)  # Bright green
        cv2.putText(visualization, "20-30m", (110, legend_y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        legend_y += 20
        cv2.circle(visualization, (100, legend_y - 5), 5, (255, 255, 255), -1)  # White
        cv2.putText(visualization, ">30m", (110, legend_y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Save visualization to file
        output_dir = self.output_dirs.get(camera_name)

        # Determine appropriate subfolder based on whether this is fusion or original
        if is_fusion:
            subfolder = os.path.join(output_dir, 'fusion_with_lidar')
        else:
            subfolder = os.path.join(output_dir, 'lidar_viz')

        os.makedirs(subfolder, exist_ok=True)

        # Save the visualization
        viz_path = os.path.join(subfolder, f"{frame_num:06d}.png")
        cv2.imwrite(viz_path, visualization)

    # Helper function for CARLA Depth Conversion (assuming saved with carla.ColorConverter.Depth)
    def carla_depth_to_meters(self, depth_img_array_uint8):
        """
        Converts a CARLA depth image (saved as PNG with Depth converter) to meters.
        Assumes input is a NumPy array with dtype uint8 and shape (H, W, 3) or (H, W, 4).
        Based on CARLA documentation: https://carla.readthedocs.io/en/latest/ref_sensors/#depth-camera
        """
        if depth_img_array_uint8.shape[-1] < 3:
            print("Warning: Expected 3 or 4 channel depth image for standard CARLA depth conversion.")
            # Fallback or error handling needed here - maybe assume grayscale direct depth?
            # For now, let's assume it *was* meant to be encoded and try using the first channel
            R = depth_img_array_uint8.astype(np.float32)
            G = depth_img_array_uint8.astype(np.float32)  # Repeat R if single channel
            B = depth_img_array_uint8.astype(np.float32)  # Repeat R if single channel
        else:
            R = depth_img_array_uint8[:, :, 0].astype(np.float32)
            G = depth_img_array_uint8[:, :, 1].astype(np.float32)
            B = depth_img_array_uint8[:, :, 2].astype(np.float32)

        # Formula: normalized = (R + G*256 + B*256*256) / (256*256*256 - 1)
        # Formula: depth_meters = 1000 * normalized
        normalized_depth = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)
        depth_in_meters = 1000.0 * normalized_depth
        return depth_in_meters

    # Helper function for CARLA Logarithmic Depth Conversion (if saved with LogarithmicDepth)
    # def carla_log_depth_to_meters(depth_img_array_uint8):
    #     """ Converts CARLA Logarithmic depth """
    #     # Assuming saved as grayscale PNG
    #     if len(depth_img_array_uint8.shape) > 2:
    #         gray_depth = depth_img_array_uint8[:,:,0] # Take one channel
    #     else:
    #         gray_depth = depth_img_array_uint8
    #
    #     # Formula from docs (approximate): Depth = 1000 * (DepthImage / 255.0)
    #     # This might need tuning based on actual far plane used during saving
    #     depth_in_meters = 1000.0 * (gray_depth.astype(np.float32) / 255.0)
    #     return depth_in_meters

    def perform_carla_sensor_fusion(self, camera_images):
        """
        Perform sensor fusion on multiple camera images.
        Returns a dictionary of fused images for each camera.
        """
        # Dictionary to store the fused outputs for each camera
        fused_outputs = {}

        # Make sure we have at least two cameras to fuse
        if len(camera_images) < 2:
            print("Not enough cameras for fusion")
            return fused_outputs

        # Process each camera
        for target_camera_name, target_image in camera_images.items():
            # Create a copy of the target image for fusion
            fused_image = target_image.copy()
            target_camera = self.sensors[target_camera_name]

            # Get the world transform for the target camera
            target_to_world = target_camera.get_transform().get_matrix()

            # Get the camera intrinsic matrix (K) for the target camera
            target_K = self.frame_counters[target_camera_name]['K']
            target_K_inv = np.linalg.inv(target_K)

            # Get image dimensions
            target_h, target_w = target_image.shape[:2]

            # Create a grid of pixel coordinates for the target image
            target_y, target_x = np.indices((target_h, target_w))
            target_pixels = np.stack((target_x.flatten(), target_y.flatten(), np.ones_like(target_x.flatten())), axis=0)

            # Project each source camera onto target camera
            for source_camera_name, source_image in camera_images.items():
                # Skip if source is the same as target
                if source_camera_name == target_camera_name:
                    continue

                source_camera = self.sensors[source_camera_name]
                source_to_world = source_camera.get_transform().get_matrix()
                world_to_source = np.linalg.inv(source_to_world)

                # Get camera intrinsics for source
                source_K = self.frame_counters[source_camera_name]['K']

                # Get image dimensions for source
                source_h, source_w = source_image.shape[:2]

                # Define the source camera in the world
                source_camera_in_world = source_to_world

                # Define the target camera in the world
                target_camera_in_world = target_to_world

                # Define the transformation from target to source camera
                target_to_source = np.dot(world_to_source, target_camera_in_world)

                # Transform the target pixel grid to camera rays
                rays = np.dot(target_K_inv, target_pixels)

                # We need to estimate the depth for each pixel
                # Let's use a constant depth for visualization
                # In a real-world scenario, you'd use a depth map from sensors
                depth = 50.0  # Arbitrary depth value to use for testing

                # Scale rays by depth
                rays_3d = np.multiply(rays, depth)

                # Add homogeneous coordinate
                rays_3d_homogeneous = np.vstack((rays_3d, np.ones((1, rays_3d.shape[1]))))

                # Transform rays from target camera to world
                rays_world = np.dot(target_camera_in_world, rays_3d_homogeneous)

                # Transform rays from world to source camera
                rays_source = np.dot(world_to_source, rays_world)

                # Project rays to source image space
                rays_source = rays_source[:3, :]
                rays_source_pixels = np.dot(source_K, rays_source)

                # Normalize homogeneous coordinates
                rays_source_pixels = rays_source_pixels / rays_source_pixels[2, :]

                # Reshape back to image dimensions
                source_x = rays_source_pixels[0, :].reshape(target_h, target_w)
                source_y = rays_source_pixels[1, :].reshape(target_h, target_w)

                # Create a mask for valid projection (inside source image bounds)
                valid_mask = (
                        (source_x >= 0) & (source_x < source_w) &
                        (source_y >= 0) & (source_y < source_h)
                )

                # Check if we have depth information from source camera
                if 'depth' in source_camera_name:
                    # Process depth image if available
                    pass  # Implement depth processing if needed

                # Sample source image at projected coordinates for valid pixels
                for y in range(target_h):
                    for x in range(target_w):
                        if valid_mask[y, x]:
                            # Get the source pixel coordinates
                            sx, sy = int(source_x[y, x]), int(source_y[y, x])

                            # Sample color from source image
                            source_color = source_image[sy, sx]

                            # Simple alpha blending for visualization
                            # Adjust alpha based on how central the pixel is in the source image
                            # This reduces the prominence of distortions at the edges
                            cx, cy = source_w // 2, source_h // 2
                            dx, dy = abs(sx - cx) / cx, abs(sy - cy) / cy
                            distance_to_center = np.sqrt(dx ** 2 + dy ** 2)
                            alpha = max(0, 1 - distance_to_center)

                            # Apply the blending
                            fused_image[y, x] = alpha * source_color + (1 - alpha) * fused_image[y, x]

            # Store the fused image for this camera
            fused_outputs[target_camera_name] = fused_image

            # Save the fused image
            output_dir = self.output_dirs.get(target_camera_name)
            if output_dir:
                fusion_dir = os.path.join(output_dir, 'fusion')
                os.makedirs(fusion_dir, exist_ok=True)

                camera_frame_num = self.frame_counters[target_camera_name]['counter'] - 1
                fusion_path = os.path.join(fusion_dir, f"{camera_frame_num:06d}.png")
                cv2.imwrite(fusion_path, fused_image)

        return fused_outputs

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
        Cleanup resources and create videos from saved images
        """
        print("Cleaning up resources...")

        # Destroy the sensors
        for sensor_name, sensor in self.sensors.items():
            try:
                sensor.destroy()
                print(f"Sensor {sensor_name} destroyed successfully")
            except Exception as e:
                print(f"Error destroying sensor {sensor_name}: {e}")

        # Destroy the ego vehicle
        try:
            self.vehicle.destroy()
            print("Ego vehicle destroyed successfully")
        except Exception as e:
            print(f"Error destroying ego vehicle: {e}")

        # Restore original world settings
        if self.original_settings:
            self.world.apply_settings(self.original_settings)
            print("Original world settings restored")

        # Create videos from saved images for all camera outputs and their subfolders
        for camera_name, output_dir in self.output_dirs.items():
            if 'camera' in camera_name:
                # Base camera output directory
                self.create_video_from_images(output_dir, f"{camera_name}_video.mp4")

                # Check for lidar visualization subfolder
                lidar_viz_dir = os.path.join(output_dir, 'lidar_viz')
                if os.path.exists(lidar_viz_dir) and os.path.isdir(lidar_viz_dir):
                    self.create_video_from_images(lidar_viz_dir, f"{camera_name}_lidar_viz_video.mp4")

                # Check for fusion with lidar subfolder
                fusion_lidar_dir = os.path.join(output_dir, 'fusion_with_lidar')
                if os.path.exists(fusion_lidar_dir) and os.path.isdir(fusion_lidar_dir):
                    self.create_video_from_images(fusion_lidar_dir, f"{camera_name}_fusion_with_lidar_video.mp4")

                # Check for normal fusion subfolder
                fusion_dir = os.path.join(output_dir, 'fusion')
                if os.path.exists(fusion_dir) and os.path.isdir(fusion_dir):
                    self.create_video_from_images(fusion_dir, f"{camera_name}_fusion_video.mp4")

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
