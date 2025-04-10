import argparse
import glob
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from queue import Empty, Queue

import carla
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
        'position': (1.8, -0.5, 0.7),
        'rotation': (0.0, 0.0, 0.0)
    },
    {
        'type': 'sensor.camera.depth',
        'name': 'front_depth_camera',
        'attributes': {
            'image_size_x': '1280',
            'image_size_y': '720',
            'fov': '90'
        },
        'position': (1.8, 0.5, 0.7),
        'rotation': (0.0, 0.0, 0.0),
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
        'position': (0.0, 0.0, 2.0),
        'rotation': (0.0, 0.0, 0.0),
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

        print(config)

        # Use position from config if available, otherwise fall back to provided or default
        if position is None:
            position = (1.6, 0.0, 1.6)
        print(position)

        # Use rotation from config if available, otherwise fall back to provided or default
        if rotation is None:
            rotation = (0.0, 0.0, 0.0)
            print(rotation)

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

                # Create output directory if it doesn't exist
                if camera_name not in self.output_dirs:
                    # Main directory for fused images (keep original behavior)
                    fused_output_dir = os.path.join(args.save_dir, camera_name)
                    os.makedirs(fused_output_dir, exist_ok=True)
                    self.output_dirs[camera_name] = fused_output_dir

                    # Create a separate directory for original images
                    orig_output_dir = os.path.join(args.save_dir, f"{camera_name}_original")
                    os.makedirs(orig_output_dir, exist_ok=True)

                # Skip if projections not available
                if camera_name not in projections:
                    continue

                # Get projection data
                proj_data = projections[camera_name]
                points_2d = proj_data['points_2d']
                intensity = proj_data['intensity']

                # Get distances if available (or calculate them)
                if 'distances' in proj_data:
                    distances = proj_data['distances']
                else:
                    # If distances aren't available, use intensity as a proxy
                    distances = intensity

                # Create image from raw data
                im_array = np.copy(np.frombuffer(camera_data.raw_data, dtype=np.dtype("uint8")))
                im_array = np.reshape(im_array, (camera_data.height, camera_data.width, 4))
                im_array = im_array[:, :, :3][:, :, ::-1]  # BGRA to RGB

                frame_count = self.frame_counters[camera_name]['counter']
                orig_file_name = f"{frame_count:06d}.png"
                orig_file_path = os.path.join(args.save_dir, f"{camera_name}_original", orig_file_name)

                # Save original image using PIL
                original_image = Image.fromarray(im_array)
                original_image.save(orig_file_path)

                # Create a separate overlay for LiDAR points for better visibility
                lidar_overlay = np.zeros_like(im_array)

                # Extract screen coordinates
                u_coord = points_2d[:, 0].astype(np.int_)
                v_coord = points_2d[:, 1].astype(np.int_)

                # Create color mapping for points with enhanced vibrancy
                intensity = 4 * intensity - 3
                color_map = np.array([
                    np.interp(intensity, self.VID_RANGE, self.VIRIDIS[:, 0]) * 255.0,
                    np.interp(intensity, self.VID_RANGE, self.VIRIDIS[:, 1]) * 255.0,
                    np.interp(intensity, self.VID_RANGE, self.VIRIDIS[:, 2]) * 255.0
                ]).astype(np.int_).T

                # ENHANCED VISUALIZATION: Larger points with outlines for better visibility
                lidar_point_radius = 3  # Increased from default dot_extent

                # Base radius on args if available, but ensure minimum visibility
                if hasattr(args, 'dot_extent'):
                    lidar_point_radius = max(3, args.dot_extent)  # Use at least 3 for visibility

                # Draw the 2D points on the overlay with enhanced visualization
                for i in range(len(points_2d)):
                    if 0 <= u_coord[i] < camera_data.width and 0 <= v_coord[i] < camera_data.height:
                        # Draw outline (black) for better contrast against any background
                        point_size = int(
                            lidar_point_radius * (1.5 - 0.5 * distances[i]))  # Vary between 50-150% of base size
                        cv2.circle(lidar_overlay, (u_coord[i], v_coord[i]),
                                   point_size + 1, (0, 0, 0), -1)  # Outline
                        cv2.circle(lidar_overlay, (u_coord[i], v_coord[i]),
                                   point_size, color_map[i].tolist(), -1)  # Colored point

                # Blend the lidar overlay with the camera image
                # Adjust alpha values to control LiDAR prominence (0.7 and 0.4 in this example)
                # Higher alpha for lidar_overlay makes points more visible
                blended_image = cv2.addWeighted(im_array, 0.7, lidar_overlay, 0.4, 0)

                # Save the frame as an image
                frame_count = self.frame_counters[camera_name]['counter']
                file_name = f"{frame_count:06d}.png"
                file_path = os.path.join(self.output_dirs[camera_name], file_name)

                # Save image using PIL
                image = Image.fromarray(blended_image)
                image.save(file_path)

                # Increment frame counter
                self.frame_counters[camera_name]['counter'] += 1

            except Empty:
                print(f"[Warning] Camera {camera_name} data was missed")
                continue

        # Increment lidar frame counter
        self.frame_counters[lidar_name]['counter'] += 1

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


    def perform_carla_sensor_fusion(self):
        """
        Use CARLA's built-in capabilities and correct transformations to perform
        sensor fusion between RGB and Depth cameras.
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

        # Get the frame number - IMPORTANT: Assumes sensors are synchronized!
        # If not running in synchronous mode, this can lead to mismatches.
        frame_num = max(
            counter_data['counter'] for sensor_name, counter_data in self.frame_counters.items()
            if 'counter' in counter_data
        ) - 1 if self.frame_counters else 0

        # Get main reference sensor (e.g., RGB camera)
        reference_sensor_name = next((name for name in camera_sensors if 'rgb' in name or 'front_camera' in name), None)
        if not reference_sensor_name:
            print("Warning: Could not find a primary RGB camera ('rgb' or 'front_camera'). Using first available.")
            reference_sensor_name = next(iter(camera_sensors))  # Fallback

        reference_sensor = self.sensors[reference_sensor_name]
        reference_transform = reference_sensor.get_transform()  # World transform of RGB cam

        # Get reference image
        ref_img_path = os.path.join(
            self.output_dirs.get(reference_sensor_name,
                                 os.path.join(self.output_dirs.get('base', 'sensor_output'), reference_sensor_name)),
            f"{self.frame_counters[reference_sensor_name]['counter'] - 1:06d}.png"
        )
        if not os.path.exists(ref_img_path):
            print(f"Reference image {ref_img_path} not found for frame {frame_num}")
            return

        reference_img = cv2.imread(ref_img_path)
        if reference_img is None:
            print(f"Error reading reference image {ref_img_path}")
            return
        rgb_height, rgb_width, _ = reference_img.shape

        # --- Find and Process Depth Sensor ---
        depth_sensor_name = next((name for name in camera_sensors if 'depth' in name), None)
        if not depth_sensor_name:
            print("No depth sensor found for fusion.")
            return

        depth_sensor = self.sensors[depth_sensor_name]
        depth_transform = depth_sensor.get_transform()  # World transform of Depth cam

        # Get depth image path
        depth_img_path = os.path.join(
            self.output_dirs.get(depth_sensor_name + "_original",
                                 os.path.join(self.output_dirs.get('base', 'sensor_output'),
                                              depth_sensor_name + "_original")),
            f"{self.frame_counters[depth_sensor_name]['counter'] - 1:06d}.png"
        )

        if not os.path.exists(depth_img_path):
            print(f"Depth image {depth_img_path} not found for frame {frame_num}")
            return

        # --- Correct Depth Loading and Conversion ---
        try:
            # Load using PIL to handle different PNG types, then convert to NumPy
            depth_img_pil = Image.open(depth_img_path)
            depth_img_raw = np.array(depth_img_pil, dtype=np.uint8)  # Assume uint8 PNG output

            print(f"Loaded Depth image shape: {depth_img_raw.shape}, dtype: {depth_img_raw.dtype}")

            # *** CRITICAL STEP: Choose the correct conversion based on how depth was saved ***
            # Option 1: Assume saved using carla.ColorConverter.Depth (Encodes depth in RGB channels)
            depth_in_meters = self.carla_depth_to_meters(depth_img_raw)


        except Exception as e:
            print(f"Error processing depth image {depth_img_path}: {e}")
            return

        depth_height, depth_width = depth_in_meters.shape  # Should now be 2D

        # --- Calculate Depth Camera Intrinsics ---
        try:
            depth_fov = float(depth_sensor.attributes['fov'])
        except KeyError:
            print("Warning: Depth sensor 'fov' attribute missing. Assuming 90.")
            depth_fov = 90.0
        depth_focal = depth_width / (2.0 * math.tan(math.radians(depth_fov) / 2.0))
        depth_cx = depth_width / 2.0
        depth_cy = depth_height / 2.0

        # --- Generate Point Cloud in World Coordinates ---
        world_points = []
        subsample = 5  # Process every 5th pixel for efficiency
        for y in range(0, depth_height, subsample):
            for x in range(0, depth_width, subsample):
                depth_value = depth_in_meters[y, x]

                # Skip invalid depth points (e.g., sky, far distance)
                # CARLA depth is typically max 1000m, but check your sensor's range attribute
                if depth_value <= 0 or depth_value >= 1000.0:
                    continue

                # --- Inverse Projection: 2D Pixel (Depth Cam) -> 3D Point (Depth Cam Local) ---
                # CARLA Camera Local Coords: X=forward, Y=right, Z=up
                # Image Coords: x increases right, y increases down
                # Formulas derived from pinhole model:
                # x_img = cx + f * (Y_cam / X_cam)  => Y_cam = X_cam * (x_img - cx) / f
                # y_img = cy - f * (Z_cam / X_cam)  => Z_cam = -X_cam * (y_img - cy) / f
                x_cam = float(depth_value)  # X_cam is the depth value
                y_cam = x_cam * (x - depth_cx) / depth_focal  # Y_cam is right
                z_cam = -x_cam * (y - depth_cy) / depth_focal  # Z_cam is up (y increases down)
                point_in_cam_local = carla.Location(x=x_cam, y=y_cam, z=z_cam)

                # --- Transform Point from Depth Camera Local Space to World Space ---
                # Uses the depth sensor's world transform
                world_point = depth_transform.transform(point_in_cam_local)
                world_points.append(world_point)

        print(f"Generated {len(world_points)} points from depth map (subsampled).")

        # --- Project World Points onto Reference (RGB) Camera Image ---
        fused_img = reference_img.copy()

        # --- Calculate RGB Camera Intrinsics ---
        try:
            rgb_fov = float(reference_sensor.attributes['fov'])
        except KeyError:
            print("Warning: RGB sensor 'fov' attribute missing. Assuming 90.")
            rgb_fov = 90.0
        rgb_focal = rgb_width / (2.0 * math.tan(math.radians(rgb_fov) / 2.0))
        rgb_cx = rgb_width / 2.0
        rgb_cy = rgb_height / 2.0

        # --- Get the Transformation Matrix from World to RGB Camera Local Space ---
        # We need the inverse of the RGB camera's world transform
        world_to_rgb_cam_matrix = np.array(reference_transform.get_inverse_matrix())

        point_count = 0
        visible_count = 0

        for world_point in world_points:
            point_count += 1

            # --- Transform World Point to RGB Camera Local Coordinates ---
            # Convert carla.Location to homogeneous coordinates [x, y, z, 1] for matrix multiplication
            wp = np.array([world_point.x, world_point.y, world_point.z, 1.0])
            # Apply the inverse transform matrix
            point_in_rgb_cam_h = world_to_rgb_cam_matrix @ wp
            # Result is [X_cam, Y_cam, Z_cam, 1] in the RGB camera's local frame

            # Get non-homogeneous coords (X=forward, Y=right, Z=up)
            x_rgb_cam = point_in_rgb_cam_h[0]
            y_rgb_cam = point_in_rgb_cam_h[1]
            z_rgb_cam = point_in_rgb_cam_h[2]

            # --- Forward Projection: 3D Point (RGB Cam Local) -> 2D Pixel (RGB Image) ---
            # Check if point is behind the camera (X_cam <= 0)
            if x_rgb_cam <= 1e-6:  # Add small epsilon for stability
                continue

            # Apply pinhole projection formulas (consistent with inverse projection derivation)
            # x_img = cx + f * (Y_cam / X_cam)
            # y_img = cy - f * (Z_cam / X_cam)
            image_x = rgb_cx + rgb_focal * (y_rgb_cam / x_rgb_cam)
            image_y = rgb_cy - rgb_focal * (z_rgb_cam / x_rgb_cam)

            # Convert to integer pixel coordinates
            px = int(round(image_x))
            py = int(round(image_y))

            depth_values = np.zeros((rgb_height, rgb_width), dtype=np.float32)
            point_counts = np.zeros((rgb_height, rgb_width), dtype=np.uint8)

            # Check if the projected point is within the image bounds
            if 0 <= px < rgb_width and 0 <= py < rgb_height:
                visible_count += 1

                # Use normalized depth for RGB gradient coloring
                depth = x_rgb_cam  # This is the actual distance along the camera's forward axis
                depth_normalized = min(depth / 100.0, 1.0)  # Normalize depth (capping at 100m)

                # Define RGB gradient (from red to green to blue)
                if depth_normalized < 0.5:
                    r = int(255 * (1 - 2 * depth_normalized))  # Red decreases
                    g = int(255 * (2 * depth_normalized))  # Green increases
                    b = 0
                else:
                    r = 0
                    g = int(255 * (2 - 2 * depth_normalized))  # Green decreases
                    b = int(255 * (2 * depth_normalized - 1))  # Blue increases
                color = (b, g, r)  # OpenCV uses BGR format

                # Draw the projected point (use cv2.circle for better visibility)
                cv2.circle(fused_img, (px, py), radius=2, color=color, thickness=-1)  # Filled circle

        # --- Save Results ---
        print(f"Projected {point_count} points, {visible_count} visible in RGB frame.")

        fusion_path = os.path.join(fusion_dir, f"carla_fusion_{frame_num:06d}.png")
        cv2.imwrite(fusion_path, fused_img)

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
            position = sensor_config.get('position', None)
            rotation = sensor_config.get('rotation', None)

            # Add the sensor
            self.add_sensor(sensor_type, sensor_name, config=sensor_attributes, position=position, rotation=rotation)

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
