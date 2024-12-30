import sys
import struct
import os
import cv2
import rclpy
from rclpy.node import Node
import csv
from scipy.spatial.transform import Rotation
from cv_bridge import CvBridge
from sensor_msgs.msg import Imu, Image, PointCloud2, PointField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
import numpy as np
import time
from std_msgs.msg import Header
from tqdm import tqdm


def adjust_rgb_and_camera_matrix(rgb_image, depth_image, camera_matrix):
    """
    Adjust RGB image size to match Depth image and update camera matrix.

    Args:
        rgb_image (numpy.ndarray): Original RGB image.
        depth_image (numpy.ndarray): Depth image.
        camera_matrix (numpy.ndarray): Original camera intrinsic matrix.

    Returns:
        resized_rgb (numpy.ndarray): Resized RGB image.
        adjusted_camera_matrix (numpy.ndarray): Adjusted camera intrinsic matrix.
    """

    depth_h, depth_w = depth_image.shape[:2]
    rgb_h, rgb_w = rgb_image.shape[:2]

    # Calculate scale factors
    scale_x = depth_w / rgb_w
    scale_y = depth_h / rgb_h

    # Resize RGB image
    resized_rgb = cv2.resize(
        rgb_image, (depth_w, depth_h), interpolation=cv2.INTER_LINEAR
    )

    # Adjust camera matrix
    adjusted_camera_matrix = camera_matrix.copy()
    adjusted_camera_matrix[0, 0] *= scale_x  # fx
    adjusted_camera_matrix[1, 1] *= scale_y  # fy
    adjusted_camera_matrix[0, 2] *= scale_x  # cx
    adjusted_camera_matrix[1, 2] *= scale_y  # cy

    return resized_rgb, adjusted_camera_matrix


def create_pointcloud2(points_with_rgb):
    """
    Create a PointCloud2 message from 3D points with RGB.

    Args:
        points_with_rgb (numpy.ndarray): Nx6 array of 3D points with RGB (x, y, z, r, g, b).

    Returns:
        pointcloud_msg (PointCloud2): ROS 2 PointCloud2 message.
    """
    pointcloud_msg = PointCloud2()

    # Header
    pointcloud_msg.header.frame_id = "camera_depth_frame"

    # Define fields for PointCloud2
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
    ]
    pointcloud_msg.fields = fields

    # Calculate point step and row step
    pointcloud_msg.point_step = 16  # 4 bytes (float32) * 3 + 4 bytes (uint32)
    pointcloud_msg.row_step = pointcloud_msg.point_step * len(points_with_rgb)

    # Convert points to byte data
    buffer = []
    for point in points_with_rgb:
        x, y, z, r, g, b = point
        # Ensure RGB values are in 0-255 range and convert to uint32
        r, g, b = int(r), int(g), int(b)
        rgb = struct.unpack("I", struct.pack("BBBB", b, g, r, 0))[0]
        buffer.append(struct.pack("fffI", x, y, z, rgb))

    pointcloud_msg.data = b"".join(buffer)
    pointcloud_msg.is_bigendian = False
    pointcloud_msg.height = 1
    pointcloud_msg.width = len(points_with_rgb)
    pointcloud_msg.is_dense = True

    return pointcloud_msg


def unproject_depth(depth_image, rgb_image, camera_matrix, mask=None):
    """
    Generate a 3D point cloud with color from a depth image and camera matrix.
    Only include points where mask is True.

    Args:
        depth_image (numpy.ndarray): Depth image in mm (uint16).
        rgb_image (numpy.ndarray): RGB image matching the depth image size.
        camera_matrix (numpy.ndarray): Camera intrinsic matrix.
        mask (numpy.ndarray, optional): Boolean mask to filter points.

    Returns:
        point_cloud (numpy.ndarray): Nx6 array of 3D points with RGB colors (x, y, z, r, g, b).
    """
    # Get camera intrinsics
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    # Create a grid of pixel coordinates
    depth_h, depth_w = depth_image.shape
    u, v = np.meshgrid(np.arange(depth_w), np.arange(depth_h))
    u = u.flatten()
    v = v.flatten()

    # Flatten depth image and filter valid depths
    z = depth_image.flatten().astype(np.float32) / 1000.0  # Convert mm to meters
    valid = z > 0  # Ignore invalid depth values

    if mask is not None:
        mask = mask.flatten().astype(bool)
        valid = valid & mask

    z = z[valid]
    u = u[valid]
    v = v[valid]

    # Backproject to 3D
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points_3d = np.stack((x, y, z), axis=-1)  # Shape: (N, 3)

    # Add color from RGB image
    rgb_flat = rgb_image.reshape(-1, 3)  # Flatten RGB image
    colors = rgb_flat[valid]  # Get colors for valid depth points

    # Combine points and colors
    return np.concatenate((points_3d, colors), axis=-1)  # Shape: (N, 6)


def read_csv(file_path):
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    return data

def skew_symmetric(v):
    """
    Creates a skew-symmetric matrix from a vector.
    """
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def rotate_to_earth(points_with_rgb):
    """
    Preprocess the point cloud by downsampling and transforming coordinates.

    Args:
        points_with_rgb (np.ndarray): Input point cloud with RGB, shape (N, 6).
        voxel_size (float): The size of the voxel grid for downsampling in meters.

    Returns:
        np.ndarray: Preprocessed point cloud with RGB, shape (M, 6).
    """
    if points_with_rgb.size == 0:
        return points_with_rgb

    # Separate XYZ and RGB
    points_xyz = points_with_rgb[:, :3]
    points_rgb = points_with_rgb[:, 3:]

    # Coordinate Transformation
    # Define rotation matrix to convert from camera to base coordinates
    # Camera (x_right, y_down, z_forward) -> Base (x_forward, y_left, z_up)
    R = np.array([
        [0,  0, -1],
        [1, 0, 0],
        [0, -1, 0]
    ])

    # Apply rotation
    transformed_xyz = points_xyz @ R

    # Combine transformed XYZ with RGB
    preprocessed_points = np.hstack((transformed_xyz, points_rgb))

    return preprocessed_points

def combine_and_sort_data(imu_data, odometry_data, rgb_data, depth_data, k=1):
    """
    Combine and sort sensor data, skipping k-1 entries for each type.

    Args:
        imu_data (list): IMU data from CSV.
        odometry_data (list): Odometry data from CSV.
        rgb_data (list): RGB image data.
        depth_data (list): Depth image data.
        k (int): Step size for skipping data. Default is 1 (no skipping).

    Returns:
        list: Combined and sorted data.
    """
    combined_data = []

    # Combine IMU data with skipping
    for i, row in enumerate(imu_data[1:]):  # Skip header
        if i % k == 0:
            combined_data.append({"timestamp": float(row[0]), "type": "imu", "data": row})

    # Combine Odometry data with skipping
    for i, row in enumerate(odometry_data[1:]):  # Skip header
        if i % k == 0:
            combined_data.append(
                {"timestamp": float(row[0]), "type": "odometry", "data": row}
            )

    # Combine RGB data with skipping
    for i, image_info in enumerate(rgb_data):
        if i % k == 0:
            combined_data.append(
                {"timestamp": image_info["timestamp"], "type": "rgb", "data": image_info}
            )

    # Combine Depth data with skipping
    for i, image_info in enumerate(depth_data):
        if i % k == 0:
            combined_data.append(
                {"timestamp": image_info["timestamp"], "type": "depth", "data": image_info}
            )

    # Sort by timestamp
    combined_data.sort(key=lambda x: x["timestamp"])
    return combined_data


class StrayScannerDataPublisher(Node):
    def __init__(self, data_dir):
        super().__init__("csv_and_video_publisher")

        # Publishers
        self.imu_pub = self.create_publisher(Imu, "/imu", 100)
        self.odometry_pub = self.create_publisher(Odometry, "/odometry", 100)
        self.rgb_pub = self.create_publisher(Image, "/camera/rgb", 100)
        self.depth_pub = self.create_publisher(Image, "/camera/depth", 100)
        self.pointcloud_pub_body = self.create_publisher(PointCloud2, "/pointcloud_body", 100)
        self.pointcloud_pub_world = self.create_publisher(PointCloud2, "/pointcloud_world", 100)

        # Load CSV data
        imu_data = read_csv(f"{data_dir}/imu.csv")[1:]  # Skip header
        odometry_data = read_csv(f"{data_dir}/odometry.csv")[1:]  # Skip header

        # Prepare Depth data with Confidence
        self.depth_dir = os.path.join(data_dir, "depth")
        depth_data = self.prepare_image_paths(self.depth_dir, odometry_data)
        self.num_frames = len(depth_data)

        # Convert the video to images 
        video_path = f"{data_dir}/rgb.mp4"
        print(f"video_path: {video_path}")
        self.rgb_dir = os.path.join(data_dir, "images")
        # Check if rgb_dir exists
        if os.path.exists(self.rgb_dir) and len(os.listdir(self.rgb_dir)) == self.num_frames:
            print(f"{self.rgb_dir} already exists. Skipping save_frames.")
        else:
            os.makedirs(self.rgb_dir, exist_ok=True) 
            self.save_frames(video_path)

        # Prepare RGB data
        rgb_data = self.prepare_image_paths(self.rgb_dir, odometry_data)

        # Prepare intrinsic matrix
        camera_matrix_csv = f"{data_dir}/camera_matrix.csv"
        if os.path.exists(camera_matrix_csv):
            camera_matrix_data = read_csv(camera_matrix_csv)
            self.rgb_intrinsic_matrix = np.array(camera_matrix_data, dtype=float)
        else:
            self.get_logger().error(f"Camera matrix file not found: {camera_matrix_csv}")
            raise FileNotFoundError(f"Camera matrix file not found: {camera_matrix_csv}")

        # pub option
        self.skip_frame_k = 1

        self.pose_matrix = None 

        # Combine and sort all data
        self.sorted_data = combine_and_sort_data(
            imu_data, odometry_data, rgb_data, depth_data, self.skip_frame_k
        )
        self.current_index = 0
        self.bridge = CvBridge()

        # Initialize progress bar
        self.progress_bar = tqdm(total=len(self.sorted_data), desc="Publishing data")

        # Start publishing
        self.start_time = time.time()
        self.initial_timestamp = self.sorted_data[0]["timestamp"]
        self.timer = self.create_timer(0.001, self.publish_data)  # High frequency timer

    def save_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        rgb_data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_path = os.path.join(self.rgb_dir, f"{frame_idx:06d}.jpg")
            cv2.imwrite(image_path, frame)

            rgb_data.append(image_path)
            frame_idx += 1
        
        cap.release()
        self.num_frames = len(rgb_data)
        print(f"num_frames: {self.num_frames}")
        return rgb_data

    def prepare_image_paths(self, image_dir, odometry_data):
        image_data = []
        for row in odometry_data:
            frame_id = int(row[1])
            timestamp = float(row[0])
            if "depth" in image_dir:
                image_path = os.path.join(
                    image_dir, f"{frame_id:06d}.png"
                )
                confidence_dir = image_dir.replace("depth", "confidence")
                confidence_path = os.path.join(confidence_dir, f"{frame_id:06d}.png")
                if os.path.exists(image_path) and os.path.exists(confidence_path):
                    image_data.append({
                        "timestamp": timestamp,
                        "depth_path": image_path,
                        "confidence_path": confidence_path
                    })
            elif "confidence" in image_dir:
                # Confidence images are handled alongside depth images
                pass
            else:
                # For RGB images
                image_path = os.path.join(
                    image_dir, f"{frame_id:06d}.jpg"
                )
                if os.path.exists(image_path):
                    image_data.append({"timestamp": timestamp, "path": image_path})
        return image_data


    def publish_data(self):
        if self.current_index >= len(self.sorted_data):
            self.progress_bar.close()  # Close progress bar
            self.get_logger().info("All data published.")
            self.destroy_timer(self.timer)
            return

        current_time = time.time()
        elapsed_time = current_time - self.start_time

        while (
            self.current_index < len(self.sorted_data)
            and self.sorted_data[self.current_index]["timestamp"]
            - self.initial_timestamp
            <= elapsed_time
        ):
            entry = self.sorted_data[self.current_index]
            timestamp = entry["timestamp"]

            if entry["type"] == "imu":
                # IMU Publishing
                imu_row = entry["data"]
                imu_msg = Imu()
                imu_msg.header.stamp = rclpy.time.Time(seconds=timestamp).to_msg()
                imu_msg.header.frame_id = "imu_frame"
                imu_msg.linear_acceleration.x = float(imu_row[1])
                imu_msg.linear_acceleration.y = float(imu_row[2])
                imu_msg.linear_acceleration.z = float(imu_row[3])
                imu_msg.angular_velocity.x = float(imu_row[4])
                imu_msg.angular_velocity.y = float(imu_row[5])
                imu_msg.angular_velocity.z = float(imu_row[6])
                self.imu_pub.publish(imu_msg)

            elif entry["type"] == "odometry":
                # Odometry Publishing
                odom_row = entry["data"]
                odom_msg = Odometry()
                odom_msg.header.stamp = rclpy.time.Time(seconds=timestamp).to_msg()
                odom_msg.header.frame_id = "odom_frame"
                odom_msg.child_frame_id = "base_link"
                odom_msg.pose.pose.position.x = float(odom_row[2])
                odom_msg.pose.pose.position.y = float(odom_row[3])
                odom_msg.pose.pose.position.z = float(odom_row[4])
                odom_msg.pose.pose.orientation = Quaternion(
                    x=float(odom_row[5]),
                    y=float(odom_row[6]),
                    z=float(odom_row[7]),
                    w=float(odom_row[8]),
                )

                # Extract position and orientation from odom_row
                position = np.array([float(odom_row[2]), float(odom_row[3]), float(odom_row[4])])
                orientation = [float(odom_row[5]), float(odom_row[6]), float(odom_row[7]), float(odom_row[8])]

                # Convert quaternion to rotation matrix using scipy
                rotation_matrix = Rotation.from_quat(orientation).as_matrix()

                # Construct SE(3) transformation matrix
                pose_matrix = np.eye(4)  # Initialize 4x4 identity matrix
                pose_matrix[:3, :3] = rotation_matrix  # Top-left 3x3 is the rotation matrix
                pose_matrix[:3, 3] = position  # Top-right 3x1 is the translation vector
                self.pose_matrix = pose_matrix # update 

                # Publish the updated odometry message
                self.odometry_pub.publish(odom_msg)

            elif entry["type"] == "rgb":
                # RGB Publishing
                image_info = entry["data"]
                bgr_img = cv2.imread(image_info["path"])
                if bgr_img is not None:
                    img_msg = self.bridge.cv2_to_imgmsg(bgr_img, encoding="bgr8")
                    img_msg.header.stamp = rclpy.time.Time(seconds=timestamp).to_msg()
                    img_msg.header.frame_id = "camera_rgb_frame"
                    self.rgb_pub.publish(img_msg)

                rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

            elif entry["type"] == "depth" and (self.pose_matrix is not None):
                # Depth and Confidence Handling
                image_info = entry["data"]
                depth_img = cv2.imread(image_info["depth_path"], cv2.IMREAD_UNCHANGED)
                confidence_img = cv2.imread(image_info["confidence_path"], cv2.IMREAD_UNCHANGED) # 0, 1, 2

                if depth_img is not None and depth_img.dtype == np.uint16 and confidence_img is not None:
                    # Create mask to filter out noisy points (== 0)
                    mask = (confidence_img == 1) | (confidence_img == 2)

                    # Publish depth image
                    depth_msg = self.bridge.cv2_to_imgmsg(depth_img, encoding="mono16")
                    depth_msg.header.stamp = rclpy.time.Time(seconds=timestamp).to_msg()
                    depth_msg.header.frame_id = "camera_depth_frame"
                    self.depth_pub.publish(depth_msg)

                    # Adjust RGB image and camera matrix
                    resized_rgb, adjusted_camera_matrix = adjust_rgb_and_camera_matrix(
                        rgb_img, depth_img, self.rgb_intrinsic_matrix
                    )
                    points_with_rgb = unproject_depth(
                        depth_img, resized_rgb, adjusted_camera_matrix, mask=mask
                    )

                    if points_with_rgb.size > 0:
                        # Convert points to PointCloud2 message
                        pointcloud_msg = create_pointcloud2(points_with_rgb)
                        pointcloud_msg.header.stamp = depth_msg.header.stamp  # Sync with depth image
                        self.pointcloud_pub_body.publish(pointcloud_msg)

                        # using the frame pose, pub within world 
                        # Convert points to PointCloud2 message
                        points_with_rgb_world = points_with_rgb.copy()

                        # body to world 
                        points_xyz = points_with_rgb_world[:, :3] 
                        points_xyz_local_homg = np.hstack((points_xyz, np.ones((points_xyz.shape[0], 1))))
                        points_xyz_world_homg = (self.pose_matrix @ points_xyz_local_homg.T).T

                        # let world cloud parr to the z=0 ground 
                        T_cam_to_FLU = np.array([[0, 0, -1, 0], 
                                                [1, 0, 0, 0], 
                                                [0, -1, 0, 0], 
                                                [0, 0, 0, 1]])
                        points_xyz_world_homg = (T_cam_to_FLU @ points_xyz_world_homg.T).T

                        # reset with the transformed cloud 
                        points_with_rgb_world[:, :3]  = points_xyz_world_homg[:, :3]

                        # for the lightweight visualization 
                        world_cloud_skip_k = 100
                        points_with_rgb_world_skip_k = points_with_rgb_world[::world_cloud_skip_k]

                        pointcloud_world_msg = create_pointcloud2(points_with_rgb_world_skip_k)
                        pointcloud_world_msg.header.stamp = depth_msg.header.stamp  # Sync with depth image
                        self.pointcloud_pub_world.publish(pointcloud_world_msg)

                    else:
                        self.get_logger().info(
                            f"No valid points with confidence=0 in frame {self.current_index}."
                        )
                else:
                    self.get_logger().warn(
                        f"Depth or confidence image missing or invalid for frame {self.current_index}."
                    )

            # Update progress bar
            self.progress_bar.update(1)

            self.current_index += 1


def main(args=None):
    rclpy.init(args=args)

    if len(sys.argv) < 2:
        print("Usage: python3 publish_ros2_msgs.py <data_directory>")
        rclpy.shutdown()
        return

    data_dir = sys.argv[1]
    node = StrayScannerDataPublisher(data_dir)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
