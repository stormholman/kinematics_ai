#!/usr/bin/env python3
"""
Standalone ArUco Marker Detector and Reference Frame Tracker
Reads RGB data from RGBD camera and detects ArUco markers as reference frames
"""

import cv2
import numpy as np
from record3d import Record3DStream
from threading import Event
from typing import Optional, Tuple, Dict, Any
import json

class ArucoTransformer:
    """
    Transforms coordinates from camera frame to ArUco marker frame
    """
    
    def __init__(self, camera_matrix: np.ndarray = None, dist_coeffs: np.ndarray = None, marker_size: float = 0.04):
        """
        Initialize ArUco transformer
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Camera distortion coefficients
            marker_size: Physical size of ArUco marker in meters (default 5cm)
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_size = marker_size
        
        # Initialize ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Optimize parameters for better detection
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 23
        self.aruco_params.adaptiveThreshWinSizeStep = 10
        self.aruco_params.adaptiveThreshConstant = 7
        self.aruco_params.minMarkerPerimeterRate = 0.1
        self.aruco_params.maxMarkerPerimeterRate = 0.8
        self.aruco_params.polygonalApproxAccuracyRate = 0.05
        self.aruco_params.minCornerDistanceRate = 0.1
        self.aruco_params.minDistanceToBorder = 3
        self.aruco_params.minOtsuStdDev = 5.0
        self.aruco_params.perspectiveRemovePixelPerCell = 4
        self.aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.13
        self.aruco_params.maxErroneousBitsInBorderRate = 0.2
        self.aruco_params.minMarkerDistanceRate = 0.05
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementWinSize = 5
        self.aruco_params.cornerRefinementMaxIterations = 30
        self.aruco_params.cornerRefinementMinAccuracy = 0.1
        
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Default camera parameters (will be updated from RGBD stream)
        if self.camera_matrix is None:
            self.camera_matrix = np.array([[800, 0, 320],
                                         [0, 800, 240],
                                         [0, 0, 1]], dtype=np.float32)
        
        if self.dist_coeffs is None:
            self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    
    def detect_aruco_marker(self, rgb_image: np.ndarray, target_id: int = 0) -> Optional[Dict[str, Any]]:
        """
        Detect ArUco marker and estimate its pose
        
        Args:
            rgb_image: RGB image from camera
            target_id: ArUco marker ID to detect (default 0)
            
        Returns:
            Dictionary with marker pose information or None if not found
        """
        try:
            # Convert to grayscale for ArUco detection
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            
            # Detect markers
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
            
            if ids is None or len(ids) == 0:
                return None
            
            # Find target marker ID
            target_idx = None
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == target_id:
                    target_idx = i
                    break
            
            if target_idx is None:
                return None
            
            # Create 3D object points for ArUco marker corners
            # OpenCV ArUco detection returns corners in clockwise order starting from top-left:
            # Corner 0: top-left, Corner 1: top-right, Corner 2: bottom-right, Corner 3: bottom-left
            # Standard ArUco coordinate system: origin at first detected corner (top-left)
            # +X axis points from corner 0 to corner 1 (rightward along top edge)
            # +Y axis points from corner 0 to corner 3 (downward along left edge)
            # +Z axis points out of marker plane toward camera
            marker_corners_3d = np.array([
                [0, 0, 0],                                    # corner 0: top-left (origin)
                [self.marker_size, 0, 0],                     # corner 1: top-right (+X direction)
                [self.marker_size, self.marker_size, 0],      # corner 2: bottom-right (+X,+Y)
                [0, self.marker_size, 0]                      # corner 3: bottom-left (+Y direction)
            ], dtype=np.float32)
            
            # Estimate pose using solvePnP
            success, rvec, tvec = cv2.solvePnP(
                marker_corners_3d, 
                corners[target_idx][0], 
                self.camera_matrix, 
                self.dist_coeffs
            )
            
            if not success:
                return None
            
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec.flatten())
            
            # Create 4x4 transformation matrix (camera to marker)
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = tvec.flatten()
            
            result = {
                'marker_id': target_id,
                'found': True,
                'position': {
                    'x': float(tvec.flatten()[0]),
                    'y': float(tvec.flatten()[1]),
                    'z': float(tvec.flatten()[2])
                },
                'transformation_matrix': transformation_matrix.tolist(),
                'corners': corners[target_idx][0].tolist(),
                'rotation_matrix': rotation_matrix.tolist(),
                'rotation_vector': rvec.flatten().tolist()
            }
            
            return result
            
        except Exception as e:
            print(f"[ArUco] Error detecting marker: {e}")
            return None
    
    def get_target_relative_to_aruco(self, rgb_image: np.ndarray, target_3d_camera: np.ndarray, aruco_id: int = 0) -> Optional[Dict[str, Any]]:
        """
        Get target object position relative to ArUco marker
        
        Args:
            rgb_image: RGB image from camera
            target_3d_camera: 3D point in camera coordinates [x, y, z]
            aruco_id: ArUco marker ID to use as reference (default 0)
            
        Returns:
            Dictionary with relative position information or None if failed
        """
        try:
            # Detect ArUco marker
            aruco_pose = self.detect_aruco_marker(rgb_image, aruco_id)
            if aruco_pose is None:
                return None
            
            # Transform target point to ArUco frame
            target_3d_aruco = self.transform_to_aruco_frame(target_3d_camera, aruco_pose)
            if target_3d_aruco is None:
                return None
            
            # Prepare result
            result = {
                'success': True,
                'aruco_marker': aruco_pose,
                'target_position_camera': target_3d_camera.tolist(),
                'target_position_aruco': target_3d_aruco.tolist(),
                'relative_position': {
                    'x': float(target_3d_aruco[0]),
                    'y': float(target_3d_aruco[1]),
                    'z': float(target_3d_aruco[2])
                }
            }
            
            return result
            
        except Exception as e:
            print(f"[ArUco] Error getting relative position: {e}")
            return None
    
    def transform_to_aruco_frame(self, point_camera_frame: np.ndarray, aruco_pose: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Transform point from camera frame to ArUco marker frame
        
        Args:
            point_camera_frame: 3D point in camera coordinates [x, y, z]
            aruco_pose: ArUco marker pose information
            
        Returns:
            3D point in ArUco marker coordinates [x, y, z] or None if failed
        """
        try:
            # Get transformation matrix (camera to ArUco)
            T_cam_to_aruco = np.array(aruco_pose['transformation_matrix'])
            
            # Invert to get ArUco to camera transformation
            T_aruco_to_cam = np.linalg.inv(T_cam_to_aruco)
            
            # Convert point to homogeneous coordinates
            point_homogeneous = np.append(point_camera_frame, 1.0)
            
            # Transform point to ArUco frame
            point_aruco_homogeneous = T_aruco_to_cam @ point_homogeneous
            point_aruco_frame = point_aruco_homogeneous[:3]
            
            return point_aruco_frame
            
        except Exception as e:
            print(f"[ArUco] Error transforming to ArUco frame: {e}")
            return None
    
    def pixel_to_3d_point(self, u: int, v: int, depth: float) -> Optional[np.ndarray]:
        """
        Convert pixel coordinates and depth to 3D point in camera frame
        
        Args:
            u: x pixel coordinate
            v: y pixel coordinate
            depth: depth value in meters
            
        Returns:
            3D point in camera coordinates [x, y, z] or None if failed
        """
        try:
            # Extract intrinsic parameters
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]
            
            # Convert to 3D coordinates using pinhole camera model
            X = (u - cx) * depth / fx
            Y = (v - cy) * depth / fy
            Z = depth
            
            return np.array([X, Y, Z])
            
        except Exception as e:
            print(f"[ArUco] Error converting pixel to 3D point: {e}")
            return None
    
    def rotation_vector_to_euler(self, rvec: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation vector to Euler angles (roll, pitch, yaw)
        
        Args:
            rvec: Rotation vector from solvePnP
            
        Returns:
            Tuple of (roll, pitch, yaw) in radians
        """
        try:
            rotation_matrix, _ = cv2.Rodrigues(rvec.flatten())
            
            # Extract Euler angles
            sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
            singular = sy < 1e-6
            
            if not singular:
                x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                y = np.arctan2(-rotation_matrix[2, 0], sy)
                z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                y = np.arctan2(-rotation_matrix[2, 0], sy)
                z = 0
            
            return (x, y, z)
            
        except Exception as e:
            print(f"[ArUco] Error converting rotation vector to Euler angles: {e}")
            return (0.0, 0.0, 0.0)
    
    def estimate_marker_size_from_corners(self, corners_2d: np.ndarray, tvec: np.ndarray) -> float:
        """
        Estimate actual marker size from detected corners and camera parameters
        
        Args:
            corners_2d: 2D corner coordinates in pixels
            tvec: Translation vector from camera to marker
            
        Returns:
            Estimated marker size in meters
        """
        try:
            # Calculate average side length in pixels
            side_lengths = []
            for i in range(4):
                next_i = (i + 1) % 4
                side_len = np.linalg.norm(corners_2d[i] - corners_2d[next_i])
                side_lengths.append(side_len)
            avg_side_len_pixels = np.mean(side_lengths)
            
            # Get camera parameters
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            
            # Distance to marker center
            distance = np.linalg.norm(tvec.flatten())
            
            # Estimate marker size using similar triangles
            # marker_size_meters / distance = side_length_pixels / focal_length
            estimated_size = (avg_side_len_pixels * distance) / fx
            
            return estimated_size
            
        except Exception as e:
            print(f"[ArUco] Error estimating marker size: {e}")
            return self.marker_size  # Return configured size as fallback
    
    def create_visualization(self, rgb_image: np.ndarray, aruco_pose: Dict[str, Any] = None, 
                           target_pixel: Tuple[int, int] = None) -> np.ndarray:
        """
        Create visualization showing ArUco marker and target object
        
        Args:
            rgb_image: RGB image
            aruco_pose: ArUco marker pose information
            target_pixel: Target object pixel coordinates
            
        Returns:
            Image with visualization overlay
        """
        vis_image = rgb_image.copy()
        
        # Draw ArUco marker
        if aruco_pose and aruco_pose.get('found'):
            corners = np.array(aruco_pose['corners'], dtype=np.int32)
            
            # Draw marker outline
            cv2.polylines(vis_image, [corners], True, (0, 255, 0), 2)
            
            # Draw marker ID
            center = np.mean(corners, axis=0).astype(int)
            cv2.putText(vis_image, f"ID: {aruco_pose['marker_id']}", 
                       (center[0] - 20, center[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw coordinate axes
            if 'transformation_matrix' in aruco_pose:
                # Get transformation matrix and camera parameters
                T = np.array(aruco_pose['transformation_matrix'])
                fx = self.camera_matrix[0, 0]
                fy = self.camera_matrix[1, 1]
                cx = self.camera_matrix[0, 2]
                cy = self.camera_matrix[1, 2]
                
                # Define 3D axis endpoints in marker frame
                axis_length = self.marker_size * 0.5  # 50% of marker size
                axes_3d = np.array([
                    [0, 0, 0],                    # origin
                    [axis_length, 0, 0],          # X-axis end (red)
                    [0, axis_length, 0],          # Y-axis end (green) 
                    [0, 0, axis_length]           # Z-axis end (blue)
                ], dtype=np.float32)
                
                # Transform to camera frame and project to image
                axes_cam = []
                for point_3d in axes_3d:
                    # Transform to camera frame
                    point_cam = T[:3, :3] @ point_3d + T[:3, 3]
                    if point_cam[2] > 0:  # Check if in front of camera
                        # Project to image coordinates
                        u = int(fx * point_cam[0] / point_cam[2] + cx)
                        v = int(fy * point_cam[1] / point_cam[2] + cy)
                        axes_cam.append((u, v))
                    else:
                        axes_cam.append(None)
                
                # Draw axes if all points are valid
                if all(pt is not None for pt in axes_cam):
                    origin, x_end, y_end, z_end = axes_cam
                    # Draw axes with different colors and thickness
                    cv2.line(vis_image, origin, x_end, (0, 0, 255), 3)    # X-axis: red
                    cv2.line(vis_image, origin, y_end, (0, 255, 0), 3)    # Y-axis: green
                    cv2.line(vis_image, origin, z_end, (255, 0, 0), 3)    # Z-axis: blue
                    
                    # Add axis labels
                    cv2.putText(vis_image, "X", (x_end[0]+5, x_end[1]-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(vis_image, "Y", (y_end[0]+5, y_end[1]-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(vis_image, "Z", (z_end[0]+5, z_end[1]-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw target object
        if target_pixel:
            x, y = target_pixel
            cv2.circle(vis_image, (x, y), 10, (0, 0, 255), 2)
            cv2.putText(vis_image, "Target", (x + 15, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return vis_image

class StandaloneArucoDetector:
    """
    Standalone ArUco detector that reads RGB data from RGBD camera
    """
    
    def __init__(self):
        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1
        
        # ArUco transformer
        self.aruco_transformer = None
        self.camera_matrix = None
        
        # Detection settings
        self.target_marker_id = 0
        self.detection_enabled = True
        self.show_axes = True
        self.show_info = True
        
        # Create windows
        cv2.namedWindow('ArUco Detection')
        cv2.namedWindow('Reference Frame Info')
        
        # Mouse callback for target selection
        self.target_pixel = None
        cv2.setMouseCallback('ArUco Detection', self.on_mouse)
    
    def on_mouse(self, event, x, y, flags, param):
        """Handle mouse events for target selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.target_pixel = (x, y)
            print(f"🎯 Target selected at pixel ({x}, {y})")
    
    def on_new_frame(self):
        """Called when new frame arrives"""
        self.event.set()
    
    def on_stream_stopped(self):
        """Called when stream stops"""
        print('Stream stopped')
    
    def connect_to_device(self, dev_idx=0):
        """Connect to RGBD device"""
        print('Searching for devices')
        devs = Record3DStream.get_connected_devices()
        print('{} device(s) found'.format(len(devs)))
        for dev in devs:
            print('\tID: {}\n\tUDID: {}\n'.format(dev.product_id, dev.udid))

        if len(devs) <= dev_idx:
            raise RuntimeError('Cannot connect to device #{}, try different index.'
                               .format(dev_idx))

        dev = devs[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)
    
    def get_intrinsic_mat_from_coeffs(self, coeffs):
        """Get intrinsic matrix from camera coefficients"""
        return np.array([[coeffs.fx,         0, coeffs.tx],
                         [        0, coeffs.fy, coeffs.ty],
                         [        0,         0,         1]])
    
    def initialize_aruco_transformer(self):
        """Initialize ArUco transformer with camera parameters"""
        if self.camera_matrix is not None:
            # Use actual camera matrix from RGBD stream
            camera_matrix = self.camera_matrix
            # Assume no distortion for now
            dist_coeffs = np.zeros((4, 1), dtype=np.float32)
            
            self.aruco_transformer = ArucoTransformer(
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                marker_size=0.05  # 5cm marker size
            )
            print(f"[ArUco] ✅ Initialized with camera matrix:")
            print(f"[ArUco] Focal length: ({camera_matrix[0,0]:.1f}, {camera_matrix[1,1]:.1f})")
            print(f"[ArUco] Principal point: ({camera_matrix[0,2]:.1f}, {camera_matrix[1,2]:.1f})")
            return True
        else:
            print("[ArUco] ❌ Camera matrix not available yet")
            return False
    
    def create_info_display(self, aruco_pose=None):
        """Create information display window"""
        info_image = np.ones((400, 500, 3), dtype=np.uint8) * 50
        
        # Title
        cv2.putText(info_image, "ArUco Reference Frame Detector", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        y_offset = 70
        
        if aruco_pose and aruco_pose.get('found'):
            # Marker detected
            cv2.putText(info_image, f"✅ Marker ID {aruco_pose['marker_id']} DETECTED", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 40
            
            # Position information
            pos = aruco_pose['position']
            cv2.putText(info_image, f"Position (camera frame):", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(info_image, f"  X: {pos['x']:.3f} m", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(info_image, f"  Y: {pos['y']:.3f} m", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(info_image, f"  Z: {pos['z']:.3f} m", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 30
            
            # Distance from camera
            distance = np.sqrt(pos['x']**2 + pos['y']**2 + pos['z']**2)
            cv2.putText(info_image, f"Distance: {distance:.3f} m", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 30
            
            # Euler angles
            if 'rotation_vector' in aruco_pose:
                rvec = np.array(aruco_pose['rotation_vector'])
                roll, pitch, yaw = self.aruco_transformer.rotation_vector_to_euler(rvec)
                cv2.putText(info_image, f"Orientation (radians):", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_offset += 25
                cv2.putText(info_image, f"  Roll: {roll:.3f}", (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
                cv2.putText(info_image, f"  Pitch: {pitch:.3f}", (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
                cv2.putText(info_image, f"  Yaw: {yaw:.3f}", (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 30
            
            # Target transformation
            if self.target_pixel and self.aruco_transformer:
                # Get depth at target pixel (if available)
                depth = self.session.get_depth_frame()
                if depth is not None:
                    x, y = self.target_pixel
                    if 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]:
                        depth_value = float(depth[y, x])
                        if depth_value > 0:
                            target_3d = self.aruco_transformer.pixel_to_3d_point(x, y, depth_value)
                            if target_3d is not None:
                                result = self.aruco_transformer.get_target_relative_to_aruco(
                                    cv2.cvtColor(self.session.get_rgb_frame(), cv2.COLOR_RGB2BGR), 
                                    target_3d, 
                                    self.target_marker_id
                                )
                                if result:
                                    rel_pos = result['relative_position']
                                    cv2.putText(info_image, f"Target (marker frame):", (10, y_offset), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                                    y_offset += 25
                                    cv2.putText(info_image, f"  X: {rel_pos['x']:.3f} m", (20, y_offset), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                    y_offset += 20
                                    cv2.putText(info_image, f"  Y: {rel_pos['y']:.3f} m", (20, y_offset), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                    y_offset += 20
                                    cv2.putText(info_image, f"  Z: {rel_pos['z']:.3f} m", (20, y_offset), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            # No marker detected
            cv2.putText(info_image, "❌ NO MARKER DETECTED", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 40
            cv2.putText(info_image, "Place ArUco marker ID 0 in view", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 30
            cv2.putText(info_image, "Ensure good lighting and contrast", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Controls info
        y_offset += 40
        cv2.putText(info_image, "Controls:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        cv2.putText(info_image, "  Click: Select target point", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(info_image, "  'q': Quit", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(info_image, "  'i': Toggle info display", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return info_image
    
    def start_processing_stream(self):
        """Start processing the RGBD stream"""
        print("🎯 ArUco Reference Frame Detector")
        print("=" * 50)
        print("Looking for ArUco marker ID 0 as reference frame")
        print("Click on the detection window to select target points")
        print("Press 'q' to quit, 'i' to toggle info display")
        print("=" * 50)
        
        while True:
            self.event.wait()  # Wait for new frame to arrive

            # Get RGBD data
            rgb = self.session.get_rgb_frame()
            depth = self.session.get_depth_frame()
            intrinsic_mat = self.get_intrinsic_mat_from_coeffs(self.session.get_intrinsic_mat())

            # Update camera matrix
            if self.camera_matrix is None:
                self.camera_matrix = intrinsic_mat
                self.initialize_aruco_transformer()

            # Postprocess frames
            if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                depth = cv2.flip(depth, 1)
                rgb = cv2.flip(rgb, 1)

            # Convert RGB to BGR for OpenCV
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Detect ArUco marker
            aruco_pose = None
            if self.aruco_transformer and self.detection_enabled:
                aruco_pose = self.aruco_transformer.detect_aruco_marker(rgb, self.target_marker_id)

            # Create visualization
            vis_image = self.aruco_transformer.create_visualization(rgb_bgr, aruco_pose, self.target_pixel) if self.aruco_transformer else rgb_bgr

            # Add status text
            if aruco_pose:
                status_text = f"Marker {aruco_pose['marker_id']} DETECTED"
                color = (0, 255, 0)
            else:
                status_text = "NO MARKER DETECTED"
                color = (0, 0, 255)
            
            cv2.putText(vis_image, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            if self.target_pixel:
                cv2.putText(vis_image, f"Target: {self.target_pixel}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Show detection window
            cv2.imshow('ArUco Detection', vis_image)
            
            # Show info window
            if self.show_info:
                info_image = self.create_info_display(aruco_pose)
                cv2.imshow('Reference Frame Info', info_image)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            elif key == ord('i'):
                self.show_info = not self.show_info
                if not self.show_info:
                    cv2.destroyWindow('Reference Frame Info')
                print(f"ℹ️  Info display: {'ON' if self.show_info else 'OFF'}")

            self.event.clear()

def main():
    """Main function to run the standalone ArUco detector"""
    detector = StandaloneArucoDetector()
    
    try:
        detector.connect_to_device(dev_idx=0)
        detector.start_processing_stream()
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()