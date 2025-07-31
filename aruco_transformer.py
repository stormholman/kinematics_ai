#!/usr/bin/env python3
"""
ArUco Marker Coordinate Transformer
Transforms target object positions from camera coordinates to ArUco marker coordinates
Uses ArUco marker ID 0 as reference frame for robotic manipulation
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
import json

class ArucoTransformer:
    """
    Transforms coordinates from camera frame to ArUco marker frame
    """
    
    def __init__(self, camera_matrix: np.ndarray = None, dist_coeffs: np.ndarray = None, marker_size: float = 0.05):
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
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Default camera parameters (replace with actual calibration)
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
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
            
            # Detect ArUco markers
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
            
            if ids is None or len(ids) == 0:
                print(f"[ArUco] No ArUco markers detected")
                return None
            
            # Find target marker ID
            target_idx = None
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == target_id:
                    target_idx = i
                    break
            
            if target_idx is None:
                print(f"[ArUco] Marker ID {target_id} not found")
                return None
            
            # Create 3D object points for ArUco marker corners
            marker_corners_3d = np.array([
                [-self.marker_size/2, -self.marker_size/2, 0],
                [self.marker_size/2, -self.marker_size/2, 0],
                [self.marker_size/2, self.marker_size/2, 0],
                [-self.marker_size/2, self.marker_size/2, 0]
            ], dtype=np.float32)
            
            # Estimate pose using solvePnP
            success, rvec, tvec = cv2.solvePnP(
                marker_corners_3d, 
                corners[target_idx][0], 
                self.camera_matrix, 
                self.dist_coeffs
            )
            
            if not success:
                print(f"[ArUco] Failed to estimate pose for marker {target_id}")
                return None
            
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec.flatten())
            
            # Create 4x4 transformation matrix (camera to marker)
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = tvec.flatten()
            
            # Convert rotation vector to Euler angles (rx, ry, rz)
            rx, ry, rz = self.rotation_vector_to_euler(rvec.flatten())
            
            result = {
                'marker_id': target_id,
                'found': True,
                'position': {
                    'x': float(tvec.flatten()[0]),
                    'y': float(tvec.flatten()[1]),
                    'z': float(tvec.flatten()[2])
                },
                'rotation': {
                    'rx': float(rx),
                    'ry': float(ry),
                    'rz': float(rz)
                },
                'rotation_vector': rvec.flatten().tolist(),
                'translation_vector': tvec.flatten().tolist(),
                'rotation_matrix': rotation_matrix.tolist(),
                'transformation_matrix': transformation_matrix.tolist(),
                'corners': corners[target_idx][0].tolist()
            }
            
            print(f"[ArUco] Marker {target_id} detected at position ({tvec.flatten()[0]:.3f}, {tvec.flatten()[1]:.3f}, {tvec.flatten()[2]:.3f})")
            return result
            
        except Exception as e:
            print(f"[ArUco] Error detecting marker: {e}")
            return None
    
    def pixel_to_3d_point(self, pixel_coords: Tuple[int, int], depth_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Convert pixel coordinates to 3D point in camera frame using depth
        
        Args:
            pixel_coords: (x, y) pixel coordinates
            depth_image: Depth image (same dimensions as RGB)
            
        Returns:
            3D point in camera coordinates [x, y, z] or None if invalid
        """
        try:
            x_pixel, y_pixel = pixel_coords
            
            # Get depth value at pixel location
            if depth_image is None:
                print("[ArUco] No depth image provided")
                return None
                
            if (y_pixel >= depth_image.shape[0] or x_pixel >= depth_image.shape[1] or 
                y_pixel < 0 or x_pixel < 0):
                print(f"[ArUco] Pixel coordinates ({x_pixel}, {y_pixel}) out of bounds")
                return None
            
            depth = depth_image[y_pixel, x_pixel]
            
            # Handle different depth image formats
            if depth == 0:
                print(f"[ArUco] Invalid depth at pixel ({x_pixel}, {y_pixel})")
                return None
            
            # Convert depth to meters if needed (assuming depth in mm)
            if depth > 10:  # Likely in mm
                depth = depth / 1000.0
            
            # Convert pixel to 3D point using camera intrinsics
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]
            
            # 3D point in camera frame
            x_cam = (x_pixel - cx) * depth / fx
            y_cam = (y_pixel - cy) * depth / fy
            z_cam = depth
            
            point_3d = np.array([x_cam, y_cam, z_cam])
            print(f"[ArUco] 3D point in camera frame: ({x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f})")
            
            return point_3d
            
        except Exception as e:
            print(f"[ArUco] Error converting pixel to 3D: {e}")
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
            
            print(f"[ArUco] Point in ArUco frame: ({point_aruco_frame[0]:.3f}, {point_aruco_frame[1]:.3f}, {point_aruco_frame[2]:.3f})")
            
            return point_aruco_frame
            
        except Exception as e:
            print(f"[ArUco] Error transforming to ArUco frame: {e}")
            return None
    
    def get_target_relative_to_aruco(self, rgb_image: np.ndarray, depth_image: np.ndarray, 
                                   target_pixel: Tuple[int, int], aruco_id: int = 0) -> Optional[Dict[str, Any]]:
        """
        Get target object position relative to ArUco marker
        
        Args:
            rgb_image: RGB image from camera
            depth_image: Depth image from camera
            target_pixel: (x, y) pixel coordinates of target object
            aruco_id: ArUco marker ID to use as reference (default 0)
            
        Returns:
            Dictionary with relative position information or None if failed
        """
        try:
            # Detect ArUco marker
            aruco_pose = self.detect_aruco_marker(rgb_image, aruco_id)
            if aruco_pose is None:
                return None
            
            # Convert target pixel to 3D point in camera frame
            target_3d_camera = self.pixel_to_3d_point(target_pixel, depth_image)
            if target_3d_camera is None:
                return None
            
            # Transform target point to ArUco frame
            target_3d_aruco = self.transform_to_aruco_frame(target_3d_camera, aruco_pose)
            if target_3d_aruco is None:
                return None
            
            # Prepare result
            result = {
                'success': True,
                'aruco_marker': aruco_pose,
                'target_pixel': list(target_pixel),
                'target_position_camera': target_3d_camera.tolist(),
                'target_position_aruco': target_3d_aruco.tolist(),
                'relative_position': {
                    'x': float(target_3d_aruco[0]),
                    'y': float(target_3d_aruco[1]),
                    'z': float(target_3d_aruco[2])
                }
            }
            
            print(f"[ArUco] Target relative to marker: x={target_3d_aruco[0]:.3f}m, y={target_3d_aruco[1]:.3f}m, z={target_3d_aruco[2]:.3f}m")
            
            return result
            
        except Exception as e:
            print(f"[ArUco] Error getting relative position: {e}")
            return None
    
    def rotation_vector_to_euler(self, rvec: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation vector to Euler angles (rx, ry, rz)
        
        Args:
            rvec: Rotation vector from cv2.aruco.estimatePoseSingleMarkers
            
        Returns:
            Tuple of Euler angles (rx, ry, rz) in radians
        """
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        # Extract Euler angles from rotation matrix (ZYX convention)
        sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            rx = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            ry = np.arctan2(-rotation_matrix[2, 0], sy)
            rz = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            rx = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            ry = np.arctan2(-rotation_matrix[2, 0], sy)
            rz = 0
        
        return rx, ry, rz
    
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
            if 'rotation_vector' in aruco_pose and 'translation_vector' in aruco_pose:
                rvec = np.array(aruco_pose['rotation_vector'])
                tvec = np.array(aruco_pose['translation_vector'])
                cv2.drawFrameAxes(vis_image, self.camera_matrix, self.dist_coeffs, 
                                rvec.flatten(), tvec.flatten(), self.marker_size * 0.5)
        
        # Draw target object
        if target_pixel:
            x, y = target_pixel
            cv2.circle(vis_image, (x, y), 10, (0, 0, 255), 2)
            cv2.putText(vis_image, "Target", (x + 15, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return vis_image

# Test function
def test_aruco_transformer():
    """Test the ArUco transformer with sample data"""
    print("Testing ArUco Transformer")
    print("=" * 50)
    
    transformer = ArucoTransformer()
    
    # Create test image with ArUco marker
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Generate ArUco marker
    marker = cv2.aruco.generateImageMarker(transformer.aruco_dict, 0, 200)
    
    # Place marker in image
    y_start, y_end = 100, 300
    x_start, x_end = 200, 400
    test_image[y_start:y_end, x_start:x_end] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    
    # Create mock depth image
    depth_image = np.ones((480, 640), dtype=np.float32) * 1.0  # 1 meter depth
    
    # Test detection
    aruco_pose = transformer.detect_aruco_marker(test_image, 0)
    
    if aruco_pose:
        print("\nArUco Detection Results:")
        print(json.dumps({k: v for k, v in aruco_pose.items() 
                         if k not in ['rotation_matrix', 'transformation_matrix']}, indent=2))
        
        # Test coordinate transformation
        target_pixel = (320, 240)  # Center of image
        result = transformer.get_target_relative_to_aruco(test_image, depth_image, target_pixel, 0)
        
        if result:
            print("\nTransformation Results:")
            print(json.dumps({k: v for k, v in result.items() 
                             if k != 'aruco_marker'}, indent=2))
        
        # Show visualization
        vis_image = transformer.create_visualization(test_image, aruco_pose, target_pixel)
        cv2.imshow('ArUco Transformer Test', vis_image)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("ArUco marker detection failed")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_aruco_transformer()
    else:
        print("ArUco Transformer Module")
        print("Use 'python aruco_transformer.py test' to run test")