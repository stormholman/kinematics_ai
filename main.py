import numpy as np
import cv2
from rgbd_feed import DemoApp
from vision_analyzer import VisionAnalyzer
import requests

class KinematicsApp(DemoApp):
    def __init__(self):
        super().__init__()
        self.intrinsic_mat = None  # Keep for backward compatibility
        self.depth_intrinsic_mat = None  # Depth camera intrinsics
        self.rgb_intrinsic_mat = None    # RGB camera intrinsics
        
        # Vision analyzer for AI-powered object detection
        self.vision_analyzer = VisionAnalyzer()
        self.vision_mode = False
        self.vision_result = None
        self.vision_target_description = ""
        
        # Current RGB frame for vision analysis
        self.current_rgb = None
        
        # ArUco detection - direct approach from aruco_rgbd_stream.py
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        # Add the same parameters as the working script
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 10
        params.adaptiveThreshConstant = 7
        params.minMarkerPerimeterRate = 0.1
        params.maxMarkerPerimeterRate = 0.8
        params.polygonalApproxAccuracyRate = 0.05
        params.minCornerDistanceRate = 0.1
        params.minDistanceToBorder = 3
        params.minOtsuStdDev = 5.0
        params.perspectiveRemovePixelPerCell = 4
        params.perspectiveRemoveIgnoredMarginPerCell = 0.13
        params.maxErroneousBitsInBorderRate = 0.2
        params.minMarkerDistanceRate = 0.05
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 30
        params.cornerRefinementMinAccuracy = 0.1
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, params)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        # ArUco marker settings
        self.aruco_marker_id = 0
        self.aruco_marker_size = 0.04  # 4cm marker size
        self.current_pose_T_cam_marker = None  # 4x4 homogeneous transformation
        
        # Storage for last clicked pixel and its marker-frame coordinates
        self.clicked_pixel = None           # (u,v)
        self.clicked_marker = None          # (x,y,z) in marker frame
        
        # Robot move settings
        self.robot_endpoint = "http://192.168.178.190/move/absolute?robot_id=0"
        self.position_tolerance = 0.03
        self.orientation_tolerance = 0.2

    def estimate_pose_from_corners(self, corners_2d):
        """Estimate ArUco marker pose from detected corners"""
        # marker corner positions in marker coordinate system (origin at top-left)
        obj_pts = np.array([[0, 0, 0],
                            [self.aruco_marker_size, 0, 0],
                            [self.aruco_marker_size, self.aruco_marker_size, 0],
                            [0, self.aruco_marker_size, 0]], dtype=np.float32)
        success, rvec, tvec = cv2.solvePnP(obj_pts, corners_2d, self.rgb_intrinsic_mat, self.dist_coeffs)
        if not success:
            return None
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return T

    def detect_aruco_marker(self):
        """Detect ArUco marker in current RGB frame"""
        if self.current_rgb is None or self.rgb_intrinsic_mat is None:
            return False
            
        # Convert to grayscale for ArUco detection
        gray = cv2.cvtColor(self.current_rgb, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        
        if ids is not None and self.aruco_marker_id in ids.flatten():
            idx = list(ids.flatten()).index(self.aruco_marker_id)
            self.current_pose_T_cam_marker = self.estimate_pose_from_corners(corners[idx][0])
            return self.current_pose_T_cam_marker is not None
        else:
            self.current_pose_T_cam_marker = None
            return False

    def pixel_to_marker_coordinates(self, rgb_x, rgb_y):
        """Convert RGB pixel coordinates to marker frame coordinates"""
        if self.current_pose_T_cam_marker is None:
            print("⚠️  Cannot compute – marker pose not available")
            return None
            
        # Get depth at RGB pixel (handle scaling automatically)
        depth_val = self.get_depth_at_rgb_coordinate(rgb_x, rgb_y)
        if depth_val is None or depth_val <= 0:
            print("⚠️  Invalid depth at pixel")
            return None

        # Convert pixel to 3D camera coordinates
        fx, fy = self.rgb_intrinsic_mat[0, 0], self.rgb_intrinsic_mat[1, 1]
        cx, cy = self.rgb_intrinsic_mat[0, 2], self.rgb_intrinsic_mat[1, 2]
        X = (rgb_x - cx) * depth_val / fx
        Y = (rgb_y - cy) * depth_val / fy
        Z = depth_val
        pt_cam = np.array([X, Y, Z, 1.0])
        
        # Transform to marker frame
        pt_marker = np.linalg.inv(self.current_pose_T_cam_marker) @ pt_cam
        marker_coords = pt_marker[:3]
        
        print(f"RGB pixel ({rgb_x},{rgb_y}) depth {depth_val:.3f} m → Camera ({X:.3f},{Y:.3f},{Z:.3f}) → Marker ({marker_coords[0]:.3f},{marker_coords[1]:.3f},{marker_coords[2]:.3f}) m")
        
        return marker_coords

    def convert_to_robot_frame(self, aruco_coords_m):
        """Convert ArUco-frame metres to robot-frame centimetres with proper axis mapping
        
        ArUco frame (origin at top-left corner):
        - +X: rightward along top edge
        - +Y: downward along left edge  
        - +Z: out of marker plane toward camera
        
        Robot frame mapping:
        - robot_x = -aruco_y (robot X is negative ArUco Y - moving left from marker top-left)
        - robot_y = -aruco_z (robot Y is negative ArUco Z - moving away from marker)  
        - robot_z = aruco_x + 13cm (robot Z is ArUco X + elevation offset)
        """
        ax, ay, az = aruco_coords_m  # metres
        
        # Apply the correct mapping (keep in metres first):
        robot_x_m = -ay           # robot X = -aruco Y (negative because robot left is negative ArUco Y)
        robot_y_m = -ax           # robot Y = -aruco Z (negative because robot away is negative ArUco Z)
        robot_z_m = -az - 0.13     # robot Z = aruco X + 13cm offset (robot height is ArUco X plus elevation)
        
        # Convert to centimetres
        rx_cm = robot_x_m * 100.0
        ry_cm = robot_y_m * 100.0
        rz_cm = robot_z_m * 100.0
        
        return rx_cm, ry_cm, rz_cm

    def prompt_and_move_robot(self, rel_coords):
        """Ask user for confirmation and move robot (convert to robot frame)"""
        rx_cm, ry_cm, rz_cm = self.convert_to_robot_frame(rel_coords)
        print(f"\n🛰  Debug (ArUco frame m): X={rel_coords[0]:.3f}  Y={rel_coords[1]:.3f}  Z={rel_coords[2]:.3f}")
        print(f"🛰  Converted to robot frame cm: X={rx_cm:.1f}  Y={ry_cm:.1f}  Z={rz_cm:.1f}")
        preview_curl = (
            "curl -X 'POST' \\\n"
            "  'http://192.168.178.190/move/absolute?robot_id=0' \\\n"
            "  -H 'accept: application/json' \\\n"
            "  -H 'Content-Type: application/json' \\\n"
            "  -d '{\\n"
            f"  \"x\": {rx_cm:.1f},\\n  \"y\": {ry_cm:.1f},\\n  \"z\": {rz_cm:.1f},\\n  \"open\": 0,\\n  \"max_trials\": 100,\\n  \"position_tolerance\": 0.03,\\n  \"orientation_tolerance\": 0.2\\n}}'"
        )
        print("\n🛰  Proposed robot move command:\n" + preview_curl)
        try:
            answer = input("\n🤖 Send this command? (y/n): ").strip().lower()
            if answer != 'y':
                print("🚫 Move cancelled")
                return
            payload = {
                "x": rx_cm,
                "y": ry_cm,
                "z": rz_cm,
                "open": 0,
                "max_trials": 100,
                "position_tolerance": self.position_tolerance,
                "orientation_tolerance": self.orientation_tolerance
            }
            resp = requests.post(self.robot_endpoint, json=payload, timeout=10)
            print(f"➡️  HTTP status: {resp.status_code}")
            if resp.ok:
                print("✅ Robot move command sent successfully")
            else:
                print(f"❌ Robot move failed: {resp.text}")
        except Exception as e:
            print(f"❌ Error sending move command: {e}")
    
    def pixel_to_3d_position(self, u, v, depth_value, camera_matrix=None):
        """
        Convert pixel coordinates (u,v) and depth to 3D position relative to camera.
        
        Args:
            u (int): x pixel coordinate
            v (int): y pixel coordinate  
            depth_value (float): depth in meters at that pixel
            camera_matrix (np.ndarray): Camera intrinsic matrix to use (defaults to depth camera)
            
        Returns:
            tuple: (X, Y, Z) 3D coordinates in meters relative to camera
        """
        # Use depth camera intrinsics by default
        K = camera_matrix if camera_matrix is not None else self.depth_intrinsic_mat
        
        if K is None:
            print("Warning: No camera intrinsic matrix available yet")
            return None
            
        # Extract intrinsic parameters
        fx = K[0, 0]  # focal length x
        fy = K[1, 1]  # focal length y
        cx = K[0, 2]  # principal point x
        cy = K[1, 2]  # principal point y
        
        # Convert to 3D coordinates using pinhole camera model
        X = (u - cx) * depth_value / fx
        Y = (v - cy) * depth_value / fy
        Z = depth_value
        
        return (X, Y, Z)
    
    def scale_coordinates_rgb_to_depth(self, rgb_x, rgb_y):
        """Scale coordinates from RGB frame to depth frame"""
        if self.current_rgb is None or self.depth is None:
            return rgb_x, rgb_y
            
        rgb_height, rgb_width = self.current_rgb.shape[:2]
        depth_height, depth_width = self.depth.shape[:2]
        
        if rgb_width != depth_width or rgb_height != depth_height:
            scale_x = depth_width / rgb_width
            scale_y = depth_height / rgb_height
            depth_x = int(rgb_x * scale_x)
            depth_y = int(rgb_y * scale_y)
            return depth_x, depth_y
        else:
            return rgb_x, rgb_y
    
    def get_depth_at_rgb_coordinate(self, rgb_x, rgb_y):
        """Get depth value at RGB coordinate (handles scaling automatically)"""
        if self.depth is None:
            return None
            
        depth_x, depth_y = self.scale_coordinates_rgb_to_depth(rgb_x, rgb_y)
        
        if 0 <= depth_y < self.depth.shape[0] and 0 <= depth_x < self.depth.shape[1]:
            return float(self.depth[depth_y, depth_x])
        else:
            return None
    
    def on_mouse(self, event, x, y, flags, param):
        """Override mouse callback to handle pixel-to-marker coordinate conversion"""
        if self.depth is not None and event == cv2.EVENT_MOUSEMOVE:
            if 0 <= y < self.depth.shape[0] and 0 <= x < self.depth.shape[1]:
                # x, y are in depth frame coordinates
                depth_value = float(self.depth[y, x])
                
                # Convert depth coordinates to RGB coordinates for 3D calculation
                if self.current_rgb is not None:
                    rgb_height, rgb_width = self.current_rgb.shape[:2]
                    depth_height, depth_width = self.depth.shape[:2]
                    
                    if rgb_width != depth_width or rgb_height != depth_height:
                        # Scale from depth to RGB coordinates
                        scale_x = rgb_width / depth_width
                        scale_y = rgb_height / depth_height
                        rgb_x = int(x * scale_x)
                        rgb_y = int(y * scale_y)
                    else:
                        rgb_x, rgb_y = x, y
                else:
                    rgb_x, rgb_y = x, y
                
                self.last_mouse_pos = (rgb_x, rgb_y)  # Store RGB coordinates for overlay
                self.depth_value = depth_value
                
                # Only show manual mode info if not in vision mode
                if not self.vision_mode:
                    # Calculate 3D position using RGB coordinates
                    if self.rgb_intrinsic_mat is not None:
                        pos_3d = self.pixel_to_3d_position(rgb_x, rgb_y, depth_value, self.rgb_intrinsic_mat)
                        if pos_3d:
                            X, Y, Z = pos_3d
                            print(f"Manual D({x},{y})->RGB({rgb_x},{rgb_y}) -> Depth: {depth_value:.3f}m, 3D: ({X:.3f},{Y:.3f},{Z:.3f})")
                            # Show marker coordinates if available
                            if self.current_pose_T_cam_marker is not None:
                                marker_coords = self.pixel_to_marker_coordinates(rgb_x, rgb_y)
                    else:
                        print(f"Depth at D({x},{y}): {depth_value:.3f} meters")
        
        # Handle vision mode clicks
        elif self.vision_mode and event == cv2.EVENT_LBUTTONDOWN:
            if self.depth is not None and 0 <= y < self.depth.shape[0] and 0 <= x < self.depth.shape[1]:
                depth_value = float(self.depth[y, x])
                
                # Convert depth coordinates to RGB coordinates
                if self.current_rgb is not None:
                    rgb_height, rgb_width = self.current_rgb.shape[:2]
                    depth_height, depth_width = self.depth.shape[:2]
                    
                    if rgb_width != depth_width or rgb_height != depth_height:
                        scale_x = rgb_width / depth_width
                        scale_y = rgb_height / depth_height
                        rgb_x = int(x * scale_x)
                        rgb_y = int(y * scale_y)
                    else:
                        rgb_x, rgb_y = x, y
                else:
                    rgb_x, rgb_y = x, y
                
                self.last_mouse_pos = (rgb_x, rgb_y)
                self.depth_value = depth_value
                
                if self.rgb_intrinsic_mat is not None:
                    pos_3d = self.pixel_to_3d_position(rgb_x, rgb_y, depth_value, self.rgb_intrinsic_mat)
                    if pos_3d:
                        X, Y, Z = pos_3d
                        print(f"Vision Click D({x},{y})->RGB({rgb_x},{rgb_y}) -> Depth: {depth_value:.3f}m, 3D: ({X:.3f},{Y:.3f},{Z:.3f})")
    
        # Handle manual mode left-click for robot move with marker coordinates
        if not self.vision_mode and event == cv2.EVENT_LBUTTONDOWN and self.depth is not None:
            if 0 <= y < self.depth.shape[0] and 0 <= x < self.depth.shape[1]:
                depth_value = float(self.depth[y, x])
                if depth_value <= 0:
                    print("⚠️  Invalid depth at selected pixel")
                    return
                    
                # Convert to RGB coordinates
                if self.current_rgb is not None:
                    rgb_height, rgb_width = self.current_rgb.shape[:2]
                    depth_height, depth_width = self.depth.shape[:2]
                    if rgb_width != depth_width or rgb_height != depth_height:
                        scale_x = rgb_width / depth_width
                        scale_y = rgb_height / depth_height
                        rgb_x = int(x * scale_x)
                        rgb_y = int(y * scale_y)
                    else:
                        rgb_x, rgb_y = x, y
                else:
                    rgb_x, rgb_y = x, y
                    
                # Store clicked pixel and convert to marker coordinates
                self.clicked_pixel = (rgb_x, rgb_y)
                self.clicked_marker = None
                
                if self.current_pose_T_cam_marker is not None:
                    marker_coords = self.pixel_to_marker_coordinates(rgb_x, rgb_y)
                    if marker_coords is not None:
                        self.clicked_marker = marker_coords
                        # Send to robot
                        self.prompt_and_move_robot(marker_coords)
                    else:
                        print("⚠️  Could not convert to marker coordinates")
                else:
                    print("[ArUco] ❌ No marker detected - cannot transform coordinates")
    
    def get_3d_position_from_input(self):
        """Get 3D position from user input of pixel coordinates"""
        try:
            u = int(input("Enter u (x) pixel coordinate: "))
            v = int(input("Enter v (y) pixel coordinate: "))
            
            if self.depth is None:
                print("No depth data available. Make sure the stream is running.")
                return None
                
            if not (0 <= v < self.depth.shape[0] and 0 <= u < self.depth.shape[1]):
                print(f"Coordinates ({u}, {v}) are out of bounds. Image size: {self.depth.shape[1]}x{self.depth.shape[0]}")
                return None
                
            depth_value = float(self.depth[v, u])  # Note: depth[y, x]
            
            if depth_value <= 0:
                print(f"Invalid depth value: {depth_value}")
                return None
                
            pos_3d = self.pixel_to_3d_position(u, v, depth_value)
            if pos_3d:
                X, Y, Z = pos_3d
                print(f"\n3D Position Calculation:")
                print(f"Pixel coordinates: ({u}, {v})")
                print(f"Depth: {depth_value:.3f} meters")
                print(f"3D Position (relative to camera): ({X:.3f}, {Y:.3f}, {Z:.3f}) meters")
                print(f"Distance from camera: {np.sqrt(X*X + Y*Y + Z*Z):.3f} meters")
                return pos_3d
            else:
                print("Failed to calculate 3D position")
                return None
                
        except ValueError:
            print("Invalid input. Please enter integer pixel coordinates.")
            return None
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return None
    
    def get_vision_target(self):
        """Get target description from user for vision-based selection (HYBRID mode)"""
        try:
            print("\n" + "="*60)
            print("🤖 AI VISION ANALYSIS (HYBRID MODE)")
            print("="*60)
            print("Describe what you want to target:")
            print()
            print("💡 Examples:")
            print("  • 'yellow banana'")
            print("  • 'red cup on the table'")
            print("  • 'the book with blue cover'")
            print("  • 'coffee mug'")
            print("-" * 60)
            
            target_description = input("Target description: ").strip()
            
            if not target_description:
                print("❌ No description provided")
                return False
            
            if not self.vision_analyzer.is_available():
                print("❌ Vision analyzer not available (API key required)")
                return False
            
            if self.current_rgb is None:
                print("❌ No RGB frame available")
                return False
            
            print(f"🔍 Analyzing for: '{target_description}'")
            print("⏳ Processing...")
            
            # Analyze the current RGB frame
            result = self.vision_analyzer.analyze_image(self.current_rgb, target_description, self.depth)
            
            if result and result.get('success'):
                self.vision_result = result
                self.vision_target_description = target_description
                self.vision_mode = True
                
                # Extract target coordinates
                x, y = result['target_pixel']
                
                # Scale coordinates from RGB to depth frame if needed
                rgb_height, rgb_width = self.current_rgb.shape[:2]
                depth_height, depth_width = self.depth.shape[:2]
                
                if rgb_width != depth_width or rgb_height != depth_height:
                    scale_x = depth_width / rgb_width
                    scale_y = depth_height / rgb_height
                    depth_x = int(x * scale_x)
                    depth_y = int(y * scale_y)
                else:
                    depth_x, depth_y = x, y
                
                # Get depth and calculate 3D position
                if 0 <= depth_y < self.depth.shape[0] and 0 <= depth_x < self.depth.shape[1]:
                    depth_value = float(self.depth[depth_y, depth_x])
                    if depth_value > 0 and self.rgb_intrinsic_mat is not None:
                        pos_3d = self.pixel_to_3d_position(x, y, depth_value, self.rgb_intrinsic_mat)
                        if pos_3d:
                            X, Y, Z = pos_3d
                            print(f"\n✅ TARGET ACQUIRED!")
                            print(f"Object: {result.get('object_description', 'Found')}")
                            print(f"RGB Pixel: ({x}, {y})")
                            print(f"Depth Pixel: ({depth_x}, {depth_y})")
                            print(f"Depth: {depth_value:.3f}m")
                            print(f"3D Position: ({X:.3f}, {Y:.3f}, {Z:.3f})m")
                            print(f"Confidence: {result.get('confidence', 'N/A')}")
                            
                            # Transform to ArUco marker frame if available
                            if self.current_pose_T_cam_marker is not None:
                                marker_coords = self.pixel_to_marker_coordinates(x, y)
                                if marker_coords is not None:
                                    print(f"[ArUco] Relative to marker: ({marker_coords[0]:.3f},{marker_coords[1]:.3f},{marker_coords[2]:.3f})")
                                    
                                    # Store clicked pixel and marker coordinates for visualization
                                    self.clicked_pixel = (x, y)
                                    self.clicked_marker = marker_coords
                                    
                                    # Send to robot (same as manual mode)
                                    self.prompt_and_move_robot(marker_coords)
                                else:
                                    print("[ArUco] ❌ Could not convert to marker coordinates")
                            else:
                                print("[ArUco] ❌ No marker detected - cannot transform coordinates")
                            
                            # Store the last position for overlay (RGB coordinates)
                            self.last_mouse_pos = (x, y)
                            self.depth_value = depth_value
                            
                            return True
                        else:
                            print("⚠️  Could not calculate 3D position")
                    else:
                        print(f"⚠️  Invalid depth at target: {depth_value}")
                else:
                    print(f"⚠️  Target coordinates out of bounds: RGB({x},{y}) -> Depth({depth_x},{depth_y})")
            else:
                print(f"❌ Object not found: {result.get('reasoning', 'Unknown error') if result else 'Analysis failed'}")
                self.vision_mode = False
                self.vision_result = None
            
            print("="*60)
            return result and result.get('success', False)
            
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled by user")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def toggle_vision_mode(self):
        """Toggle between manual and vision modes (HYBRID mode only)"""
        if self.vision_mode:
            self.vision_mode = False
            self.vision_result = None
            self.vision_target_description = ""
            print("🎯 Switched to MANUAL mode")
        else:
            print("🤖 Switched to AI VISION mode - use 'v' to analyze objects")
    
    def select_operating_mode(self):
        """Let user select operating mode at startup"""
        print("\n" + "="*60)
        print("KINEMATICS AI - MODE SELECTION")
        print("="*60)
        print("Choose your operating mode:")
        print()
        print("1. MANUAL MODE")
        print("   - Use mouse cursor to select pixels")
        print("   - Real-time 3D position feedback")
        print("   - Manual precision control")
        print()
        print("2. AI VISION MODE")
        print("   - Describe objects in natural language")
        print("   - AI automatically finds and targets objects")
        print("   - Examples: 'yellow banana', 'red cup on table'")
        
        if not self.vision_analyzer.is_available():
            print("   - ⚠️  NOT AVAILABLE (API key required)")
        else:
            print("   - ✅ READY")
        
        print()
        print("3. HYBRID MODE")
        print("   - Switch between modes during operation")
        print("   - Press 'm' to toggle, 'v' for AI vision")
        print()
        
        while True:
            try:
                choice = input("Select mode (1/2/3): ").strip()
                
                if choice == '1':
                    self.vision_mode = False
                    print("\n🎯 MANUAL MODE selected")
                    print("Use mouse to hover over objects in the depth window")
                    return 'manual'
                elif choice == '2':
                    if not self.vision_analyzer.is_available():
                        print("❌ AI Vision not available - API key required")
                        continue
                    self.vision_mode = True
                    print("\n🤖 AI VISION MODE selected")
                    return 'ai'
                elif choice == '3':
                    self.vision_mode = False
                    print("\n🔄 HYBRID MODE selected")
                    return 'hybrid'
                else:
                    print("Please enter 1, 2, or 3")
                    
            except KeyboardInterrupt:
                print("\n\nExiting...")
                return 'exit'
    
    def get_ai_task(self):
        """Get AI task description from user"""
        print("\n" + "="*60)
        print("AI VISION TASK INPUT")
        print("="*60)
        print("Describe what you want the AI to find and target:")
        print()
        print("💡 Examples:")
        print("   • 'pick up the yellow banana'")
        print("   • 'grab the red cup on the table'")
        print("   • 'target the book with blue cover'")
        print("   • 'find the coffee mug'")
        print("   • 'locate the phone next to laptop'")
        print()
        print("📝 Tips for better results:")
        print("   - Be specific about colors and shapes")
        print("   - Mention relative positions if helpful")
        print("   - Use simple, clear descriptions")
        print()
        
        try:
            task_description = input("Your task: ").strip()
            
            if not task_description:
                print("❌ No task description provided")
                return None
                
            print(f"\n🎯 Task: '{task_description}'")
            print("🔄 Processing... this may take a few seconds")
            
            return task_description
            
        except KeyboardInterrupt:
            print("\n\n❌ Task input cancelled")
            return None

    def start_processing_stream(self):
        """Override to store intrinsic matrix and add input option"""
        # Get operating mode from user
        mode = self.select_operating_mode()
        
        if mode == 'exit':
            return
        
        # If AI mode, get task description
        if mode == 'ai':
            task_description = self.get_ai_task()
            if not task_description:
                print("No task provided. Switching to manual mode.")
                self.vision_mode = False
                mode = 'manual'
            else:
                self.vision_target_description = task_description
        
        print("\n" + "="*60)
        print("STARTING RGBD STREAM")
        print("="*60)
        
        if mode == 'manual':
            print("📱 MANUAL MODE ACTIVE")
            print("Instructions:")
            print("  • Move mouse over depth window to see 3D positions")
            print("  • Press 'p' to input specific coordinates")
            print("  • Press 'q' to quit")
            
        elif mode == 'ai':
            print("🤖 AI VISION MODE ACTIVE")
            print(f"Task: {self.vision_target_description}")
            print("Instructions:")
            print("  • AI will analyze the first frame automatically")
            print("  • Green crosshair shows AI target")
            print("  • Press 'r' to re-analyze with new task")
            print("  • Press 'q' to quit")
            
        elif mode == 'hybrid':
            print("🔄 HYBRID MODE ACTIVE")
            print("Instructions:")
            print("  • Mouse hover for manual targeting")
            print("  • Press 'v' for AI vision analysis")
            print("  • Press 'm' to toggle modes")
            print("  • Press 'c' to clear AI target")
            print("  • Press 'p' for manual coordinates")
            print("  • Press 'q' to quit")
        
        print("\n⏳ Waiting for RGBD stream...")
        
        first_frame_processed = False
        
        while True:
            self.event.wait()  # Wait for new frame to arrive

            # Copy the newly arrived RGBD frame
            depth = self.session.get_depth_frame()
            rgb = self.session.get_rgb_frame()
            confidence = self.session.get_confidence_frame()
            
            # Get depth intrinsics (Record3D only provides depth intrinsics)
            # For iPhone cameras, RGB and depth intrinsics are very similar
            depth_intrinsic_coeffs = self.session.get_intrinsic_mat()
            
            depth_intrinsic_mat = self.get_intrinsic_mat_from_coeffs(depth_intrinsic_coeffs)
            # Use same intrinsics for RGB as they're co-located on iPhone
            rgb_intrinsic_mat = depth_intrinsic_mat.copy()
            
            camera_pose = self.session.get_camera_pose()

            # Store both intrinsic matrices
            self.intrinsic_mat = depth_intrinsic_mat  # Keep for backward compatibility
            self.depth_intrinsic_mat = depth_intrinsic_mat
            self.rgb_intrinsic_mat = rgb_intrinsic_mat
            
            # Debug print intrinsics (only first time)
            if not hasattr(self, '_intrinsics_printed'):
                print(f"[Camera] Using shared intrinsics - fx:{depth_intrinsic_mat[0,0]:.1f} fy:{depth_intrinsic_mat[1,1]:.1f} cx:{depth_intrinsic_mat[0,2]:.1f} cy:{depth_intrinsic_mat[1,2]:.1f}")
                print(f"[Camera] Note: iPhone RGB and depth cameras are co-located with very similar intrinsics")
                self._intrinsics_printed = True

            # Postprocess frames
            if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                depth = cv2.flip(depth, 1)
                rgb = cv2.flip(rgb, 1)

            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Save depth for calculations and store RGB for vision analysis
            self.depth = depth
            self.current_rgb = rgb.copy()
            
            # Detect ArUco marker each frame for ALL modes (AFTER frames are processed and stored)
            if self.current_rgb is not None and self.rgb_intrinsic_mat is not None:
                was_detected = self.current_pose_T_cam_marker is not None
                self.detect_aruco_marker()
                # Only print when marker detection status changes
                if self.current_pose_T_cam_marker is not None and not was_detected:
                    print(f"[ArUco] ✅ Marker {self.aruco_marker_id} detected and pose estimated")
                elif self.current_pose_T_cam_marker is None and was_detected:
                    print(f"[ArUco] ❌ Marker {self.aruco_marker_id} lost")
            
            # Auto-analyze first frame in AI mode (AFTER ArUco detection is done)
            if mode == 'ai' and not first_frame_processed and self.vision_target_description and self.rgb_intrinsic_mat is not None:
                print("🔍 Analyzing first frame with AI...")
                result = self.vision_analyzer.analyze_image(self.current_rgb, self.vision_target_description, self.depth)
                
                if result and result.get('success'):
                    self.vision_result = result
                    x, y = result['target_pixel']
                    
                    # Scale coordinates from RGB to depth frame if needed
                    rgb_height, rgb_width = self.current_rgb.shape[:2]
                    depth_height, depth_width = self.depth.shape[:2]
                    
                    # Scale coordinates if RGB and depth have different dimensions
                    if rgb_width != depth_width or rgb_height != depth_height:
                        scale_x = depth_width / rgb_width
                        scale_y = depth_height / rgb_height
                        depth_x = int(x * scale_x)
                        depth_y = int(y * scale_y)
                        print(f"🔄 Scaling coordinates: RGB({x},{y}) -> Depth({depth_x},{depth_y})")
                        print(f"   RGB: {rgb_width}x{rgb_height}, Depth: {depth_width}x{depth_height}")
                    else:
                        depth_x, depth_y = x, y
                    
                    # Get depth and calculate 3D position
                    if 0 <= depth_y < self.depth.shape[0] and 0 <= depth_x < self.depth.shape[1]:
                        depth_value = float(self.depth[depth_y, depth_x])
                        if depth_value > 0:
                            # Use original RGB coordinates for 3D calculation (intrinsics are for RGB)
                            pos_3d = self.pixel_to_3d_position(x, y, depth_value, self.rgb_intrinsic_mat)
                            if pos_3d:
                                X, Y, Z = pos_3d
                                print(f"\n✅ AI FOUND TARGET!")
                                print(f"Object: {result.get('object_description', 'Found')}")
                                print(f"RGB Pixel: ({x}, {y})")
                                print(f"Depth Pixel: ({depth_x}, {depth_y})")
                                print(f"Depth: {depth_value:.3f}m")
                                print(f"3D Position: ({X:.3f}, {Y:.3f}, {Z:.3f})m")
                                if self.current_pose_T_cam_marker is not None:
                                    # Transform to ArUco marker frame if available
                                    marker_coords = self.pixel_to_marker_coordinates(x, y)
                                    if marker_coords is not None:
                                        print(f"[ArUco] Relative to marker: ({marker_coords[0]:.3f},{marker_coords[1]:.3f},{marker_coords[2]:.3f})")
                                        
                                        # Send to robot
                                        self.prompt_and_move_robot(marker_coords)
                                    else:
                                        print("[ArUco] ❌ Could not convert to marker coordinates")
                                else:
                                    print("[ArUco] ❌ No marker detected - cannot transform coordinates")
                                print(f"Confidence: {result.get('confidence', 'N/A')}")
                                
                                # Store RGB coordinates for overlay (since RGB display uses RGB coords)
                                self.last_mouse_pos = (x, y)
                                self.depth_value = depth_value
                            else:
                                print("⚠️  Could not calculate 3D position")
                        else:
                            print(f"⚠️  Invalid depth at target: {depth_value}")
                    else:
                        print(f"⚠️  Target coordinates out of bounds: RGB({x},{y}) -> Depth({depth_x},{depth_y})")
                        print(f"   Depth frame size: {depth_width}x{depth_height}")
                else:
                    print("❌ AI could not find the target object")
                    if result:
                        print(f"Reason: {result.get('reasoning', 'Unknown')}")
                
                first_frame_processed = True

            # Apply vision overlay if in vision mode
            if self.vision_mode and self.vision_result:
                rgb = self.vision_analyzer.create_visualization(rgb, self.vision_result)
            
            # Overlay ArUco marker visualization on RGB
            if self.current_pose_T_cam_marker is not None:
                # Draw ArUco marker and axes
                gray = cv2.cvtColor(self.current_rgb, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = self.aruco_detector.detectMarkers(gray)
                
                if ids is not None and self.aruco_marker_id in ids.flatten():
                    idx = list(ids.flatten()).index(self.aruco_marker_id)
                    cv2.aruco.drawDetectedMarkers(rgb, [corners[idx]])
                    cv2.drawFrameAxes(rgb, self.rgb_intrinsic_mat, self.dist_coeffs, 
                                    cv2.Rodrigues(self.current_pose_T_cam_marker[:3,:3])[0], 
                                    self.current_pose_T_cam_marker[:3,3], self.aruco_marker_size*0.5)
                    
                    # If a pixel was clicked, draw its marker coordinate near the pixel
                    if self.clicked_pixel and self.clicked_marker is not None:
                        click_u, click_v = self.clicked_pixel
                        cv2.circle(rgb, (click_u, click_v), 6, (0, 0, 255), 2)
                        label = f"({self.clicked_marker[0]:.2f},{self.clicked_marker[1]:.2f},{self.clicked_marker[2]:.2f})m"
                        cv2.putText(rgb, label, (click_u + 8, click_v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            
            # Add mode indicator to RGB display
            mode_text = f"Mode: {'AI VISION' if self.vision_mode else 'MANUAL'}"
            cv2.putText(rgb, mode_text, (10, rgb.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if self.vision_mode and self.vision_target_description:
                target_text = f"Task: {self.vision_target_description[:40]}..."
                cv2.putText(rgb, target_text, (10, rgb.shape[0] - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Show the RGBD Stream
            cv2.imshow('RGB', rgb)

            # Show depth with overlay
            depth_vis = depth.copy()
            
            # Show crosshair for manual mode or vision target
            if self.last_mouse_pos and self.depth_value is not None:
                rgb_x, rgb_y = self.last_mouse_pos
                
                # Convert RGB coordinates to depth coordinates for overlay
                depth_x, depth_y = self.scale_coordinates_rgb_to_depth(rgb_x, rgb_y)
                
                # Different colors for different modes
                color = (0, 255, 0) if self.vision_mode else (255, 255, 255)
                
                # Draw crosshair on depth display using depth coordinates
                overlay = depth_vis.copy()
                if self.vision_mode:
                    # Larger crosshair for vision mode
                    cv2.line(overlay, (depth_x - 15, depth_y), (depth_x + 15, depth_y), color, 2)
                    cv2.line(overlay, (depth_x, depth_y - 15), (depth_x, depth_y + 15), color, 2)
                    cv2.circle(overlay, (depth_x, depth_y), 10, color, 2)
                else:
                    # Small circle for manual mode
                    cv2.circle(overlay, (depth_x, depth_y), 5, color, -1)
                
                alpha = 0.7 if self.vision_mode else 0.5
                depth_vis = cv2.addWeighted(overlay, alpha, depth_vis, 1 - alpha, 0)
                
                # Show 3D position if available (using RGB coordinates for calculation)
                if self.rgb_intrinsic_mat is not None:
                    pos_3d = self.pixel_to_3d_position(rgb_x, rgb_y, self.depth_value, self.rgb_intrinsic_mat)
                    if pos_3d:
                        X, Y, Z = pos_3d
                        if self.vision_mode:
                            text = f"AI: {self.depth_value:.3f}m ({X:.2f},{Y:.2f},{Z:.2f})"
                        else:
                            text = f"{self.depth_value:.3f}m ({X:.2f},{Y:.2f},{Z:.2f})"
                    else:
                        text = f"{self.depth_value:.3f} m"
                else:
                    text = f"{self.depth_value:.3f} m"
                    
                cv2.putText(depth_vis, text, (depth_x + 20, depth_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add mode indicator to depth display
            if mode == 'ai':
                mode_indicator = "AI VISION"
                color = (0, 255, 0)
            elif mode == 'manual':
                mode_indicator = "MANUAL"
                color = (255, 255, 255)
            else:  # hybrid
                mode_indicator = f"{'AI VISION' if self.vision_mode else 'MANUAL'} (HYBRID)"
                color = (0, 255, 0) if self.vision_mode else (255, 255, 255)
            
            cv2.putText(depth_vis, mode_indicator, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.imshow(self.depth_window_name, depth_vis)

            if confidence.shape[0] > 0 and confidence.shape[1] > 0:
                cv2.imshow('Confidence', confidence * 100)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            elif key == ord('p') and mode != 'ai':
                print("\n" + "="*50)
                self.get_3d_position_from_input()
                print("="*50)
            elif key == ord('r') and mode == 'ai':
                # Re-analyze with new task in AI mode
                new_task = self.get_ai_task()
                if new_task:
                    self.vision_target_description = new_task
                    print("🔄 Re-analyzing with new task...")
                    result = self.vision_analyzer.analyze_image(self.current_rgb, new_task, self.depth)
                    
                    if result and result.get('success'):
                        self.vision_result = result
                        x, y = result['target_pixel']
                        
                        # Scale coordinates from RGB to depth frame if needed
                        rgb_height, rgb_width = self.current_rgb.shape[:2]
                        depth_height, depth_width = self.depth.shape[:2]
                        
                        if rgb_width != depth_width or rgb_height != depth_height:
                            scale_x = depth_width / rgb_width
                            scale_y = depth_height / rgb_height
                            depth_x = int(x * scale_x)
                            depth_y = int(y * scale_y)
                        else:
                            depth_x, depth_y = x, y
                        
                        if 0 <= depth_y < self.depth.shape[0] and 0 <= depth_x < self.depth.shape[1]:
                            depth_value = float(self.depth[depth_y, depth_x])
                            if depth_value > 0:
                                pos_3d = self.pixel_to_3d_position(x, y, depth_value, self.rgb_intrinsic_mat)
                                if pos_3d:
                                    X, Y, Z = pos_3d
                                    print(f"\n✅ NEW TARGET FOUND!")
                                    print(f"Object: {result.get('object_description', 'Found')}")
                                    print(f"RGB Pixel: ({x}, {y})")
                                    print(f"Depth Pixel: ({depth_x}, {depth_y})")
                                    print(f"Depth: {depth_value:.3f}m")
                                    print(f"3D Position: ({X:.3f}, {Y:.3f}, {Z:.3f})m")
                                    
                                    # Transform to ArUco marker frame if available
                                    if self.current_pose_T_cam_marker is not None:
                                        marker_coords = self.pixel_to_marker_coordinates(x, y)
                                        if marker_coords is not None:
                                            print(f"[ArUco] Relative to marker: ({marker_coords[0]:.3f},{marker_coords[1]:.3f},{marker_coords[2]:.3f})")
                                            
                                            # Send to robot
                                            self.prompt_and_move_robot(marker_coords)
                                        else:
                                            print("[ArUco] ❌ Could not convert to marker coordinates")
                                    else:
                                        print("[ArUco] ❌ No marker detected - cannot transform coordinates")
                                    
                                    self.last_mouse_pos = (x, y)
                                    self.depth_value = depth_value
                    else:
                        print("❌ Could not find new target")
            elif key == ord('v') and mode == 'hybrid':
                if self.vision_analyzer.is_available():
                    self.get_vision_target()
                else:
                    print("❌ Vision analyzer not available (API key required)")
            elif key == ord('m') and mode == 'hybrid':
                self.toggle_vision_mode()
            elif key == ord('c') and mode == 'hybrid':
                if self.vision_mode:
                    self.vision_mode = False
                    self.vision_result = None
                    self.vision_target_description = ""
                    print("🧹 Cleared vision target")
                else:
                    print("ℹ️  Not in vision mode")
            elif key == ord('+') or key == ord('='):
                # Increase marker size
                self.aruco_marker_size += 0.01
                print(f"[Debug] Marker size increased to: {self.aruco_marker_size:.3f}m")
            elif key == ord('-') or key == ord('_'):
                # Decrease marker size
                self.aruco_marker_size = max(0.01, self.aruco_marker_size - 0.01)
                print(f"[Debug] Marker size decreased to: {self.aruco_marker_size:.3f}m")

            self.event.clear()

def main():
    """Main function to run the kinematics application"""
    print("Kinematics AI - RGBD 3D Position Calculator")
    print("=" * 50)
    
    app = KinematicsApp()
    
    try:
        app.connect_to_device(dev_idx=0)
        app.start_processing_stream()
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 