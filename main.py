import numpy as np
import cv2
from rgbd_feed import DemoApp
from vision_analyzer import VisionAnalyzer

class KinematicsApp(DemoApp):
    def __init__(self):
        super().__init__()
        self.intrinsic_mat = None
        
        # Vision analyzer for AI-powered object detection
        self.vision_analyzer = VisionAnalyzer()
        self.vision_mode = False
        self.vision_result = None
        self.vision_target_description = ""
        
        # Current RGB frame for vision analysis
        self.current_rgb = None
        
    def pixel_to_3d_position(self, u, v, depth_value):
        """
        Convert pixel coordinates (u,v) and depth to 3D position relative to camera.
        
        Args:
            u (int): x pixel coordinate
            v (int): y pixel coordinate  
            depth_value (float): depth in meters at that pixel
            
        Returns:
            tuple: (X, Y, Z) 3D coordinates in meters relative to camera
        """
        if self.intrinsic_mat is None:
            print("Warning: No intrinsic matrix available yet")
            return None
            
        # Extract intrinsic parameters
        fx = self.intrinsic_mat[0, 0]  # focal length x
        fy = self.intrinsic_mat[1, 1]  # focal length y
        cx = self.intrinsic_mat[0, 2]  # principal point x
        cy = self.intrinsic_mat[1, 2]  # principal point y
        
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
        """Override mouse callback to also calculate 3D position"""
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
                    if self.intrinsic_mat is not None:
                        pos_3d = self.pixel_to_3d_position(rgb_x, rgb_y, depth_value)
                        if pos_3d:
                            X, Y, Z = pos_3d
                            print(f"Manual D({x},{y})->RGB({rgb_x},{rgb_y}) -> Depth: {depth_value:.3f}m, 3D: ({X:.3f},{Y:.3f},{Z:.3f})")
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
                
                if self.intrinsic_mat is not None:
                    pos_3d = self.pixel_to_3d_position(rgb_x, rgb_y, depth_value)
                    if pos_3d:
                        X, Y, Z = pos_3d
                        print(f"Vision Click D({x},{y})->RGB({rgb_x},{rgb_y}) -> Depth: {depth_value:.3f}m, 3D: ({X:.3f},{Y:.3f},{Z:.3f})")
    
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
            result = self.vision_analyzer.analyze_image(self.current_rgb, target_description)
            
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
                    if depth_value > 0 and self.intrinsic_mat is not None:
                        pos_3d = self.pixel_to_3d_position(x, y, depth_value)
                        if pos_3d:
                            X, Y, Z = pos_3d
                            print(f"\n✅ TARGET ACQUIRED!")
                            print(f"Object: {result.get('object_description', 'Found')}")
                            print(f"RGB Pixel: ({x}, {y})")
                            print(f"Depth Pixel: ({depth_x}, {depth_y})")
                            print(f"Depth: {depth_value:.3f}m")
                            print(f"3D Position: ({X:.3f}, {Y:.3f}, {Z:.3f})m")
                            print(f"Confidence: {result.get('confidence', 'N/A')}")
                            
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
            intrinsic_mat = self.get_intrinsic_mat_from_coeffs(self.session.get_intrinsic_mat())
            camera_pose = self.session.get_camera_pose()

            # Store intrinsic matrix for 3D calculations
            self.intrinsic_mat = intrinsic_mat

            # Postprocess frames
            if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                depth = cv2.flip(depth, 1)
                rgb = cv2.flip(rgb, 1)

            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Save depth for calculations and store RGB for vision analysis
            self.depth = depth
            self.current_rgb = rgb.copy()
            
            # Auto-analyze first frame in AI mode
            if mode == 'ai' and not first_frame_processed and self.vision_target_description:
                print("🔍 Analyzing first frame with AI...")
                result = self.vision_analyzer.analyze_image(self.current_rgb, self.vision_target_description)
                
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
                            pos_3d = self.pixel_to_3d_position(x, y, depth_value)
                            if pos_3d:
                                X, Y, Z = pos_3d
                                print(f"\n✅ AI FOUND TARGET!")
                                print(f"Object: {result.get('object_description', 'Found')}")
                                print(f"RGB Pixel: ({x}, {y})")
                                print(f"Depth Pixel: ({depth_x}, {depth_y})")
                                print(f"Depth: {depth_value:.3f}m")
                                print(f"3D Position: ({X:.3f}, {Y:.3f}, {Z:.3f})m")
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
                if self.intrinsic_mat is not None:
                    pos_3d = self.pixel_to_3d_position(rgb_x, rgb_y, self.depth_value)
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
                    result = self.vision_analyzer.analyze_image(self.current_rgb, new_task)
                    
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
                                pos_3d = self.pixel_to_3d_position(x, y, depth_value)
                                if pos_3d:
                                    X, Y, Z = pos_3d
                                    print(f"\n✅ NEW TARGET FOUND!")
                                    print(f"Object: {result.get('object_description', 'Found')}")
                                    print(f"RGB Pixel: ({x}, {y})")
                                    print(f"Depth Pixel: ({depth_x}, {depth_y})")
                                    print(f"Depth: {depth_value:.3f}m")
                                    print(f"3D Position: ({X:.3f}, {Y:.3f}, {Z:.3f})m")
                                    
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