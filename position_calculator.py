import numpy as np
from typing import Tuple, Optional

class Position3DCalculator:
    """
    Utility class to convert 2D pixel coordinates to 3D positions using camera intrinsics.
    Designed for iPhone 13 Pro with Record3D stream specifications.
    """
    
    def __init__(self, intrinsic_matrix: Optional[np.ndarray] = None):
        """
        Initialize the position calculator.
        
        Args:
            intrinsic_matrix: 3x3 camera intrinsic matrix. If None, will be set later.
        """
        self.intrinsic_matrix = intrinsic_matrix
        self.fx = None
        self.fy = None 
        self.cx = None
        self.cy = None
        
        if intrinsic_matrix is not None:
            self._extract_intrinsic_params()
    
    def set_intrinsic_matrix(self, intrinsic_matrix: np.ndarray):
        """Set the camera intrinsic matrix and extract parameters."""
        self.intrinsic_matrix = intrinsic_matrix
        self._extract_intrinsic_params()
    
    def _extract_intrinsic_params(self):
        """Extract focal lengths and principal point from intrinsic matrix."""
        if self.intrinsic_matrix is not None:
            self.fx = self.intrinsic_matrix[0, 0]  # focal length x
            self.fy = self.intrinsic_matrix[1, 1]  # focal length y  
            self.cx = self.intrinsic_matrix[0, 2]  # principal point x
            self.cy = self.intrinsic_matrix[1, 2]  # principal point y
    
    def pixel_to_3d(self, u: int, v: int, depth: float) -> Optional[Tuple[float, float, float]]:
        """
        Convert pixel coordinates and depth to 3D position relative to camera.
        
        Uses the pinhole camera model:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy  
        Z = depth
        
        Args:
            u: x pixel coordinate (column)
            v: y pixel coordinate (row)
            depth: depth value in meters
            
        Returns:
            Tuple of (X, Y, Z) coordinates in meters, or None if calculation fails
        """
        if self.intrinsic_matrix is None:
            raise ValueError("Intrinsic matrix not set. Call set_intrinsic_matrix() first.")
        
        if depth <= 0:
            print(f"Warning: Invalid depth value: {depth}")
            return None
        
        # Calculate 3D coordinates using pinhole camera model
        X = (u - self.cx) * depth / self.fx
        Y = (v - self.cy) * depth / self.fy
        Z = depth
        
        return (X, Y, Z)
    
    def calculate_distance_from_camera(self, position_3d: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance from camera origin to 3D point."""
        X, Y, Z = position_3d
        return np.sqrt(X*X + Y*Y + Z*Z)
    
    def pixel_to_3d_with_info(self, u: int, v: int, depth: float) -> dict:
        """
        Convert pixel to 3D position and return detailed information.
        
        Returns:
            Dictionary containing position, distance, and calculation details
        """
        position_3d = self.pixel_to_3d(u, v, depth)
        
        if position_3d is None:
            return {
                'success': False,
                'error': 'Failed to calculate 3D position',
                'pixel_coords': (u, v),
                'depth': depth
            }
        
        X, Y, Z = position_3d
        distance = self.calculate_distance_from_camera(position_3d)
        
        return {
            'success': True,
            'pixel_coords': (u, v),
            'depth': depth,
            'position_3d': position_3d,
            'X': X,
            'Y': Y, 
            'Z': Z,
            'distance_from_camera': distance,
            'intrinsic_params': {
                'fx': self.fx,
                'fy': self.fy,
                'cx': self.cx,
                'cy': self.cy
            }
        }
    
    def print_calculation_details(self, result: dict):
        """Print detailed calculation results in a formatted way."""
        if not result['success']:
            print(f"Error: {result['error']}")
            return
        
        print("\n" + "="*60)
        print("3D POSITION CALCULATION RESULTS")
        print("="*60)
        print(f"Input:")
        print(f"  Pixel coordinates (u, v): {result['pixel_coords']}")
        print(f"  Depth: {result['depth']:.3f} meters")
        print(f"\nCamera Intrinsic Parameters:")
        params = result['intrinsic_params']
        print(f"  Focal length (fx): {params['fx']:.2f}")
        print(f"  Focal length (fy): {params['fy']:.2f}")
        print(f"  Principal point (cx): {params['cx']:.2f}")
        print(f"  Principal point (cy): {params['cy']:.2f}")
        print(f"\nCalculated 3D Position (relative to camera):")
        print(f"  X: {result['X']:+.3f} meters")
        print(f"  Y: {result['Y']:+.3f} meters") 
        print(f"  Z: {result['Z']:+.3f} meters")
        print(f"  Distance from camera: {result['distance_from_camera']:.3f} meters")
        print("="*60)
    
    @staticmethod
    def create_sample_intrinsic_matrix() -> np.ndarray:
        """
        Create a sample intrinsic matrix with typical iPhone 13 Pro values.
        Note: These are approximate values - actual values should come from Record3D.
        """
        # Approximate values for iPhone 13 Pro rear camera
        fx = 1000.0  # focal length x
        fy = 1000.0  # focal length y
        cx = 640.0   # principal point x (assuming 1280x960 resolution)
        cy = 480.0   # principal point y
        
        return np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ])

# Example usage and testing functions
def example_usage():
    """Demonstrate how to use the Position3DCalculator."""
    print("Position3D Calculator - Example Usage")
    print("="*50)
    
    # Create calculator with sample intrinsic matrix
    calculator = Position3DCalculator()
    sample_intrinsic = Position3DCalculator.create_sample_intrinsic_matrix()
    calculator.set_intrinsic_matrix(sample_intrinsic)
    
    # Example calculations
    test_cases = [
        (640, 480, 1.0),  # Center of image, 1 meter away
        (320, 240, 2.0),  # Upper left quadrant, 2 meters away
        (960, 720, 0.5),  # Lower right quadrant, 0.5 meters away
    ]
    
    for u, v, depth in test_cases:
        result = calculator.pixel_to_3d_with_info(u, v, depth)
        calculator.print_calculation_details(result)
        print()

def interactive_calculator():
    """Interactive command-line calculator for testing."""
    print("Interactive 3D Position Calculator")
    print("="*50)
    print("Note: Using sample intrinsic matrix (typical iPhone 13 Pro values)")
    print("In real usage, this should come from Record3D stream")
    print()
    
    calculator = Position3DCalculator()
    sample_intrinsic = Position3DCalculator.create_sample_intrinsic_matrix()
    calculator.set_intrinsic_matrix(sample_intrinsic)
    
    while True:
        try:
            print("\nEnter pixel coordinates and depth:")
            u = int(input("u (x pixel coordinate): "))
            v = int(input("v (y pixel coordinate): "))
            depth = float(input("depth (meters): "))
            
            result = calculator.pixel_to_3d_with_info(u, v, depth)
            calculator.print_calculation_details(result)
            
            continue_calc = input("\nCalculate another point? (y/n): ").lower()
            if continue_calc != 'y':
                break
                
        except ValueError:
            print("Invalid input. Please enter numeric values.")
        except KeyboardInterrupt:
            print("\nExiting calculator...")
            break

if __name__ == '__main__':
    # Run example usage
    example_usage()
    
    # Optionally run interactive calculator
    run_interactive = input("Run interactive calculator? (y/n): ").lower()
    if run_interactive == 'y':
        interactive_calculator() 