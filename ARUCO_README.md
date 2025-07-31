# ArUco Reference Frame Detector

A standalone application that reads RGB data from your RGBD camera and detects ArUco markers as reference frames for coordinate transformation.

## Features

- **Real-time ArUco Detection**: Detects ArUco markers in real-time using your RGBD camera
- **Reference Frame Tracking**: Uses ArUco markers as coordinate reference frames
- **3D Position Transformation**: Transforms target positions from camera coordinates to marker coordinates
- **Interactive Target Selection**: Click to select target points and see their position relative to the marker
- **Visual Feedback**: Shows marker detection status, coordinate axes, and target positions
- **Information Display**: Real-time display of marker position, orientation, and target coordinates

## Quick Start

### 1. Generate ArUco Markers

```bash
# Generate marker ID 0 (default reference frame)
python generate_aruco.py --id 0 --show

# Generate different marker IDs
python generate_aruco.py --id 1 --size 300
python generate_aruco.py --id 2 --size 400
```

### 2. Print and Place Markers

- Print the generated marker images
- Place them in your camera's field of view
- Ensure good lighting and contrast
- Keep markers roughly parallel to the camera

### 3. Run the Detector

```bash
python aruco_transformer.py
```

## Usage

### Controls

- **Mouse Click**: Select target points on the detection window
- **'q'**: Quit the application
- **'i'**: Toggle information display

### What You'll See

1. **Detection Window**: Shows the camera feed with ArUco marker overlay
   - Green outline around detected markers
   - Coordinate axes (red=X, green=Y, blue=Z)
   - Target point indicator (red circle)

2. **Info Window**: Real-time information display
   - Marker detection status
   - Position in camera frame (X, Y, Z)
   - Distance from camera
   - Orientation (roll, pitch, yaw)
   - Target position relative to marker (if target selected)

### Understanding the Output

- **Camera Frame**: 3D coordinates relative to the camera
- **Marker Frame**: 3D coordinates relative to the ArUco marker
- **Distance**: Euclidean distance from camera to marker
- **Orientation**: Euler angles in radians

## Applications

### Robotics
- Use ArUco markers as reference frames for robot positioning
- Transform target coordinates to robot coordinate system
- Precise positioning for robotic manipulation

### Augmented Reality
- Anchor virtual objects to physical markers
- Create stable reference points for AR applications
- Track object positions relative to fixed markers

### Precision Assembly
- Align parts using marker coordinates
- Measure relative positions between components
- Quality control and inspection

## Technical Details

### ArUco Marker Specifications

- **Dictionary**: DICT_ARUCO_ORIGINAL (supports IDs 0-999)
- **Physical Size**: 5cm (configurable)
- **Detection**: Real-time pose estimation using solvePnP
- **Coordinate System**: Right-handed (X-right, Y-down, Z-forward)

### Camera Requirements

- **RGBD Camera**: iPhone with Record3D app or compatible device
- **Resolution**: Any resolution supported by Record3D
- **Frame Rate**: Real-time processing
- **Calibration**: Automatic intrinsic matrix from RGBD stream

### Coordinate Transformation

The system performs the following transformations:

1. **Pixel to 3D**: Convert pixel coordinates to 3D camera coordinates
2. **Camera to Marker**: Transform from camera frame to marker frame
3. **Pose Estimation**: Estimate marker pose using solvePnP
4. **Coordinate Axes**: Visualize marker coordinate system

## Troubleshooting

### No Markers Detected

1. **Check Lighting**: Ensure good, even lighting
2. **Marker Visibility**: Make sure marker is fully in view
3. **Contrast**: Use high-contrast markers
4. **Distance**: Keep markers within camera range (0.2m to 5m)
5. **Orientation**: Keep markers roughly parallel to camera

### Poor Detection

1. **Improve Lighting**: Add more light or reduce shadows
2. **Reduce Distance**: Move camera closer to marker
3. **Check Motion**: Minimize camera movement
4. **Marker Quality**: Use high-quality printed markers

### Transformation Errors

1. **Valid Depth**: Ensure target has valid depth data
2. **Marker Detection**: Verify marker is detected first
3. **Camera Calibration**: Check intrinsic matrix values
4. **Target Selection**: Click on areas with valid depth

## Examples

### Basic Usage

```bash
# Generate a marker
python generate_aruco.py --id 0 --show

# Run detector
python aruco_transformer.py

# Click on objects to see their position relative to the marker
```

### Multiple Markers

```bash
# Generate multiple markers
python generate_aruco.py --id 0
python generate_aruco.py --id 1
python generate_aruco.py --id 2

# Place them in different locations
# The detector will find marker ID 0 as the reference frame
```

## Integration with Main Application

The ArUco transformer can also be integrated with the main kinematics application:

```python
from aruco_transformer import ArucoTransformer

# Initialize with camera parameters
transformer = ArucoTransformer(camera_matrix=camera_matrix)

# Detect marker
aruco_pose = transformer.detect_aruco_marker(rgb_image, marker_id=0)

# Transform target position
target_3d_camera = np.array([x, y, z])
result = transformer.get_target_relative_to_aruco(rgb_image, target_3d_camera, 0)
```

## File Structure

- `aruco_transformer.py`: Standalone ArUco detector application
- `generate_aruco.py`: ArUco marker generator
- `aruco_marker_0.png`: Generated marker image

## Requirements

- Python 3.7+
- OpenCV 4.5+
- NumPy
- Record3D (for RGBD camera)
- ArUco markers (generated or printed)

## Next Steps

1. **Test Detection**: Run the detector with a printed marker
2. **Experiment**: Try different marker positions and orientations
3. **Applications**: Use for robotics, AR, or precision positioning
4. **Integration**: Incorporate into your main application 