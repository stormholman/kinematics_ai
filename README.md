# Kinematics AI

A computer vision and robotics system that combines RGBD cameras, ArUco marker tracking, and AI-powered object detection for precise robotic manipulation.

## Features

- **RGBD Stream Processing**: Real-time RGB-D data from iPhone using Record3D
- **ArUco Marker Tracking**: Precise 6DOF pose estimation for coordinate frame reference
- **AI Vision Analysis**: Claude-powered object detection with natural language descriptions
- **3D Coordinate Transformation**: Pixel → Camera → ArUco Marker → Robot coordinate conversion
- **Robot Control**: HTTP API integration for robotic arm movement
- **Multiple Operation Modes**: Manual, AI Vision, and Hybrid modes

## System Architecture

```
iPhone (Record3D) → RGBD Stream → Computer Vision Pipeline
                                        ↓
ArUco Detection ← RGB Frame ← Coordinate Transformation → Robot Commands
                                        ↓
AI Object Detection ← Depth Frame ← 3D Position Calculation
```

## Installation

### Prerequisites

- Python 3.8+
- iPhone with Record3D app
- ArUco marker (4x4cm, ID 0, DICT_ARUCO_ORIGINAL)
- Anthropic API key for AI vision features

### Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

- `opencv-python` - Computer vision and ArUco detection
- `numpy` - Numerical computations
- `record3d` - iPhone RGBD streaming
- `anthropic` - AI vision analysis
- `requests` - Robot HTTP communication
- `Pillow` - Image processing

## Quick Start

1. **Setup ArUco Marker**
   ```bash
   python generate_aruco.py --id 0 --size 200 --output marker_0.png
   ```
   Print the generated marker at 4x4cm size.

2. **Configure API Key**
   ```bash
   export ANTHROPIC_API_KEY="your_api_key_here"
   ```

3. **Start Record3D** on your iPhone and connect to the same network.

4. **Run the Application**
```bash
python main.py
```

## Operation Modes

### Manual Mode
- Mouse hover shows real-time 3D coordinates
- Left-click sends coordinates to robot
- Precise manual control

### AI Vision Mode  
- Describe objects in natural language
- AI automatically finds and targets objects
- Examples: "yellow banana", "red cup on table"

### Hybrid Mode
- Switch between manual and AI modes
- Press 'v' for AI analysis, 'm' to toggle modes

## Key Components

### `main.py`
Main application with RGBD processing, ArUco detection, and robot control.

### `vision_analyzer.py`  
AI-powered object detection using Claude vision API with accuracy improvements:
- Image upsampling for better detection
- Bounding box validation
- Depth-based pixel refinement
- Precision-focused prompting

### `aruco_transformer.py`
Standalone ArUco detection and coordinate transformation (legacy).

### `aruco_rgbd_stream.py`
Simplified ArUco testing application for pixel-to-marker coordinate conversion.

### `rgbd_feed.py`
Base RGBD streaming functionality using Record3D.

### `position_calculator.py`
3D coordinate calculation utilities.

## Coordinate Systems

### ArUco Marker Frame
- **Origin**: Top-left corner of marker
- **+X**: Rightward along top edge  
- **+Y**: Downward along left edge
- **+Z**: Out of marker plane toward camera

### Robot Frame Mapping
```python
robot_x = -aruco_y    # Robot X = -ArUco Y
robot_y = -aruco_x    # Robot Y = -ArUco X  
robot_z = -aruco_z - 0.13  # Robot Z = -ArUco Z - 13cm offset
```

## Configuration

### Robot Endpoint
Update in `main.py`:
```python
self.robot_endpoint = "http://192.168.178.190/move/absolute?robot_id=0"
```

### ArUco Settings
```python
self.aruco_marker_id = 0           # Marker ID
self.aruco_marker_size = 0.04      # 4cm physical size
```

### Camera Calibration
Automatic calibration from Record3D intrinsics. Manual override available in code.

## Usage Examples

### Manual Targeting
1. Start application in Manual mode
2. Hover mouse over objects to see 3D coordinates
3. Left-click to send coordinates to robot
4. Confirm robot movement when prompted

### AI Object Detection
1. Start application in AI Vision mode
2. Describe target: "pick up the blue cup"
3. AI analyzes frame and finds object
4. Coordinates automatically sent to robot

### Hybrid Workflow
1. Start in Hybrid mode
2. Use manual mode for precise positioning
3. Press 'v' to switch to AI for object recognition
4. Press 'm' to toggle between modes

## Keyboard Controls

- `q` - Quit application
- `p` - Manual coordinate input (Manual/Hybrid mode)
- `v` - AI vision analysis (Hybrid mode)
- `m` - Toggle manual/AI modes (Hybrid mode)
- `r` - Re-analyze with new task (AI mode)
- `c` - Clear AI target (Hybrid mode)
- `+/-` - Adjust ArUco marker size for debugging

## Troubleshooting

### ArUco Detection Issues
- Ensure good lighting and contrast
- Check marker is 4x4cm and ID 0
- Verify marker is DICT_ARUCO_ORIGINAL type
- Try adjusting marker size with +/- keys

### Depth Accuracy
- Ensure RGB and depth frames are aligned
- Check for proper Record3D connection
- Verify camera intrinsics are loaded

### AI Vision Accuracy
- Use specific object descriptions
- Ensure objects are clearly visible
- Good lighting improves detection
- Bounding box visualization helps debug

### Robot Communication
- Verify robot endpoint URL
- Check network connectivity
- Confirm robot API is accessible
- Test with curl commands

## API Reference

### VisionAnalyzer.analyze_image()
```python
result = analyzer.analyze_image(rgb_image, "target description", depth_frame)
```

Returns:
```python
{
    "success": True,
    "bbox": [x0, y0, x1, y1],
    "target_pixel": [x, y], 
    "confidence": 0.95,
    "object_description": "found object",
    "reasoning": "why these coordinates"
}
```

### Coordinate Transformation
```python
# Pixel to 3D camera coordinates
pos_3d = pixel_to_3d_position(u, v, depth, camera_matrix)

# Camera to ArUco marker coordinates  
marker_coords = pixel_to_marker_coordinates(rgb_x, rgb_y)

# ArUco to robot coordinates
robot_coords = convert_to_robot_frame(marker_coords)
```

## Development

### Testing ArUco Detection
```bash
python aruco_rgbd_stream.py
```

### Generating Test Markers
```bash
python generate_aruco.py --help
```

### Vision System Testing
```bash
python vision_analyzer.py test
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Record3D for iPhone RGBD streaming
- OpenCV for computer vision capabilities  
- Anthropic Claude for AI vision analysis
- ArUco marker detection system 