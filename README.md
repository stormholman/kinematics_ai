# Kinematics AI - RGBD 3D Position Calculator

This project provides tools to capture RGBD data from iPhone 13 Pro using Record3D and convert 2D pixel coordinates to 3D positions relative to the camera.

## Files Overview

- **`rgbd_feed.py`** - Original RGBD stream capture from iPhone using Record3D
- **`main.py`** - Main application with real-time 3D position calculation and AI vision
- **`position_calculator.py`** - Utility class for 3D position calculations and testing
- **`vision_analyzer.py`** - AI-powered object detection using Claude vision API
- **`config.py.template`** - Configuration template for API keys

## Requirements

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install numpy opencv-python record3d anthropic pillow
```

You'll also need:
- iPhone 13 Pro (or compatible device) with Record3D app
- Both devices on the same WiFi network
- Anthropic API key (for AI vision features - optional)

## Setup

### API Key Configuration (for AI Vision)

1. Get an Anthropic API key from [https://console.anthropic.com/](https://console.anthropic.com/)

2. **Option 1: Using config file (recommended)**
   ```bash
   cp config.py.template config.py
   # Edit config.py and add your API key
   ```

3. **Option 2: Environment variable**
   ```bash
   export ANTHROPIC_API_KEY="your-anthropic-api-key-here"
   ```

## Usage

### 1. Real-time RGBD Stream with AI Vision & 3D Position Calculation

Run the main application:

```bash
python main.py
```

**Two Operating Modes:**

#### MANUAL Mode (Default)
- **Mouse**: Move over the depth window to see real-time 3D positions
- Click to get precise measurements

#### AI VISION Mode 
- **'v' key**: Analyze objects using natural language prompts
- Examples: "yellow banana", "red cup on the table", "coffee mug"
- AI automatically identifies and targets objects

**Controls:**
- **'v'**: Analyze objects with AI vision (describe what you want to target)
- **'m'**: Toggle between MANUAL and VISION modes
- **'c'**: Clear current vision target
- **'p'**: Input specific pixel coordinates manually  
- **'q'**: Quit the application

**Features:**
- Real-time depth and RGB streaming from iPhone
- Two modes: Manual cursor selection vs AI-powered object detection
- Mouse hover shows 3D position coordinates in manual mode
- AI vision identifies objects from natural language descriptions
- Visual overlays showing depth values and 3D coordinates
- Smart crosshairs with different colors for each mode

### 2. Standalone 3D Position Calculator

Test the position calculation independently:

```bash
python position_calculator.py
```

This provides:
- Example calculations with sample data
- Interactive calculator for testing conversions
- Detailed output showing calculation steps

### 3. Original RGBD Feed

Run the original feed viewer:

```bash
python rgbd_feed.py
```

### 4. Test AI Vision (Standalone)

Test the vision analyzer independently:

```bash
python vision_analyzer.py test
```

## AI Vision Examples

The AI vision system can identify objects using natural language descriptions:

**Simple Objects:**
- "yellow banana"
- "red apple" 
- "coffee mug"
- "water bottle"

**Complex Descriptions:**
- "the book with blue cover"
- "red cup on the table"
- "yellow banana on the left"
- "the phone next to the laptop"
- "green bottle in the center"

**Tips for Better Results:**
- Be specific about colors, shapes, and positions
- Use simple, clear descriptions
- Objects should be clearly visible in the RGB frame
- Good lighting improves detection accuracy

## How 3D Position Calculation Works

The conversion from 2D pixel coordinates to 3D positions uses the **pinhole camera model**:

```
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy  
Z = depth
```

Where:
- `(u, v)` = pixel coordinates
- `(cx, cy)` = principal point (camera center)
- `fx, fy` = focal lengths
- `Z` = depth value from RGBD sensor
- `(X, Y, Z)` = 3D coordinates relative to camera

## Camera Coordinate System

- **X-axis**: Right (positive to the right)
- **Y-axis**: Down (positive downward) 
- **Z-axis**: Forward (positive away from camera)
- **Origin**: Camera center

## Example Output

```
Pixel (320, 240) -> Depth: 1.500m, 3D Position: (-0.156, -0.117, 1.500)

============================================================
3D POSITION CALCULATION RESULTS
============================================================
Input:
  Pixel coordinates (u, v): (320, 240)
  Depth: 1.500 meters

Camera Intrinsic Parameters:
  Focal length (fx): 1000.25
  Focal length (fy): 1000.30
  Principal point (cx): 640.12
  Principal point (cy): 480.08

Calculated 3D Position (relative to camera):
  X: -0.156 meters
  Y: -0.117 meters
  Z: +1.500 meters
  Distance from camera: 1.509 meters
============================================================
```

## iPhone 13 Pro Setup

1. Install **Record3D** app on your iPhone 13 Pro
2. Connect both iPhone and computer to the same WiFi network
3. Open Record3D app and start streaming
4. Run the Python scripts on your computer

## Code Structure

### Main Application (`main.py`)

```python
from main import KinematicsApp

app = KinematicsApp()
app.connect_to_device(dev_idx=0)
app.start_processing_stream()
```

### Utility Calculator (`position_calculator.py`)

```python
from position_calculator import Position3DCalculator

calculator = Position3DCalculator()
calculator.set_intrinsic_matrix(intrinsic_matrix)
position_3d = calculator.pixel_to_3d(u, v, depth)
```

## Troubleshooting

**Connection Issues:**
- Ensure both devices are on the same WiFi network
- Check that Record3D app is running and streaming
- Try different device indices: `dev_idx=0, 1, 2...`

**Invalid Depth Values:**
- Some pixels may have no depth data (value = 0)
- TrueDepth camera has limited range (~0.2m to 5m)
- LiDAR has different characteristics than TrueDepth

**Coordinate System:**
- Remember that image coordinates (0,0) are at top-left
- Depth array indexing is `depth[y, x]` (row, column)
- 3D coordinates are in camera reference frame

**AI Vision Issues:**
- "Vision analyzer not available" → Check API key configuration
- "Object not found" → Try different descriptions or better lighting
- Slow response → Normal for first API call, subsequent calls are faster
- Wrong object detected → Use more specific descriptions with colors/positions

## Applications

This system can be used for:
- Object distance measurement
- 3D reconstruction and mapping
- Robotic navigation and manipulation
- Augmented reality applications
- Computer vision research

## Technical Notes

- **Intrinsic Matrix**: Automatically obtained from Record3D stream
- **Depth Units**: Meters (from Record3D)
- **Frame Rate**: Depends on iPhone model and settings
- **Resolution**: Typically 640x480 for depth, higher for RGB
- **Accuracy**: Limited by iPhone's depth sensor precision
- **AI Vision Model**: Claude Sonnet 4 (requires internet connection)
- **API Costs**: ~$0.001-0.01 per vision analysis (varies by image size) 