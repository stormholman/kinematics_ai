#!/usr/bin/env python3
"""
ArUco Marker Generator
Generate ArUco markers for testing the standalone detector
"""

import cv2
import numpy as np
import argparse

def generate_aruco_marker(marker_id=0, marker_size=200, dictionary_type=cv2.aruco.DICT_ARUCO_ORIGINAL):
    """
    Generate an ArUco marker image
    
    Args:
        marker_id: ID of the marker (0-999 for DICT_ARUCO_ORIGINAL)
        marker_size: Size of the marker image in pixels
        dictionary_type: Type of ArUco dictionary to use
        
    Returns:
        Generated marker image
    """
    # Get the ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
    
    # Generate the marker
    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    
    return marker

def save_aruco_marker(marker_id=0, filename=None, marker_size=200):
    """Save ArUco marker to file"""
    if filename is None:
        filename = f"aruco_marker_{marker_id}.png"
    
    marker = generate_aruco_marker(marker_id, marker_size)
    cv2.imwrite(filename, marker)
    print(f"✅ Saved ArUco marker ID {marker_id} to {filename}")
    return filename

def main():
    parser = argparse.ArgumentParser(description="ArUco Marker Generator")
    parser.add_argument("--id", type=int, default=0, help="Marker ID (default: 0)")
    parser.add_argument("--size", type=int, default=200, help="Marker size in pixels (default: 200)")
    parser.add_argument("--output", type=str, help="Output filename")
    parser.add_argument("--show", action="store_true", help="Show the generated marker")
    
    args = parser.parse_args()
    
    # Generate marker
    marker_id = args.id
    filename = args.output or f"aruco_marker_{marker_id}.png"
    
    print(f"🎯 Generating ArUco marker ID {marker_id}")
    print(f"📁 Output file: {filename}")
    print(f"📏 Size: {args.size}x{args.size} pixels")
    
    # Generate and save marker
    save_aruco_marker(marker_id, filename, args.size)
    
    # Show marker if requested
    if args.show:
        marker = generate_aruco_marker(marker_id, args.size)
        cv2.imshow(f'ArUco Marker ID {marker_id}', marker)
        print(f"\nDisplaying marker ID {marker_id}. Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("\n📝 Instructions:")
    print("1. Print the generated marker image")
    print("2. Place it in your camera's field of view")
    print("3. Run 'python aruco_transformer.py' to test detection")
    print("4. Ensure good lighting and contrast for best detection")

if __name__ == "__main__":
    main() 