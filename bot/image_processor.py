import cv2
import numpy as np
import argparse
import os
import platform

# Only set QT_QPA_PLATFORM on Linux systems
if platform.system() != "Windows":
    os.environ["QT_QPA_PLATFORM"] = "xcb"  # Force XCB platform for Linux

def process_game_state(screen_bytes) -> dict:
    """
    Processes the game state from the screen image by cropping the upper half and segmenting it horizontally.

    Args:
        screen_bytes: The raw bytes of the screen image in RGBA format (1280x720x4).

    Returns:
        dict: A dictionary containing the processed segments and their classifications.
    """
    # Convert bytes to numpy array and reshape to (720, 1280, 4)
    nparr = np.frombuffer(screen_bytes, np.uint8)
    img = nparr.reshape((720, 1280, 4))
    
    # Convert RGBA to BGR (OpenCV format)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    
    # Crop the upper half of the image
    height, width = img.shape[:2]
    cropped_height = height // 2
    cropped_img = img[:cropped_height, :]
    
    # Split the cropped image into 4 horizontal segments
    segment_height = cropped_height // 4
    segments = []
    for i in range(4):
        start_y = i * segment_height
        end_y = (i + 1) * segment_height
        segment = cropped_img[start_y:end_y, :]
        segments.append(segment)
    
    return {"road_segments_ahead": classify_segments(segments)}

def classify_segments(segments: list) -> list:
    """
    Classifies the segments of the game state.
    
    Args:
        segments: List of image segments to classify
        
    Returns:
        list: List of classifications for each segment
    """
    # TODO: Implement actual classification logic
    return segments

def display_segments(segments: list):
    """
    Displays the segments vertically stacked with borders between them.
    
    Args:
        segments: List of image segments to display
    """
    # Create a border image (white line)
    border_height = 2
    border = np.ones((border_height, segments[0].shape[1], 3), dtype=np.uint8) * 255
    
    # Stack segments vertically with borders
    display_img = segments[0]
    for segment in segments[1:]:
        display_img = np.vstack((display_img, border, segment))
    
    # Show the image
    cv2.imshow('Segments', display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Process game state images')
    parser.add_argument('input_file', help='Path to the input image file')
    args = parser.parse_args()

    # Read the input image
    with open(args.input_file, 'rb') as f:
        screen_bytes = f.read()

    # Process the image
    result = process_game_state(screen_bytes)
    
    # Display segments
    display_segments(result["road_segments_ahead"])

if __name__ == "__main__":
    main() 