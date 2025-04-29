import cv2
import numpy as np
import argparse
import os

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

def save_segments(segments: list, output_dir: str):
    """
    Saves the segments as separate image files for debugging.
    
    Args:
        segments: List of image segments to save
        output_dir: Directory to save the segments in
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, segment in enumerate(segments):
        output_path = os.path.join(output_dir, f"segment_{i}.png")
        cv2.imwrite(output_path, segment)

def main():
    parser = argparse.ArgumentParser(description='Process game state images')
    parser.add_argument('input_file', help='Path to the input image file')
    parser.add_argument('--output-dir', default='debug_segments', help='Directory to save debug segments')
    args = parser.parse_args()

    # Read the input image
    with open(args.input_file, 'rb') as f:
        screen_bytes = f.read()

    # Process the image
    result = process_game_state(screen_bytes)
    
    # Save segments for debugging
    save_segments(result["road_segments_ahead"], args.output_dir)
    
    print(f"Processed image and saved segments to {args.output_dir}")

if __name__ == "__main__":
    main() 