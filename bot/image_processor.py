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
    
    # Calculate curvature values for each segment
    curvature_values = classify_segments(segments)
    
    return {
        "road_segments_ahead": segments,
        "curvature_values": curvature_values
    }

def detect_road_curvature(img) -> float:
    """
    Detects road curvature using a geometric approach.
    Args:
        img: The input image in BGR format.
    Returns:
        The radius of curvature.
    """
    # Step 1: Convert to grayscale and create a binary mask for black stripes
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Step 2: Apply a bird's-eye warp (perspective transform)
    height, width = binary_mask.shape
    src = np.float32([[0, height], [width, height], [0, 0], [width, 0]])
    dst = np.float32([[width*0.2, height], [width*0.8, height], [0, 0], [width, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(binary_mask, M, (width, height))
    
    # Display the warped image
    cv2.imshow('Warped View', warped)
    cv2.waitKey(1)  # Wait for 1ms to update the window
    
    # Step 3: Sliding window to collect stripe center pixels
    histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)
    midpoint = int(histogram.shape[0]//2)
    base = np.argmax(histogram[:midpoint])
    
    # Step 4: Fit a second-order polynomial
    y_indices = np.linspace(0, height-1, num=height)
    x_indices = base + (y_indices - height/2) * 0.5  # Example calculation
    fit = np.polyfit(y_indices, x_indices, 2)
    
    # Step 5: Calculate the radius of curvature
    A, B, _ = fit
    y_eval = np.max(y_indices)
    curvature_radius = ((1 + (2*A*y_eval + B)**2)**1.5) / np.abs(2*A)
    return curvature_radius

def classify_segments(segments: list) -> list:
    """
    Classifies the segments of the game state.
    
    Args:
        segments: List of image segments to classify
        
    Returns:
        list: List of classifications for each segment
    """
    return [detect_road_curvature(segment) for segment in segments]

def detect_edges_laplacian(img) -> np.ndarray:
    """
    Detects edges in an image using the Laplacian method.
    
    Args:
        img: Input image in BGR format
        
    Returns:
        Edge-detected image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply Laplacian edge detection
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # Convert to absolute values and scale to 0-255
    abs_laplacian = np.uint8(np.absolute(laplacian))
    
    # Convert back to BGR for display
    return cv2.cvtColor(abs_laplacian, cv2.COLOR_GRAY2BGR)

def count_edges(img) -> int:
    """
    Counts the number of significant edges in an image.
    
    Args:
        img: Input image in BGR format
        
    Returns:
        Number of significant edges detected
    """
    # Get edge detection
    edge_detected = detect_edges_laplacian(img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(edge_detected, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get binary image
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Count significant contours (filter out very small ones)
    significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 15]
    
    return len(significant_contours)

def display_segments(segments: list, curvature_values: list):
    """
    Displays the edge-detected segments vertically stacked with borders between them.
    Each segment is annotated with its edge count in the top right corner.
    
    Args:
        segments: List of image segments to display
        curvature_values: List of curvature values corresponding to each segment
    """
    # Create a border image (white line)
    border_height = 2
    border = np.ones((border_height, segments[0].shape[1], 3), dtype=np.uint8) * 255
    
    # Process each segment to get edge detection and count edges
    processed_segments = []
    edge_counts = []
    for segment in segments:
        # Get edge detection
        edge_detected = detect_edges_laplacian(segment)
        # Count edges
        edge_count = count_edges(segment)
        edge_counts.append(edge_count)
        processed_segments.append(edge_detected)
    
    # Stack segments vertically with borders
    display_img = processed_segments[0]
    for segment in processed_segments[1:]:
        display_img = np.vstack((display_img, border, segment))
    
    # Add edge counts to each segment
    segment_height = segments[0].shape[0]
    for i, (segment, edge_count) in enumerate(zip(processed_segments, edge_counts)):
        # Create a copy of the segment for annotation
        annotated_segment = display_img[i * (segment_height + border_height):(i + 1) * segment_height + i * border_height].copy()
        
        # Add text with background for better visibility
        text = f"Edges: {edge_count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (255, 255, 255)  # White text
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Add background rectangle
        padding = 5
        cv2.rectangle(annotated_segment, 
                     (annotated_segment.shape[1] - text_width - padding, padding),
                     (annotated_segment.shape[1] - padding, text_height + padding),
                     (0, 0, 0), -1)  # Black background
        
        # Add text
        cv2.putText(annotated_segment, text,
                   (annotated_segment.shape[1] - text_width - padding, text_height + padding),
                   font, font_scale, color, thickness)
        
        # If more than 2 edges are detected, add a warning box
        if edge_count > 2:
            warning_text = "Multiple Edges Detected!"
            warning_color = (0, 0, 255)  # Red color for warning
            cv2.putText(annotated_segment, warning_text,
                       (10, 30), font, font_scale, warning_color, thickness)
        
        # Replace the original segment with the annotated one
        start_y = i * (segment_height + border_height)
        end_y = start_y + segment_height
        display_img[start_y:end_y] = annotated_segment
    
    # Show the image
    cv2.imshow('Edge Detection Results', display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_png_to_edges(input_file: str, output_dir: str) -> None:
    """
    Processes a PNG image and saves an edge-detected version.
    
    Args:
        input_file: Path to the input PNG file
        output_dir: Directory to save the edge-detected image
    """
    # Read the input image
    img = cv2.imread(input_file)
    if img is None:
        print(f"Error: Could not read image {input_file}")
        return
    
    # Scale down the image by 50%
    height, width = img.shape[:2]
    new_height = height // 2
    new_width = width // 2
    img = cv2.resize(img, (new_width, new_height))
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Convert back to BGR for saving
    edge_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Generate output filename
    base_name = os.path.basename(input_file)
    name_without_ext = os.path.splitext(base_name)[0]
    output_file = os.path.join(output_dir, f"{name_without_ext}_edges.png")
    
    # Save the edge-detected image
    cv2.imwrite(output_file, edge_img)
    print(f"Saved edge-detected image to {output_file}")

def process_folder(input_dir: str, output_dir: str) -> None:
    """
    Processes all PNG images in the input directory and saves edge-detected versions.
    
    Args:
        input_dir: Directory containing input PNG files
        output_dir: Directory to save edge-detected images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PNG files in the input directory
    png_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    
    if not png_files:
        print(f"No PNG files found in {input_dir}")
        return
    
    print(f"Found {len(png_files)} PNG files to process")
    
    # Process each PNG file
    for png_file in png_files:
        input_path = os.path.join(input_dir, png_file)
        process_png_to_edges(input_path, output_dir)
    
    print(f"Finished processing {len(png_files)} images")

def main():
    parser = argparse.ArgumentParser(description='Process game state images')
    parser.add_argument('input_dir', help='Directory containing input PNG files')
    parser.add_argument('--output-dir', default='edge_images', help='Directory to save edge-detected images')
    args = parser.parse_args()

    # Process all PNG files in the input directory
    process_folder(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main() 