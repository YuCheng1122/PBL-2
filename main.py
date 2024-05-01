import cv2
import numpy as np
import logging

# Setup basic configuration for logging
logging.basicConfig(filename='player_detection_log.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def get_dynamic_thresholds(y, height):
    # Define thresholds for different vertical segments of the frame
    if y > height * 0.66:  # Lower third
        return 0.25, (0.5, 1.5)
    elif y > height * 0.33:  # Middle third
        return 0.5, (0.3, 1.3)
    else:  # Upper third
        return 0.6, (0.2, 1.2)

# Enhanced function with dynamic area threshold and adjusted aspect ratio limits for better detection
def detect_players_optimized(video_path, speed_factor=1):
    # Load the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Error: Cannot open video.")
        return

    # Retrieve the original FPS of the video to control playback speed
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int((1 / fps) * 1000 * speed_factor)  # Calculate delay based on speed factor

    # Define the region of interest (ROI) for the court area manually based on typical dimensions
    x_start, y_start, width, height = 100, 100, 1100, 700
    frame_counter = 0

    # Define the refined color range for player detection
    refined_lower = np.array([110, 50, 50], np.uint8)
    refined_upper = np.array([155, 255, 160], np.uint8)

    # Initialize variables to help with dynamic thresholding
    average_area = 0
    contour_count = 0

    # Read the first frame for initialization
    _, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)[y_start:y_start + height, x_start:x_start + width]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("End of video or error reading frame.")
            break

        frame_counter += 1
        # Focus on the court area for detection
        court_roi = frame[y_start:y_start + height, x_start:x_start + width]
        gray_roi = cv2.cvtColor(court_roi, cv2.COLOR_BGR2GRAY)

        # Convert the ROI to HSV color space
        hsv_roi = cv2.cvtColor(court_roi, cv2.COLOR_BGR2HSV)

        # Mask for the specified colors
        color_mask = cv2.inRange(hsv_roi, refined_lower, refined_upper)

        # Use frame difference for moving object detection
        frame_diff = cv2.absdiff(gray_roi, prev_gray)
        _, motion_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        combined_mask = cv2.bitwise_and(color_mask, motion_mask)

        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=3)

        # Find contours in the mask within ROI
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Dynamic area and aspect ratio adjustments
        total_area = sum(cv2.contourArea(c) for c in contours)
        num_contours = len(contours)
        if num_contours > 0:
            average_area = (average_area * contour_count + total_area) / (contour_count + num_contours)
            contour_count += num_contours

        # Draw bounding boxes around detected players in the full frame and log details
        # Draw bounding boxes around detected players in the full frame and log details
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            # Adjust aspect ratio and area thresholds based on dynamic calculations
            area_factor, aspect_ratio_range = get_dynamic_thresholds(y_start + y, height)
            dynamic_area_threshold = average_area * area_factor
            
            if area > dynamic_area_threshold and aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1]:
                cv2.rectangle(frame, (x_start + x, y_start + y), (x_start + x + w, y_start + y + h), (0, 255, 0), 2)
                # Output the score next to the frame
                cv2.putText(frame, f"Score: {area}", (x_start + x, y_start + y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                logging.info(f"Frame {frame_counter}: Detected player with area {area}, aspect_ratio {aspect_ratio}, at ({x_start+x}, {y_start+y})")
            else:
                logging.info(f"Frame {frame_counter}: Rejected contour with area {area}, aspect_ratio {aspect_ratio}, dynamic area threshold {dynamic_area_threshold}, aspect range {aspect_ratio_range}, at ({x_start+x}, {y_start+y})")

        # Update the previous frame for the next loop
        prev_gray = gray_roi

        # Optional: Resize frame for display if the window is too large
        display_frame = cv2.resize(frame, (960, 540))  # Resize to 960x540 for display


        # Show the resized frame
        cv2.imshow('Player Detection Full Frame', display_frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Function call (commented out for now)
# Example: Play video at half speed
detect_players_optimized('./input_video.mp4')
