import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import yaml # For loading config
import os
import copy

def load_config(config_path='config.yaml'):
    """Loads parameters from a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the loaded configuration parameters.

    Raises:
        FileNotFoundError: If the config file cannot be found.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded successfully from {config_path}")
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        print(f"An error occurred loading the config: {e}")
        raise


def load_and_binarize_image(image_path, threshold_value=127, invert=True):
    """Loads an image in grayscale and applies binary thresholding.

    Args:
        image_path (str): Path to the input image file.
        threshold_value (int): Pixel value threshold.
        invert (bool): If True, uses THRESH_BINARY_INV (pixels <= thresh become 255).
                       If False, uses THRESH_BINARY (pixels > thresh become 255).

    Returns:
        numpy.ndarray: The binary image (0 or 255 values).

    Raises:
        FileNotFoundError: If the image file cannot be found.
        IOError: If OpenCV cannot load the image.
    """
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        if not os.path.exists(image_path):
             raise FileNotFoundError(f"Image file not found at: {image_path}")
        # If file exists but cannot be read by OpenCV
        raise IOError(f"Could not load image (OpenCV error): {image_path}")

    if invert:
        _, bin_img = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    else:
        _, bin_img = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)
    print(f"Image '{os.path.basename(image_path)}' binarized.")
    return bin_img


def clean_image(binary_img, min_contour_area=50, close_kernel_size=3, close_iterations=1):
    """Removes noise from a binary image using contour area filtering and morphological closing.

    Args:
        binary_img (numpy.ndarray): Input binary image.
        min_contour_area (int): Minimum area for contours to be kept.
        close_kernel_size (int): Size of the kernel for morphological closing.
        close_iterations (int): Number of iterations for closing.

    Returns:
        numpy.ndarray: The cleaned binary image.

    Raises:
        ValueError: If the input image is empty or None.
    """
    if binary_img is None or binary_img.size == 0:
        raise ValueError("Input image to clean_image is empty or None")
    binary_img = np.uint8(binary_img) # Ensure correct type

    # Find contours and filter by area
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Draw large contours onto a blank image
    cleaned_img = np.zeros_like(binary_img)
    cv2.drawContours(cleaned_img, large_contours, -1, (255), thickness=cv2.FILLED)

    # Apply morphological closing if kernel size > 0
    if close_kernel_size > 0 and close_iterations > 0:
        kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        cleaned_img = cv2.morphologyEx(cleaned_img, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    
    print(f"Image cleaned (min_area={min_contour_area}, close_k={close_kernel_size}, close_iter={close_iterations}).")
    return cleaned_img


def detect_lines(bin_img, rho=1, theta_rad=np.pi/180, threshold=30, min_line_length=30, max_line_gap=10):
    """Detects line segments in a binary image using the Probabilistic Hough Transform.

    Args:
        bin_img (numpy.ndarray): Input binary image (should be 8-bit).
        rho (float): Distance resolution of the accumulator in pixels.
        theta_rad (float): Angle resolution of the accumulator in radians.
        threshold (int): Accumulator threshold parameter.
        min_line_length (int): Minimum line length.
        max_line_gap (int): Maximum allowed gap between points on the same line.

    Returns:
        list: A list of detected lines, where each line is [x1, y1, x2, y2]. Returns an empty list if no lines are found.

    Raises:
        ValueError: If the input image is empty or None.
    """
    if bin_img is None or bin_img.size == 0:
        raise ValueError("Input image to detect_lines is empty")
    if bin_img.dtype != np.uint8:
         bin_img = np.uint8(bin_img) # Ensure correct type

    lines = cv2.HoughLinesP(bin_img, rho=rho, theta=theta_rad, threshold=threshold,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
                            
    return lines[:, 0].tolist() if lines is not None else []


def categorize_and_snap_lines(lines, angle_tolerance_deg=5):
    """Categorizes lines into horizontal, vertical, or slanted, snapping H/V lines.

    Args:
        lines (list): List of lines, each as [x1, y1, x2, y2, ...].
        angle_tolerance_deg (float): Tolerance in degrees to classify as H or V.

    Returns:
        tuple: A tuple containing three lists:
               (horizontal_lines, vertical_lines, slanted_lines).
               H/V lines are snapped and include length: [x_start, y, x_end, y, length] or [x, y_start, x, y_end, length].
               Slanted lines include original coords and length: [x1, y1, x2, y2, length].
    """
    horizontal_lines, vertical_lines, slanted_lines = [], [], []
    angle_tolerance_rad = np.deg2rad(angle_tolerance_deg)
    if lines is None: return [], [], []
    
    for line in lines:
        if line is None or len(line) < 4: continue
        x1, y1, x2, y2 = line[:4]
        dx, dy = x2 - x1, y2 - y1
        
        # Skip zero-length lines
        if abs(dx) < 1e-6 and abs(dy) < 1e-6: continue
        
        angle = math.atan2(dy, dx)
        length = math.sqrt(dx*dx + dy*dy)
        
        # Check orientation within tolerance
        is_horizontal = abs(angle) < angle_tolerance_rad or \
                        abs(angle - math.pi) < angle_tolerance_rad or \
                        abs(angle + math.pi) < angle_tolerance_rad
        is_vertical = abs(angle - math.pi/2) < angle_tolerance_rad or \
                      abs(angle + math.pi/2) < angle_tolerance_rad

        if is_horizontal:
            # Snap Horizontal: average y, ensure x1 <= x2
            y_avg = int(round((y1 + y2) / 2))
            x_start, x_end = min(x1, x2), max(x1, x2)
            horizontal_lines.append([x_start, y_avg, x_end, y_avg, length])
        elif is_vertical:
            # Snap Vertical: average x, ensure y1 <= y2
            x_avg = int(round((x1 + x2) / 2))
            y_start, y_end = min(y1, y2), max(y1, y2)
            vertical_lines.append([x_avg, y_start, x_avg, y_end, length])
        else:
            # Keep as slanted, store original coords and length
            slanted_lines.append([x1, y1, x2, y2, length])
            
    return horizontal_lines, vertical_lines, slanted_lines


def filter_slanted_lines_by_length(slanted_lines, min_length=20):
    """Filters a list of slanted lines, keeping only those above a minimum length.

    Args:
        slanted_lines (list): List of slanted lines, expecting [x1, y1, x2, y2, length].
        min_length (float): The minimum length required to keep a line.

    Returns:
        list: A list of slanted lines (without length) that meet the minimum length requirement.
    """
    if slanted_lines is None: return []
    # Filter based on length (5th element), return only coords [x1, y1, x2, y2]
    filtered = [line[:4] for line in slanted_lines if len(line) >= 5 and line[4] >= min_length]
    return filtered


def merge_lines(lines, is_horizontal, merge_gap=15, axis_tolerance=2):
    """Merges close, collinear lines (only for horizontal or vertical lines).

    Args:
        lines (list): List of H or V lines, expecting format like [c1, fixed, c2, fixed, length].
        is_horizontal (bool): True if merging horizontal lines, False for vertical.
        merge_gap (int): Maximum gap allowed along the sweep direction for merging.
        axis_tolerance (int): Maximum difference allowed in the fixed coordinate for merging.

    Returns:
        list: List of merged lines in [x1, y1, x2, y2] format.
    """
    if not lines: return []
    
    fixed_coord_idx = 1 if is_horizontal else 0 # y for H, x for V
    sweep_coord_idx = 0 if is_horizontal else 1 # x for H, y for V
    
    # Sort primarily by fixed coordinate, then by the start of the sweep coordinate
    lines.sort(key=lambda l: (l[fixed_coord_idx], min(l[sweep_coord_idx], l[sweep_coord_idx+2])))
    
    merged_lines = []
    if not lines: return merged_lines
    
    # Start with the first line
    current_line_full = list(lines[0]) # Keep original including length if needed
    current_line = current_line_full[:4] # Work with [x1, y1, x2, y2]

    for i in range(1, len(lines)):
        next_line_full = lines[i]
        next_line = next_line_full[:4]
        
        # Get coordinates for comparison
        fixed_coord_current = current_line[fixed_coord_idx]
        fixed_coord_next = next_line[fixed_coord_idx]
        current_start = min(current_line[sweep_coord_idx], current_line[sweep_coord_idx + 2])
        current_end = max(current_line[sweep_coord_idx], current_line[sweep_coord_idx + 2])
        next_start = min(next_line[sweep_coord_idx], next_line[sweep_coord_idx + 2])
        next_end = max(next_line[sweep_coord_idx], next_line[sweep_coord_idx + 2])
        
        # Check merge conditions: close fixed coordinate and overlapping/close sweep coordinates
        if abs(fixed_coord_current - fixed_coord_next) <= axis_tolerance and \
           (max(current_start, next_start) <= min(current_end, next_end) + merge_gap):
            
            # Merge: Update sweep range and average fixed coordinate
            new_start = min(current_start, next_start)
            new_end = max(current_end, next_end)
            avg_fixed_coord = int(round((fixed_coord_current + fixed_coord_next) / 2)) # Average position
            
            if is_horizontal:
                 current_line[0], current_line[2] = new_start, new_end # Update x range
                 current_line[1] = current_line[3] = avg_fixed_coord # Update y
            else: # Vertical
                 current_line[1], current_line[3] = new_start, new_end # Update y range
                 current_line[0] = current_line[2] = avg_fixed_coord # Update x
        else:
            # No merge: Finalize the current_line and add it to the results
            final_sweep_start = min(current_line[sweep_coord_idx], current_line[sweep_coord_idx + 2])
            final_sweep_end = max(current_line[sweep_coord_idx], current_line[sweep_coord_idx + 2])
            if is_horizontal:
                 merged_lines.append([final_sweep_start, current_line[1], final_sweep_end, current_line[3]])
            else: # Vertical
                 merged_lines.append([current_line[0], final_sweep_start, current_line[2], final_sweep_end])
                 
            # Start the next line segment
            current_line_full = list(next_line_full)
            current_line = current_line_full[:4]

    # Add the last processed line segment after finalizing its coordinates
    final_sweep_start = min(current_line[sweep_coord_idx], current_line[sweep_coord_idx + 2])
    final_sweep_end = max(current_line[sweep_coord_idx], current_line[sweep_coord_idx + 2])
    if is_horizontal:
        merged_lines.append([final_sweep_start, current_line[1], final_sweep_end, current_line[3]])
    else: # Vertical
        merged_lines.append([current_line[0], final_sweep_start, current_line[2], final_sweep_end])
        
    return merged_lines


def connect_corners(h_lines, v_lines, threshold=15):
    """Connects endpoints of horizontal and vertical lines that are close together.

    Modifies the line segments in place by extending them to meet at a calculated corner point.

    Args:
        h_lines (list): List of horizontal lines [x1, y, x2, y].
        v_lines (list): List of vertical lines [x, y1, x, y2].
        threshold (int): Maximum pixel distance between endpoints to consider them a potential corner.

    Returns:
        tuple: A tuple containing two lists: (final_h_lines, final_v_lines)
               with coordinates adjusted and lines re-validated as strictly H or V.
    """
    connected_h = copy.deepcopy(h_lines)
    connected_v = copy.deepcopy(v_lines)
    threshold_sq = threshold * threshold

    snapped_h_endpoints = set() # Store (line_idx, end_coord_idx) e.g., (0, 0) or (0, 2)
    snapped_v_endpoints = set() # Store (line_idx, end_coord_idx) e.g., (1, 1) or (1, 3)

    # Iterate twice to allow connections to propagate slightly
    for iteration in range(2):
        snaps_made_in_iteration = 0
        for i in range(len(connected_h)):
            # Ensure H line is horizontal
            hy = int(round((connected_h[i][1] + connected_h[i][3]) / 2))
            connected_h[i][1] = connected_h[i][3] = hy
            hx1, _, hx2, _ = connected_h[i]
            h_endpoints = [ ((hx1, hy), 0), ((hx2, hy), 2) ] # ((x, y), coord_idx)

            for j in range(len(connected_v)):
                # Ensure V line is vertical
                vx = int(round((connected_v[j][0] + connected_v[j][2]) / 2))
                connected_v[j][0] = connected_v[j][2] = vx
                _, vy1, _, vy2 = connected_v[j]
                v_endpoints = [ ((vx, vy1), 1), ((vx, vy2), 3) ] # ((x, y), coord_idx)

                for h_pt_tuple, h_idx in h_endpoints:
                    current_endpoint_id_h = (i, h_idx)
                    # If already snapped in first iteration, skip to avoid fighting over snaps
                    if current_endpoint_id_h in snapped_h_endpoints and iteration == 0 : continue

                    best_v_match = None
                    min_dist_sq = threshold_sq

                    for v_pt_tuple, v_idx in v_endpoints:
                        current_endpoint_id_v = (j, v_idx)
                        if current_endpoint_id_v in snapped_v_endpoints and iteration == 0: continue

                        # Calculate distance between H endpoint and V endpoint
                        dist_sq = (h_pt_tuple[0] - v_pt_tuple[0])**2 + (h_pt_tuple[1] - v_pt_tuple[1])**2

                        if dist_sq < min_dist_sq:
                             # Heuristic check to prevent snapping nearly parallel lines' ends together
                            other_h_idx = 2 if h_idx == 0 else 0
                            other_v_idx = 3 if v_idx == 1 else 1
                            other_h_pt_x = connected_h[i][other_h_idx]
                            other_v_pt_y = connected_v[j][other_v_idx]
                            # Check if the *other* endpoints are also close - if so, likely parallel, don't snap
                            if (other_h_pt_x - v_pt_tuple[0])**2 + (h_pt_tuple[1] - other_v_pt_y)**2 > threshold_sq * 4:
                                min_dist_sq = dist_sq
                                best_v_match = (v_pt_tuple, v_idx, j) # Store best match info

                    # If a suitable nearby V endpoint was found
                    if best_v_match is not None:
                        v_pt_tuple, v_idx, v_line_idx = best_v_match
                        current_endpoint_id_v = (v_line_idx, v_idx)

                        # Final check if endpoints were snapped by another pairing in this iteration
                        if current_endpoint_id_h in snapped_h_endpoints or current_endpoint_id_v in snapped_v_endpoints: continue

                        # Calculate the ideal corner intersection point
                        corner_x = v_pt_tuple[0] # x from vertical line
                        corner_y = h_pt_tuple[1] # y from horizontal line

                        # Check if a change is actually needed
                        if connected_h[i][h_idx] != corner_x or connected_v[v_line_idx][v_idx] != corner_y:
                            # Modify the coordinates of the endpoints to meet at the corner
                            connected_h[i][h_idx] = corner_x # Adjust H line's x
                            connected_v[v_line_idx][v_idx] = corner_y # Adjust V line's y

                            # Mark endpoints as snapped to prevent further modification in this iter
                            snapped_h_endpoints.add(current_endpoint_id_h)
                            snapped_v_endpoints.add(current_endpoint_id_v)
                            snaps_made_in_iteration += 1

        # If no snaps were made in the second iteration, stop early
        if snaps_made_in_iteration == 0 and iteration > 0:
             break

    # Final cleanup: Ensure lines are strictly H or V after potential coordinate averaging/snapping
    final_h, final_v = [], []
    for line in connected_h:
        x1, y1, x2, y2 = line
        y_avg = int(round((y1+y2)/2)) # Enforce same y
        final_h.append([min(x1,x2), y_avg, max(x1,x2), y_avg]) # Ensure x1 <= x2

    for line in connected_v:
        x1, y1, x2, y2 = line
        x_avg = int(round((x1+x2)/2)) # Enforce same x
        final_v.append([x_avg, min(y1,y2), x_avg, max(y1,y2)]) # Ensure y1 <= y2

    return final_h, final_v


def draw_lines(img, lines, color=(0, 0, 255), thickness=1):
    """Draws line segments onto an image.

    Args:
        img (numpy.ndarray): The image to draw on (can be grayscale or BGR).
        lines (list): A list of lines, each formatted as [x1, y1, x2, y2].
        color (tuple): The BGR color for the lines.
        thickness (int): The thickness of the lines.

    Returns:
        numpy.ndarray: The image with lines drawn on it (will be BGR).

    Raises:
        ValueError: If the input image is None.
    """
    if img is None: raise ValueError("Input image to draw_lines is None")
    
    # Ensure image is BGR for colored drawing
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = img.copy()
        
    if lines is None: return vis_img
    
    for line in lines:
        if line is not None and len(line) >= 4:
            x1, y1, x2, y2 = map(int, line[:4])
            # Avoid drawing zero-length lines that might result from processing
            if x1 == x2 and y1 == y2: continue
            cv2.line(vis_img, (x1, y1), (x2, y2), color, thickness)
            
    return vis_img

def main():
    """Main function to execute the floor plan processing pipeline."""
    try:
        # Load Configuration
        config = load_config('config.yaml')

        # Extract parameters
        img_proc_cfg = config.get('image_processing', {})
        bin_cfg = config.get('binarization', {})
        clean_cfg = config.get('cleaning', {})
        detect_cfg = config.get('line_detection', {})
        cat_cfg = config.get('line_categorization', {})
        slant_cfg = config.get('slanted_line_filtering', {})
        ortho_proc_cfg = config.get('orthogonal_line_processing', {})
        output_cfg = config.get('output', {})
        
        image_path = img_proc_cfg.get('image_path', 'map.pgm')
        output_dir = img_proc_cfg.get('output_dir', 'output_images')

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output will be saved to: {output_dir}")
        
        # Convert degrees to radians for Hough theta
        hough_theta_rad = np.deg2rad(detect_cfg.get('hough_theta_degrees', 1.0))

        # --- Pipeline ---

        # 1. Load & Binarize
        print("1. Loading and Binarizing...")
        bin_img = load_and_binarize_image(
            image_path, 
            threshold_value=bin_cfg.get('threshold_value', 150), 
            invert=bin_cfg.get('invert', True)
        )

        # 2. Clean Noise
        print("2. Cleaning Noise...")
        cleaned_img = clean_image(
            bin_img, 
            min_contour_area=clean_cfg.get('min_contour_area', 50), 
            close_kernel_size=clean_cfg.get('close_kernel_size', 3), 
            close_iterations=clean_cfg.get('close_iterations', 1)
        )
        print("Plotting 2. Cleaned Image")
        plt.figure(figsize=(7,7)); plt.imshow(cleaned_img, cmap='gray'); plt.title("2. Cleaned Image"); plt.axis('off'); plt.show()
        cv2.imwrite(os.path.join(output_dir, "2_cleaned.png"), cleaned_img)

        kernel = np.ones((10,5), np.uint8)  
        d_im = cv2.dilate(np.float32(cleaned_img), kernel, iterations=2)
        e_im = cv2.erode(d_im, kernel, iterations=1)
        # Convert e_im to 8-bit single-channel image
        e_im_8u = cv2.convertScaleAbs(e_im)
        # Apply Canny edge detector
        edges = cv2.Canny(e_im_8u, 10, 150, apertureSize=5) 

        # 3. Detect Lines
        print("3. Detecting Lines (Hough)...")
        raw_lines = detect_lines(
            edges, 
            rho=detect_cfg.get('hough_rho', 1), 
            theta_rad=hough_theta_rad, 
            threshold=detect_cfg.get('hough_threshold', 25), 
            min_line_length=detect_cfg.get('hough_min_line_length', 20), 
            max_line_gap=detect_cfg.get('hough_max_line_gap', 20)
        )
        print(f"   Detected {len(raw_lines)} raw lines.")

        # 4. Categorize & Snap H/V Lines
        print("4. Categorizing Lines and Snapping H/V...")
        h_lines, v_lines, slanted_lines_raw = categorize_and_snap_lines(
            raw_lines, 
            angle_tolerance_deg=cat_cfg.get('angle_tolerance_degrees', 7.0)
        )
        print(f"   Found {len(h_lines)} H, {len(v_lines)} V, {len(slanted_lines_raw)} Slanted candidates.")

        # 4b. Filter Slanted Lines
        print("4b. Filtering Slanted Lines by length...")
        slanted_lines_filtered = filter_slanted_lines_by_length(
            slanted_lines_raw, 
            min_length=slant_cfg.get('slanted_min_length', 25)
        )
        print(f"   Kept {len(slanted_lines_filtered)} slanted lines after length filter.")

        # 5. Merge H/V Collinear Lines
        print("5. Merging H/V Collinear Lines...")
        merged_h_lines = merge_lines(
            h_lines, is_horizontal=True, 
            merge_gap=ortho_proc_cfg.get('merge_gap_pixels', 25), 
            axis_tolerance=ortho_proc_cfg.get('merge_axis_tolerance', 4)
        )
        merged_v_lines = merge_lines(
            v_lines, is_horizontal=False, 
            merge_gap=ortho_proc_cfg.get('merge_gap_pixels', 25), 
            axis_tolerance=ortho_proc_cfg.get('merge_axis_tolerance', 4)
        )
        print(f"   Merged into {len(merged_h_lines)} H and {len(merged_v_lines)} V lines.")

        # 5b. Connect H/V Corners
        print("5b. Connecting H/V Corners...")
        final_h_lines, final_v_lines = connect_corners(
            merged_h_lines, merged_v_lines, 
            threshold=ortho_proc_cfg.get('corner_connection_threshold', 20)
        )
        print(f"   Connected H/V lines: {len(final_h_lines)} H and {len(final_v_lines)} V.")

        # Visualize combined lines before final draw
        print("Plotting 5c. Processed Orthogonal and Filtered Slanted Lines")
        vis_all_lines = np.zeros_like(cv2.cvtColor(cleaned_img, cv2.COLOR_GRAY2BGR))
        vis_all_lines = draw_lines(vis_all_lines, final_h_lines, color=(0, 255, 255), thickness=output_cfg.get('line_thickness', 2)) # Yellow H
        vis_all_lines = draw_lines(vis_all_lines, final_v_lines, color=(255, 0, 255), thickness=output_cfg.get('line_thickness', 2)) # Magenta V
        vis_all_lines = draw_lines(vis_all_lines, slanted_lines_filtered, color=(0, 255, 0), thickness=output_cfg.get('line_thickness', 2)) # Green Slanted
        plt.figure(figsize=(7,7)); plt.imshow(vis_all_lines); plt.title("5c. H(Yellow), V(Magenta), Slanted(Green)"); plt.axis('off'); plt.show()
        cv2.imwrite(os.path.join(output_dir, "5c_all_processed_lines.png"), vis_all_lines)

        # 6. Draw Final Output (Combined)
        print("6. Drawing Final Combined Output...")
        final_cad_img = np.ones_like(cv2.cvtColor(cleaned_img, cv2.COLOR_GRAY2BGR)) * 255 # White background
        final_cad_img = draw_lines(final_cad_img, final_h_lines, color=(0, 0, 0), thickness=output_cfg.get('line_thickness', 2))
        final_cad_img = draw_lines(final_cad_img, final_v_lines, color=(0, 0, 0), thickness=output_cfg.get('line_thickness', 2))
        final_cad_img = draw_lines(final_cad_img, slanted_lines_filtered, color=(0, 0, 0), thickness=output_cfg.get('line_thickness', 2))
        print("Plotting 6. Final Output (Combined)")
        plt.figure(figsize=(7,7)); plt.imshow(final_cad_img); plt.title("6. Final Output (Combined)"); plt.axis('off'); plt.show()
        cv2.imwrite(os.path.join(output_dir, "6_final_combined_output.png"), final_cad_img)

        print(f"\nProcessing Complete. Output saved in '{output_dir}'.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except IOError as e:
        print(f"Error reading image file: {e}")
    except ValueError as e:
        print(f"Data Error: {e}")
    except yaml.YAMLError as e:
        print(f"YAML Configuration Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
