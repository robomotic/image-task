from image_segmentation.app.version import __version__
import base64
import json
import os


def test_regions():
    """
    These files are loaded
    """
    import cv2
    import numpy as np
    # Check that all required files exist
    assert os.path.exists("./data/landmarks.txt"), "landmarks.txt is missing"
    assert os.path.exists("./data/original_image.png"), "original_image.png is missing"
    assert os.path.exists("./data/segmentation_map.png"), "segmentation_map.png is missing"

    with open("./data/landmarks.txt", "r") as file:
        landmarks_str = file.read()
        try:
            landmarks = json.loads(landmarks_str)
        except json.JSONDecodeError:
            import ast
            landmarks = ast.literal_eval(landmarks_str)

    # Load segmentation map with OpenCV
    segmap = cv2.imread("./data/segmentation_map.png", cv2.IMREAD_UNCHANGED)
    assert segmap is not None, "Failed to load segmentation_map.png"
    print(f"Segmentation map shape: {segmap.shape}")
    # Print total RGB levels before conversion (if color)
    if len(segmap.shape) == 3:
        total_rgb_levels = np.sum(segmap, axis=(0, 1))
        print(f"Total RGB levels before conversion: R={total_rgb_levels[2]}, G={total_rgb_levels[1]}, B={total_rgb_levels[0]}")
        segmap_gray = cv2.cvtColor(segmap, cv2.COLOR_BGR2GRAY)
    else:
        print(f"Total grayscale levels before conversion: {np.sum(segmap)}")
        segmap_gray = segmap
    print(f"Segmentation map grayscale shape: {segmap_gray.shape}")
    # Count unique regions (excluding background if needed)
    unique_regions = np.unique(segmap_gray)
    print(f"Unique region values: {unique_regions}")
    # Optionally, exclude background (e.g., 0)
    region_values = unique_regions[unique_regions != 0]
    num_regions = len(region_values)
    print(f"Number of regions (excluding background=0): {num_regions}")
    assert num_regions > 0, "No regions found in segmentation map"

    # Create a color map for the regions
    color_palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 128), (128, 128, 0), (0, 128, 128), (128, 128, 128)
    ]
    # Create a blank color image
    color_img = np.zeros((*segmap_gray.shape, 3), dtype=np.uint8)
    region_pixel_counts = []
    for idx, region_val in enumerate(region_values):
        mask = segmap_gray == region_val
        color = color_palette[idx % len(color_palette)]
        color_img[mask] = color
        # Add label at the centroid of the region
        ys, xs = np.where(mask)
        pixel_count = np.sum(mask)
        region_pixel_counts.append((region_val, pixel_count))
        print(f"Region {region_val}: {pixel_count} pixels")
        if len(xs) > 0 and len(ys) > 0:
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            cv2.putText(color_img, f"{region_val}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    # Save the new image
    cv2.imwrite("./data/segmentation_map_colored.png", color_img)
    print("Saved colored segmentation map with labels as ./data/segmentation_map_colored.png")

    # Sort regions by area (pixel count) and print
    region_pixel_counts.sort(key=lambda x: x[1], reverse=True)
    print("\nRegions sorted by area (pixel count):")
    for region_val, pixel_count in region_pixel_counts:
        print(f"Region {region_val}: {pixel_count} pixels")

    # I suspect those are the 478 points from Face Mesh Media pipe 
    # however we have to annotate
    landmark = landmarks["landmarks"][0]

    print("Total landmark points %d" % len(landmark))

def test_estimate_face_rotation():
    """
    Estimate the rotation angle of the face based on the center between the left and right eye landmarks.
    """
    import json
    import math
    # Load landmarks
    with open("./data/landmarks.txt", "r") as file:
        landmarks_str = file.read()
        try:
            landmarks = json.loads(landmarks_str)
        except json.JSONDecodeError:
            import ast
            landmarks = ast.literal_eval(landmarks_str)
    landmark_points = landmarks["landmarks"][0]
    # MediaPipe Face Mesh standard indices for left and right eye center
    # These are approximate: 33 (right eye outer), 133 (right eye inner), 362 (left eye outer), 263 (left eye inner)
    right_eye_idx = 33
    left_eye_idx = 263
    right_eye = landmark_points[right_eye_idx]
    left_eye = landmark_points[left_eye_idx]
    # Compute the angle in degrees
    dx = left_eye["x"] - right_eye["x"]
    dy = left_eye["y"] - right_eye["y"]
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    print(f"Estimated face rotation angle (degrees): {angle_deg:.2f}")
    # Optionally, assert the angle is within a reasonable range
    assert -90 <= angle_deg <= 90, "Face rotation angle out of expected range!"

def test_rotate_landmarks_and_image():
    """
    Rotate the landmarks and the image by the negative of the estimated face angle so the eyes are horizontal.
    Save the rotated image with numbered landmarks.
    """
    import json
    import math
    import cv2
    import numpy as np
    # Load landmarks
    with open("./data/landmarks.txt", "r") as file:
        landmarks_str = file.read()
        try:
            landmarks = json.loads(landmarks_str)
        except json.JSONDecodeError:
            import ast
            landmarks = ast.literal_eval(landmarks_str)
    landmark_points = landmarks["landmarks"][0]
    # Eye indices
    right_eye_idx = 33
    left_eye_idx = 263
    right_eye = landmark_points[right_eye_idx]
    left_eye = landmark_points[left_eye_idx]
    # Compute angle
    dx = left_eye["x"] - right_eye["x"]
    dy = left_eye["y"] - right_eye["y"]
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    print(f"Original face angle: {angle_deg:.2f} degrees")
    
    # Load image
    img = cv2.imread("./data/original_image.png", cv2.IMREAD_COLOR)
    assert img is not None, "Failed to load original_image.png"
    h, w = img.shape[:2]
    
    # Center of rotation: midpoint between eyes
    cx = (left_eye["x"] + right_eye["x"]) / 2
    cy = (left_eye["y"] + right_eye["y"]) / 2
    print(f"Rotation center: ({cx:.1f}, {cy:.1f})")
    
    # Rotation matrix (rotate by +angle_deg to make eyes horizontal)
    # If the face is tilted at +22° (clockwise), we need to rotate +22° (clockwise) to straighten it
    rotation_angle = angle_deg
    M = cv2.getRotationMatrix2D((cx, cy), rotation_angle, 1.0)
    print(f"Rotating by {rotation_angle:.2f} degrees to align eyes horizontally")
    
    rotated_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
    
    # Also rotate the segmentation map by the same amount
    segmap = cv2.imread("./data/segmentation_map.png", cv2.IMREAD_UNCHANGED)
    assert segmap is not None, "Failed to load segmentation_map.png"
    rotated_segmap = cv2.warpAffine(segmap, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    cv2.imwrite("./data/segmentation_map_rotated.png", rotated_segmap)
    print("Saved rotated segmentation map as ./data/segmentation_map_rotated.png")
    
    # Rotate landmarks
    rotated_landmarks = []
    for idx, pt in enumerate(landmark_points):
        x, y = pt["x"], pt["y"]
        x_rot = M[0,0]*x + M[0,1]*y + M[0,2]
        y_rot = M[1,0]*x + M[1,1]*y + M[1,2]
        rotated_landmarks.append({"x": x_rot, "y": y_rot, "index": idx})
    
    # Verify rotation worked - check if eyes are now horizontal
    rotated_right_eye = rotated_landmarks[right_eye_idx]
    rotated_left_eye = rotated_landmarks[left_eye_idx]
    new_dx = rotated_left_eye["x"] - rotated_right_eye["x"]
    new_dy = rotated_left_eye["y"] - rotated_right_eye["y"]
    new_angle = math.degrees(math.atan2(new_dy, new_dx))
    print(f"After rotation, eye angle is: {new_angle:.2f} degrees (should be close to 0)")
    
    # Save rotated landmarks to JSON
    with open("./data/rotated_landmarks.json", "w") as f:
        json.dump(rotated_landmarks, f, indent=2)
    # Draw rotated landmarks on rotated image
    for pt in rotated_landmarks:
        x, y, idx = int(round(pt["x"])), int(round(pt["y"])), pt["index"]
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(rotated_img, (x, y), 2, (0, 255, 0), -1)
            cv2.putText(rotated_img, str(idx), (x+3, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)
    cv2.imwrite("./data/original_image_rotated_landmarks.png", rotated_img)
    print("Saved rotated image with numbered landmarks as ./data/original_image_rotated_landmarks.png")
    print("Saved rotated landmarks as ./data/rotated_landmarks.json")

def test_load_mediapipe_face_annotations():
    """
    Load MediaPipe Face Mesh default annotations for the 478 landmark points.
    This provides semantic labels for each landmark index.
    """
    # MediaPipe Face Mesh landmark annotations
    # Based on the 468 face landmarks from MediaPipe Face Mesh
    mediapipe_face_annotations = {
        # Face oval
        10: "face_oval", 151: "face_oval", 9: "face_oval", 8: "face_oval", 
        # Left eyebrow (user's left)
        70: "left_eyebrow_outer", 63: "left_eyebrow", 105: "left_eyebrow", 66: "left_eyebrow", 107: "left_eyebrow_inner",
        # Right eyebrow (user's right)
        55: "right_eyebrow_outer", 65: "right_eyebrow", 52: "right_eyebrow", 53: "right_eyebrow", 46: "right_eyebrow_inner",
        # Left eye (user's left)
        33: "left_eye_outer", 7: "left_eye_top", 163: "left_eye_top", 144: "left_eye_top", 145: "left_eye_top",
        153: "left_eye_top", 154: "left_eye_top", 155: "left_eye_top", 133: "left_eye_inner",
        173: "left_eye_bottom", 157: "left_eye_bottom", 158: "left_eye_bottom", 159: "left_eye_bottom",
        160: "left_eye_bottom", 161: "left_eye_bottom", 246: "left_eye_bottom",
        # Right eye (user's right)
        362: "right_eye_outer", 398: "right_eye_top", 384: "right_eye_top", 385: "right_eye_top", 386: "right_eye_top",
        387: "right_eye_top", 388: "right_eye_top", 466: "right_eye_top", 263: "right_eye_inner",
        249: "right_eye_bottom", 390: "right_eye_bottom", 373: "right_eye_bottom", 374: "right_eye_bottom",
        380: "right_eye_bottom", 381: "right_eye_bottom", 382: "right_eye_bottom",
        # Nose
        1: "nose_tip", 2: "nose_tip", 5: "nose_bridge", 4: "nose_bridge", 6: "nose_bridge",
        19: "nose_bridge", 20: "nose_bridge", 94: "nose_nostril", 125: "nose_nostril",
        141: "nose_nostril", 235: "nose_nostril", 236: "nose_nostril", 3: "nose_bridge",
        51: "nose_nostril", 48: "nose_nostril", 115: "nose_nostril", 131: "nose_nostril",
        134: "nose_nostril", 102: "nose_nostril", 49: "nose_nostril", 220: "nose_nostril",
        305: "nose_nostril", 290: "nose_nostril", 294: "nose_nostril", 278: "nose_nostril",
        # Lips
        0: "lip_center", 17: "upper_lip", 18: "upper_lip", 200: "upper_lip",
        199: "upper_lip", 175: "upper_lip", 12: "upper_lip", 15: "upper_lip",
        16: "upper_lip", 85: "upper_lip", 16: "upper_lip", 15: "upper_lip",
        14: "lower_lip", 13: "lower_lip", 82: "lower_lip", 81: "lower_lip",
        80: "lower_lip", 78: "lower_lip", 95: "lower_lip", 88: "lower_lip",
        178: "lower_lip", 87: "lower_lip", 14: "lower_lip", 317: "lower_lip",
        402: "lower_lip", 318: "lower_lip", 324: "lower_lip", 308: "lower_lip",
        # Additional key points
        10: "forehead_center", 151: "chin_center", 234: "left_cheek", 454: "right_cheek"
    }
    
    # Create a more comprehensive mapping based on MediaPipe's standard regions
    region_mappings = {
        "face_oval": list(range(0, 17)) + [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323],
        "left_eyebrow": [46, 53, 52, 65, 55, 70],
        "right_eyebrow": [107, 66, 105, 63, 70, 46],
        "left_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
        "right_eye": [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382],
        # Region 2 - Left eye below
        "region_2": [
            22,35,143,111,117,118,119,120,121,122,128,31,228,229,230,231,232,233
        ],
        # Region 3 - Right eye below
        "region_3": [
            265,372,340,346,347,348,349,343,357,451,450,449,448,261
        ],
        "region_5": [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 290, 294, 278],
        "upper_lip": [0, 17, 18, 200, 199, 175, 12, 15, 16, 85, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
        "lower_lip": [14, 13, 82, 81, 80, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80],
        # Expanded chin region (region 4) - including jawline and lower face landmarks
        "region_4": [
            137,132,215,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288,435,401,360,435,345,346,344,347,329,366,346,420,429,423,393,
            164,167,165,206,203,36,50,123,
            116
        ],
        "region_1": [300,301,251,284,332,297,336,10,109,67,103,54,21,162,71,68,66,107,9,336,296,334,293,300]
    }
    
    print("MediaPipe Face Mesh Landmark Annotations loaded:")
    print(f"Total annotated landmarks: {len(mediapipe_face_annotations)}")
    print("\nRegion mappings:")
    for region, indices in region_mappings.items():
        print(f"  {region}: {len(indices)} landmarks")
    
    # Save annotations to JSON
    annotation_data = {
        "individual_landmarks": mediapipe_face_annotations,
        "region_mappings": region_mappings,
        "total_landmarks": 478,
        "description": "MediaPipe Face Mesh landmark annotations"
    }
    
    with open("./data/mediapipe_face_annotations.json", "w") as f:
        json.dump(annotation_data, f, indent=2)
    print("\nSaved MediaPipe annotations to ./data/mediapipe_face_annotations.json")
    
    return annotation_data

def test_create_svg_masks_for_regions():
    """
    Create SVG fill masks for specific face regions (region_2, region_3, forehead, nose).
    Loads rotated image, landmarks, annotations, and segmentation map to generate SVG masks.
    """
    import json
    import cv2
    import numpy as np
    from xml.etree.ElementTree import Element, SubElement, tostring
    from xml.dom import minidom
    
    # a) Load the rotated original image
    rotated_img = cv2.imread("./data/original_image_rotated_landmarks.png", cv2.IMREAD_COLOR)
    assert rotated_img is not None, "Failed to load rotated original image"
    h, w = rotated_img.shape[:2]
    print(f"Loaded rotated image with dimensions: {w}x{h}")
    
    # b) Load the rotated landmark coordinates
    with open("./data/rotated_landmarks.json", "r") as f:
        rotated_landmarks = json.load(f)
    print(f"Loaded {len(rotated_landmarks)} rotated landmarks")
    
    # c) Load the mediapipe_face_annotations.json
    with open("./data/mediapipe_face_annotations.json", "r") as f:
        face_annotations = json.load(f)
    region_mappings = face_annotations["region_mappings"]
    print("Loaded MediaPipe face annotations")
    
    # d) Load the rotated segmentation map
    rotated_segmap = cv2.imread("./data/segmentation_map_rotated.png", cv2.IMREAD_UNCHANGED)
    assert rotated_segmap is not None, "Failed to load rotated segmentation map"
    print(f"Loaded rotated segmentation map with shape: {rotated_segmap.shape}")
    
    # Convert segmentation map to grayscale if needed
    if len(rotated_segmap.shape) == 3:
        segmap_gray = cv2.cvtColor(rotated_segmap, cv2.COLOR_BGR2GRAY)
    else:
        segmap_gray = rotated_segmap
    
    # Target regions for SVG mask creation
    target_regions = ["region_1", "region_2","region_3","region_4","region_5"]
    
    # Create SVG masks for each target region
    for region_name in target_regions:
        if region_name not in region_mappings:
            print(f"Warning: {region_name} not found in region mappings, skipping...")
            continue
            
        print(f"\nCreating SVG mask for {region_name}...")
        landmark_indices = region_mappings[region_name]
        
        # Get coordinates for this region's landmarks
        region_points = []
        for idx in landmark_indices:
            if idx < len(rotated_landmarks):
                landmark = rotated_landmarks[idx]
                x, y = landmark["x"], landmark["y"]
                # Ensure coordinates are within image bounds
                if 0 <= x < w and 0 <= y < h:
                    region_points.append((x, y))
        
        if len(region_points) < 3:
            print(f"Warning: Not enough valid points for {region_name} (found {len(region_points)}), skipping...")
            continue
        
        print(f"Found {len(region_points)} valid landmarks for {region_name}")
        
        # Create boundary points - use convex hull for most regions, but not for region_4
        points_array = np.array(region_points, dtype=np.int32)
        if region_name == "region_4":
            # For region_4 (chin), sort points clockwise to avoid zigzag pattern
            # Calculate centroid
            centroid = np.mean(points_array, axis=0)
            
            # Calculate angles from centroid to each point
            angles = np.arctan2(points_array[:, 1] - centroid[1], points_array[:, 0] - centroid[0])
            
            # Sort points by angle (clockwise)
            sorted_indices = np.argsort(-angles)  # Negative for clockwise
            hull_points = points_array[sorted_indices]
        else:
            # For other regions, use convex hull to get a smooth boundary
            hull = cv2.convexHull(points_array)
            hull_points = hull.reshape(-1, 2)
        
        # Create SVG
        svg = Element('svg')
        svg.set('width', str(w))
        svg.set('height', str(h))
        svg.set('viewBox', f'0 0 {w} {h}')
        svg.set('xmlns', 'http://www.w3.org/2000/svg')
        
        # Create a group for this region
        group = SubElement(svg, 'g')
        group.set('id', f'{region_name}_mask')
        
        # Create polygon path from hull points
        polygon = SubElement(group, 'polygon')
        points_str = ' '.join([f'{pt[0]},{pt[1]}' for pt in hull_points])
        polygon.set('points', points_str)
        polygon.set('fill', 'white')
        polygon.set('fill-opacity', '1.0')
        polygon.set('stroke', 'none')
        
        # Add a background rectangle (black) for proper mask
        bg_rect = Element('rect')
        bg_rect.set('width', str(w))
        bg_rect.set('height', str(h))
        bg_rect.set('fill', 'black')
        svg.insert(0, bg_rect)
        
        # Pretty print and save SVG
        rough_string = tostring(svg, 'unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_svg = reparsed.toprettyxml(indent="  ")
        
        # Remove empty lines and XML declaration for cleaner output
        pretty_lines = [line for line in pretty_svg.split('\n') if line.strip()]
        if pretty_lines[0].startswith('<?xml'):
            pretty_lines = pretty_lines[1:]
        pretty_svg = '\n'.join(pretty_lines)
        
        # Save SVG file
        svg_filename = f"./data/{region_name}_mask.svg"
        with open(svg_filename, 'w') as f:
            f.write(pretty_svg)
        print(f"Saved SVG mask: {svg_filename}")
        
        # Also create a visual debug image showing the mask overlay
        debug_img = rotated_img.copy()
        
        # Draw the hull on the debug image
        cv2.fillPoly(debug_img, [hull_points], (0, 255, 0))  # Green fill
        cv2.polylines(debug_img, [hull_points], True, (255, 0, 0), 2)  # Blue border
        
        # Draw landmark points
        for point in region_points:
            cv2.circle(debug_img, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)  # Red dots
        
        # Add region label
        if len(hull_points) > 0:
            centroid_x = int(np.mean(hull_points[:, 0]))
            centroid_y = int(np.mean(hull_points[:, 1]))
            cv2.putText(debug_img, region_name, (centroid_x-30, centroid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        debug_filename = f"./data/{region_name}_mask_debug.png"
        cv2.imwrite(debug_filename, debug_img)
        print(f"Saved debug image: {debug_filename}")
    
    print(f"\nSVG mask creation completed for regions: {target_regions}")
    print("Files saved in ./data/ directory:")
    print("- SVG mask files: *_mask.svg")
    print("- Debug visualization images: *_mask_debug.png")
    
    return True


