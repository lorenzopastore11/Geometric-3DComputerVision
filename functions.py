import cv2
import numpy as np
from matplotlib.path import Path


def undistort_frame(frame: np.ndarray, K: np.ndarray, dist: np.ndarray):
    return cv2.undistort(frame, K, dist)


def read_undistorted_frames(VIDEO_PATH: str, K: np.ndarray, dist: np.ndarray):
    """Generator yielding undistorted frames, one at a time, from a video file."""
    cap = cv2.VideoCapture(VIDEO_PATH)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield undistort_frame(frame, K, dist)
    cap.release()


###############################################################
#
#       PART 1: RECTANGULAR ESTIMATION
#
#       This section of the code is used to locate, within the camera image,
#       the two physical rectangular markers positioned in the real world, and to convert these markers
#       into numerical data (the coordinates of their four corners). This is important because the system
#       needs to understand where the camera is located in relation to the work surface and how to map
#       the image pixels onto real-world coordinates.
#
#       The basic idea is as follows:
#       - Threshold the image to isolate the thick black outline of the marker against a light background.
#       - Extract the contours (with findContours).
#       - Approximate the contours using approxPolyDP and retain only those with 4 vertices.
#       - Identify which contour corresponds to rectangle 1 and which to 2
#         (a reliable criterion is needed, e.g. position in the image, area, or both visible simultaneously).
#       - Order the four corners of each rectangle consistently (e.g. clockwise starting from the
#         top-left corner) so that they always correspond to the same known 3D points: (0,0), (W,0), (W,H), (0,H).
#
###############################################################


def order_corners(pts: np.ndarray):
    """This code sorts 4 points - the vertices of a quadrilateral, here top-left, top-right, bottom-right, bottom-left
    """

    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-Left (x + y is minimum)
    rect[2] = pts[np.argmax(s)]  # Bottom-Right (x + y is maximum)

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-Right (y - x is minimum)
    rect[3] = pts[np.argmax(diff)]  # Bottom-Left (y - x is maximum)

    return rect


def find_rectangles(frame: np.ndarray, min_area: int = 2000):
    """Detect quadrilateral contours (candidate markers) in an undistorted frame.
    It takes as input the frame (set min_area as threshold in pixels below
    which a contour is discarded it is too small to be a genuine marker) and
    returns a list of (4x2 array of ordered corners, area) sorted by area,
    descending.
    """

    # Converts the frame to greyscale, which is necessary for thresholding.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # This was the first choice, then adaptiveThreshold for optimization
    # _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        3,
    )

    # Find all the contours in the binary image
    # cv2.CHAIN_APPROX_SIMPLE retains only the essential points
    # The function returns a sequence of points (x, y) along the boundary of a region and
    # an array describing the nesting relationships between the contours (ignored here)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # For each outline:
    # - discard those with an area that is too small (min_area);
    # - approximate the outline to a polygon using approxPolyDP;
    # - if the polygon has exactly 4 vertices, treat it as a quadrilateral candidate
    # - sort its corners using order_corners.

    candidates = []

    for cnt in contours:

        # This function calculates the area enclosed by a contour,
        # i.e. how many pixels (approximately) the region bounded by that contour occupies.
        area = cv2.contourArea(cnt)
        if area >= min_area:

            # This function takes as input the contour, epsilon=10 as the maximum tolerance, in pixels,
            # between the original outline and its approximation. The higher the epsilon value,
            # the more points are ‘discarded’ and the rougher/simpler the resulting polygon becomes.
            approx = cv2.approxPolyDP(cnt, 10, closed=True)
            if len(approx) == 4:

                # This function takes the raw output from `approxPolyDP`,
                # reformats it into a more convenient structure,
                # passes it to the function that determines the consistent order of the angles
                # The ordering requires a point-to-point correspondence between the angles of the actual marker and the angles detected in the image.
                corners = order_corners(approx.reshape(4, 2).astype(np.float32))
                candidates.append((corners, area))

    # Largest contours first: the two markers are expected to be the biggest quads in the scene
    candidates.sort(key=lambda c: c[1], reverse=True)
    return candidates


def assign_rectangles(candidates: list[tuple[np.ndarray, float]]):
    """Pick the two markers among the candidates and label them as rect1 / rect2.
    """

    # If there are not at least two candidate quadrilaterals, it cannot assign anything.
    # It handles the case of a failed or partial detection.
    if len(candidates) < 2:
        return None, None  # not enough markers detected in this frame

    # It only takes the corners (excluding the area) of the two largest outlines,
    # assuming that these are the two markers being sought
    quads = [c[0] for c in candidates[:2]]  # two largest quadrilaterals found

    # Compute centroids (x, y) for both quadrilaterals
    center0 = quads[0].mean(axis=0)
    center1 = quads[1].mean(axis=0)

    # Compare Y coordinates (center[1]): smaller Y means higher up on screen
    if center0[1] < center1[1]:
        rect_wall = quads[0]  # Top
        rect_table = quads[1]  # Bottom
    else:
        rect_wall = quads[1]  # Top
        rect_table = quads[0]  # Bottom

    return rect_wall, rect_table


####################################################################
#
#       PART 2: Estimation of the 3D orientation of the two rectangles
#       to obtain planes P1 and P2
#
####################################################################


def estimate_plane_from_rectangle(object_points: np.ndarray, corners_2d: np.ndarray, K: np.ndarray):
    """Given the the 3D coordinates of the four corners, the 4 detected 2D
    corners of a marker and the camera intrinsics K, estimate the 3D plane (p,
    n) the marker lies on.
    """

    # Ensure that the data is in the correct format for solvePnP (float32)
    obj_pts = np.asarray(object_points, dtype=np.float32)
    img_pts = np.asarray(corners_2d, dtype=np.float32)
    
    # distCoeffs=None because the frame was already
    # undistorted, so no further distortion correction is needed here.
    # Input: array of 3D points, array of 2D and intrinsic matrix K
    # The SOLVEPNP_IPPE (Infinitesimal Plane-Based Pose Estimation) algorithm is optimised and extremely fast for flat objects
    # Output: Output rotation vector and output translation vector
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K, distCoeffs=None, flags=cv2.SOLVEPNP_IPPE
    )
    assert ok

    # Get the rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # A point on the plane: marker's origin, in camera coords
    p = tvec.flatten()

    # Obtain the normal vector 
    n = R[:, 2].copy()

    # Normalization
    n = n / np.linalg.norm(n)

    # Point the normal towards the camera
    if n[2] > 0:
        n = -n

    return p, n


###########################################################################
#
# PART 3: LASER LINE DETECTION
# Isolate the red laser pixels in an undistorted frame
#
############################################################################

def detect_laser_points(frame: np.ndarray, 
                        value_thresh: int = 20, 
                        sat_thresh: int = 30, 
                        dominance_ratio: float = 1.10,
                        min_peak_diff: int = 15):
    
    # --- 1. Split BGR and HSV channels --------------------------------------
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv_image)

    blue = frame[:, :, 0].astype(np.float32)
    green = frame[:, :, 1].astype(np.float32)
    red = frame[:, :, 2].astype(np.float32)

    # --- 2. Adaptive Red Laser Detection ------------------------------------
    hue_is_red = (hue < 12) | (hue > 168)
    
    # Absolute difference between Red and the brightest of Green/Blue
    max_gb = np.maximum(green, blue)
    red_diff = red - max_gb

    # Standard saturated red
    std_red = hue_is_red & (sat > sat_thresh) & (val > value_thresh) & (red > dominance_ratio * max_gb)

    # Overexposed / Desaturated core on white plastic
    # High value, clear red dominance even if saturation drops near white
    bright_core = (val > 140) & (red_diff > min_peak_diff) & (red > green) & (red > blue)

    laser_mask = std_red | bright_core

    # Compute continuous energy map (Difference map highlights the laser line best)
    weighted_red = np.maximum(0, red_diff) * laser_mask.astype(np.float32)

    height, width = weighted_red.shape
    
    x_peaks = np.argmax(weighted_red, axis=1)
    y_indices = np.arange(height)
    
    peak_values = weighted_red[y_indices, x_peaks]
    
    # Valid row must have a significant peak intensity
    valid_mask = (peak_values > min_peak_diff) & (x_peaks > 0) & (x_peaks < width - 1)
    
    valid_y = y_indices[valid_mask]
    valid_x = x_peaks[valid_mask]
        
    if np.any(valid_mask):
        laser_points = np.column_stack((valid_x, valid_y)).astype(np.float32)
    else:
        laser_points = np.empty((0, 2), dtype=np.float32)
        

    return laser_points


############################################################################
#
# PART 5: CLASSIFICATION OF THE LASER'S PIXELS
# This section takes all the laser line points detected and sorts them into three groups,
# depending on where they fall in the image in relation to the two rectangular markers:
# inside rectangle 1, inside rectangle 2, or outside both.
#
###########################################################################

def classify_laser_points(laser_points: np.ndarray, rect1: np.ndarray, rect2: np.ndarray):
    """Classifies laser points into three groups (rect1, rect2, on_object) using
    vectorized polygon path tests.
    """
    
    if laser_points.shape[0] == 0:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty, empty

    # Ensure point coordinates are in the correct format
    pts1 = np.asarray(rect1, dtype=np.float32)
    pts2 = np.asarray(rect2, dtype=np.float32)

    # 1. Create spatial polygon paths
    path1 = Path(pts1)
    path2 = Path(pts2)

    # 2. Perform vectorized inclusion tests.
    # Optimization: not use of expensive for
    # Dimensions: a 1D array of exactly N elements (the laser points).
    in_mask1 = path1.contains_points(laser_points)
    in_mask2 = path2.contains_points(laser_points)

    # 4. Extract arrays using boolean indexing
    # Form: (N, 2), where N is simply the number of points at which the mask is True
    in_rect1 = laser_points[in_mask1]
    in_rect2 = laser_points[in_mask2]
    on_object = laser_points[~(in_mask1 | in_mask2)]

    return in_rect1, in_rect2, on_object


###################################################################
#
# PART 6: BACKPROJECTION OF PIXELS ONTO 3D RAYS
#
###################################################################


def backproject_points(points_2d: np.ndarray, K_inv: np.ndarray):
    """Back-projects 2D pixel coordinates to 3D unit direction rays in camera space.
    Expects pre-computed inverse intrinsic matrix K_inv for performance.
    """
    n = points_2d.shape[0]
    if n == 0:
        return np.empty((0, 3), dtype=np.float32)

    # 1. Construct homogeneous coordinates (n, 3)
    points_h = np.empty((n, 3), dtype=np.float32)
    points_h[:, :2] = points_2d
    points_h[:, 2] = 1.0

    # 2. Compute 3D direction vectors directly using matrix product transpose rule
    # (K_inv @ p.T).T is mathematically equivalent to: p @ K_inv.T
    directions = points_h @ K_inv.T

    # 3. Normalize vectors to unit length
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= np.maximum(norms, 1e-8)

    # Directions is an array (N, 3) containing the unit 3D direction vectors oriented from the camera origin
    return directions


######################################################################
#
# PART 7: ESTIMATE OF THE PL LASER PLANE FOR THE CURRENT FRAME
#
######################################################################

def intersect_rays_with_plane(directions: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray):
    """
    Intersects 3D rays (originating at camera [0,0,0]) with a 3D plane (p, n).
    Returns intersecting 3D points and a boolean mask of valid rays.
    """
    if directions.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=bool)

    # Ensure 1D arrays for plane point and normal
    p = np.asarray(plane_point, dtype=np.float32).flatten()
    n = np.asarray(plane_normal, dtype=np.float32).flatten()
    dirs = np.asarray(directions, dtype=np.float32)

    # 1. Compute dot products d · n for all direction vectors
    denom = dirs @ n

    # 2. Distance from camera origin (0,0,0) to plane along normal axis
    d_plane = np.dot(p, n)

    # 3. Calculate ray scale parameter s = (p · n) / (d · n)
    # Temporarily disable warning messages regarding division by zero if a ray is exactly parallel to the plane
    with np.errstate(divide="ignore", invalid="ignore"):
        zeta = d_plane / denom

    # 4. Keep only rays that intersect validly in front of the camera
    valid = (np.abs(denom) > 1e-6) & (zeta > 0)

    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), valid

    # 5. Compute 3D points via element-wise vector scaling: P = s * d
    # s[valid, None] converts the 1D scalar vector of valid rays into a column (K, 1)
    points_3d = dirs[valid] * zeta[valid, None]

    # points_3d is an array (K, 3) containing the Cartesian coordinates (X, Y, Z) of the 3D points calculated in camera space
    return points_3d, valid


def fit_plane(points_3d: np.ndarray):
    """
    Fits a 3D plane (p, n) to a set of points using PCA:
    - p is the centroid
    - n is the eigenvector corresponding to the smallest eigenvalue
    """
    
    pts = np.asarray(points_3d, dtype=np.float32)
    n_points = pts.shape[0]

    # 1. Require at least 3 points to define a plane
    if n_points < 3:
        return None, None

    # 2. Compute centroid and center the point cloud
    p = pts.mean(axis=0)
    centered = pts - p

    # 3. Compute normalized 3x3 covariance matrix
    cov = (centered.T @ centered) / n_points

    # 4. Compute eigenvalues and eigenvectors for symmetric matrix
    eigvals, eigvecs = np.linalg.eigh(cov)

    # 5. Extract normal vector corresponding to the smallest eigenvalue
    n = eigvecs[:, 0].copy()

    # 6. Normalize vector length
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-8:
        return None, None
    n = n / n_norm

    # 7. Ensure normal points toward the camera origin (Z < 0 in camera space)
    if n[2] > 0:
        n = -n

    return p, n


############################################################################
#
# Point 8: RECONSTUCT OBJECT'S 3D POINTS USING ESTIMATING LASER PLANE PL
#
############################################################################


def get_pixel_colors(points_2d: np.ndarray, frame: np.ndarray):
    """Given a set of 2D points and the original frame, sample the colour of the image at each point"""
    if points_2d.shape[0] == 0:
        return np.empty((0, 3), dtype=np.uint8)

    # points_2d[:, 0] and points_2d[:, 1]: extract all the x and y coordinates respectively
    # .round().astype(int): given that a pixel in the image must have integer coordinates,
    # this rounds to the nearest pixel (hence nearest pixel in the docstring)
    # and converts it to an integer
    # np.clip clips any value outside the range to the nearest boundary,
    xs = np.clip(points_2d[:, 0].round().astype(int), 0, frame.shape[1] - 1)
    ys = np.clip(points_2d[:, 1].round().astype(int), 0, frame.shape[0] - 1)

    # Fancy indexing to extract the frame colour at all (y, x) positions simultaneously.
    colors_bgr = frame[ys, xs]
    colors_rgb = colors_bgr[:, ::-1]  # BGR -> RGB

    return colors_rgb