import cv2
import numpy as np


def preprocess_image(frame):
    """
    Convert the frame to a binary image using gradient and color thresholds.
    """
    # Convert to HLS color space and extract the S channel.
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]

    # Convert to grayscale and apply Sobel operator in the x direction.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Apply threshold on Sobel result.
    sx_thresh_min = 20
    sx_thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh_min) & (scaled_sobel <= sx_thresh_max)] = 1

    # Apply threshold on the S channel.
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Combine both thresholds.
    binary_output = np.zeros_like(sxbinary)
    binary_output[(sxbinary == 1) | (s_binary == 1)] = 255

    return binary_output


def perspective_transform(img):
    """
    Warp the binary image to a bird's eye view.
    """
    img_size = (img.shape[1], img.shape[0])
    # These source points should be tuned for your specific camera/view.
    src = np.float32([[580, 460], [700, 460], [1040, 680], [260, 680]])
    # Destination points form a rectangle.
    dst = np.float32([[260, 0], [1040, 0], [1040, 720], [260, 720]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M


def sliding_window_search(binary_warped):
    """
    Find lane pixels using a sliding window search and fit a second order polynomial.
    Returns polynomial coefficients for the left and right lanes.
    """
    # Compute histogram of the bottom half of the image.
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2 :, :], axis=0)

    # Find peaks for left and right lanes.
    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Parameters for sliding windows.
    nwindows = 9
    window_height = binary_warped.shape[0] // nwindows
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    minpix = 50

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window.
        good_left_inds = (
            (nonzeroy >= win_y_low)
            & (nonzeroy < win_y_high)
            & (nonzerox >= win_xleft_low)
            & (nonzerox < win_xleft_high)
        ).nonzero()[0]
        good_right_inds = (
            (nonzeroy >= win_y_low)
            & (nonzeroy < win_y_high)
            & (nonzerox >= win_xright_low)
            & (nonzerox < win_xright_high)
        ).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # Recenter next window if enough pixels were found.
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions.
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each.
    left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) and len(lefty) else None
    right_fit = np.polyfit(righty, rightx, 2) if len(rightx) and len(righty) else None

    return left_fit, right_fit


def draw_lane(original_img, binary_warped, left_fit, right_fit, M_inv):
    """
    Draw the detected lane area back onto the original image.
    """
    # Create an image to draw the lane on.
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    if left_fit is not None:
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    else:
        left_fitx = None
    if right_fit is not None:
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
    else:
        right_fitx = None

    # Draw the lane onto the warped blank image.
    if left_fitx is not None and right_fitx is not None:
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space.
    newwarp = cv2.warpPerspective(
        color_warp, M_inv, (original_img.shape[1], original_img.shape[0])
    )
    # Combine the result with the original image.
    result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)
    return result


def detect_lanes(frame):
    """
    Complete lane detection pipeline:
      1. Preprocess the image.
      2. Warp the image to a bird's eye view.
      3. Find lane pixels using sliding windows and fit a polynomial.
      4. Draw the lane area back onto the original frame.
    """
    # Preprocess to get a binary image.
    binary = preprocess_image(frame)

    # Perspective transform.
    binary_warped, M = perspective_transform(binary)
    img_size = (frame.shape[1], frame.shape[0])
    # Compute inverse transform.
    src = np.float32([[580, 460], [700, 460], [1040, 680], [260, 680]])
    dst = np.float32([[260, 0], [1040, 0], [1040, 720], [260, 720]])
    M_inv = cv2.getPerspectiveTransform(dst, src)

    # Use sliding window search to fit lane lines.
    left_fit, right_fit = sliding_window_search(binary_warped)

    # Draw the detected lane on the original image.
    result = draw_lane(frame, binary_warped, left_fit, right_fit, M_inv)
    return result
