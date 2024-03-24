import cv2
import numpy as np

def perform_suace(src, distance, sigma):
    assert src.dtype == np.uint8, "Input image must be of type CV_8UC1"
    assert distance > 0 and sigma > 0, "Distance and sigma must be greater than 0"

    dst = np.zeros_like(src, dtype=np.uint8)
    smoothed = cv2.GaussianBlur(src, (0, 0), sigma)

    half_distance = distance // 2
    distance_d = float(distance)

    for x in range(src.shape[1]):
        for y in range(src.shape[0]):
            val = src[y, x]
            adjuster = smoothed[y, x]

            if (val - adjuster) > distance_d:
                adjuster += (val - adjuster) * 0.5

            adjuster = max(adjuster, half_distance)
            b = adjuster + half_distance
            b = min(b, 255)
            a = max(b - distance, 0)

            if a <= val <= b:
                dst[y, x] = int(((val - a) / distance_d) * 255)
            elif val < a:
                dst[y, x] = 0
            elif val > b:
                dst[y, x] = 255

    return dst

# Example usage
