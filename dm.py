import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import cv2


def euclidianDistanceMap(mask, min_value=0.0, max_value=1.0, mean=None, mapper = lambda x: x):
    """
            Parameters
            ----------
            mask : np.array
                The image mask as a float32 numpy array
            level : int
                exponent of the inverted normal distance
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    b = np.zeros(mask.shape)
    contim = cv2.drawContours(b, contours, -1, 1.0, 0)
    dstmap = distance_transform_edt(1.0-contim)

    dstmap = mapper(dstmap)

    dstmap = (dstmap - dstmap.min()) / (dstmap.max() - dstmap.min()) # mapped to 0-1
    dstmap = min_value + dstmap * (max_value - min_value) # mapped to min-max

    # Apply mean
    if mean is not None:
        dstmap = dstmap / np.average(dstmap) * mean

    return dstmap


img = cv2.imread("546.png", cv2.IMREAD_GRAYSCALE)
out = euclidianDistanceMap(img, mean=1.0)
cv2.imshow("Windows", out)
cv2.waitKey()