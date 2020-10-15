import numpy as np
import cv2


def euclidianDistanceMap(mask, level=1):
    """
            Parameters
            ----------
            mask : np.array
                The image mask as an uint8 numpy array
            level : int
                exponent of the inverted normal distance
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    b = np.zeros(mask.shape)
    contim = cv2.drawContours(b, contours, -1, 255, 0).astype("uint8")
    invcont = cv2.bitwise_not(contim)
    dstmap = cv2.distanceTransform(invcont, cv2.DIST_L2, 3)

    dstmap = dstmap / dstmap.max()
    normalized = (255 * (1.0 - dstmap) ** level).astype("uint8")
    return normalized


img = cv2.imread("546.png", cv2.IMREAD_GRAYSCALE)
out = euclidianDistanceMap(img)
cv2.imshow("Windows", out)
cv2.waitKey()