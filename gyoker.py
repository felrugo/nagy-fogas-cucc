import numpy as np
import cv2
import sys
import pathlib

src = sys.argv[1]

srcD = pathlib.Path(src+"/binary")
dstBin = pathlib.Path(src+"/roots")

if not dstBin.exists():
    dstBin.mkdir()

cv = cv2

def processOneImage(path):
    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    roots = [contours[i] for i in range(len(contours)) if hierarchy[0][i, 3] != -1]
    nm = np.zeros(gray.shape, np.uint8)
    nm = cv.drawContours(nm, roots, -1, 255, -1)
    ret, nmth = cv.threshold(nm, 1, 255, cv2.THRESH_BINARY)
    return nmth

for f in srcD.iterdir():
    if f.is_file():
        rimg = processOneImage(f)
        cv2.imwrite(str(dstBin.joinpath(f.name)), rimg)

