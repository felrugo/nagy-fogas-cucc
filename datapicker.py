import sys
import pathlib
import shutil
import pandas as pd

if len(sys.argv) < 3:
    raise Exception("Not enough argument")

gid = 0

src = sys.argv[1]
dst = sys.argv[2]

dstOrig = pathlib.Path(dst+"/original")
dstBin = pathlib.Path(dst+"/binary")

if not dstOrig.exists():
    dstOrig.mkdir()

if not dstBin.exists():
    dstBin.mkdir()

srcD = pathlib.Path(src)

segments = pd.read_csv(str(srcD)+"/intervals.csv", index_col=0, dtype={"start": "int32", "len": "int32"})

def processDir(d: pathlib.Path):
    originals = [f for f in d.iterdir() if f.is_file() and f.suffix == ".png"]
    originals.sort(key= lambda f: int(f.stem))
    binDir = d.joinpath("axis/binaris")
    if not binDir.exists():
        return
    bins = [f for f in binDir.iterdir() if f.is_file() and f.suffix == ".png"]
    bins.sort(key= lambda f: int(f.stem))

    seg = segments[segments.dir == d.stem]
    ser = seg["start"]
    start = ser.values[0]

    global gid

    for i in range(len(bins)):
        shutil.copy(str(originals[i+start].absolute()), str(dstOrig.joinpath(f"{gid}.png").absolute()))
        shutil.copy(str(bins[i].absolute()), str(dstBin.joinpath(f"{gid}.png").absolute()))
        gid += 1


for d in srcD.iterdir():
    if d.is_dir():
        print(f"Processing: {d.stem}")
        processDir(d)


