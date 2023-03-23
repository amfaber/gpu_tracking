import trackpy as tp
import tifffile
import time
import gpu_tracking as gt
import numpy as np

import GPUtil
import platform
import cpuinfo
def format_row(times, func_name):
    mean = np.mean(times)
    std = np.std(times) / len(times)
    try:
        gpu = GPUtil.getGPUs()[0].name
    except IndexError:
        gpu = "No GPU found"
    return f"{platform.system()}, {cpuinfo.get_cpu_info()['brand_raw']}, {gpu}, {func_name}: {mean} ± {std}\n"

MINMASS = 700

def time_func(to_time, n = 5, **kwargs):
    start = time.perf_counter()
    times = []
    for _ in range(n):
        start = time.perf_counter()
        df = to_time(**kwargs)
        times.append(time.perf_counter() - start)
    times = np.array(times)
    df.to_csv(f"{to_time.__name__}.csv")
    if "key" in kwargs and kwargs["key"] is not None:
        n = len(kwargs["key"])
    else:
        n = 5000
    print(len(df) / n)
    with open(f"time.txt", "a") as file:
        file.write(format_row(times, to_time.__name__))
    return df, times
    


def trackpy(key = None):
    vid = tifffile.imread("../../gpu_tracking_testing/easy_test_data.tif", key = key).astype("float32")
    locate_result = tp.batch(vid, 9, minmass = MINMASS, convert_to_int = False)
    tracked = tp.link(locate_result, 10)
    return tracked

def gpu_tracking(key = None):
    return gt.batch("../../gpu_tracking_testing/easy_test_data.tif", 9, characterize = True, minmass = MINMASS, search_range = 10, keys = key, tqdm = False)


if __name__ == "__main__":
    repeats = 5
    n = None
    tpdf, _ = time_func(trackpy, 5, key = n)
    gtdf, _ = time_func(gpu_tracking, 5, key = n)
