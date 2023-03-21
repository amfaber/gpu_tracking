import trackpy as tp
import tifffile
if __name__ == "__main__":
    vid = tifffile.imread("../../emily_tracking/sample_vids/s_20.tif").astype("float32")
    # batch_result = tp.batch(vid, 9, threshold = 0, percentile = 0, )
    locate_result = tp.batch(vid, 9, threshold = 0, percentile = 0, minmass = 0, characterize = False, processes=0)
    import pickle
    
    with open("locations.pkl", "wb") as file:
        pickle.dump(locate_result, file)
