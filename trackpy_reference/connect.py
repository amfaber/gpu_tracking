import pandas as pd
import gpu_tracking as gt


tp_df = pd.read_csv("trackpy.csv")
gt_df = pd.read_csv("gpu_tracking.csv")

df = gt.connect(tp_df, gt_df, 0.2)
na = df.isna().any(axis = 1)
total = len(df)
print(na.sum())
print(total)
print(na.sum()/total)

print(df[na][["y_x", "x_x", "y_y", "x_y", "frame_x", "frame_y"]])
