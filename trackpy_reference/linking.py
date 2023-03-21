import trackpy as tp
import pickle
import time
with open("locations.pkl", "rb") as file:
    data = pickle.load(file)
# tp.quiet()
now = time.time()
res = tp.link(data, 9, memory = 0)
print(time.time()-now)
with open("linked.pkl", "wb") as file:
    pickle.dump(res, file)