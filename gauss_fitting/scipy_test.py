import scipy.optimize as opt
import numpy as np
import time

# vec![-1.0f64, 3.0],
# vec![2.0, 1.5],
# vec![2.0, -1.0],

initial_simplex = np.array([
	[-1., 3.],
	[2, 1.5],
	[2, -1],
])

options = dict(initial_simplex = initial_simplex)
now = time.time()
# dummy_x0 = np.array([[-1, 3]])

for i in range(1_000):
	res = opt.minimize(opt.rosen, x0 = initial_simplex[0], options = options, method = "Nelder-Mead")
	# break
elapsed = time.time() - now
print(res)

print(elapsed)
