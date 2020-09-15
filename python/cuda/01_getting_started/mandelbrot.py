import math
import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(z, max_iterations):
    c = z
    for i in range(max_iterations):
        if abs(z) > 2:
            return i
        z = z*z + c
    return max_iterations

width = 1000
height = 1000
max_iterations = 100
xmin = -2.5
xmax = 1.5
ymin = -2.0
ymax = 2.0
reals = np.linspace(xmin, xmax, width)
imaginaries = np.linspace(ymax, ymin, height)
res = np.empty((width, height))

for i in range(width):
    for j in range(height):
        res[i, j] = mandelbrot(complex(reals[j], imaginaries[i]), max_iterations)

# Displaying the Mandelbrot set
fig, ax = plt.subplots()

ax.imshow(res, interpolation='bicubic', cmap=plt.get_cmap("terrain"))
plt.axis("off")
plt.tight_layout()

plt.show()
