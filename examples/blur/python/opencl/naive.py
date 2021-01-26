from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math
import time
import pyopencl as cl

# Getting context for running on the GPU
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

prg = cl.Program(ctx, """
#define FILTER_SIZE 21

typedef struct {
  float red;
  float green;
  float blue;
} FloatPixel;

__constant float filter[FILTER_SIZE] = {0.00448085286008595, 0.008088921017904009, 0.013721954636352889, 0.021874457565195425, 0.032768377542725796, 0.046128493416378785, 0.06102117792005995, 0.07585585938388222, 0.08861252263650558, 0.09727441460033705, 0.10034593684114455, 0.09727441460033709, 0.08861252263650556, 0.07585585938388223, 0.06102117792005995, 0.046128493416378785, 0.0327683775427258, 0.021874457565195425, 0.013721954636352883, 0.008088921017904007, 0.004480852860085952};

__kernel void blur(__global const FloatPixel *pixels_in, __global FloatPixel *pixels_out, int width) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  /* ANCHOR: gaussianblur */
  float red = 0.0;
  float green = 0.0;
  float blue = 0.0;
  for (int dy = -FILTER_SIZE / 2; dy <= FILTER_SIZE / 2; dy++) {
    for (int dx = -FILTER_SIZE / 2; dx <= FILTER_SIZE / 2; dx++) {
      float filter_value = filter[dy + FILTER_SIZE / 2] * filter[dx + FILTER_SIZE / 2];
      int ny = y + dy;
      int nx = x + dx;
      if (0 <= ny && 0 <= nx) {
        red += filter_value * pixels_in[ny * width + nx].red;
        green += filter_value * pixels_in[ny * width + nx].green;
        blue += filter_value * pixels_in[ny * width + nx].blue;
      }
    }
  }
  pixels_out[y * width + x].red = red;
  pixels_out[y * width + x].green += green;
  pixels_out[y * width + x].blue += blue;
  /* ANCHOR_END: gaussianblur */
}
""").build()

im = Image.open("../../butterfly.bmp")

(width, height) = im.size
im = np.asarray(im, np.float32)

res = np.empty(height*width*3).astype(np.float32)

dim_size = 32
block_size = (dim_size,dim_size,1)

mf = cl.mem_flags
im_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=im)
res_dev = cl.Buffer(ctx, mf.WRITE_ONLY, size=res.nbytes)

start_time = time.time()
prg.blur(
  queue,
  (width, height),
  None,
  im_dev,
  res_dev,
  np.int32(width)
)

cl.enqueue_copy(queue, res, res_dev).wait()

total_time = time.time() - start_time
print("Elapsed time:", total_time)

# Setting shape of array to help displaying it
res.shape = (height, width, 3)

# Displaying the image
fig, ax = plt.subplots()

ax.imshow(np.asarray(res, np.int32))
plt.axis("off")
plt.tight_layout()

plt.show()
