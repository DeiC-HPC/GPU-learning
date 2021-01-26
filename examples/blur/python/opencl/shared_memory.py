from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math
import time
import pyopencl as cl

dim_size = 32

# Getting context for running on the GPU
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

prg = cl.Program(ctx, """
#define T %d
#define FILTER_SIZE 21

typedef struct {
  float red;
  float green;
  float blue;
} FloatPixel;

__constant float filter[FILTER_SIZE] = {0.00448085286008595, 0.008088921017904009, 0.013721954636352889, 0.021874457565195425, 0.032768377542725796, 0.046128493416378785, 0.06102117792005995, 0.07585585938388222, 0.08861252263650558, 0.09727441460033705, 0.10034593684114455, 0.09727441460033709, 0.08861252263650556, 0.07585585938388223, 0.06102117792005995, 0.046128493416378785, 0.0327683775427258, 0.021874457565195425, 0.013721954636352883, 0.008088921017904007, 0.004480852860085952};

__kernel void blur(__global FloatPixel *pixels_in, __global FloatPixel *pixels_out, int width, int height) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  int loc_x = get_local_id(0);
  int loc_y = get_local_id(1);

  /* ANCHOR: gaussianblur */
  __local FloatPixel shared_pixels[T + FILTER_SIZE - 1][T + FILTER_SIZE - 1];

  for (int dy = -FILTER_SIZE / 2; dy <= T + FILTER_SIZE / 2; dy += T) {
    for (int dx = -FILTER_SIZE / 2; dx <= T + FILTER_SIZE / 2; dx += T) {
      int nx = x + dx;
      int ny = y + dy;
      int sx = dx + loc_x + FILTER_SIZE / 2;
      int sy = dy + loc_y + FILTER_SIZE / 2;
      if (sx < T + FILTER_SIZE - 1 && sy < T + FILTER_SIZE - 1) {
        if (0 <= nx && nx < width && 0 <= ny && ny < height) {
          shared_pixels[sy][sx] = pixels_in[ny * width + nx];
        } else {
          shared_pixels[sy][sx].red = 0.0;
          shared_pixels[sy][sx].blue = 0.0;
          shared_pixels[sy][sx].green = 0.0;
        }
      }
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (x < width && y < height) {
    float red = 0.0;
    float green = 0.0;
    float blue = 0.0;
    for (int dy = -FILTER_SIZE / 2; dy <= FILTER_SIZE / 2; dy++) {
      for (int dx = -FILTER_SIZE / 2; dx <= FILTER_SIZE / 2; dx++) {
        float filter_value = filter[dy + FILTER_SIZE / 2] * filter[dx + FILTER_SIZE / 2];
        int sx = loc_x + dx + FILTER_SIZE / 2;
        int sy = loc_y + dy + FILTER_SIZE / 2;
        red += filter_value * shared_pixels[sy][sx].red;
        green += filter_value * shared_pixels[sy][sx].green;
        blue += filter_value * shared_pixels[sy][sx].blue;
      }
    }
    pixels_out[y * width + x].red = red;
    pixels_out[y * width + x].green += green;
    pixels_out[y * width + x].blue += blue;
  }
  /* ANCHOR_END: gaussianblur */
}
""" % (dim_size)).build()

im = Image.open("../../butterfly.bmp")

(width, height) = im.size
im = np.asarray(im, np.float32)

res = np.empty(height*width*3).astype(np.float32)

grid_size = (dim_size * int(math.ceil(width/float(dim_size))),
             dim_size * int(math.ceil(height/float(dim_size))))

mf = cl.mem_flags
im_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=im)
res_dev = cl.Buffer(ctx, mf.WRITE_ONLY, size=res.nbytes)

start_time = time.time()
prg.blur(
  queue,
  grid_size,
  (dim_size, dim_size),
  im_dev,
  res_dev,
  np.int32(width),
  np.int32(height)
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
