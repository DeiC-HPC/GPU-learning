from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

dim_size = 32
filter_size = 21

mod = SourceModule("""
#define T %d
#define FILTER_SIZE %d
#define SHARED_SIZE (T+FILTER_SIZE-1)

struct FloatPixel {
  float red;
  float green;
  float blue;
};

__device__ float filter[FILTER_SIZE] = {0.00448085286008595, 0.008088921017904009, 0.013721954636352889, 0.021874457565195425, 0.032768377542725796, 0.046128493416378785, 0.06102117792005995, 0.07585585938388222, 0.08861252263650558, 0.09727441460033705, 0.10034593684114455, 0.09727441460033709, 0.08861252263650556, 0.07585585938388223, 0.06102117792005995, 0.046128493416378785, 0.0327683775427258, 0.021874457565195425, 0.013721954636352883, 0.008088921017904007, 0.004480852860085952};

__global__ void blur(const FloatPixel *pixels_in, FloatPixel *pixels_out,
                     int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  /* ANCHOR: gaussianblur */
  extern __shared__ FloatPixel shared_pixels[];

  for (int dy = -FILTER_SIZE / 2; dy <= T + FILTER_SIZE / 2; dy += T) {
    for (int dx = -FILTER_SIZE / 2; dx <= T + FILTER_SIZE / 2; dx += T) {
      int nx = x + dx;
      int ny = y + dy;
      int sx = dx + threadIdx.x + FILTER_SIZE / 2;
      int sy = dy + threadIdx.y + FILTER_SIZE / 2;
      if (sx < T + FILTER_SIZE - 1 && sy < T + FILTER_SIZE - 1) {
        if (0 <= nx && nx < width && 0 <= ny && ny < height) {
          shared_pixels[sy*SHARED_SIZE+sx] = pixels_in[ny * width + nx];
        } else {
          shared_pixels[sy*SHARED_SIZE+sx] = FloatPixel{0.0, 0.0, 0.0};
        }
      }
    }
  }

  __syncthreads();

  if (x < width && y < height) {
    float red = 0.0;
    float green = 0.0;
    float blue = 0.0;
    for (int dy = -FILTER_SIZE / 2; dy <= FILTER_SIZE / 2; dy++) {
      for (int dx = -FILTER_SIZE / 2; dx <= FILTER_SIZE / 2; dx++) {
        float filter_value = filter[dy + FILTER_SIZE / 2] * filter[dx + FILTER_SIZE / 2];
        int sx = threadIdx.x + dx + FILTER_SIZE / 2;
        int sy = threadIdx.y + dy + FILTER_SIZE / 2;
        red += filter_value * shared_pixels[sy*SHARED_SIZE+sx].red;
        green += filter_value * shared_pixels[sy*SHARED_SIZE+sx].green;
        blue += filter_value * shared_pixels[sy*SHARED_SIZE+sx].blue;
      }
    }
    pixels_out[y * width + x].red = red;
    pixels_out[y * width + x].green += green;
    pixels_out[y * width + x].blue += blue;
  }
  /* ANCHOR_END: gaussianblur */
}

""" % (dim_size, filter_size))

im = Image.open("../../butterfly.bmp")

(width, height) = im.size
im = np.asarray(im, np.float32)

res = np.empty(height*width*3).astype(np.float32)

block_size = (dim_size,dim_size,1)

dimx = int(math.ceil(width / float(dim_size)))
dimy = int(math.ceil(height / float(dim_size)))
grid_size = (dimx, dimy, 1)

im_gpu = cuda.mem_alloc(im.nbytes)
res_gpu = cuda.mem_alloc(res.nbytes)

cuda.memcpy_htod(im_gpu, im)

blur = mod.get_function("blur")

start_time = time.time()
# ANCHOR: call
blur(
    im_gpu,
    res_gpu,
    np.uint32(width),
    np.uint32(height),
    block=block_size,
    grid=grid_size,
    shared=(dim_size+filter_size-1)**2*(3*4)
)
# ANCHOR_END: call
cuda.Context.synchronize()

total_time = time.time() - start_time
print("Elapsed time:", total_time)

cuda.memcpy_dtoh(res, res_gpu)

# Setting shape of array to help displaying it
res.shape = (height, width, 3)

# Displaying the image
fig, ax = plt.subplots()

ax.imshow(np.asarray(res, np.int32))
plt.axis("off")
plt.tight_layout()

plt.show()
