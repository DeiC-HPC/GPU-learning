from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""
#define FILTER_SIZE 21

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

  if (x < width && y < height) {
    /* ANCHOR: gaussianblur */
    float red = 0.0;
    float green = 0.0;
    float blue = 0.0;
    for (int dy = -FILTER_SIZE / 2; dy <= FILTER_SIZE / 2; dy++) {
      for (int dx = -FILTER_SIZE / 2; dx <= FILTER_SIZE / 2; dx++) {
        float filter_value = filter[dy + FILTER_SIZE / 2] * filter[dx + FILTER_SIZE / 2];
        int ny = y + dy;
        int nx = x + dx;
        if (0 <= ny && ny < height && 0 <= nx && nx < width) {
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
}
""")

im = Image.open("../../butterfly.bmp")

(width, height) = im.size
im = np.asarray(im, np.float32)

res = np.empty(height*width*3).astype(np.float32)

dim_size = 32
block_size = (dim_size,dim_size,1)

dimx = int(math.ceil(width / float(dim_size)))
dimy = int(math.ceil(height / float(dim_size)))
grid_size = (dimx, dimy, 1)

im_gpu = cuda.mem_alloc(im.nbytes)
res_gpu = cuda.mem_alloc(res.nbytes)

cuda.memcpy_htod(im_gpu, im)

blur = mod.get_function("blur")

start_time = time.time()
blur(
    im_gpu,
    res_gpu,
    np.uint32(width),
    np.uint32(height),
    block=block_size,
    grid=grid_size
)
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
