#include "hip/hip_runtime.h"
#include <bmp.h>
#include <timer.h>

#define T 32
#define FILTER_SIZE 21

using namespace std;

__device__ float filter[FILTER_SIZE] = {0.00448085286008595, 0.008088921017904009, 0.013721954636352889, 0.021874457565195425, 0.032768377542725796, 0.046128493416378785, 0.06102117792005995, 0.07585585938388222, 0.08861252263650558, 0.09727441460033705, 0.10034593684114455, 0.09727441460033709, 0.08861252263650556, 0.07585585938388223, 0.06102117792005995, 0.046128493416378785, 0.0327683775427258, 0.021874457565195425, 0.013721954636352883, 0.008088921017904007, 0.004480852860085952};

__global__ void blur(const FloatPixel *pixels_in, FloatPixel *pixels_out,
                     size_t width, size_t height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  /* ANCHOR: gaussianblur */
  __shared__ FloatPixel shared_pixels[T + FILTER_SIZE - 1][T + FILTER_SIZE - 1];

  for (int dy = -FILTER_SIZE / 2; dy <= T + FILTER_SIZE / 2; dy += T) {
    for (int dx = -FILTER_SIZE / 2; dx <= T + FILTER_SIZE / 2; dx += T) {
      int nx = x + dx;
      int ny = y + dy;
      int sx = dx + threadIdx.x + FILTER_SIZE / 2;
      int sy = dy + threadIdx.y + FILTER_SIZE / 2;
      if (sx < T + FILTER_SIZE - 1 && sy < T + FILTER_SIZE - 1) {
        if (0 <= nx && nx < width && 0 <= ny && ny < height) {
          shared_pixels[sy][sx] = pixels_in[ny * width + nx];
        } else {
          shared_pixels[sy][sx] = FloatPixel{0.0, 0.0, 0.0};
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

int main() {
  Bmp bmp("../../butterfly.bmp");
  FloatImage im = bmp.get_pixel_data();

  size_t pixel_data_size = im.width * im.height * sizeof(FloatPixel);

  FloatPixel *device_pixels_in = nullptr;
  FloatPixel *device_pixels_out = nullptr;

  hipMalloc((void **)&device_pixels_in, pixel_data_size);
  hipMalloc((void **)&device_pixels_out, pixel_data_size);
  hipMemcpy(device_pixels_in, im.data, pixel_data_size,
             hipMemcpyHostToDevice);

  int dimx = ceil(((float)im.width) / T);
  int dimy = ceil(((float)im.height) / T);
  dim3 block(T, T, 1), grid(dimx, dimy, 1);

  timer time;
  hipLaunchKernelGGL(blur, grid, block, 0, 0, device_pixels_in, device_pixels_out, im.width,
                        im.height);
  hipDeviceSynchronize();
  cout << "Elapsed time: " << time.getTime() << endl;

  hipMemcpy(im.data, device_pixels_out, pixel_data_size,
             hipMemcpyDeviceToHost);

  bmp.set_pixel_data(im);
  bmp.save("shared_memory.bmp");
}
