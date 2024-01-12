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

  if (x < width && y < height) {
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
  }
}

int main() {
  Bmp bmp("../../butterfly.bmp");
  FloatImage im = bmp.get_pixel_data();

  size_t pixel_data_size = im.width * im.height * sizeof(FloatPixel);

  FloatPixel *device_pixels_in = nullptr;
  FloatPixel *device_pixels_out = nullptr;

  cudaMalloc((void **)&device_pixels_in, pixel_data_size);
  cudaMalloc((void **)&device_pixels_out, pixel_data_size);
  cudaMemcpy(device_pixels_in, im.data, pixel_data_size,
             cudaMemcpyHostToDevice);

  int dimx = ceil(((float)im.width) / T);
  int dimy = ceil(((float)im.height) / T);
  dim3 block(T, T, 1), grid(dimx, dimy, 1);

  timer time;
  blur<<<grid, block>>>(device_pixels_in, device_pixels_out, im.width,
                        im.height);
  cudaDeviceSynchronize();
  cout << "Elapsed time: " << time.getTime() << endl;

  cudaMemcpy(im.data, device_pixels_out, pixel_data_size,
             cudaMemcpyDeviceToHost);

  bmp.set_pixel_data(im);
  bmp.save("naive.bmp");
}
