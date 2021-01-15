#include <bmp.h>
#include <timer.h>

#define T 32
#define FILTER_SIZE 21
#define SHARED_SIZE (T+FILTER_SIZE-1)

using namespace std;

__device__ float filter[FILTER_SIZE][FILTER_SIZE] = {
  {0.00002007804235374044, 0.00003624526487808453, 0.00006148605967827151, 0.00009801622574383467, 0.0001468302782326991,  0.0002066949916562368, 0.00027342691960891425, 0.00033989894447454625, 0.000397059675495217, 0.00043587233887510676, 0.00044963537809264664, 0.0004358723388751069, 0.0003970596754952169,  0.00033989894447454636, 0.00027342691960891425, 0.0002066949916562368,  0.00014683027823269912, 0.00009801622574383467, 0.00006148605967827148, 0.00003624526487808452, 0.000020078042353740445},
  {0.00003624526487808453, 0.00006543064323388922, 0.00011099580726472024, 0.00017694075955435861, 0.0002650608178279684,  0.00037312973991999304, 0.0004935954886148329,  0.0006135920553014558,  0.0007167796968039247, 0.0007868450567649749,  0.0008116903575756024,  0.0007868450567649753, 0.0007167796968039245,  0.000613592055301456,   0.0004935954886148329, 0.00037312973991999304, 0.00026506081782796845, 0.00017694075955435861, 0.0001109958072647202,  0.00006543064323388921, 0.00003624526487808454},
  {0.00006148605967827151, 0.00011099580726472024, 0.00018829203904212653, 0.00030016031440443787, 0.0004496461901481681,  0.0006329730941028526, 0.0008373298352758811,  0.0010408906613671954,  0.001215937015830923, 0.0013347951044236082,  0.0013769423932765176,  0.0013347951044236086, 0.0012159370158309228,  0.0010408906613671954,  0.0008373298352758811, 0.0006329730941028526,  0.0004496461901481682,  0.00030016031440443787, 0.00018829203904212647, 0.00011099580726472022, 0.00006148605967827152},
  {0.00009801622574383467, 0.00017694075955435861, 0.00030016031440443787, 0.00047849189377153535, 0.0007167904840386581,  0.0010090357717829743, 0.0013348051669905914,  0.0016593057771641627,  0.0019383508661571605, 0.002127825054354299,   0.002195012937271397,   0.0021278250543543, 0.0019383508661571598,  0.001659305777164163,   0.0013348051669905914, 0.0010090357717829743,  0.0007167904840386584,  0.00047849189377153535, 0.00030016031440443776, 0.0001769407595543586,  0.00009801622574383471},
  {0.0001468302782326991,  0.0002650608178279684,  0.0004496461901481681, 0.0007167904840386581,  0.0010737665667826164,  0.0015115558877450413, 0.0019995649961863675,  0.002485673439118972,   0.002903688596766351, 0.003187524742871483,   0.003288173543289142,   0.003187524742871484, 0.00290368859676635,    0.0024856734391189724,  0.0019995649961863675, 0.0015115558877450413,  0.0010737665667826166,  0.0007167904840386581, 0.00044964619014816793, 0.00026506081782796834, 0.00014683027823269914},
  {0.0002066949916562368, 0.00037312973991999304, 0.0006329730941028526, 0.0010090357717829743, 0.0015115558877450413,  0.002127837904864901, 0.0028148150039451637, 0.0034991165101831657,  0.004087562167046764, 0.004487122193473748,  0.004628806886937098,   0.00448712219347375, 0.004087562167046763,  0.003499116510183166,   0.0028148150039451637, 0.002127837904864901,  0.0015115558877450417,  0.0010090357717829743, 0.0006329730941028524, 0.000373129739919993,   0.00020669499165623686},
  {0.00027342691960891425, 0.0004935954886148329, 0.0008373298352758811, 0.0013348051669905914,  0.0019995649961863675, 0.0028148150039451637, 0.0037235841547516115,  0.004628813891742926,  0.005407240509747547, 0.005935799360396844,   0.00612322726553858,   0.005935799360396846, 0.005407240509747545,   0.004628813891742927,  0.0037235841547516115, 0.0028148150039451637,  0.001999564996186368,  0.0013348051669905914, 0.0008373298352758808,  0.0004935954886148328, 0.00027342691960891436},
  {0.00033989894447454625, 0.0006135920553014558, 0.0010408906613671954, 0.0016593057771641627,  0.002485673439118972,  0.0034991165101831657, 0.004628813891742926,   0.005754111402867312,  0.006721779056765847, 0.007378834315572626,   0.007611827274765787,  0.00737883431557263, 0.006721779056765845,   0.0057541114028673125, 0.004628813891742926, 0.0034991165101831657,  0.0024856734391189724, 0.0016593057771641627, 0.001040890661367195,   0.0006135920553014557, 0.0003398989444745464},
  {0.000397059675495217,  0.0007167796968039247, 0.001215937015830923, 0.0019383508661571605, 0.002903688596766351,  0.004087562167046764, 0.005407240509747547,  0.006721779056765847,  0.007852179168005215, 0.008619731265725196,  0.00889190659981728,   0.008619731265725199, 0.007852179168005211,  0.006721779056765849,  0.005407240509747547, 0.004087562167046764,  0.0029036885967663514, 0.0019383508661571605, 0.0012159370158309226, 0.0007167796968039246, 0.00039705967549521717},
  {0.00043587233887510676, 0.0007868450567649749, 0.0013347951044236082, 0.002127825054354299,   0.003187524742871483,  0.004487122193473748, 0.005935799360396844,   0.007378834315572626,  0.008619731265725196, 0.009462311735838265,   0.009761092263744732,  0.00946231173583827, 0.008619731265725194,   0.007378834315572628,  0.005935799360396844, 0.004487122193473748,   0.0031875247428714837, 0.002127825054354299, 0.0013347951044236075,  0.0007868450567649748, 0.0004358723388751069},
  {0.00044963537809264664, 0.0008116903575756024, 0.0013769423932765176, 0.002195012937271397,   0.003288173543289142,  0.004628806886937098, 0.00612322726553858,    0.007611827274765787,  0.00889190659981728, 0.009761092263744732,   0.010069307040526972,  0.009761092263744735, 0.008891906599817279,   0.007611827274765789,  0.00612322726553858, 0.004628806886937098,   0.003288173543289143,  0.002195012937271397, 0.0013769423932765172,  0.0008116903575756022, 0.00044963537809264686},
  {0.0004358723388751069, 0.0007868450567649753, 0.0013347951044236086, 0.0021278250543543,    0.003187524742871484,  0.00448712219347375, 0.005935799360396846,  0.00737883431557263,   0.008619731265725199, 0.00946231173583827,   0.009761092263744735,  0.009462311735838273, 0.008619731265725197,  0.0073788343155726304, 0.005935799360396846, 0.00448712219347375,   0.003187524742871485,  0.0021278250543543, 0.0013347951044236082, 0.0007868450567649751, 0.0004358723388751071},
  {0.0003970596754952169, 0.0007167796968039245, 0.0012159370158309228, 0.0019383508661571598, 0.00290368859676635,   0.004087562167046763, 0.005407240509747545,  0.006721779056765845,  0.007852179168005211, 0.008619731265725194,  0.008891906599817279,  0.008619731265725197, 0.00785217916800521,   0.006721779056765846,  0.005407240509747545, 0.004087562167046763,  0.0029036885967663505, 0.0019383508661571598, 0.0012159370158309221, 0.0007167796968039244, 0.00039705967549521706},
  {0.00033989894447454636, 0.000613592055301456,  0.0010408906613671954, 0.001659305777164163,   0.0024856734391189724, 0.003499116510183166, 0.004628813891742927,   0.0057541114028673125, 0.006721779056765849, 0.007378834315572628,   0.007611827274765789,  0.0073788343155726304, 0.006721779056765846,   0.005754111402867313,  0.004628813891742927, 0.003499116510183166,   0.002485673439118973,  0.001659305777164163, 0.0010408906613671952,  0.0006135920553014558, 0.00033989894447454647},
  {0.00027342691960891425, 0.0004935954886148329, 0.0008373298352758811, 0.0013348051669905914,  0.0019995649961863675, 0.0028148150039451637, 0.0037235841547516115,  0.004628813891742926,  0.005407240509747547, 0.005935799360396844,   0.00612322726553858,   0.005935799360396846, 0.005407240509747545,   0.004628813891742927,  0.0037235841547516115, 0.0028148150039451637,  0.001999564996186368,  0.0013348051669905914, 0.0008373298352758808,  0.0004935954886148328, 0.00027342691960891436},
  {0.0002066949916562368, 0.00037312973991999304, 0.0006329730941028526, 0.0010090357717829743, 0.0015115558877450413,  0.002127837904864901, 0.0028148150039451637, 0.0034991165101831657,  0.004087562167046764, 0.004487122193473748,  0.004628806886937098,   0.00448712219347375, 0.004087562167046763,  0.003499116510183166,   0.0028148150039451637, 0.002127837904864901,  0.0015115558877450417,  0.0010090357717829743, 0.0006329730941028524, 0.000373129739919993,   0.00020669499165623686},
  {0.00014683027823269912, 0.00026506081782796845, 0.0004496461901481682, 0.0007167904840386584,  0.0010737665667826166,  0.0015115558877450417, 0.001999564996186368,   0.0024856734391189724,  0.0029036885967663514, 0.0031875247428714837,  0.003288173543289143,   0.003187524742871485, 0.0029036885967663505,  0.002485673439118973,   0.001999564996186368, 0.0015115558877450417,  0.0010737665667826168,  0.0007167904840386584, 0.00044964619014816804, 0.0002650608178279684,  0.0001468302782326992},
  {0.00009801622574383467, 0.00017694075955435861, 0.00030016031440443787, 0.00047849189377153535, 0.0007167904840386581,  0.0010090357717829743, 0.0013348051669905914,  0.0016593057771641627,  0.0019383508661571605, 0.002127825054354299,   0.002195012937271397,   0.0021278250543543, 0.0019383508661571598,  0.001659305777164163,   0.0013348051669905914, 0.0010090357717829743,  0.0007167904840386584,  0.00047849189377153535, 0.00030016031440443776, 0.0001769407595543586,  0.00009801622574383471},
  {0.00006148605967827148, 0.0001109958072647202,  0.00018829203904212647, 0.00030016031440443776, 0.00044964619014816793, 0.0006329730941028524, 0.0008373298352758808,  0.001040890661367195,   0.0012159370158309226, 0.0013347951044236075,  0.0013769423932765172,  0.0013347951044236082, 0.0012159370158309221,  0.0010408906613671952,  0.0008373298352758808, 0.0006329730941028524,  0.00044964619014816804, 0.00030016031440443776, 0.0001882920390421264,  0.00011099580726472018, 0.00006148605967827151},
  {0.00003624526487808452, 0.00006543064323388921, 0.00011099580726472022, 0.0001769407595543586,  0.00026506081782796834, 0.000373129739919993, 0.0004935954886148328,  0.0006135920553014557,  0.0007167796968039246, 0.0007868450567649748,  0.0008116903575756022,  0.0007868450567649751, 0.0007167796968039244,  0.0006135920553014558,  0.0004935954886148328, 0.000373129739919993,   0.0002650608178279684,  0.0001769407595543586, 0.00011099580726472018, 0.0000654306432338892,  0.000036245264878084536},
  {0.000020078042353740445, 0.00003624526487808454,  0.00006148605967827152, 0.00009801622574383471,  0.00014683027823269914,  0.00020669499165623686, 0.00027342691960891436,  0.0003398989444745464,   0.00039705967549521717, 0.0004358723388751069,   0.00044963537809264686,  0.0004358723388751071, 0.00039705967549521706,  0.00033989894447454647,  0.00027342691960891436, 0.00020669499165623686,  0.0001468302782326992,   0.00009801622574383471, 0.00006148605967827151,  0.000036245264878084536, 0.000020078042353740452},
};

__global__ void blur(const FloatPixel *pixels_in, FloatPixel *pixels_out,
                     size_t width, size_t height) {
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
        float filter_value = filter[dy + FILTER_SIZE / 2][dx + FILTER_SIZE / 2];
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
  /* ANCHOR: call */
  blur<<<grid, block, SHARED_SIZE*SHARED_SIZE*sizeof(FloatPixel)>>>(device_pixels_in, device_pixels_out, im.width,
                        im.height);
  /* ANCHOR_END: call */
  cudaDeviceSynchronize();
  cout << "Elapsed time: " << time.getTime() << endl;

  cudaMemcpy(im.data, device_pixels_out, pixel_data_size,
             cudaMemcpyDeviceToHost);

  bmp.set_pixel_data(im);
  bmp.save("dynamic_shared_memory.bmp");
}