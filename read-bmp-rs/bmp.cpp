#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace std;

/* Utility functions */
vector<uint8_t> read_file(string path) {
  ifstream instream(path, ios::in | ios::binary);
  if (!instream.is_open()) {
    cerr << "Could not open the file - '" << path << "'" << endl;
    exit(EXIT_FAILURE);
  }
  vector<uint8_t> data((istreambuf_iterator<char>(instream)),
                       istreambuf_iterator<char>());
  return data;
}

void write_file(string path, vector<uint8_t> &data) {
  ofstream outstream(path, ios::out | ios::binary);
  if (!outstream.is_open()) {
    cerr << "Could not open the file - '" << path << "'" << endl;
    exit(EXIT_FAILURE);
  }
  outstream.write((const char *)&data[0], data.size());
}

uint16_t read_u16_little_endian(uint8_t values[2]) {
  return ((uint16_t)values[0]) | (((uint16_t)values[1]) << 8);
}

uint32_t read_u32_little_endian(uint8_t values[4]) {
  return ((uint32_t)values[0]) | (((uint32_t)values[1]) << 8) |
         (((uint32_t)values[2]) << 16) | (((uint32_t)values[3]) << 24);
}

int32_t read_i32_little_endian(uint8_t values[4]) {
  return ((int32_t)values[0]) | (((int32_t)values[1]) << 8) |
         (((int32_t)values[2]) << 16) | (((int32_t)values[3]) << 24);
}

uint32_t checked_add(uint32_t a, uint32_t b) {
  uint32_t result = a + b;
  assert(result >= a && result >= b);
  return result;
}

uint32_t checked_sub(uint32_t a, uint32_t b) {
  assert(a >= b);
  return a - b;
}

uint32_t checked_mul(uint32_t a, uint32_t b) {
  uint32_t result = a * b;
  assert(a == 0 || result / a == b);
  return result;
}

uint32_t round_up(uint32_t value, uint32_t divisor) {
  return checked_add(value, divisor - 1) / divisor * divisor;
}

/* BMP file types */
struct BmpFileHeader {
  uint8_t bfType[2];
  uint8_t _bfSize[4];
  uint8_t _bfReserved1[2];
  uint8_t _bfReserved2[2];
  uint8_t _bfOffBits[4];

  uint32_t bfSize() { return read_u32_little_endian(_bfSize); }
  uint16_t bfReserved1() { return read_u16_little_endian(_bfReserved1); }
  uint16_t bfReserved2() { return read_u16_little_endian(_bfReserved2); }
  uint32_t bfOffBits() { return read_u32_little_endian(_bfOffBits); }

  bool is_valid() { return bfType[0] == 'B' && bfType[1] == 'M'; }
};

struct BmpInfoHeader {
  uint8_t _biSize[4];
  uint8_t _biWidth[4];
  uint8_t _biHeight[4];
  uint8_t _biPlanes[2];
  uint8_t _biBitCount[2];
  uint8_t _biCompression[4];
  uint8_t _biSizeImage[4];
  uint8_t _biXPelsPerMeter[4];
  uint8_t _biYPelsPerMeter[4];
  uint8_t _biClrUsed[4];
  uint8_t _biClrImportant[4];

  uint32_t biSize() { return read_u32_little_endian(_biSize); }
  int32_t biWidth() { return read_i32_little_endian(_biWidth); }
  int32_t biHeight() { return read_i32_little_endian(_biHeight); }
  uint16_t biPlanes() { return read_u16_little_endian(_biPlanes); }
  uint16_t biBitCount() { return read_u16_little_endian(_biBitCount); }
  uint32_t biCompression() { return read_u32_little_endian(_biCompression); }
  uint32_t biSizeImage() { return read_u32_little_endian(_biSizeImage); }
  int32_t biXPelsPerMeter() { return read_i32_little_endian(_biXPelsPerMeter); }
  int32_t biYPelsPerMeter() { return read_i32_little_endian(_biYPelsPerMeter); }
  uint32_t biClrUsed() { return read_u32_little_endian(_biClrUsed); }
  uint32_t biClrImportant() { return read_u32_little_endian(_biClrImportant); }

  uint32_t bytes_per_row() { return round_up(checked_mul(biWidth(), 3), 4); }
  uint32_t total_bytes() { return checked_mul(bytes_per_row(), biHeight()); }

  bool is_valid() {
    return biSize() == sizeof(BmpInfoHeader) && biPlanes() == 1 &&
           biBitCount() == 24 && biCompression() == 0 && biClrUsed() == 0 &&
           biClrImportant() == 0 && biWidth() > 0 && biHeight() > 0 &&
           (biSizeImage() == 0 || total_bytes() == biSizeImage());
  }
};

struct BmpHeader {
  BmpFileHeader file_header;
  BmpInfoHeader info_header;

  bool is_valid() { return file_header.is_valid() && info_header.is_valid(); }
};

struct BmpPixel {
  uint8_t blue;
  uint8_t green;
  uint8_t red;
};

struct BmpPixelData {
  uint8_t *data;
  uint32_t width;
  uint32_t height;
  uint32_t bytes_per_row;

  BmpPixel *get_row(uint32_t index) {
    static_assert(sizeof(BmpPixel) == 3);
    static_assert(alignof(BmpPixel) == 1);
    return (BmpPixel *)&data[checked_mul(bytes_per_row,
                                         checked_sub(height, index + 1))];
  }
};

struct FloatPixel {
  float red;
  float green;
  float blue;
};

struct FloatImage {
  uint32_t width;
  uint32_t height;
  FloatPixel *data;
};

class Bmp {
private:
  vector<uint8_t> buf;

  BmpHeader *header() {
    static_assert(sizeof(BmpFileHeader) == 14);
    static_assert(sizeof(BmpInfoHeader) == 40);
    static_assert(sizeof(BmpHeader) == 14 + 40);
    static_assert(alignof(BmpFileHeader) == 1);
    static_assert(alignof(BmpInfoHeader) == 1);
    static_assert(alignof(BmpHeader) == 1);

    assert(buf.size() >= sizeof(BmpHeader));

    return (BmpHeader *)&buf[0];
  }

  BmpPixelData get_raw_pixels() {
    BmpHeader *h = header();
    assert(buf.size() >= checked_add(h->file_header.bfOffBits(),
                                     h->info_header.total_bytes()));
    BmpPixelData out;
    out.data = &buf[h->file_header.bfOffBits()];
    out.width = (uint32_t)h->info_header.biWidth();
    out.height = (uint32_t)h->info_header.biHeight();
    out.bytes_per_row = h->info_header.bytes_per_row();
    return out;
  }

public:
  Bmp() = delete;
  Bmp(const Bmp &) = delete;
  Bmp &operator=(const Bmp &) = delete;

  Bmp(string path) : buf(read_file(path)) { assert(header()->is_valid()); }
  void save(string path) { write_file(path, buf); }
  FloatImage get_pixel_data() {
    BmpPixelData pixel_data = get_raw_pixels();
    uint32_t width = pixel_data.width;
    uint32_t height = pixel_data.height;

    FloatImage out;
    out.width = pixel_data.width;
    out.height = pixel_data.height;
    out.data = new FloatPixel[checked_mul(width, height)];

    for (uint64_t row = 0; row < height; row++) {
      for (uint64_t col = 0; col < width; col++) {
        auto pixel = &pixel_data.get_row(row)[col];
        out.data[row * width + col].red = pixel->red / 255.0;
        out.data[row * width + col].green = pixel->green / 255.0;
        out.data[row * width + col].blue = pixel->blue / 255.0;
      }
    }
    return out;
  }

  void set_pixel_data(FloatImage &im) {
    BmpPixelData pixel_data = get_raw_pixels();
    uint32_t width = pixel_data.width;
    uint32_t height = pixel_data.height;

    assert(im.width == width);
    assert(im.height == height);

    for (uint64_t row = 0; row < height; row++) {
      for (uint64_t col = 0; col < width; col++) {
        auto pixel = &pixel_data.get_row(row)[col];
        pixel->red =
            round(clamp(im.data[row * width + col].red, 0.0f, 1.0f) * 255.0);
        pixel->green =
            round(clamp(im.data[row * width + col].green, 0.0f, 1.0f) * 255.0);
        pixel->blue =
            round(clamp(im.data[row * width + col].blue, 0.0f, 1.0f) * 255.0);
      }
    }
  }
};

int main() {
  Bmp bmp("gottagofast.bmp");
  FloatImage im = bmp.get_pixel_data();
  for (uint64_t i = 0; i < im.height * im.width; i++) {
    swap(im.data[i].red, im.data[i].green);
  }
  bmp.set_pixel_data(im);
  bmp.save("converted.bmp");
}