#![allow(non_snake_case, dead_code)]

fn round_up(n: u32, m: u32) -> u32 {
    (n + m - 1) / m * m
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct BmpFileHeader {
    bfType: [u8; 2],
    bfSize: [u8; 4],
    bfReserved1: [u8; 2],
    bfReserved2: [u8; 2],
    bfOffBits: [u8; 4],
}

static_assertions::assert_eq_size!(BmpFileHeader, [u8; 14]);
static_assertions::assert_eq_align!(BmpFileHeader, u8);

impl BmpFileHeader {
    fn parse(s: &[u8]) -> Option<&Self> {
        if s.len() >= std::mem::size_of::<Self>() {
            Some(unsafe { &*(s.as_ptr() as *const Self) })
        } else {
            None
        }
    }

    fn is_valid(&self) -> bool {
        &self.bfType() == b"BM"
    }

    fn bfType(&self) -> [u8; 2] {
        self.bfType
    }
    fn bfSize(&self) -> u32 {
        u32::from_le_bytes(self.bfSize)
    }
    fn bfReserved1(&self) -> u16 {
        u16::from_le_bytes(self.bfReserved1)
    }
    fn bfReserved2(&self) -> u16 {
        u16::from_le_bytes(self.bfReserved2)
    }

    fn bfOffBits(&self) -> u32 {
        u32::from_le_bytes(self.bfOffBits)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct BmpInfoHeader {
    biSize: [u8; 4],
    biWidth: [u8; 4],
    biHeight: [u8; 4],
    biPlanes: [u8; 2],
    biBitCount: [u8; 2],
    biCompression: [u8; 4],
    biSizeImage: [u8; 4],
    biXPelsPerMeter: [u8; 4],
    biYPelsPerMeter: [u8; 4],
    biClrUsed: [u8; 4],
    biClrImportant: [u8; 4],
}

static_assertions::assert_eq_size!(BmpInfoHeader, [u8; 40]);
static_assertions::assert_eq_align!(BmpInfoHeader, u8);

impl BmpInfoHeader {
    fn parse(s: &[u8]) -> Option<&Self> {
        if s.len() >= std::mem::size_of::<Self>() {
            Some(unsafe { &*(s.as_ptr() as *const Self) })
        } else {
            None
        }
    }

    fn is_valid(&self) -> bool {
        self.biSize() as usize == std::mem::size_of::<Self>()
            && self.biPlanes() == 1
            && self.biBitCount() == 24
            && self.biCompression() == 0
            && self.biClrUsed() == 0
            && self.biClrImportant() == 0
            && self.biWidth() > 0
            && self.biHeight() > 0
            && (self.biSizeImage() == 0 || self.total_bytes() == self.biSizeImage() as usize)
    }

    fn bytes_per_row(&self) -> usize {
        round_up((self.biWidth() as u32).checked_mul(3).unwrap(), 4) as usize
    }

    fn total_bytes(&self) -> usize {
        self.bytes_per_row()
            .checked_mul(self.biHeight() as usize)
            .unwrap()
    }

    fn biSize(&self) -> u32 {
        u32::from_le_bytes(self.biSize)
    }
    fn biWidth(&self) -> i32 {
        i32::from_le_bytes(self.biWidth)
    }
    fn biHeight(&self) -> i32 {
        i32::from_le_bytes(self.biHeight)
    }
    fn biPlanes(&self) -> u16 {
        u16::from_le_bytes(self.biPlanes)
    }
    fn biBitCount(&self) -> u16 {
        u16::from_le_bytes(self.biBitCount)
    }
    fn biCompression(&self) -> u32 {
        u32::from_le_bytes(self.biCompression)
    }
    fn biSizeImage(&self) -> u32 {
        u32::from_le_bytes(self.biSizeImage)
    }
    fn biXPelsPerMeter(&self) -> i32 {
        i32::from_le_bytes(self.biXPelsPerMeter)
    }
    fn biYPelsPerMeter(&self) -> i32 {
        i32::from_le_bytes(self.biYPelsPerMeter)
    }
    fn biClrUsed(&self) -> u32 {
        u32::from_le_bytes(self.biClrUsed)
    }
    fn biClrImportant(&self) -> u32 {
        u32::from_le_bytes(self.biClrImportant)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct BmpHeader {
    file_header: BmpFileHeader,
    info_header: BmpInfoHeader,
}

static_assertions::assert_eq_size!(BmpHeader, [u8; 14 + 40]);
static_assertions::assert_eq_align!(BmpHeader, u8);

impl BmpHeader {
    fn parse(s: &[u8]) -> Option<&Self> {
        if s.len() >= std::mem::size_of::<Self>() {
            Some(unsafe { &*(s.as_ptr() as *const Self) })
        } else {
            None
        }
    }

    fn is_valid(&self) -> bool {
        self.file_header.is_valid() && self.info_header.is_valid()
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Pixel {
    blue: u8,
    green: u8,
    red: u8,
}

static_assertions::assert_eq_size!(Pixel, [u8; 3]);
static_assertions::assert_eq_align!(Pixel, u8);

impl Pixel {
    fn from_slice(s: &[u8]) -> &[Pixel] {
        unsafe { std::slice::from_raw_parts(s.as_ptr() as *const Pixel, s.len() / 3) }
    }

    fn from_mut_slice(s: &mut [u8]) -> &mut [Pixel] {
        unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut Pixel, s.len() / 3) }
    }
}

#[derive(Debug)]
struct PixelData<'a> {
    data: &'a mut [u8],
    width: usize,
    bytes_per_row: usize,
    height: usize,
}

impl<'a> std::ops::Index<usize> for PixelData<'a> {
    type Output = [Pixel];

    fn index(&self, index: usize) -> &Self::Output {
        Pixel::from_slice(
            &self.data[self.bytes_per_row * (self.height - index - 1)
                ..self.bytes_per_row * (self.height - index - 1) + self.width * 3],
        )
    }
}

impl<'a> std::ops::IndexMut<usize> for PixelData<'a> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        Pixel::from_mut_slice(
            &mut self.data[self.bytes_per_row * (self.height - index - 1)
                ..self.bytes_per_row * (self.height - index - 1) + self.width * 3],
        )
    }
}

fn main() {
    let mut data = std::fs::read("gottagofast.bmp").unwrap();
    {
        let header = *BmpHeader::parse(&data).unwrap();
        println!("{:?}", header);
        assert!(header.is_valid());
        dbg!(header.info_header.biWidth());
        dbg!(header.info_header.bytes_per_row());
        dbg!(header.info_header.total_bytes());
        let data = &mut data[header.file_header.bfOffBits() as usize..];
        let data = &mut data[..header.info_header.total_bytes()];
        let mut data = PixelData {
            data,
            width: header.info_header.biWidth() as usize,
            bytes_per_row: header.info_header.bytes_per_row(),
            height: header.info_header.biHeight() as usize,
        };

        for row in 0..data.height {
            for col in 0..data.width {
                let pixel = &mut data[row][col];
                std::mem::swap(&mut pixel.red, &mut pixel.green);
            }
        }
    }
    std::fs::write("converted.bmp", data).unwrap();
}
