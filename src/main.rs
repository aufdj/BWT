use std::io::{Read, Write, BufReader, BufWriter, BufRead};
use std::fs::{File, remove_file, metadata};
use std::cmp::{Ordering, min};
use std::path::Path;
use std::time::Instant;
use std::env;

const BLOCK_SIZE: usize = 1_048_576;


#[derive(PartialEq, Eq)]
pub enum BufferState {
    NotEmpty,
    Empty,
}
impl BufferState {
    fn is_eof(&self) -> bool {
        *self == BufferState::Empty
    }
}

pub trait BufferedRead {
    fn read_byte(&mut self) -> u8;
    fn read_u64(&mut self) -> u64;
    fn fill_buffer(&mut self) -> BufferState;
    fn read_byte_checked(&mut self) -> Option<u8>;
}
impl BufferedRead for BufReader<File> {
    fn read_byte(&mut self) -> u8 {
        let mut byte = [0u8; 1];

        if self.read(&mut byte).is_ok() {
            if self.buffer().is_empty() {
                self.consume(self.capacity());
                self.fill_buf().unwrap();
            }
        }
        else {
            println!("Function read_byte failed.");
        }
        u8::from_le_bytes(byte)
    }

    fn read_byte_checked(&mut self) -> Option<u8> {
        let mut byte = [0u8; 1];

        let bytes_read = self.read(&mut byte).unwrap();
        if self.buffer().len() <= 0 { 
            self.consume(self.capacity()); 
            self.fill_buf().unwrap();
        }
        if bytes_read == 0 {
            return None;
        }
        Some(u8::from_le_bytes(byte))
    }
    
    fn read_u64(&mut self) -> u64 {
        let mut bytes = [0u8; 8];

        if let Ok(len) = self.read(&mut bytes) {
            if self.buffer().is_empty() {
                self.consume(self.capacity());
                self.fill_buf().unwrap();
                if len < 8 {
                    self.read_exact(&mut bytes[len..]).unwrap();
                }
            }
        }
        else {
            println!("Function read_u64 failed.");
        }
        u64::from_le_bytes(bytes)
    }

    fn fill_buffer(&mut self) -> BufferState {
        self.consume(self.capacity());
        self.fill_buf().unwrap();
        if self.buffer().is_empty() {
            return BufferState::Empty;
        }
        BufferState::NotEmpty
    }
}

pub trait BufferedWrite {
    fn write_byte(&mut self, output: u8);
    fn write_u64(&mut self, output: u64);
    fn flush_buffer(&mut self);
}
impl BufferedWrite for BufWriter<File> {
    fn write_byte(&mut self, output: u8) {
        self.write(&[output]).unwrap();
        
        if self.buffer().len() >= self.capacity() {
            self.flush().unwrap();
        }
    }

    fn write_u64(&mut self, output: u64) {
        self.write(&output.to_le_bytes()[..]).unwrap();
        
        if self.buffer().len() >= self.capacity() {
            self.flush().unwrap();
        }
    }

    fn flush_buffer(&mut self) {
        self.flush().unwrap(); 
    }
}


fn new_input_file(capacity: usize, file_name: &str) -> BufReader<File> {
    BufReader::with_capacity(
        capacity, File::open(file_name).unwrap()
    )
}

fn new_output_file(capacity: usize, file_name: &str) -> BufWriter<File> {
    BufWriter::with_capacity(
        capacity, File::create(file_name).unwrap()
    )
}



struct Bwt {
    file_in: BufReader<File>,
    file_out: BufWriter<File>,
}
impl Bwt {
    fn new(file_in: BufReader<File>, file_out: BufWriter<File>) -> Self {
        Self {
            file_in,
            file_out,
        }
    }

    fn transform(&mut self) {
        loop {
            if self.file_in.fill_buffer().is_eof() { break; }
            let len = self.file_in.buffer().len();

            let mut indices = (0..len as u32).collect::<Vec<u32>>();

            indices.sort_by(|a, b| { 
                block_cmp(*a as usize, *b as usize, self.file_in.buffer())
            });

            let primary_index = indices.iter().position(|&i| i == 1).unwrap();

            let bwt = (0..len).zip(indices.iter()).map(|(_, &idx)| {
                if idx == 0 { 
                    self.file_in.buffer()[len - 1]
                } 
                else {
                    self.file_in.buffer()[(idx as usize) - 1]
                }   
            })
            .collect::<Vec<u8>>();
        
            self.file_out.write_u64(primary_index as u64);
            self.file_out.write_all(&bwt).unwrap();
        }  
        self.file_out.flush_buffer(); 
    }

    fn inverse_transform(&mut self) {
        let mut transform = vec![0u32; BLOCK_SIZE];

        loop {
            if self.file_in.fill_buffer().is_eof() { break; }
        
            let mut index = self.file_in.read_u64() as usize;

            let mut count = [0u32; 256];
            let mut cumul = [0u32; 256];

            for byte in self.file_in.buffer().iter() {
                count[*byte as usize] += 1;    
            }

            let mut sum = 0;
            for i in 0..256 {
                cumul[i] = sum;
                sum += count[i];
                count[i] = 0;
            }

            for (i, byte) in self.file_in.buffer().iter().enumerate() {
                let byte = *byte as usize;
                transform[(count[byte] + cumul[byte]) as usize] = i as u32;
                count[byte] += 1;
            }

            for _ in 0..self.file_in.buffer().len() { 
                self.file_out.write_byte(self.file_in.buffer()[index]);
                index = transform[index] as usize;
            }
        } 
        self.file_out.flush().unwrap();
    }
}

fn block_cmp(a: usize, b: usize, block: &[u8]) -> Ordering {
    let min = min(block[a..].len(), block[b..].len());

    // Lexicographical comparison
    let result = block[a..a + min].cmp(&block[b..b + min]);
    
    // Wraparound if needed
    if result == Ordering::Equal {
        let remainder_a = [&block[a + min..], &block[0..a]].concat();
        let remainder_b = [&block[b + min..], &block[0..b]].concat();
        return remainder_a.cmp(&remainder_b);
    }
    result   
}


const STATE_TABLE: [[u8; 2]; 256] = [
[  1,  2],[  3,  5],[  4,  6],[  7, 10],[  8, 12],[  9, 13],[ 11, 14], // 0
[ 15, 19],[ 16, 23],[ 17, 24],[ 18, 25],[ 20, 27],[ 21, 28],[ 22, 29], // 7
[ 26, 30],[ 31, 33],[ 32, 35],[ 32, 35],[ 32, 35],[ 32, 35],[ 34, 37], // 14
[ 34, 37],[ 34, 37],[ 34, 37],[ 34, 37],[ 34, 37],[ 36, 39],[ 36, 39], // 21
[ 36, 39],[ 36, 39],[ 38, 40],[ 41, 43],[ 42, 45],[ 42, 45],[ 44, 47], // 28
[ 44, 47],[ 46, 49],[ 46, 49],[ 48, 51],[ 48, 51],[ 50, 52],[ 53, 43], // 35
[ 54, 57],[ 54, 57],[ 56, 59],[ 56, 59],[ 58, 61],[ 58, 61],[ 60, 63], // 42
[ 60, 63],[ 62, 65],[ 62, 65],[ 50, 66],[ 67, 55],[ 68, 57],[ 68, 57], // 49
[ 70, 73],[ 70, 73],[ 72, 75],[ 72, 75],[ 74, 77],[ 74, 77],[ 76, 79], // 56
[ 76, 79],[ 62, 81],[ 62, 81],[ 64, 82],[ 83, 69],[ 84, 71],[ 84, 71], // 63
[ 86, 73],[ 86, 73],[ 44, 59],[ 44, 59],[ 58, 61],[ 58, 61],[ 60, 49], // 70
[ 60, 49],[ 76, 89],[ 76, 89],[ 78, 91],[ 78, 91],[ 80, 92],[ 93, 69], // 77
[ 94, 87],[ 94, 87],[ 96, 45],[ 96, 45],[ 48, 99],[ 48, 99],[ 88,101], // 84
[ 88,101],[ 80,102],[103, 69],[104, 87],[104, 87],[106, 57],[106, 57], // 91
[ 62,109],[ 62,109],[ 88,111],[ 88,111],[ 80,112],[113, 85],[114, 87], // 98
[114, 87],[116, 57],[116, 57],[ 62,119],[ 62,119],[ 88,121],[ 88,121], // 105
[ 90,122],[123, 85],[124, 97],[124, 97],[126, 57],[126, 57],[ 62,129], // 112
[ 62,129],[ 98,131],[ 98,131],[ 90,132],[133, 85],[134, 97],[134, 97], // 119
[136, 57],[136, 57],[ 62,139],[ 62,139],[ 98,141],[ 98,141],[ 90,142], // 126
[143, 95],[144, 97],[144, 97],[ 68, 57],[ 68, 57],[ 62, 81],[ 62, 81], // 133
[ 98,147],[ 98,147],[100,148],[149, 95],[150,107],[150,107],[108,151], // 140
[108,151],[100,152],[153, 95],[154,107],[108,155],[100,156],[157, 95], // 147
[158,107],[108,159],[100,160],[161,105],[162,107],[108,163],[110,164], // 154
[165,105],[166,117],[118,167],[110,168],[169,105],[170,117],[118,171], // 161
[110,172],[173,105],[174,117],[118,175],[110,176],[177,105],[178,117], // 168
[118,179],[110,180],[181,115],[182,117],[118,183],[120,184],[185,115], // 175
[186,127],[128,187],[120,188],[189,115],[190,127],[128,191],[120,192], // 182
[193,115],[194,127],[128,195],[120,196],[197,115],[198,127],[128,199], // 189
[120,200],[201,115],[202,127],[128,203],[120,204],[205,115],[206,127], // 196
[128,207],[120,208],[209,125],[210,127],[128,211],[130,212],[213,125], // 203
[214,137],[138,215],[130,216],[217,125],[218,137],[138,219],[130,220], // 210
[221,125],[222,137],[138,223],[130,224],[225,125],[226,137],[138,227], // 217
[130,228],[229,125],[230,137],[138,231],[130,232],[233,125],[234,137], // 224
[138,235],[130,236],[237,125],[238,137],[138,239],[130,240],[241,125], // 231
[242,137],[138,243],[130,244],[245,135],[246,137],[138,247],[140,248], // 238
[249,135],[250, 69],[ 80,251],[140,252],[249,135],[250, 69],[ 80,251], // 245
[140,252],[  0,  0],[  0,  0],[  0,  0]];  // 252


fn next_state(state: u8, bit: i32) -> u8 {
    STATE_TABLE[state as usize][bit as usize]
}

#[allow(overflowing_literals)]
const PR_MSK: i32 = 0xFFFFFC00; // High 22 bit mask
const LIMIT: usize = 127; // Controls rate of adaptation (higher = slower) (0..512)

struct StateMap {
    cxt:     usize,     
    cxt_map: Vec<u32>, // Maps a context to a prediction and a count 
    rec_t:   Vec<u16>, // Reciprocal table: controls adjustment to cxt_map
}
impl StateMap {
    fn new(n: usize) -> StateMap {
        StateMap { 
            cxt:      0,
            cxt_map:  vec![1 << 31; n],
            rec_t:    (0..512).map(|i| 16384/(i+i+3)).collect(),
        }
    }

    fn p(&mut self, cxt: usize) -> i32 {                   
        self.cxt = cxt;
        (self.cxt_map[self.cxt] >> 16) as i32  
    }

    fn update(&mut self, bit: i32) {
        assert!(bit == 0 || bit == 1);  
        let count = (self.cxt_map[self.cxt] & 1023) as usize; // Low 10 bits
        let pr    = (self.cxt_map[self.cxt] >> 10 ) as i32;   // High 22 bits

        if count < LIMIT { self.cxt_map[self.cxt] += 1; }

        // Update cxt_map based on prediction error
        let pr_err = ((bit << 22) - pr) >> 3; // Prediction error
        let rec_v = self.rec_t[count] as i32; // Reciprocal value
        self.cxt_map[self.cxt] = 
        self.cxt_map[self.cxt].wrapping_add((pr_err * rec_v & PR_MSK) as u32); 
    }
}

struct Predictor {
    cxt:   usize,
    sm:    StateMap,
    state: [u8; 256],
}
impl Predictor {
    fn new() -> Predictor {
        Predictor {
            cxt:    0,
            sm:     StateMap::new(65536),
            state:  [0; 256],
        }
    }

    fn p(&mut self) -> i32 { 
        self.sm.p(self.cxt * 256 + self.state[self.cxt] as usize) 
    } 

    fn update(&mut self, bit: i32) {
        self.sm.update(bit);

        self.state[self.cxt] = next_state(self.state[self.cxt], bit);

        self.cxt += self.cxt + bit as usize;
        if self.cxt >= 256 { self.cxt = 0; }
    }
}


struct Encoder {
    high:      u32,
    low:       u32,
    predictor: Predictor,
    archive:   BufWriter<File>,
}
impl Encoder {
    fn new(archive: BufWriter<File>) -> Self {
        Self {
            high: 0xFFFFFFFF, 
            low: 0, 
            predictor: Predictor::new(), 
            archive
        }
    }

    fn compress_bit(&mut self, bit: i32) {
        let mut p = self.predictor.p() as u32;
        if p < 2048 { p += 1; }

        let range = self.high - self.low;
        let mid: u32 = self.low + (range >> 16) * p
                       + ((range & 0x0FFF) * p >> 16);
                       
        if bit == 1 {
            self.high = mid;
        } 
        else {
            self.low = mid + 1;
        }
        self.predictor.update(bit);
        
        while ( (self.high ^ self.low) & 0xFF000000) == 0 {
            self.archive.write_byte((self.high >> 24) as u8);
            self.high = (self.high << 8) + 255;
            self.low <<= 8;  
        }
    }

    fn flush(&mut self) {
        while ( (self.high ^ self.low) & 0xFF000000) == 0 {
            self.archive.write_byte((self.high >> 24) as u8);
            self.high = (self.high << 8) + 255;
            self.low <<= 8; 
        }
        self.archive.write_byte((self.high >> 24) as u8);
        self.archive.flush_buffer();
    }
}

struct Decoder {
    high:      u32,
    low:       u32,
    predictor: Predictor,
    archive:   BufReader<File>,
    x:         u32, 
}
impl Decoder {
    fn new(archive: BufReader<File>) -> Self {
        let mut dec = Self {
            high: 0xFFFFFFFF, 
            low: 0, 
            x: 0, 
            predictor: Predictor::new(), 
            archive,
        };
        for _ in 0..4 {
            dec.x = (dec.x << 8) + dec.archive.read_byte() as u32;
        }
        dec
    }
    
    fn decompress_bit(&mut self) -> i32 {
        let mut p = self.predictor.p() as u32;
        if p < 2048 { p += 1; }

        let range = self.high - self.low;
        let mid: u32 = self.low + (range >> 16) * p
                       + ((range & 0x0FFF) * p >> 16);

        let mut bit: i32 = 0;
        if self.x <= mid {
            bit = 1;
            self.high = mid;
        } 
        else {
            self.low = mid + 1;
        }
        self.predictor.update(bit);
        
        while ( (self.high ^ self.low) & 0xFF000000) == 0 {
            self.high = (self.high << 8) + 255;
            self.low <<= 8; 
            self.x = (self.x << 8) + self.archive.read_byte() as u32; 
        }
        bit
    }
}


fn main() {
    let start_time = Instant::now();
    let args: Vec<String> = env::args().collect();
    let temp = "temp";

    match (&args[1]).as_str() {
        "c" => {
            let file_in = new_input_file(BLOCK_SIZE, &args[2]);
            let bwt_out = new_output_file(4096, temp);
            Bwt::new(file_in, bwt_out).transform();

            let mut bwt_in = new_input_file(4096, temp);
            let file_out = new_output_file(4096, &args[3]);
            let mut enc = Encoder::new(file_out);
            bwt_in.fill_buf().unwrap();
    
            while let Some(byte) = bwt_in.read_byte_checked() {
                enc.compress_bit(1);
                for i in (0..=7).rev() {
                    enc.compress_bit(((byte >> i) & 1).into());
                } 
            }   
            enc.compress_bit(0);
            enc.flush(); 
            
            remove_file(temp).unwrap();

            println!("Finished Compressing.");     
        }
        "d" => {
            let file_in = new_input_file(4096, &args[2]);
            let mut bwt_out = new_output_file(4096, temp);
            let mut dec = Decoder::new(file_in);
    
            while dec.decompress_bit() != 0 {   
                let mut decoded_byte: i32 = 1;
                while decoded_byte < 256 {
                    decoded_byte += decoded_byte + dec.decompress_bit();
                }
                decoded_byte -= 256;
                bwt_out.write_byte(decoded_byte as u8);
            }
            bwt_out.flush().unwrap();
            drop(bwt_out);

            let bwt_in = new_input_file(BLOCK_SIZE + 8, temp);
            let file_out = new_output_file(4096, &args[3]);
            Bwt::new(bwt_in, file_out).inverse_transform();
            remove_file(temp).unwrap();

            println!("Finished Decompressing.");   
        }
        _ => { 
            println!("To Compress: c input output");
            println!("To Decompress: d input output");
        }
    } 
    println!("{} bytes -> {} bytes in {:.2?}", 
        metadata(Path::new(&args[2])).unwrap().len(), 
        metadata(Path::new(&args[3])).unwrap().len(), 
        start_time.elapsed()
    );   
}
