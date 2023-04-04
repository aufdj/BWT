use std::io::{Read, Write, BufReader, BufWriter, BufRead};
use std::fs::{File, remove_file, metadata};
use std::cmp::{Ordering, min};
use std::path::Path;
use std::time::Instant;
use std::env;

// Convenience functions for buffered IO ---------------------------
fn write(buf_out: &mut BufWriter<File>, output: &[u8]) {
    buf_out.write(output).unwrap();
    if buf_out.buffer().len() >= buf_out.capacity() { 
        buf_out.flush().unwrap(); 
    }
}
fn read(buf_in: &mut BufReader<File>, input: &mut [u8; 1]) -> usize {
    let bytes_read = buf_in.read(input).unwrap();
    if buf_in.buffer().len() <= 0 { 
        buf_in.consume(buf_in.capacity()); 
        buf_in.fill_buf().unwrap();
    }
    bytes_read
}
fn read8(buf_in: &mut BufReader<File>, input: &mut [u8; 8]) -> usize {
    let bytes_read = buf_in.read(input).unwrap();
    if buf_in.buffer().len() <= 0 { 
        buf_in.consume(buf_in.capacity()); 
        buf_in.fill_buf().unwrap();
    }
    bytes_read
}
// -----------------------------------------------------------------


// BWT Transform ---------------------------------------------------
fn block_compare(a: usize, b: usize, block: &[u8]) -> Ordering {
    let min = min(block[a..].len(), block[b..].len());

    // Lexicographical comparison
    let result = block[a..a + min].cmp(
                &block[b..b + min]    );
    
    // Implement wraparound if needed
    if result == Ordering::Equal {
        return [&block[a + min..], &block[0..a]].concat().cmp(
              &[&block[b + min..], &block[0..b]].concat()    );
    }
    result   
}
fn bwt_transform(file_in: &mut BufReader<File>, file_out: &mut BufWriter<File>, block_size: usize) { 
    let mut primary_index: usize = 0; // starting position for inverse transform
    let mut indexes: Vec<u32> = Vec::with_capacity(block_size); // indexes into block
    let mut bwt: Vec<u8> = Vec::with_capacity(block_size); // BWT output
    
    loop {
        file_in.consume(file_in.capacity());
        file_in.fill_buf().unwrap();
        if file_in.buffer().is_empty() { break; }
        
        // Create indexes into block
        indexes.resize(file_in.buffer().len(), 0);
        for i in 0..indexes.len() { indexes[i as usize] = i as u32; }
        
        // Sort indexes
        indexes[..].sort_by(|a, b| block_compare(*a as usize, *b as usize, file_in.buffer()));
        
        // Get primary index and BWT output
        bwt.resize(file_in.buffer().len(), 0);
        for i in 0..bwt.len() {
            if indexes[i] == 1 {
                primary_index = i;
            }
            if indexes[i] == 0 { 
                bwt[i] = file_in.buffer()[file_in.buffer().len() - 1];
            } else {
                bwt[i] = file_in.buffer()[(indexes[i] as usize) - 1];
            }    
        } 
    
        write(file_out, &primary_index.to_le_bytes()[..]);
        for byte in bwt.iter() {
            write(file_out, &byte.to_le_bytes()[..]);
        }
    }  
    file_out.flush().unwrap();  
}
fn inverse_bwt_transform(file_in: &mut BufReader<File>, file_out: &mut BufWriter<File>, block_size: usize) {
    let mut transform_vector: Vec<u32> = Vec::with_capacity(block_size); // For inverting transform

    loop {
        file_in.consume(file_in.capacity());
        file_in.fill_buf().unwrap();
        if file_in.buffer().is_empty() { break; }
    
        // Read primary index
        let mut primary_index = [0u8; 8];
        read8(file_in, &mut primary_index);
        let primary_index = usize::from_le_bytes(primary_index);

        let mut counts = [0u32; 256];
        let mut cumul_counts = [0u32; 256];

        // Get number of occurences for each byte
        for byte in file_in.buffer().iter() {
            counts[*byte as usize] += 1;    
        }

        // Get cumulative counts for each byte
        let mut sum = 0;
        for i in 0..256 {
            cumul_counts[i] = sum;
            sum += counts[i];
            counts[i] = 0;
        }

        // Build transformation vector
        transform_vector.resize(file_in.buffer().len(), 0);
        for i in 0..file_in.buffer().len() {
            let index = file_in.buffer()[i] as usize; 
            transform_vector[(counts[index] + cumul_counts[index]) as usize] = i as u32;
            counts[index] += 1;
        }

        // Invert transform and output original data
        let mut index = primary_index;
        for _ in 0..file_in.buffer().len() { 
            write(file_out, &file_in.buffer()[index].to_le_bytes()[..]);
            index = transform_vector[index] as usize;
        }
    } 
    file_out.flush().unwrap();   
}
// -----------------------------------------------------------------


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

const N: usize = 65_536;  // number of contexts
const LIMIT: usize = 127; // controls rate of adaptation (higher = slower) (0..512)


// StateMap --------------------------------------------------------
struct StateMap {
    context:        usize,         // context of last prediction
    context_map:    Box<[u32; N]>, // maps a context to a prediction and a count (allocate on heap to avoid stack overflow)
    recipr_table:  [i32; 512],     // controls the size of each adjustment to context_map
}
impl StateMap {
    fn new() -> StateMap {
        let mut statemap = StateMap { 
            context:        0,
            context_map:    Box::new([0; N]),
            recipr_table:  [0; 512],
        };

        for i in 0..N { 
            statemap.context_map[i] = 1 << 31; 
        }
        if statemap.recipr_table[0] == 0 {
            for i in 0..512 { 
                statemap.recipr_table[i] = (32_768 / (i + i + 5)) as i32; 
            }
        }
        statemap
    }
    fn p(&mut self, cx: usize) -> u32 {
        self.context = cx;
        self.context_map[self.context] >> 16
    }
    fn update(&mut self, bit: i32) {
        let count: usize = (self.context_map[self.context] & 511) as usize;  // low 9 bits
        let prediction: i32 = (self.context_map[self.context] >> 14) as i32; // high 18 bits

        if count < LIMIT { self.context_map[self.context] += 1; }

        // updates context_map based on the difference between the predicted and actual bit
        #[allow(overflowing_literals)]
        let high_23_bit_mask: i32 = 0xfffffe00;
        self.context_map[self.context] = self.context_map[self.context].wrapping_add(
        (((bit << 18) - prediction) * self.recipr_table[count] & high_23_bit_mask) as u32);
        
    }
}
// -----------------------------------------------------------------


// Predictor -------------------------------------------------------
struct Predictor {
    context:   usize,
    statemap:  StateMap,
    state:     [u8; 256],
}
impl Predictor {
    fn new() -> Predictor {
        Predictor {
            context:   0,
            statemap:  StateMap::new(),
            state:     [0; 256],
        }
    }
    fn p(&mut self) -> u32 { 
        self.statemap.p(self.context * 256 + self.state[self.context] as usize) 
    } 
    fn update(&mut self, bit: i32) {
        self.statemap.update(bit);

        self.state[self.context] = STATE_TABLE[self.state[self.context] as usize][bit as usize];

        self.context += self.context + bit as usize;
        if self.context >= 256 { self.context = 0; }
    }
}
// -----------------------------------------------------------------


// Encoder ---------------------------------------------------------
#[derive(PartialEq, Eq)]
enum Mode {
    Compress,
    Decompress,
}

#[allow(dead_code)]
struct Encoder {
    high:       u32,
    low:        u32,
    predictor:  Predictor,
    file_in:    BufReader<File>,
    file_out:   BufWriter<File>,
    x:          u32,
    mode:       Mode,
}
impl Encoder {
    fn new(predictor: Predictor, file_in: BufReader<File>, file_out: BufWriter<File>, mode: Mode) -> Encoder {
        let mut e = Encoder {
            high: 0xFFFFFFFF, 
            low: 0, 
            x: 0, 
            predictor, file_in, file_out, mode
        };

        // During decompression, initialize x to 
        // first 4 bytes of compressed data
        if e.mode == Mode::Decompress {
            let mut byte = [0; 1];
            for _ in 0..4 {
                read(&mut e.file_in, &mut byte);
                e.x = (e.x << 8) + byte[0] as u32;
            }
        }
        e
    }
    fn compress_bit(&mut self, bit: i32) {
        // Compress bit
        let p: u32 = self.predictor.p();
        let mid: u32 = self.low + ((self.high - self.low) >> 16) * p + ((self.high - self.low & 0xFFFF) * p >> 16);
        if bit == 1 {
            self.high = mid;
        } else {
            self.low = mid + 1;
        }

        // Update model with new bit
        self.predictor.update(bit);

        // Write identical leading MSB to output
        while ( (self.high ^ self.low) & 0xFF000000) == 0 {
            write(&mut self.file_out, &self.high.to_le_bytes()[3..4]);
            self.high = (self.high << 8) + 255;
            self.low <<= 8;  
        }
    }
    fn decompress_bit(&mut self) -> i32 {
        // Decompress bit
        let p: u32 = self.predictor.p();
        let mid: u32 = self.low + ((self.high - self.low) >> 16) * p + ((self.high - self.low & 0xFFFF) * p >> 16);
        let mut bit: i32 = 0;
        if self.x <= mid {
            bit = 1;
            self.high = mid;
        } else {
            self.low = mid + 1;
        }

        // Update model with new bit
        self.predictor.update(bit);
        
        // Write identical leading MSB to output and read new byte 
        let mut byte = [0; 1];
        while ( (self.high ^ self.low) & 0xFF000000) == 0 {
            self.high = (self.high << 8) + 255;
            self.low <<= 8;
            read(&mut self.file_in, &mut byte); 
            self.x = (self.x << 8) + byte[0] as u32; 
        }
        bit
    }
    fn flush(&mut self) {
        while ( (self.high ^ self.low) & 0xFF000000) == 0 {
            write(&mut self.file_out, &self.high.to_le_bytes()[3..4]);
            self.high = (self.high << 8) + 255;
            self.low <<= 8; 
        }
        write(&mut self.file_out, &self.high.to_le_bytes()[3..4]);  
        self.file_out.flush().unwrap();
    }
}
// -----------------------------------------------------------------


fn main() {
    let start_time = Instant::now();
    let args: Vec<String> = env::args().collect();
    
    // Encoder buffers
    let e_file_in  =   BufReader::with_capacity(4096, File::open(&args[2]).unwrap());
    let e_file_out =   BufWriter::with_capacity(4096, File::create(&args[3]).unwrap());

    let block_size: usize = 1_048_576;

    match (&args[1]).as_str() {
        "c" => {  
            let mut e = Encoder::new(Predictor::new(), e_file_in, e_file_out, Mode::Compress);
            // Transform               | Compression
            // file_in -> bwt_file_out | bwt_file_in -> file_out
            
            // BWT ----------------------------------------------
            let mut file_in = BufReader::with_capacity(block_size, File::open(&args[2]).unwrap());
            let mut bwt_file_out = BufWriter::with_capacity(4096, File::create("bwt_temp.bin").unwrap());
            bwt_transform(&mut file_in, &mut bwt_file_out, block_size);
            drop(bwt_file_out);
            // --------------------------------------------------

            // Compression --------------------------------------
            let mut bwt_file_in = BufReader::with_capacity(4096, File::open("bwt_temp.bin").unwrap());
            bwt_file_in.fill_buf().unwrap();
    
            let mut byte = [0; 1];
            while read(&mut bwt_file_in, &mut byte) != 0 {
                e.compress_bit(1);
                for i in (0..=7).rev() {
                    e.compress_bit(((byte[0] >> i) & 1).into());
                } 
            }   
            e.compress_bit(0);
            e.flush(); 
            // --------------------------------------------------
            remove_file("bwt_temp.bin").unwrap();

            let file_in_size  = metadata(Path::new(&args[2])).unwrap().len();
            let file_out_size = metadata(Path::new(&args[3])).unwrap().len();
            println!("Finished Compressing.");   
            println!("{} bytes -> {} bytes in {:.2?}", file_in_size, file_out_size, start_time.elapsed());    
        }
        "d" => {
            let mut e = Encoder::new(Predictor::new(), e_file_in, e_file_out, Mode::Decompress);
            // Decompression           | Inverse Transform
            // file_in -> bwt_file_out | bwt_file_in -> file_out
            
            // Decompression ------------------------------------
            let mut bwt_file_out = BufWriter::with_capacity(4096, File::create("bwt_i_temp.bin").unwrap());
            while e.decompress_bit() != 0 {   
                let mut decoded_byte: i32 = 1;
                while decoded_byte < 256 {
                    decoded_byte += decoded_byte + e.decompress_bit();
                }
                decoded_byte -= 256;
                write(&mut bwt_file_out, &decoded_byte.to_le_bytes()[0..1]);
            }
            bwt_file_out.flush().unwrap();
            drop(bwt_file_out);
            // --------------------------------------------------

            // Inverse BWT --------------------------------------
            let mut bwt_file_in = BufReader::with_capacity(block_size + 8, File::open("bwt_i_temp.bin").unwrap());
            let mut file_out = BufWriter::with_capacity(4096, File::create(&args[3]).unwrap());
            inverse_bwt_transform(&mut bwt_file_in, &mut file_out, block_size);
            // --------------------------------------------------
            remove_file("bwt_i_temp.bin").unwrap();

            let file_in_size  = metadata(Path::new(&args[2])).unwrap().len();
            let file_out_size = metadata(Path::new(&args[3])).unwrap().len();
            println!("Finished Decompressing.");  
            println!("{} bytes -> {} bytes in {:.2?}", file_in_size, file_out_size, start_time.elapsed());   
        }
        _ => { 
            println!("To Compress: c input output");
            println!("To Decompress: d input output");
        }
    } 
}
