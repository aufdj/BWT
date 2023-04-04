use std::io::{Read, Write, BufReader, BufWriter, BufRead};
use std::fs::File;
use std::env;
use std::time::Instant;
use std::cmp::Ordering;
use std::cmp::min;

// Convenience functions for buffered IO ---------------------------
fn write(buf_out: &mut BufWriter<File>, output: &[u8]) {
    buf_out.write(output).unwrap();
    if buf_out.buffer().len() >= buf_out.capacity() { 
        buf_out.flush().unwrap(); 
    }
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
        for i in 0..bwt.len() {
            write(file_out, &bwt[i].to_le_bytes()[..]);
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
        for i in 0..file_in.buffer().len() {
            counts[file_in.buffer()[i] as usize] += 1;    
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

fn main() {
    let start = Instant::now();
    let args: Vec<String> = env::args().collect();
    let mut file_out = BufWriter::with_capacity(4096, File::create(&args[3]).unwrap());

    let block_size: usize = 1_048_576;

    match (&args[1]).as_str() {
        "c" => {
            let mut file_in = BufReader::with_capacity(block_size, File::open(&args[2]).unwrap());
            bwt_transform(&mut file_in, &mut file_out, block_size);    
        }
        "d" => {
            // Extra 8 bytes for primary index
            let mut file_in = BufReader::with_capacity(block_size + 8, File::open(&args[2]).unwrap());
            inverse_bwt_transform(&mut file_in, &mut file_out, block_size);
        }
        _ => { 
            println!("Transform: c input output"); 
            println!("Inverse Transform: d input output"); 
        }
    }
    println!("{:.2?}", start.elapsed());
}
