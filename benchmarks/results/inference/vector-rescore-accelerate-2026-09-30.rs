use simsimd::SpatialSimilarity;
use numkong::Angular;
use std::hint::black_box;
use std::time::Instant;

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sdot(n: i32, x: *const f32, incx: i32, y: *const f32, incy: i32) -> f32;
}

fn accelerate_dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    unsafe { cblas_sdot(a.len() as i32, a.as_ptr(), 1, b.as_ptr(), 1) }
}

fn accelerate(query: &[f32], rows: &[Vec<f32>]) -> Vec<f32> {
    let q = accelerate_dot(query, query).sqrt();
    rows.iter().map(|row| {
        let length = q * accelerate_dot(row, row).sqrt();
        if length > 0.0 { accelerate_dot(query, row) / length } else { 0.0 }
    }).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn scalar(query: &[f32], rows: &[Vec<f32>]) -> Vec<f32> {
    let q = dot(query, query).sqrt();
    rows.iter()
        .map(|row| {
            let length = q * dot(row, row).sqrt();
            if length > 0.0 { dot(query, row) / length } else { 0.0 }
        })
        .collect()
}

fn simd(query: &[f32], rows: &[Vec<f32>]) -> Vec<f32> {
    let q = <f32 as SpatialSimilarity>::dot(query, query).unwrap().sqrt();
    rows.iter()
        .map(|row| {
            let length = q * <f32 as SpatialSimilarity>::dot(row, row).unwrap().sqrt();
            if length == 0.0 {
                0.0
            } else {
                (<f32 as SpatialSimilarity>::dot(query, row).unwrap() / length) as f32
            }
        })
        .collect()
}

fn numkong(query: &[f32], rows: &[Vec<f32>]) -> Vec<f32> {
    rows.iter().map(|row| {
        let similarity = 1.0 - <f32 as Angular>::angular(query, row).unwrap();
        if similarity.is_finite() { similarity as f32 } else { 0.0 }
    }).collect()
}

fn run(name: &str, compute: fn(&[f32], &[Vec<f32>]) -> Vec<f32>, query: &[f32], rows: &[Vec<f32>]) -> f64 {
    let started = Instant::now();
    for _ in 0..400 {
        black_box(compute(black_box(query), black_box(rows)));
    }
    let us = started.elapsed().as_secs_f64() * 1e6 / 400.0;
    println!("{name} {us:.3} us per 100x1024");
    us
}

fn main() {
    let query: Vec<f32> = (0..1024).map(|i| ((i * 37 % 101) as f32 - 50.0) / 100.0).collect();
    let mut rows: Vec<Vec<f32>> = (0..100)
        .map(|j| (0..1024).map(|i| query[i] + ((i * 19 + j * 7) % 43) as f32 / 1000.0).collect())
        .collect();
    rows[99].fill(0.0);
    let old = scalar(&query, &rows);
    let oracle: Vec<f64> = rows.iter().map(|row| {
        let accum = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x,y)| f64::from(*x) * f64::from(*y)).sum::<f64>();
        let length = (accum(&query, &query) * accum(row, row)).sqrt();
        if length > 0.0 { accum(&query, row) / length } else { 0.0 }
    }).collect();
    let oracle_error = |got: &[f32]| got.iter().zip(&oracle).map(|(a,b)| (f64::from(*a)-b).abs()).fold(0.0f64, f64::max);
    println!("scalar max error vs FP64 oracle {:.9}", oracle_error(&old));
    let new = simd(&query, &rows);
    let error = old.iter().zip(&new).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
    println!("maximum absolute score difference {error:.9}");
    let numkong_score = numkong(&query, &rows);
    let error = old.iter().zip(&numkong_score).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
    println!("numkong maximum absolute score difference {error:.9}");
    println!("numkong max error vs FP64 oracle {:.9}", oracle_error(&numkong_score));
    let accelerate_score = accelerate(&query, &rows);
    let error = old.iter().zip(&accelerate_score).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
    println!("accelerate maximum absolute score difference {error:.9}");
    println!("accelerate max error vs FP64 oracle {:.9}", oracle_error(&accelerate_score));
    let mut times = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for turn in 0..6 {
        if turn % 2 == 0 {
            times[0].push(run("scalar", scalar, &query, &rows));
            times[1].push(run("simd", simd, &query, &rows));
            times[2].push(run("numkong", numkong, &query, &rows));
            times[3].push(run("accelerate", accelerate, &query, &rows));
        } else {
            times[3].push(run("accelerate", accelerate, &query, &rows));
            times[2].push(run("numkong", numkong, &query, &rows));
            times[1].push(run("simd", simd, &query, &rows));
            times[0].push(run("scalar", scalar, &query, &rows));
        }
    }
    for (name, values) in ["scalar", "simd", "numkong", "accelerate"].into_iter().zip(times.iter_mut()) {
        values.sort_by(f64::total_cmp);
        println!("{name} median {:.3} us", (values[2] + values[3]) / 2.0);
    }

    // Deliberately clustered FP16-rounded vectors from the earlier
    // 100-row CoreML numerical screen.
    let mut state = 1234567u64;
    let mut random = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (((state >> 32) as u32) as f64 / u32::MAX as f64 - 0.5) as f32
    };
    let q: Vec<f32> = (0..1024).map(|_| random()).collect();
    let mut near: Vec<Vec<f32>> = (0..100)
        .map(|_| (0..1024).map(|_| half::f16::from_f32(random()).to_f32()).collect())
        .collect();
    for row in near.iter_mut().take(20) {
        for (element, query) in row.iter_mut().zip(&q) {
            *element = half::f16::from_f32(*query + random() * 0.002).to_f32();
        }
    }
    near[99].fill(0.0);
    let truth: Vec<f64> = near.iter().map(|row| {
        let accum = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x,y)| f64::from(*x) * f64::from(*y)).sum::<f64>();
        let length = (accum(&q, &q) * accum(row, row)).sqrt();
        if length > 0.0 { accum(&q, row) / length } else { 0.0 }
    }).collect();
    let order = |scores: &[f64]| {
        let mut ranked: Vec<usize> = (0..scores.len()).collect();
        ranked.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        ranked[..10].to_vec()
    };
    let truth_top = order(&truth);
    for (name, values) in [("scalar", scalar(&q, &near)), ("accelerate", accelerate(&q, &near))] {
        let as_double: Vec<f64> = values.iter().copied().map(f64::from).collect();
        let got_top = order(&as_double);
        let matching = got_top.iter().filter(|index| truth_top.contains(index)).count();
        let error = as_double.iter().zip(&truth).map(|(a,b)| (a-b).abs()).fold(0.0f64, f64::max);
        println!("near-tie {name}: top10 agreement {matching}/10, max error {error:.9}");
    }
}
