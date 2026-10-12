//! Component screen of `recall_dense`'s host-side rescore, three arms:
//!   main    - scalar fp16 decode (collect) + sequential f32 dot
//!   dot     - scalar fp16 decode (collect) + dispatched SIMD dot (255fbe1)
//!   convert - `half` slice conversion + dispatched SIMD dot (01614fb)
//! Per call: 1 query norm, then for each of 100 candidates widen 1024 fp16
//! codes, take its norm and its dot with the query, as projection.rs does.
//! Excludes the zvec query, document iteration, sort and everything else.
#![allow(dead_code)]
mod vector_math;
include!("decode.rs");
use half::slice::{HalfBitsSliceExt, HalfFloatSliceExt};
use std::hint::black_box;
use std::time::Instant;

fn seq_dot(l: &[f32], r: &[f32]) -> f32 { l.iter().zip(r).map(|(a, b)| a * b).sum() }
fn widen_scalar(codes: &[u16]) -> Vec<f32> { codes.iter().map(|c| decode(*c)).collect() }
fn widen_half(codes: &[u16]) -> Vec<f32> {
    let mut v = vec![0.0; codes.len()];
    codes.reinterpret_cast::<half::f16>().convert_to_f32_slice(&mut v);
    v
}
fn rescore(q: &[f32], rows: &[Vec<u16>], widen: fn(&[u16]) -> Vec<f32>, dot: fn(&[f32], &[f32]) -> f32) -> Vec<f32> {
    let length = dot(q, q).sqrt();
    let mut out = Vec::with_capacity(rows.len());
    for codes in rows {
        let v = widen(codes);
        let lengths = length * dot(&v, &v).sqrt();
        out.push(if lengths > 0.0 { dot(q, &v) / lengths } else { 0.0 });
    }
    out
}

fn main() {
    let mut state = 0x9e3779b97f4a7c15u64;
    let mut next = move || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; ((state >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * 0.06 };
    let q: Vec<f32> = (0..1024).map(|_| next()).collect();
    let rows: Vec<Vec<u16>> = (0..100).map(|_| (0..1024).map(|_| half::f16::from_f32(next()).to_bits()).collect()).collect();
    let arms: [(&str, fn(&[u16]) -> Vec<f32>, fn(&[f32], &[f32]) -> f32); 3] = [
        ("main", widen_scalar, seq_dot),
        ("dot", widen_scalar, vector_math::dot),
        ("convert", widen_half, vector_math::dot),
    ];
    // premise: all arms produce the same ranking and near-identical scores
    let reference: Vec<f64> = rows.iter().map(|r| {
        let v: Vec<f64> = r.iter().map(|c| decode(*c) as f64).collect();
        let qq: f64 = q.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let vv: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        q.iter().zip(&v).map(|(a, b)| *a as f64 * b).sum::<f64>() / (qq * vv)
    }).collect();
    for (name, w, d) in arms {
        let s = rescore(&q, &rows, w, d);
        let err = s.iter().zip(&reference).map(|(a, b)| (*a as f64 - b).abs()).fold(0.0, f64::max);
        let mut order: Vec<usize> = (0..100).collect();
        order.sort_by(|a, b| s[*b].total_cmp(&s[*a]));
        let mut truth: Vec<usize> = (0..100).collect();
        truth.sort_by(|a, b| reference[*b].total_cmp(&reference[*a]));
        let top10 = order[..10].iter().filter(|i| truth[..10].contains(i)).count();
        println!("premise {name:>8}: max |cos - f64 oracle| = {err:.3e}, top-10 overlap with oracle {top10}/10");
    }
    let rounds = 16;
    let mut samples = vec![Vec::new(); 3];
    for round in 0..rounds {
        let order: [usize; 3] = match round % 3 { 0 => [0, 1, 2], 1 => [2, 0, 1], _ => [1, 2, 0] };
        let order = if round % 2 == 0 { order } else { [order[2], order[1], order[0]] };
        for i in order {
            let (_, w, d) = arms[i];
            for _ in 0..100 {
                let t = Instant::now();
                black_box(rescore(black_box(&q), black_box(&rows), w, d));
                samples[i].push(t.elapsed().as_secs_f64() * 1e6);
            }
        }
    }
    for (i, (name, _, _)) in arms.iter().enumerate() {
        let s = &mut samples[i];
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("{name:>8}: p50 {:8.2} us  p95 {:8.2} us  (n={}, per 100x1024 rescore)", s[s.len() / 2], s[s.len() * 95 / 100], s.len());
    }
}
