use std::{time::Instant, collections::BTreeMap};
use ort::{session::{Session,builder::GraphOptimizationLevel},value::TensorRef,AsPointer};
const ROWS:usize=100; const WIDTH:usize=1024;
fn dot(a:&[f32],b:&[f32])->f32{a.iter().zip(b).map(|(a,b)|a*b).sum()}
fn scalar(v:&[f32],q:&[f32])->Vec<f32>{let qn=dot(q,q).sqrt();v.chunks_exact(WIDTH).map(|r|{let n=qn*dot(r,r).sqrt();if n>0.0{dot(r,q)/n}else{0.0}}).collect()}
fn run(s:&mut Session,v:&[f32],q:&[f32])->ort::Result<Vec<f32>>{
 let input=TensorRef::from_array_view(([ROWS,WIDTH],v))?;
 let query=TensorRef::from_array_view(([1,WIDTH],q))?;
 let out=s.run(ort::inputs!["vectors"=>input,"query"=>query])?;
 Ok(out["cosines"].try_extract_tensor::<f32>()?.1.to_vec())
}
fn run64(s:&mut Session,v:&[f32],q:&[f32])->ort::Result<Vec<f64>>{
 let vectors:Vec<f64>=v.iter().copied().map(f64::from).collect();
 let query:Vec<f64>=q.iter().copied().map(f64::from).collect();
 let input=TensorRef::from_array_view(([ROWS,WIDTH],vectors.as_slice()))?;
 let query=TensorRef::from_array_view(([1,WIDTH],query.as_slice()))?;
 let out=s.run(ort::inputs!["vectors"=>input,"query"=>query])?;
 Ok(out["cosines"].try_extract_tensor::<f64>()?.1.to_vec())
}
fn assignments(s:&Session)->ort::Result<BTreeMap<String,usize>>{
 let mut graphs=std::ptr::null();let mut n=0;
 unsafe{ort::Error::result_from_status((ort::api().Session_GetEpGraphAssignmentInfo)(s.ptr(),&mut graphs,&mut n))?;}
 let mut counts=BTreeMap::new();
 if n==0{return Ok(counts)}
 assert!(!graphs.is_null());
 for &g in unsafe{std::slice::from_raw_parts(graphs,n)}{
  assert!(!g.is_null());let mut name=std::ptr::null();let mut nodes=std::ptr::null();let mut size=0;
  unsafe{ort::Error::result_from_status((ort::api().EpAssignedSubgraph_GetEpName)(g,&mut name))?;ort::Error::result_from_status((ort::api().EpAssignedSubgraph_GetNodes)(g,&mut nodes,&mut size))?;}
  assert!(!name.is_null());let name=unsafe{std::ffi::CStr::from_ptr(name)}.to_string_lossy().into_owned();*counts.entry(name).or_default()+=size;
 }Ok(counts)
}
fn main()->Result<(),Box<dyn std::error::Error>>{
 let root=std::path::PathBuf::from(std::env::args().nth(1).expect("experiment directory"));
 let mut state=1234567u64;let mut random=||{state=state.wrapping_mul(6364136223846793005).wrapping_add(1);(((state>>32)as u32)as f64/u32::MAX as f64-0.5)as f32};
 let q:Vec<f32>=(0..WIDTH).map(|_|random()).collect();
 let mut v:Vec<f32>=(0..ROWS*WIDTH).map(|_|half::f16::from_f32(random()).to_f32()).collect();
 for r in 0..20{for k in 0..WIDTH{v[r*WIDTH+k]=half::f16::from_f32(q[k]+random()*0.002).to_f32();}}
 v[(ROWS-1)*WIDTH..].fill(0.0);
 let oracle:Vec<f64>=v.chunks_exact(WIDTH).map(|r|{let d:f64=r.iter().zip(&q).map(|(a,b)|f64::from(*a)*f64::from(*b)).sum();let a:f64=r.iter().map(|x|f64::from(*x).powi(2)).sum();let b:f64=q.iter().map(|x|f64::from(*x).powi(2)).sum();if a*b>0.0{d/(a*b).sqrt()}else{0.0}}).collect();
 let mut sessions=Vec::new();let mut start_rows=Vec::new();
 for (label,provider) in [
  ("ort_cpu_simd",ort::ep::CPU::default().with_arena_allocator(false).build()),
  ("coreml_all",ort::ep::CoreML::default().with_model_format(ort::ep::coreml::ModelFormat::MLProgram).with_compute_units(ort::ep::coreml::ComputeUnits::All).with_model_cache_dir(root.join(format!("all-cache-{ROWS}-{WIDTH}")).to_string_lossy()).build().error_on_failure()),
  ("coreml_cpu_only",ort::ep::CoreML::default().with_model_format(ort::ep::coreml::ModelFormat::MLProgram).with_compute_units(ort::ep::coreml::ComputeUnits::CPUOnly).with_model_cache_dir(root.join(format!("cpu-cache-{ROWS}-{WIDTH}")).to_string_lossy()).build().error_on_failure()),
  ("coreml_cpu_gpu",ort::ep::CoreML::default().with_model_format(ort::ep::coreml::ModelFormat::MLProgram).with_compute_units(ort::ep::coreml::ComputeUnits::CPUAndGPU).with_model_cache_dir(root.join(format!("gpu-cache-{ROWS}-{WIDTH}")).to_string_lossy()).build().error_on_failure())]{
  let t=Instant::now();let result=Session::builder()?.with_execution_providers([provider])?.with_intra_threads(1)?.with_intra_op_spinning(false)?.with_config_entry("session.record_ep_graph_assignment_info","1")?.with_optimization_level(GraphOptimizationLevel::Level3)?.with_dimension_override("batch_size",ROWS as i64)?.with_dimension_override("sequence_length",WIDTH as i64)?.commit_from_file(root.join("cosine.onnx"));
  match result{Ok(mut s)=>{let nodes=assignments(&s)?;let values=run(&mut s,&v,&q)?;assert!(values.iter().all(|x|x.is_finite()));let error=values.iter().zip(&oracle).map(|(x,o)|(f64::from(*x)-o).abs()).fold(0.0,f64::max);start_rows.push(serde_json::json!({"arm":label,"startup_seconds":t.elapsed().as_secs_f64(),"assigned_nodes":nodes,"maximum_abs_error_vs_f64":error,"scores":values}));for _ in 0..10{run(&mut s,&v,&q)?;}sessions.push((label,s));},Err(e)=>start_rows.push(serde_json::json!({"arm":label,"load_error":e.to_string()}))}
 }
 let t=Instant::now();
 let mut double=Session::builder()?.with_execution_providers([ort::ep::CPU::default().with_arena_allocator(false).build()])?.with_intra_threads(1)?.with_intra_op_spinning(false)?.with_config_entry("session.record_ep_graph_assignment_info","1")?.with_optimization_level(GraphOptimizationLevel::Level3)?.with_dimension_override("batch_size",ROWS as i64)?.with_dimension_override("sequence_length",WIDTH as i64)?.commit_from_file(root.join("cosine-f64.onnx"))?;
 let values=run64(&mut double,&v,&q)?;
 let err=values.iter().zip(&oracle).map(|(x,o)|(x-o).abs()).fold(0.0,f64::max);
 start_rows.push(serde_json::json!({"arm":"ort_cpu_f64_with_conversion","startup_seconds":t.elapsed().as_secs_f64(),"assigned_nodes":assignments(&double)?,"maximum_abs_error_vs_f64":err,"scores":values}));
 for _ in 0..10{run64(&mut double,&v,&q)?;}
let base=scalar(&v,&q);let mut clocks:BTreeMap<String,Vec<f64>>=BTreeMap::new();
 for round in 0..6{for step in 0..sessions.len()+2{let arm=(step+round)%(sessions.len()+2);for _ in 0..100{
  let t=Instant::now();
  if arm==0{std::hint::black_box(scalar(&v,&q));}
  else if arm==sessions.len()+1{std::hint::black_box(run64(&mut double,&v,&q)?);}
  else{std::hint::black_box(run(&mut sessions[arm-1].1,&v,&q)?);}
  let label=if arm==0{"rust_scalar"}else if arm==sessions.len()+1{"ort_cpu_f64_with_conversion"}else{sessions[arm-1].0};
  clocks.entry(label.into()).or_default().push(t.elapsed().as_secs_f64());
 }}}
 let summary:Vec<_>=clocks.iter_mut().map(|(label,v)|{v.sort_by(f64::total_cmp);serde_json::json!({"arm":label,"samples":v.len(),"p50_seconds":v[v.len()/2],"p95_seconds":v[v.len()*95/100]})}).collect();
 let result=serde_json::json!({"runtime":ort::info(),"rows":ROWS,"width":WIDTH,"scope":"same FP16-rounded resident candidates and F32 query, host call boundary; borrowed input tensors; session construction separate; shared host","startup":start_rows,"scalar_scores":base,"oracle_scores":oracle,"summary":summary,"raw_seconds":clocks});std::fs::write(root.join("result.json"),serde_json::to_vec_pretty(&result)?)?;println!("{}",serde_json::to_string_pretty(&result["summary"])?);Ok(())
}
