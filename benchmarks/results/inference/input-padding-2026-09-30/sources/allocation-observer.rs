#[cfg(test)]
mod scratch_padding_allocations {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    use super::*;
    thread_local! {
        static ACTIVE: Cell<bool> = const { Cell::new(false) };
        static COUNT: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
    }
    struct Counted;
    fn note(bytes: usize) {
        if ACTIVE.try_with(Cell::get).unwrap_or(false) {
            COUNT.with(|count| { let (calls, allocated) = count.get(); count.set((calls+1,allocated+bytes)); });
        }
    }
    unsafe impl GlobalAlloc for Counted {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            note(layout.size());
            unsafe { System.alloc(layout) }
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            note(layout.size());
            unsafe { System.alloc_zeroed(layout) }
        }
        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            note(size);
            unsafe { System.realloc(pointer,layout,size) }
        }
        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            unsafe { System.dealloc(pointer,layout) }
        }
    }
    #[global_allocator]
    static ALLOCATOR: Counted = Counted;
    fn measure(run: impl FnOnce() -> [Vec<i64>; 2]) -> ([Vec<i64>; 2], (usize,usize)) {
        COUNT.with(|count| count.set((0,0)));
        ACTIVE.with(|active| active.set(true));
        let result = std::hint::black_box(run());
        ACTIVE.with(|active| active.set(false));
        (result,COUNT.with(Cell::get))
    }
    fn legacy(encodings: &[Encoding], [rows,tokens]:[usize;2]) -> [Vec<i64>;2] {
        let mut fixed_encodings;
        let encodings = if (rows,tokens) != (encodings.len(),encodings[0].len()) {
            fixed_encodings = encodings.to_vec();
            for encoding in &mut fixed_encodings { encoding.pad(tokens,1,0,"<pad>",tokenizers::PaddingDirection::Right); }
            while fixed_encodings.len() < rows { fixed_encodings.push(fixed_encodings.last().unwrap().clone()); }
            &fixed_encodings
        } else { encodings };
        let column = |field: fn(&Encoding)->&[u32]| encodings.iter().flat_map(|encoding|field(encoding).iter().map(|value|i64::from(*value))).collect();
        [column(Encoding::get_ids),column(Encoding::get_attention_mask)]
    }
    #[test]
    #[ignore="reads the cached tokenizer and counts only host input preparation allocations"]
    fn scratch_padding_allocations() {
        let models = std::path::PathBuf::from(std::env::var("PAMIN_EVAL_HOME").unwrap()).join("models");
        let repository = Repository::open(&models,"onnx-community/bge-reranker-v2-m3-ONNX").unwrap();
        let tokenizer = crate::tokenizer::load(&repository,256).unwrap();
        let mut records = Vec::new();
        for (name,count,rows,tokens,document) in [
            ("short_partial",1,4,64,"a migration rolls back".to_owned()),
            ("short_full",4,4,64,"a migration rolls back".to_owned()),
            ("medium_partial",3,4,128,"database migration rollback ".repeat(20)),
            ("long_partial",1,2,256,"database migration rollback ".repeat(300)),
        ] {
            let mut encodings=tokenizer.encode_batch(vec![("measurement warmup 0: migration failure",document.as_str());count],true).unwrap();
            pad(&tokenizer,&mut encodings).unwrap();
            assert!(encodings[0].len()<=tokens);
            let (old,old_cost)=measure(||legacy(&encodings,[rows,tokens]));
            let (new,new_cost)=measure(||[input_values(&encodings,[rows,tokens],Encoding::get_ids,1),input_values(&encodings,[rows,tokens],Encoding::get_attention_mask,0)]);
            assert_eq!(old,new);
            records.push(serde_json::json!({"case":name,"logical_rows":count,"logical_tokens":encodings[0].len(),"rows":rows,"tokens":tokens,"before_calls":old_cost.0,"before_bytes_requested":old_cost.1,"after_calls":new_cost.0,"after_bytes_requested":new_cost.1}));
        }
        std::fs::write(std::env::var("PAD_ALLOC_OUTPUT").unwrap(),serde_json::to_vec_pretty(&records).unwrap()).unwrap();
    }
}
