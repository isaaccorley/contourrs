use contourrs::{contours, polygonize, AffineTransform, Connectivity, RasterGrid};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
struct Meter;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Meter {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() {
            let n = LIVE.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            PEAK.fetch_max(n, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l);
        LIVE.fetch_sub(l.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        let q = System.realloc(p, l, n);
        if !q.is_null() {
            let v = if n >= l.size() {
                LIVE.fetch_add(n - l.size(), Ordering::Relaxed) + n - l.size()
            } else {
                LIVE.fetch_sub(l.size() - n, Ordering::Relaxed) - (l.size() - n)
            };
            PEAK.fetch_max(v, Ordering::Relaxed);
        }
        q
    }
}
#[allow(dead_code)]
#[cfg_attr(feature = "measure-heap", global_allocator)]
static ALLOC: Meter = Meter;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let kind = &args[1];
    let n: usize = args[2].parse().unwrap();
    let mut state = 42u64;
    let data: Vec<f64> = (0..n * n)
        .map(|i| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let rand = ((state >> 32) as u32) as f64 / u32::MAX as f64;
            match kind.as_str() {
                "uniform" => 1.,
                "categorical" => (rand * 5.).floor(),
                "smooth" => {
                    let x = (i % n) as f64 / n as f64;
                    let y = (i / n) as f64 / n as f64;
                    ((x * 8.).sin() + (y * 6.).cos() + 2.) / 4.
                }
                _ => rand,
            }
        })
        .collect();
    let grid = RasterGrid::new(&data, n, n);
    let thresholds = [0.1, 0.25, 0.5, 0.75, 0.9];
    let run = || {
        if kind == "uniform" || kind == "categorical" {
            polygonize(&grid, None, Connectivity::Four, AffineTransform::identity())
        } else {
            contours(&grid, &thresholds, None, AffineTransform::identity())
        }
    };
    for _ in 0..2 {
        std::hint::black_box(run());
    }
    let mut timings = Vec::with_capacity(7);
    let mut peaks = Vec::with_capacity(7);
    let mut count = 0;
    for _ in 0..7 {
        let before = LIVE.load(Ordering::Relaxed);
        PEAK.store(before, Ordering::Relaxed);
        let start = Instant::now();
        let out = run();
        timings.push(start.elapsed().as_secs_f64() * 1000.);
        count = out.len();
        peaks.push(PEAK.load(Ordering::Relaxed).saturating_sub(before));
        std::hint::black_box(out);
    }
    timings.sort_by(|a, b| a.total_cmp(b));
    peaks.sort();
    println!(
        "{},{},{:.3},{:.3},{}",
        kind,
        n,
        timings[3],
        peaks[3] as f64 / 1048576.,
        count
    );
}
