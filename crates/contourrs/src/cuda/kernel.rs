//! CUDA kernel wrappers for connected-component labeling.

use crate::connectivity::Connectivity;
use crate::label::LabelResult;
use cudarc::driver::{CudaDevice as CudarcDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{compile_ptx, Ptx};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

const MODULE_NAME: &str = "contourrs_ccl";
const INIT_KERNEL: &str = "init_labels";
const MERGE_KERNEL: &str = "merge_labels";
const FLATTEN_KERNEL: &str = "flatten_labels";
const COLLECT_KERNEL: &str = "collect_roots";
const MERGE_BATCH_ITERS: usize = 8;

static PTX_CACHE: OnceLock<Result<Ptx, String>> = OnceLock::new();
static DEVICE_CACHE: OnceLock<Mutex<HashMap<usize, Arc<CudarcDevice>>>> = OnceLock::new();

const CUDA_SRC: &str = r#"
__device__ __forceinline__ unsigned int find_root(const unsigned int* labels, unsigned int x) {
    unsigned int parent = labels[x];
    while (parent != labels[parent]) {
        parent = labels[parent];
    }
    return parent;
}

extern "C" __global__ void init_labels(
    unsigned int* labels,
    int width,
    int height,
    unsigned long long mask_ptr,
    int has_mask
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = width * height;
    if (idx >= n) return;

    if (has_mask) {
        const unsigned char* mask = reinterpret_cast<const unsigned char*>(mask_ptr);
        labels[idx] = mask[idx] ? (unsigned int)idx : 0xFFFFFFFFu;
    } else {
        labels[idx] = (unsigned int)idx;
    }
}

extern "C" __global__ void merge_labels(
    unsigned long long values_ptr,
    unsigned int* labels,
    int width,
    int height,
    int connectivity,
    int* changed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = width * height;
    if (idx >= n) return;

    unsigned int my_label = labels[idx];
    if (my_label == 0xFFFFFFFFu) return;

    unsigned int my_root = find_root(labels, my_label);

    const int* values = reinterpret_cast<const int*>(values_ptr);
    int row = idx / width;
    int col = idx % width;
    int my_val = values[idx];

    int dx[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    int dy[8] = {0, 0, -1, 1, -1, 1, -1, 1};
    int n_neighbors = (connectivity == 8) ? 8 : 4;

    unsigned int best_label = my_root;
    for (int d = 0; d < n_neighbors; d++) {
        int nc = col + dx[d];
        int nr = row + dy[d];
        if (nc < 0 || nc >= width || nr < 0 || nr >= height) continue;

        int nidx = nr * width + nc;
        unsigned int neigh_label = labels[nidx];
        if (neigh_label == 0xFFFFFFFFu) continue;
        if (values[nidx] != my_val) continue;

        unsigned int neigh_root = find_root(labels, neigh_label);

        if (neigh_root < best_label) {
            best_label = neigh_root;
        }
    }

    if (best_label < my_root) {
        atomicMin(&labels[my_root], best_label);
        atomicMin(&labels[idx], best_label);
        *changed = 1;
    } else if (my_root < my_label) {
        atomicMin(&labels[idx], my_root);
    }
}

extern "C" __global__ void flatten_labels(unsigned int* labels, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (labels[idx] == 0xFFFFFFFFu) return;

    unsigned int root = labels[idx];
    while (root != labels[root]) {
        root = labels[root];
    }
    labels[idx] = root;
}

extern "C" __global__ void collect_roots(
    unsigned long long values_ptr,
    const unsigned int* labels,
    int n,
    unsigned int* out_root_labels,
    int* out_root_values,
    unsigned int* out_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned int label = labels[idx];
    if (label == 0xFFFFFFFFu) return;
    if (label != (unsigned int)idx) return;

    const int* values = reinterpret_cast<const int*>(values_ptr);
    unsigned int slot = atomicAdd(out_count, (unsigned int)1);
    out_root_labels[slot] = (unsigned int)idx;
    out_root_values[slot] = values[idx];
}
"#;

/// Handle to a CUDA device.
pub struct CudaDevice {
    pub(crate) inner: Arc<CudarcDevice>,
}

fn ptx_program() -> Result<Ptx, String> {
    match PTX_CACHE.get_or_init(|| {
        compile_ptx(CUDA_SRC).map_err(|e| format!("PTX compile failed: {e:?}"))
    }) {
        Ok(ptx) => Ok(ptx.clone()),
        Err(err) => Err(err.clone()),
    }
}

fn ensure_kernels_loaded(inner: &Arc<CudarcDevice>) -> Result<(), String> {
    if inner.has_func(MODULE_NAME, INIT_KERNEL) {
        return Ok(());
    }

    let ptx = ptx_program()?;
    inner
        .load_ptx(
            ptx,
            MODULE_NAME,
            &[INIT_KERNEL, MERGE_KERNEL, FLATTEN_KERNEL, COLLECT_KERNEL],
        )
        .map_err(|e| format!("PTX load failed: {e:?}"))
}

impl CudaDevice {
    /// Initialize CUDA device by ordinal and load CCL kernels.
    pub fn new(ordinal: usize) -> Result<Self, String> {
        let cache = DEVICE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        {
            let guard = cache
                .lock()
                .map_err(|_| "CUDA device cache poisoned".to_string())?;
            if let Some(inner) = guard.get(&ordinal) {
                return Ok(Self {
                    inner: inner.clone(),
                });
            }
        }

        let inner = CudarcDevice::new(ordinal).map_err(|e| format!("CUDA init failed: {e:?}"))?;
        ensure_kernels_loaded(&inner)?;

        let mut guard = cache
            .lock()
            .map_err(|_| "CUDA device cache poisoned".to_string())?;
        guard.insert(ordinal, inner.clone());

        Ok(Self { inner })
    }
}

/// Result of GPU-based connected-component labeling.
pub struct GpuLabelResult {
    pub labels: Vec<u32>,
    pub width: usize,
    pub height: usize,
    pub root_labels: Vec<u32>,
    pub root_values: Vec<i32>,
}

impl From<GpuLabelResult> for LabelResult {
    fn from(gpu: GpuLabelResult) -> Self {
        LabelResult {
            labels: gpu.labels,
            width: gpu.width,
            height: gpu.height,
        }
    }
}

fn launch_cfg(n: usize) -> Result<LaunchConfig, String> {
    let n_u32 = u32::try_from(n).map_err(|_| "grid too large for CUDA launch".to_string())?;
    let block_dim = 256u32;
    let grid_dim = (n_u32.saturating_add(block_dim - 1)) / block_dim;
    Ok(LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    })
}

/// Run connected-component labeling on GPU, return label grid and root values on CPU.
pub fn gpu_label_regions(
    device: &CudaDevice,
    gpu_ptr: u64,
    width: usize,
    height: usize,
    mask_gpu_ptr: Option<u64>,
    connectivity: Connectivity,
) -> Result<GpuLabelResult, String> {
    let n = width
        .checked_mul(height)
        .ok_or_else(|| "width*height overflow".to_string())?;
    let cfg = launch_cfg(n)?;

    let width_i32 = i32::try_from(width).map_err(|_| "width exceeds i32".to_string())?;
    let height_i32 = i32::try_from(height).map_err(|_| "height exceeds i32".to_string())?;
    let n_i32 = i32::try_from(n).map_err(|_| "n exceeds i32".to_string())?;
    let conn_i32 = match connectivity {
        Connectivity::Four => 4,
        Connectivity::Eight => 8,
    };

    let mut labels_dev = device
        .inner
        .alloc_zeros::<u32>(n)
        .map_err(|e| format!("labels alloc failed: {e:?}"))?;
    let mut changed_dev = device
        .inner
        .alloc_zeros::<i32>(1)
        .map_err(|e| format!("changed alloc failed: {e:?}"))?;

    let mut root_labels_dev = device
        .inner
        .alloc_zeros::<u32>(n)
        .map_err(|e| format!("root_labels alloc failed: {e:?}"))?;
    let mut root_values_dev = device
        .inner
        .alloc_zeros::<i32>(n)
        .map_err(|e| format!("root_values alloc failed: {e:?}"))?;
    let mut root_count_dev = device
        .inner
        .alloc_zeros::<u32>(1)
        .map_err(|e| format!("root_count alloc failed: {e:?}"))?;

    let has_mask = i32::from(mask_gpu_ptr.is_some());
    let mask_ptr = mask_gpu_ptr.unwrap_or(0);

    let init = device
        .inner
        .get_func(MODULE_NAME, INIT_KERNEL)
        .ok_or_else(|| "missing init kernel".to_string())?;
    let merge = device
        .inner
        .get_func(MODULE_NAME, MERGE_KERNEL)
        .ok_or_else(|| "missing merge kernel".to_string())?;
    let flatten = device
        .inner
        .get_func(MODULE_NAME, FLATTEN_KERNEL)
        .ok_or_else(|| "missing flatten kernel".to_string())?;
    let collect = device
        .inner
        .get_func(MODULE_NAME, COLLECT_KERNEL)
        .ok_or_else(|| "missing collect kernel".to_string())?;

    // SAFETY: kernel signatures and parameter order/types match CUDA source.
    unsafe {
        init.launch(
            cfg,
            (&mut labels_dev, width_i32, height_i32, mask_ptr, has_mask),
        )
        .map_err(|e| format!("init kernel failed: {e:?}"))?;
    }

    let mut changed = true;
    let mut batches = 0usize;
    let max_batches = (n.max(1).saturating_add(MERGE_BATCH_ITERS - 1)) / MERGE_BATCH_ITERS;
    while changed {
        batches += 1;
        if batches > max_batches {
            return Err("CCL merge exceeded iteration cap".to_string());
        }

        device
            .inner
            .memset_zeros(&mut changed_dev)
            .map_err(|e| format!("reset changed failed: {e:?}"))?;

        for _ in 0..MERGE_BATCH_ITERS {
            // SAFETY: kernel signatures and parameter order/types match CUDA source.
            unsafe {
                merge
                    .clone()
                    .launch(
                        cfg,
                        (
                            gpu_ptr,
                            &mut labels_dev,
                            width_i32,
                            height_i32,
                            conn_i32,
                            &mut changed_dev,
                        ),
                    )
                    .map_err(|e| format!("merge kernel failed: {e:?}"))?;
                flatten
                    .clone()
                    .launch(cfg, (&mut labels_dev, n_i32))
                    .map_err(|e| format!("flatten kernel failed: {e:?}"))?;
            }
        }

        let changed_host = device
            .inner
            .dtoh_sync_copy(&changed_dev)
            .map_err(|e| format!("copy changed failed: {e:?}"))?;
        changed = changed_host.first().copied().unwrap_or(0) != 0;
    }

    // SAFETY: kernel signatures and parameter order/types match CUDA source.
    unsafe {
        flatten
            .clone()
            .launch(cfg, (&mut labels_dev, n_i32))
            .map_err(|e| format!("flatten kernel failed: {e:?}"))?;
    }

    // SAFETY: kernel signatures and parameter order/types match CUDA source.
    unsafe {
        collect
            .launch(
                cfg,
                (
                    gpu_ptr,
                    &labels_dev,
                    n_i32,
                    &mut root_labels_dev,
                    &mut root_values_dev,
                    &mut root_count_dev,
                ),
            )
            .map_err(|e| format!("collect roots kernel failed: {e:?}"))?;
    }

    let labels = device
        .inner
        .dtoh_sync_copy(&labels_dev)
        .map_err(|e| format!("copy labels failed: {e:?}"))?;

    let root_count_vec = device
        .inner
        .dtoh_sync_copy(&root_count_dev)
        .map_err(|e| format!("copy root count failed: {e:?}"))?;
    let root_count = usize::try_from(root_count_vec.first().copied().unwrap_or(0))
        .map_err(|_| "invalid root count".to_string())?;
    if root_count > n {
        return Err("invalid root count returned by kernel".to_string());
    }

    let mut root_labels = device
        .inner
        .dtoh_sync_copy(&root_labels_dev)
        .map_err(|e| format!("copy root labels failed: {e:?}"))?;
    let mut root_values = device
        .inner
        .dtoh_sync_copy(&root_values_dev)
        .map_err(|e| format!("copy root values failed: {e:?}"))?;

    root_labels.truncate(root_count);
    root_values.truncate(root_count);

    Ok(GpuLabelResult {
        labels,
        width,
        height,
        root_labels,
        root_values,
    })
}
