//! CUDA-accelerated connected-component labeling.
//!
//! Gated behind the `cuda` feature flag.
//!
//! # Pipeline
//!
//! 1. Accept a raw GPU pointer (from PyTorch CUDA tensor or cudarc allocation)
//! 2. Run connected-component labeling (CCL) kernel on GPU
//! 3. Transfer label grid + compact root-value table back to CPU
//! 4. Run boundary tracing on CPU (inherently serial)
//!
//! This avoids the full GPU→CPU transfer of the source raster when the data
//! is already on GPU from model inference.

mod kernel;
pub use kernel::{CudaDevice, GpuLabelResult};

use crate::connectivity::Connectivity;
use crate::polygon;
use crate::trace;
use crate::transform::AffineTransform;
use geo_types::Polygon;

/// Polygonize a raster that is already on GPU.
///
/// # Arguments
/// * `device` - CUDA device handle
/// * `gpu_ptr` - Device pointer to int32 raster data (row-major, contiguous)
/// * `width` - Raster width in pixels
/// * `height` - Raster height in pixels
/// * `mask_gpu_ptr` - Optional device pointer to bool mask
/// * `connectivity` - 4 or 8
/// * `transform` - Affine transform for output coordinates
///
/// # Returns
/// Vec of (polygon, value) pairs, same as CPU `polygonize()`.
pub fn polygonize_gpu(
    device: &CudaDevice,
    gpu_ptr: u64,
    width: usize,
    height: usize,
    mask_gpu_ptr: Option<u64>,
    connectivity: Connectivity,
    transform: AffineTransform,
) -> Result<Vec<(Polygon<f64>, f64)>, String> {
    // Pass 1: GPU CCL — returns label grid on CPU
    let label_result =
        kernel::gpu_label_regions(device, gpu_ptr, width, height, mask_gpu_ptr, connectivity)?;

    let max_label = label_result
        .labels
        .iter()
        .filter(|&&l| l != u32::MAX)
        .max()
        .copied()
        .unwrap_or(0);

    let mut values = vec![0.0f64; max_label as usize + 1];
    for (label, value) in label_result
        .root_labels
        .iter()
        .copied()
        .zip(label_result.root_values.iter().copied())
    {
        if label != u32::MAX {
            values[label as usize] = f64::from(value);
        }
    }

    // Pass 2: CPU boundary tracing (same as CPU path)
    let mut polygons = trace::trace_polygons(&label_result.into(), &values, &transform);

    polygons = polygons
        .into_iter()
        .map(|(poly, val)| (polygon::normalize_polygon(poly), val))
        .collect();

    Ok(polygons)
}
