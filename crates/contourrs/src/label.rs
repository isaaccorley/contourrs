use crate::connectivity::Connectivity;
use crate::raster::{RasterGrid, RasterValue};
use crate::union_find::UnionFind;

/// Result of region labeling pass.
pub struct LabelResult {
    /// Label grid — each pixel gets a region label (canonical root after resolve).
    pub labels: Vec<u32>,
    pub width: usize,
    pub height: usize,
}

/// Pass 1: Assign region labels using connected-component labeling.
///
/// Each contiguous region of equal values (respecting connectivity and mask)
/// gets a unique label. Returns the label grid with canonical labels.
pub fn label_regions<T: RasterValue>(
    grid: &RasterGrid<T>,
    mask: Option<&[bool]>,
    connectivity: Connectivity,
) -> LabelResult {
    let w = grid.width;
    let h = grid.height;
    let n = w * h;

    let mut labels = vec![0u32; n];
    let mut uf = UnionFind::new(n);
    let mut next_label = 0u32;

    for row in 0..h {
        for col in 0..w {
            let idx = row * w + col;

            // Skip masked-out pixels
            if let Some(m) = mask {
                if !m[idx] {
                    labels[idx] = u32::MAX; // sentinel for masked
                    continue;
                }
            }

            let val = grid.get(col, row);
            let mut merged_label: Option<u32> = None;

            // Check west neighbor
            if col > 0 {
                let nidx = idx - 1;
                if labels[nidx] != u32::MAX && grid.get(col - 1, row) == val {
                    let root = uf.find(labels[nidx]);
                    merged_label = Some(match merged_label {
                        Some(existing) => uf.union(existing, root),
                        None => root,
                    });
                }
            }

            // Check north neighbor
            if row > 0 {
                let nidx = idx - w;
                if labels[nidx] != u32::MAX && grid.get(col, row - 1) == val {
                    let root = uf.find(labels[nidx]);
                    merged_label = Some(match merged_label {
                        Some(existing) => uf.union(existing, root),
                        None => root,
                    });
                }
            }

            if connectivity == Connectivity::Eight {
                // Check northwest neighbor
                if col > 0 && row > 0 {
                    let nidx = (row - 1) * w + (col - 1);
                    if labels[nidx] != u32::MAX && grid.get(col - 1, row - 1) == val {
                        let root = uf.find(labels[nidx]);
                        merged_label = Some(match merged_label {
                            Some(existing) => uf.union(existing, root),
                            None => root,
                        });
                    }
                }

                // Check northeast neighbor
                if col + 1 < w && row > 0 {
                    let nidx = (row - 1) * w + (col + 1);
                    if labels[nidx] != u32::MAX && grid.get(col + 1, row - 1) == val {
                        let root = uf.find(labels[nidx]);
                        merged_label = Some(match merged_label {
                            Some(existing) => uf.union(existing, root),
                            None => root,
                        });
                    }
                }
            }

            labels[idx] = match merged_label {
                Some(root) => root,
                None => {
                    let l = next_label;
                    next_label += 1;
                    l
                }
            };
        }
    }

    // Resolve all labels to canonical roots
    for label in labels.iter_mut() {
        if *label != u32::MAX {
            *label = uf.find(*label);
        }
    }

    LabelResult {
        labels,
        width: w,
        height: h,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster::RasterGrid;

    #[test]
    fn test_uniform_grid() {
        let data = vec![1u8; 9];
        let grid = RasterGrid::new(&data, 3, 3);
        let result = label_regions(&grid, None, Connectivity::Four);
        let root = result.labels[0];
        assert!(result.labels.iter().all(|&l| l == root));
    }

    #[test]
    fn test_two_regions() {
        // 1 1 2
        // 1 1 2
        let data = vec![1u8, 1, 2, 1, 1, 2];
        let grid = RasterGrid::new(&data, 3, 2);
        let result = label_regions(&grid, None, Connectivity::Four);
        // First 4 pixels (value 1) should share a label
        let l1 = result.labels[0];
        assert_eq!(result.labels[1], l1);
        assert_eq!(result.labels[3], l1);
        assert_eq!(result.labels[4], l1);
        // Last 2 pixels (value 2) should share a different label
        let l2 = result.labels[2];
        assert_eq!(result.labels[5], l2);
        assert_ne!(l1, l2);
    }

    #[test]
    fn test_checkerboard_4conn() {
        // 1 2 1
        // 2 1 2
        let data = vec![1u8, 2, 1, 2, 1, 2];
        let grid = RasterGrid::new(&data, 3, 2);
        let result = label_regions(&grid, None, Connectivity::Four);
        // With 4-connectivity, each cell is its own region
        let labels: std::collections::HashSet<u32> = result.labels.iter().copied().collect();
        assert_eq!(labels.len(), 6);
    }

    #[test]
    fn test_checkerboard_8conn() {
        // 1 2 1
        // 2 1 2
        let data = vec![1u8, 2, 1, 2, 1, 2];
        let grid = RasterGrid::new(&data, 3, 2);
        let result = label_regions(&grid, None, Connectivity::Eight);
        // With 8-connectivity, diagonals connect:
        // value=1: (0,0),(2,0),(1,1) share label; value=2: (1,0),(0,1),(2,1) share label
        let l1 = result.labels[0]; // (0,0) value=1
        assert_eq!(result.labels[2], l1); // (2,0) connects via (1,1) diagonal? No — (2,0) and (1,1) are diagonal
                                          // Actually: (0,0) val=1, (1,1) val=1, diagonal => same region in 8-conn
        assert_eq!(result.labels[4], l1); // (1,1) value=1
    }

    #[test]
    fn test_mask() {
        let data = vec![1u8; 4];
        let mask = vec![true, false, true, true];
        let grid = RasterGrid::new(&data, 2, 2);
        let result = label_regions(&grid, Some(&mask), Connectivity::Four);
        assert_eq!(result.labels[1], u32::MAX); // masked out
        assert_ne!(result.labels[0], u32::MAX);
    }

    #[test]
    fn test_l_shaped_merge() {
        // Tests union-find merge when regions connect later:
        // 1 2
        // 1 1
        let data = vec![1u8, 2, 1, 1];
        let grid = RasterGrid::new(&data, 2, 2);
        let result = label_regions(&grid, None, Connectivity::Four);
        let l1 = result.labels[0];
        assert_eq!(result.labels[2], l1);
        assert_eq!(result.labels[3], l1);
        assert_ne!(result.labels[1], l1);
    }

    #[test]
    fn test_u_shaped_merge_west_and_north() {
        // Forces the west+north merge path (merged_label is Some when north checked)
        // 1 1 1
        // 1 2 1
        // 1 1 1
        // When processing (2,2)=1: west=(1,2)=2 no match; north=(2,1)=1 match → first merge
        // When processing (1,2)=1: west=(0,2)=1 → first merge; north=(1,1)=2 no match
        // But we need BOTH west AND north to match to exercise line 51
        // This happens when a U-shaped region reconnects:
        // 1 2 1
        // 1 1 1
        let data = vec![1u8, 2, 1, 1, 1, 1];
        let grid = RasterGrid::new(&data, 3, 2);
        let result = label_regions(&grid, None, Connectivity::Four);
        // All 1s should be unified
        let l1 = result.labels[0];
        assert_eq!(result.labels[2], l1);
        assert_eq!(result.labels[3], l1);
        assert_eq!(result.labels[4], l1);
        assert_eq!(result.labels[5], l1);
        // The 2 should be different
        assert_ne!(result.labels[1], l1);
    }

    #[test]
    fn test_8conn_all_diagonals() {
        // Forces all diagonal neighbor checks (NW and NE)
        // 1 2 1
        // 2 1 2
        // 1 2 1
        let data = vec![1u8, 2, 1, 2, 1, 2, 1, 2, 1];
        let grid = RasterGrid::new(&data, 3, 3);
        let result = label_regions(&grid, None, Connectivity::Eight);
        // 8-connectivity: all 1s connected diagonally
        let l1 = result.labels[0];
        assert_eq!(result.labels[2], l1); // (2,0) → via (1,1)
        assert_eq!(result.labels[4], l1); // (1,1) center
        assert_eq!(result.labels[6], l1); // (0,2)
        assert_eq!(result.labels[8], l1); // (2,2)
    }
}
