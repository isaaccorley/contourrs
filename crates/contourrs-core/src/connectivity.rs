/// Pixel connectivity mode for region labeling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Connectivity {
    /// 4-connected: shares an edge (up, down, left, right).
    Four,
    /// 8-connected: shares an edge or corner (includes diagonals).
    Eight,
}
