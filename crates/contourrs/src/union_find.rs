/// Disjoint set (union-find) with path compression and union by rank.
pub struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            rank: vec![0; n],
        }
    }

    /// Find root with path compression.
    #[inline]
    pub fn find(&mut self, mut x: u32) -> u32 {
        loop {
            let p = self.parent[x as usize];
            if p == x {
                return x;
            }
            let gp = self.parent[p as usize];
            self.parent[x as usize] = gp;
            x = gp;
        }
    }

    /// Union two sets by rank. Returns the new root.
    #[inline]
    pub fn union(&mut self, a: u32, b: u32) -> u32 {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return ra;
        }
        match self.rank[ra as usize].cmp(&self.rank[rb as usize]) {
            std::cmp::Ordering::Less => {
                self.parent[ra as usize] = rb;
                rb
            }
            std::cmp::Ordering::Greater => {
                self.parent[rb as usize] = ra;
                ra
            }
            std::cmp::Ordering::Equal => {
                self.parent[rb as usize] = ra;
                self.rank[ra as usize] += 1;
                ra
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_union_find() {
        let mut uf = UnionFind::new(5);
        assert_ne!(uf.find(0), uf.find(1));
        uf.union(0, 1);
        assert_eq!(uf.find(0), uf.find(1));
        uf.union(2, 3);
        assert_eq!(uf.find(2), uf.find(3));
        assert_ne!(uf.find(0), uf.find(2));
        uf.union(1, 3);
        assert_eq!(uf.find(0), uf.find(3));
    }

    #[test]
    fn test_path_compression() {
        let mut uf = UnionFind::new(100);
        for i in 0..99 {
            uf.union(i, i + 1);
        }
        let root = uf.find(0);
        // After find with path compression, all should converge quickly
        for i in 0..100 {
            assert_eq!(uf.find(i), root);
        }
    }

    #[test]
    fn test_union_rank_less_branch() {
        // Create two sets of different rank, then union small into large
        let mut uf = UnionFind::new(4);
        // union(0,1) → rank[0]=1 (equal case)
        uf.union(0, 1);
        // Now rank[root_of_01] = 1, rank[2] = 0
        // union(2, 0) should hit Ordering::Less: rank[2] < rank[root_of_01]
        let root = uf.union(2, 0);
        assert_eq!(uf.find(2), root);
        assert_eq!(uf.find(0), root);
        assert_eq!(uf.find(1), root);
    }
}
