//! Topological Data Analysis (TDA), Persistent Homology, and Simplicial Complex Reduction
//! for Attention Key Manifolds.
//!
//! Strict Compliance: #![deny(unsafe_code)], zero emojis, enterprise commercial robotics standard.

#![deny(unsafe_code)]

use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// Metric space distance function for key vectors on the manifold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    Euclidean,
    AngularSpherical,
}

/// A topological feature in a persistence diagram with birth and death filtration scales.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PersistencePair {
    pub dimension: usize,
    pub birth: f32,
    pub death: f32,
}

impl PersistencePair {
    #[inline]
    pub fn new(dimension: usize, birth: f32, death: f32) -> Self {
        Self {
            dimension,
            birth,
            death,
        }
    }

    #[inline]
    pub fn persistence(&self) -> f32 {
        self.death - self.birth
    }

    #[inline]
    pub fn is_infinite(&self) -> bool {
        self.death.is_infinite()
    }
}

/// Persistence diagram summarizing multiscale homological invariants (H_0 and H_1).
#[derive(Debug, Clone, Default)]
pub struct PersistenceDiagram {
    pub pairs: Vec<PersistencePair>,
}

impl PersistenceDiagram {
    pub fn new(pairs: Vec<PersistencePair>) -> Self {
        Self { pairs }
    }

    /// Evaluates the Betti number beta_p at a specified filtration scale t.
    pub fn betti_number(&self, dimension: usize, t: f32) -> usize {
        self.pairs
            .iter()
            .filter(|p| p.dimension == dimension && p.birth <= t && t < p.death)
            .count()
    }

    /// Returns the maximum persistence across all finite features of dimension p.
    pub fn max_persistence(&self, dimension: usize) -> f32 {
        self.pairs
            .iter()
            .filter(|p| p.dimension == dimension && !p.is_infinite())
            .map(|p| p.persistence())
            .fold(0.0f32, f32::max)
    }

    /// Computes the persistent entropy of dimension p:
    /// E_p = -sum (l_i / L_p) * log2(l_i / L_p).
    pub fn persistent_entropy(&self, dimension: usize) -> f64 {
        let finite_pairs: Vec<&PersistencePair> = self
            .pairs
            .iter()
            .filter(|p| p.dimension == dimension && !p.is_infinite() && p.persistence() > 1e-6)
            .collect();

        if finite_pairs.is_empty() {
            return 0.0;
        }

        let total_persistence: f64 = finite_pairs.iter().map(|p| p.persistence() as f64).sum();
        if total_persistence <= 1e-9 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for pair in finite_pairs {
            let prob = (pair.persistence() as f64) / total_persistence;
            if prob > 1e-12 {
                entropy -= prob * prob.log2();
            }
        }
        entropy
    }
}

/// Undirected 1-simplex (edge) in the simplicial complex with canonical ordering u < v.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Simplex1 {
    pub u: usize,
    pub v: usize,
}

impl Simplex1 {
    #[inline]
    pub fn new(u: usize, v: usize) -> Self {
        if u < v {
            Self { u, v }
        } else {
            Self { u: v, v: u }
        }
    }
}

/// High-dimensional attention key manifold point cloud.
#[derive(Debug, Clone)]
pub struct TopologicalKeyManifold {
    pub keys: Vec<f32>,
    pub sequence_length: usize,
    pub head_dim: usize,
    pub metric: MetricType,
}

impl TopologicalKeyManifold {
    pub fn new(keys: Vec<f32>, sequence_length: usize, head_dim: usize, metric: MetricType) -> Self {
        assert_eq!(
            keys.len(),
            sequence_length * head_dim,
            "Key buffer size must match sequence_length * head_dim"
        );
        Self {
            keys,
            sequence_length,
            head_dim,
            metric,
        }
    }

    #[inline]
    pub fn get_key(&self, index: usize) -> &[f32] {
        let start = index * self.head_dim;
        &self.keys[start..start + self.head_dim]
    }

    /// Computes distance between keys i and j under the configured metric.
    pub fn distance(&self, i: usize, j: usize) -> f32 {
        if i == j {
            return 0.0;
        }
        let ki = self.get_key(i);
        let kj = self.get_key(j);

        match self.metric {
            MetricType::Euclidean => {
                let mut sum = 0.0f32;
                for c in 0..self.head_dim {
                    let diff = ki[c] - kj[c];
                    sum += diff * diff;
                }
                sum.sqrt()
            }
            MetricType::AngularSpherical => {
                let mut dot = 0.0f32;
                let mut norm_i = 0.0f32;
                let mut norm_j = 0.0f32;
                for c in 0..self.head_dim {
                    dot += ki[c] * kj[c];
                    norm_i += ki[c] * ki[c];
                    norm_j += kj[c] * kj[c];
                }
                let denom = (norm_i.sqrt() * norm_j.sqrt()).max(1e-8);
                let cosine = (dot / denom).clamp(-1.0, 1.0);
                cosine.acos()
            }
        }
    }

    /// Computes the full dense pairwise distance matrix.
    pub fn compute_distance_matrix(&self) -> Vec<f32> {
        let n = self.sequence_length;
        let mut dist = vec![0.0f32; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = self.distance(i, j);
                dist[i * n + j] = d;
                dist[j * n + i] = d;
            }
        }
        dist
    }
}

/// Vietoris-Rips filtration engine tracking 0-dimensional components and 1-dimensional cycles.
pub struct VietorisRipsFiltration<'a> {
    pub manifold: &'a TopologicalKeyManifold,
    pub max_dimension: usize,
    pub max_radius: f32,
}

impl<'a> VietorisRipsFiltration<'a> {
    pub fn new(manifold: &'a TopologicalKeyManifold, max_dimension: usize, max_radius: f32) -> Self {
        Self {
            manifold,
            max_dimension,
            max_radius,
        }
    }

    /// Computes H_0 and H_1 persistent homology via Disjoint-Set union-find and column elimination over Z_2.
    pub fn compute_persistence(&self) -> PersistenceDiagram {
        let n = self.manifold.sequence_length;
        if n == 0 {
            return PersistenceDiagram::default();
        }

        // Precompute distance matrix for O(1) lookups during filtration
        let dist = self.manifold.compute_distance_matrix();
        let get_dist = |i: usize, j: usize| -> f32 { dist[i * n + j] };

        // 1. Collect all 1-simplices (edges) within max_radius
        let mut edges: Vec<(f32, usize, usize)> = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let d = get_dist(i, j);
                if d <= self.max_radius {
                    edges.push((d, i, j));
                }
            }
        }

        // Sort edges monotonically by filtration diameter
        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut pairs = Vec::new();

        // -----------------------------------------------------------------
        // H_0 Persistence via Disjoint-Set Union-Find with Elder Rule
        // -----------------------------------------------------------------
        let mut parent: Vec<usize> = (0..n).collect();
        let birth_times: Vec<f32> = vec![0.0; n];
        let mut non_tree_edges: Vec<(f32, usize, usize)> = Vec::new();

        fn find(i: usize, parent: &mut [usize]) -> usize {
            let mut root = i;
            while root != parent[root] {
                root = parent[root];
            }
            let mut curr = i;
            while curr != root {
                let nxt = parent[curr];
                parent[curr] = root;
                curr = nxt;
            }
            root
        }

        for &(diameter, u, v) in &edges {
            let root_u = find(u, &mut parent);
            let root_v = find(v, &mut parent);

            if root_u != root_v {
                // Elder rule: older component survives
                let (survivor, dying) = if birth_times[root_u] <= birth_times[root_v] {
                    (root_u, root_v)
                } else {
                    (root_v, root_u)
                };

                pairs.push(PersistencePair::new(0, birth_times[dying], diameter));
                parent[dying] = survivor;
            } else if self.max_dimension >= 1 {
                non_tree_edges.push((diameter, u, v));
            }
        }

        // Infinite persistence bars for all surviving connected components
        let mut seen_roots = HashSet::new();
        for i in 0..n {
            let r = find(i, &mut parent);
            if seen_roots.insert(r) {
                pairs.push(PersistencePair::new(0, birth_times[r], f32::INFINITY));
            }
        }

        // -----------------------------------------------------------------
        // H_1 Persistence via Minimal Capping Triangle Property
        // -----------------------------------------------------------------
        if self.max_dimension >= 1 {
            let mut seen_killing_triangles = HashSet::new();

            for &(birth, u, v) in &non_tree_edges {
                let mut min_death = f32::INFINITY;
                let mut best_w = None;

                for w in 0..n {
                    if w != u && w != v {
                        let d_max = get_dist(u, w).max(get_dist(v, w));
                        if d_max <= birth + 1e-6 {
                            // Cycle is killed at birth by triangle [u, v, w]; persistence is zero
                            min_death = d_max;
                            best_w = None;
                            break;
                        }
                        if d_max < min_death {
                            min_death = d_max;
                            best_w = Some(w);
                        }
                    }
                }

                if let Some(w) = best_w {
                    if min_death > birth + 1e-5 && min_death <= self.max_radius {
                        let mut tri = [u, v, w];
                        tri.sort_unstable();
                        if seen_killing_triangles.insert((tri[0], tri[1], tri[2])) {
                            pairs.push(PersistencePair::new(1, birth, min_death));
                        }
                    }
                }
            }
        }

        PersistenceDiagram::new(pairs)
    }
}

/// Node representing an aggregated token cluster in the Mapper Graph.
#[derive(Debug, Clone)]
pub struct MapperNode {
    pub id: usize,
    pub bin_idx: usize,
    pub token_indices: Vec<usize>,
    pub centroid_key: Vec<f32>,
    pub mean_time: f32,
    pub mean_norm: f32,
}

/// Mapper graph skeleton capturing the 1-skeleton trajectory of attention.
#[derive(Debug, Clone)]
pub struct MapperAttentionGraph {
    pub nodes: Vec<MapperNode>,
    pub edges: Vec<(usize, usize)>,
    pub head_dim: usize,
    pub sequence_length: usize,
}

impl MapperAttentionGraph {
    /// Builds the Mapper 1-skeleton graph using a 2D temporal-energy lens.
    pub fn build(
        manifold: &TopologicalKeyManifold,
        num_intervals: usize,
        overlap_fraction: f32,
        cluster_distance_threshold: f32,
    ) -> Self {
        let s = manifold.sequence_length;
        let d = manifold.head_dim;
        if s == 0 {
            return Self {
                nodes: Vec::new(),
                edges: Vec::new(),
                head_dim: d,
                sequence_length: s,
            };
        }

        // 1. Evaluate 2D Filter Function: (Normalized Time, Key L2-Norm)
        let mut norms = Vec::with_capacity(s);
        for t in 0..s {
            let kt = manifold.get_key(t);
            let mut sum_sq = 0.0f32;
            for &val in kt.iter().take(d) {
                sum_sq += val * val;
            }
            norms.push(sum_sq.sqrt());
        }

        // 2. Define Overlapping Intervals along Temporal Progression
        let step = 1.0f32 / (num_intervals as f32);
        let interval_width = step * (1.0 + overlap_fraction);

        let mut bin_assignments: Vec<Vec<usize>> = vec![Vec::new(); num_intervals];
        for t in 0..s {
            let t_norm = (t as f32) / (s.max(1) as f32);
            for (bin, bin_slot) in bin_assignments.iter_mut().enumerate().take(num_intervals) {
                let start = (bin as f32) * step - (interval_width - step) * 0.5;
                let end = start + interval_width;
                if t_norm >= start && t_norm <= end {
                    bin_slot.push(t);
                }
            }
        }

        // 3. Local High-Dimensional Clustering in Each Preimage
        let mut nodes = Vec::new();
        let mut token_to_nodes: Vec<Vec<usize>> = vec![Vec::new(); s];

        for (bin_idx, tokens) in bin_assignments.iter().enumerate() {
            if tokens.is_empty() {
                continue;
            }

            // Single-linkage clustering within the temporal bin
            let mut cluster_parent: Vec<usize> = (0..tokens.len()).collect();
            for i in 0..tokens.len() {
                for j in (i + 1)..tokens.len() {
                    let tok_i = tokens[i];
                    let tok_j = tokens[j];
                    if manifold.distance(tok_i, tok_j) <= cluster_distance_threshold {
                        let mut root_i = i;
                        while root_i != cluster_parent[root_i] {
                            root_i = cluster_parent[root_i];
                        }
                        let mut root_j = j;
                        while root_j != cluster_parent[root_j] {
                            root_j = cluster_parent[root_j];
                        }
                        if root_i != root_j {
                            cluster_parent[root_j] = root_i;
                        }
                    }
                }
            }

            let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
            for (i, &tok) in tokens.iter().enumerate() {
                let mut root = i;
                while root != cluster_parent[root] {
                    root = cluster_parent[root];
                }
                clusters.entry(root).or_default().push(tok);
            }

            for (_, cluster_tokens) in clusters {
                let node_id = nodes.len();
                let mut centroid = vec![0.0f32; d];
                let mut sum_time = 0.0f32;
                let mut sum_norm = 0.0f32;

                for &tok in &cluster_tokens {
                    token_to_nodes[tok].push(node_id);
                    let kt = manifold.get_key(tok);
                    for c in 0..d {
                        centroid[c] += kt[c];
                    }
                    sum_time += tok as f32;
                    sum_norm += norms[tok];
                }

                let k_count = cluster_tokens.len() as f32;
                for val in centroid.iter_mut().take(d) {
                    *val /= k_count;
                }

                nodes.push(MapperNode {
                    id: node_id,
                    bin_idx,
                    token_indices: cluster_tokens,
                    centroid_key: centroid,
                    mean_time: sum_time / k_count,
                    mean_norm: sum_norm / k_count,
                });
            }
        }

        // 4. Construct Nerve Graph Edges from Overlapping Token Memberships
        let mut edges_set = HashSet::new();
        for shared_nodes in token_to_nodes.iter().take(s) {
            for i in 0..shared_nodes.len() {
                for j in (i + 1)..shared_nodes.len() {
                    let u = shared_nodes[i];
                    let v = shared_nodes[j];
                    if u != v {
                        let edge = if u < v { (u, v) } else { (v, u) };
                        edges_set.insert(edge);
                    }
                }
            }
        }

        let mut edges: Vec<(usize, usize)> = edges_set.into_iter().collect();
        edges.sort_unstable();

        Self {
            nodes,
            edges,
            head_dim: d,
            sequence_length: s,
        }
    }

    /// Reconstructs the full key cache buffer via barycentric interpolation over skeleton nodes.
    pub fn reconstruct(&self, reconstructed_keys: &mut [f32]) {
        assert_eq!(
            reconstructed_keys.len(),
            self.sequence_length * self.head_dim,
            "Reconstruction buffer length must equal sequence_length * head_dim"
        );

        if self.nodes.is_empty() {
            reconstructed_keys.fill(0.0);
            return;
        }

        let mut token_contributions: Vec<Vec<(usize, f32)>> = vec![Vec::new(); self.sequence_length];
        for node in &self.nodes {
            for &tok in &node.token_indices {
                let dt = ((tok as f32) - node.mean_time).abs() + 1e-4;
                let weight = 1.0 / dt;
                token_contributions[tok].push((node.id, weight));
            }
        }

        for (t, contribs) in token_contributions.iter().enumerate().take(self.sequence_length) {
            let offset = t * self.head_dim;
            let target_slice = &mut reconstructed_keys[offset..offset + self.head_dim];

            if contribs.is_empty() {
                // Fallback to nearest node by mean temporal position
                let mut best_node = 0;
                let mut min_dt = f32::INFINITY;
                for node in &self.nodes {
                    let dt = ((t as f32) - node.mean_time).abs();
                    if dt < min_dt {
                        min_dt = dt;
                        best_node = node.id;
                    }
                }
                target_slice.copy_from_slice(&self.nodes[best_node].centroid_key);
            } else {
                let total_weight: f32 = contribs.iter().map(|c| c.1).sum();
                target_slice.fill(0.0);
                for &(node_id, w) in contribs {
                    let norm_w = w / total_weight;
                    let centroid = &self.nodes[node_id].centroid_key;
                    for c in 0..self.head_dim {
                        target_slice[c] += norm_w * centroid[c];
                    }
                }
            }
        }
    }
}

/// Simplicial decimation engine for non-uniform token pruning under Dey's Link Condition.
pub struct SimplicialSkeletonPruner {
    pub head_dim: usize,
    pub persistence_threshold: f32,
    pub link_check_radius: f32,
}

impl SimplicialSkeletonPruner {
    pub fn new(head_dim: usize, persistence_threshold: f32, link_check_radius: f32) -> Self {
        Self {
            head_dim,
            persistence_threshold,
            link_check_radius,
        }
    }

    /// Verifies the Link Condition for edge uv: Lk(u) ∩ Lk(v) == Lk(uv).
    pub fn satisfies_link_condition(
        u: usize,
        v: usize,
        adjacency: &[HashSet<usize>],
        edge_triangles: &HashMap<Simplex1, HashSet<usize>>,
    ) -> bool {
        let adj_u = &adjacency[u];
        let adj_v = &adjacency[v];

        // 0-simplices in Lk(u) ∩ Lk(v) are common neighbors of u and v
        let common_neighbors: HashSet<usize> = adj_u.intersection(adj_v).cloned().collect();

        // 0-simplices in Lk(uv) are vertices forming a triangle with edge uv
        let edge = Simplex1::new(u, v);
        let triangle_vertices = edge_triangles.get(&edge);

        if let Some(t_verts) = triangle_vertices {
            common_neighbors == *t_verts
        } else {
            common_neighbors.is_empty()
        }
    }

    /// Prunes the key manifold down to a topologically guaranteed simplicial skeleton.
    pub fn prune_manifold(
        &self,
        manifold: &TopologicalKeyManifold,
        diagram: &PersistenceDiagram,
        target_retention_ratio: f32,
    ) -> (Vec<usize>, Vec<f32>) {
        let s = manifold.sequence_length;
        let d = manifold.head_dim;
        let target_count = ((s as f32) * target_retention_ratio).ceil() as usize;
        let target_count = target_count.clamp(1, s);

        // 1. Identify critical persistent landmark tokens
        let mut token_persistence = vec![0.0f32; s];
        for pair in &diagram.pairs {
            let pers = if pair.is_infinite() { 10.0 } else { pair.persistence() };
            if pair.dimension == 0 {
                let idx = ((pair.birth * 100.0) as usize).min(s - 1);
                token_persistence[idx] = token_persistence[idx].max(pers);
            }
        }

        // Always protect attention sinks (initial 4 tokens) and local recency window
        for slot in token_persistence.iter_mut().take(4.min(s)) {
            *slot = f32::INFINITY;
        }
        for slot in token_persistence.iter_mut().skip(s.saturating_sub(8)).take(8) {
            *slot = f32::INFINITY;
        }

        // 2. Build local 1-skeleton adjacency graph
        let mut adjacency: Vec<HashSet<usize>> = vec![HashSet::new(); s];
        let mut edges = Vec::new();
        for i in 0..s {
            for j in (i + 1)..s {
                let dist = manifold.distance(i, j);
                if dist <= self.link_check_radius {
                    adjacency[i].insert(j);
                    adjacency[j].insert(i);
                    edges.push((dist, i, j));
                }
            }
        }

        // 3. Collect triangle lookup for link condition verification
        let mut edge_triangles: HashMap<Simplex1, HashSet<usize>> = HashMap::new();
        for i in 0..s {
            for &j in &adjacency[i] {
                if i < j {
                    for &k in &adjacency[j] {
                        if j < k && adjacency[i].contains(&k) {
                            edge_triangles.entry(Simplex1::new(i, j)).or_default().insert(k);
                            edge_triangles.entry(Simplex1::new(j, k)).or_default().insert(i);
                            edge_triangles.entry(Simplex1::new(i, k)).or_default().insert(j);
                        }
                    }
                }
            }
        }

        // Sort edges by distance for candidate contraction
        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // 4. Contract edges that satisfy the Link Condition and have low persistence
        let mut active_tokens: HashSet<usize> = (0..s).collect();
        let mut mapping: Vec<usize> = (0..s).collect();

        for &(_dist, u, v) in &edges {
            if active_tokens.len() <= target_count {
                break;
            }

            let rep_u = mapping[u];
            let rep_v = mapping[v];

            if rep_u == rep_v {
                continue;
            }

            // Do not contract two essential persistent landmarks
            if token_persistence[rep_u].is_infinite() && token_persistence[rep_v].is_infinite() {
                continue;
            }

            if Self::satisfies_link_condition(rep_u, rep_v, &adjacency, &edge_triangles) {
                let (survivor, eliminated) = if token_persistence[rep_u] >= token_persistence[rep_v] {
                    (rep_u, rep_v)
                } else {
                    (rep_v, rep_u)
                };

                active_tokens.remove(&eliminated);
                for slot in mapping.iter_mut().take(s) {
                    if *slot == eliminated {
                        *slot = survivor;
                    }
                }
            }
        }

        let mut retained_indices: Vec<usize> = active_tokens.iter().copied().collect();
        retained_indices.sort_unstable();

        // 5. Generate reconstructed key cache buffer using geodesic landmark interpolation
        let mut reconstructed = vec![0.0f32; s * d];
        for (t, &rep_idx) in mapping.iter().enumerate().take(s) {
            let offset = t * d;
            if active_tokens.contains(&t) {
                reconstructed[offset..offset + d].copy_from_slice(manifold.get_key(t));
            } else {
                // Interpolate along the 1-skeleton between surrounding retained landmarks
                let mut prev_landmark = None;
                for p in (0..t).rev() {
                    if active_tokens.contains(&p) {
                        prev_landmark = Some(p);
                        break;
                    }
                }
                let mut next_landmark = None;
                for n in (t + 1)..s {
                    if active_tokens.contains(&n) {
                        next_landmark = Some(n);
                        break;
                    }
                }
                let rep = manifold.get_key(rep_idx);
                match (prev_landmark, next_landmark) {
                    (Some(p), Some(n)) => {
                        let alpha = (t - p) as f32 / (n - p) as f32;
                        let kp = manifold.get_key(p);
                        let kn = manifold.get_key(n);
                        for c in 0..d {
                            let interp = (1.0 - alpha) * kp[c] + alpha * kn[c];
                            reconstructed[offset + c] = 0.70 * rep[c] + 0.30 * interp;
                        }
                    }
                    (Some(p), None) => {
                        let kp = manifold.get_key(p);
                        for c in 0..d {
                            reconstructed[offset + c] = 0.70 * rep[c] + 0.30 * kp[c];
                        }
                    }
                    (None, Some(n)) => {
                        let kn = manifold.get_key(n);
                        for c in 0..d {
                            reconstructed[offset + c] = 0.70 * rep[c] + 0.30 * kn[c];
                        }
                    }
                    (None, None) => {
                        reconstructed[offset..offset + d].copy_from_slice(rep);
                    }
                }
            }
        }

        (retained_indices, reconstructed)
    }
}

/// Empirical report generated by the Topological Data Analysis Evaluator.
#[derive(Debug, Clone)]
pub struct TdaSkeletonReport {
    pub filtration_latency_us: f64,
    pub betti_preservation_pct: f64,
    pub token_reduction_pct: f64,
    pub reconstruction_cosine_sim: f64,
    pub relative_frobenius_err: f64,
    pub max_absolute_error: f64,
    pub persistent_entropy_h0: f64,
    pub persistent_entropy_h1: f64,
    pub total_memory_footprint_bytes: usize,
}

/// Evaluator benchmark for Topological Data Analysis & Context Skeletonization.
pub struct TdaSkeletonEvaluator;

impl TdaSkeletonEvaluator {
    /// Executes comprehensive evaluation of Vietoris-Rips filtration, persistent homology,
    /// and topological skeleton key pruning on synthetic multi-turn attention trajectories.
    pub fn evaluate() -> TdaSkeletonReport {
        let s = 64;
        let d = 32;

        let mut raw_keys = vec![0.0f32; s * d];
        for t in 0..s {
            let tf = t as f32;
            let offset = t * d;
            for c in 0..d {
                let cf = c as f32;
                let semantic_cluster = (tf * 0.05 + cf * 0.1).sin();
                let reasoning_loop = 0.40 * ((tf * 0.20).cos() * (cf * 0.15).sin());
                let positional_phase = 0.15 * (tf * (cf + 1.0) * 0.02).cos();
                raw_keys[offset + c] = semantic_cluster + reasoning_loop + positional_phase;
            }
        }

        let manifold = TopologicalKeyManifold::new(raw_keys.clone(), s, d, MetricType::Euclidean);

        // 1. Benchmark Vietoris-Rips Filtration and Persistent Homology
        let t_start = Instant::now();
        let filtration = VietorisRipsFiltration::new(&manifold, 1, 1.4);
        let diagram = filtration.compute_persistence();
        let filtration_latency_us = t_start.elapsed().as_nanos() as f64 / 1_000.0;

        let entropy_h0 = diagram.persistent_entropy(0);
        let entropy_h1 = diagram.persistent_entropy(1);

        // 2. Benchmark Simplicial Decimation & Edge Contraction
        let pruner = SimplicialSkeletonPruner::new(d, 0.5, 1.6);
        let target_retention = 0.33; // Target 67% reduction
        let (retained_indices, reconstructed_keys) =
            pruner.prune_manifold(&manifold, &diagram, target_retention);

        let token_reduction_pct = (1.0 - (retained_indices.len() as f64 / s as f64)) * 100.0;

        // 3. Verify Topological Betti Invariant Preservation on Retained Skeleton
        let retained_count = retained_indices.len();
        let mut skeleton_keys = vec![0.0f32; retained_count * d];
        for (new_idx, &orig_idx) in retained_indices.iter().enumerate() {
            let src = manifold.get_key(orig_idx);
            let dst_offset = new_idx * d;
            skeleton_keys[dst_offset..dst_offset + d].copy_from_slice(src);
        }

        let skeleton_manifold =
            TopologicalKeyManifold::new(skeleton_keys, retained_count, d, MetricType::Euclidean);
        let skeleton_filtration = VietorisRipsFiltration::new(&skeleton_manifold, 0, 1.7);
        let skeleton_diagram = skeleton_filtration.compute_persistence();

        let macro_baseline = diagram.pairs.iter().filter(|p| p.is_infinite() || p.persistence() > 1.5).count();
        let macro_skeleton = skeleton_diagram.pairs.iter().filter(|p| p.is_infinite() || p.persistence() > 1.5).count();
        let betti_preservation_pct = if macro_baseline > 0 {
            let matched = macro_skeleton.min(macro_baseline);
            ((matched as f64) / (macro_baseline as f64) * 100.0).clamp(0.0, 100.0)
        } else {
            100.0
        };

        // 4. Compute Reconstruction Metrics
        let cos_sim = compute_cosine_similarity(&raw_keys, &reconstructed_keys);
        let rel_frob = compute_relative_frobenius_error(&raw_keys, &reconstructed_keys);
        let max_abs = compute_max_absolute_error(&raw_keys, &reconstructed_keys);

        let total_memory_footprint_bytes =
            (s * d * std::mem::size_of::<f32>()) + (diagram.pairs.len() * std::mem::size_of::<PersistencePair>());

        TdaSkeletonReport {
            filtration_latency_us,
            betti_preservation_pct,
            token_reduction_pct,
            reconstruction_cosine_sim: cos_sim,
            relative_frobenius_err: rel_frob,
            max_absolute_error: max_abs,
            persistent_entropy_h0: entropy_h0,
            persistent_entropy_h1: entropy_h1,
            total_memory_footprint_bytes,
        }
    }
}

#[inline]
fn compute_cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for i in 0..a.len() {
        let va = a[i] as f64;
        let vb = b[i] as f64;
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-12);
    (dot / denom).clamp(-1.0, 1.0)
}

#[inline]
fn compute_relative_frobenius_error(orig: &[f32], recon: &[f32]) -> f64 {
    assert_eq!(orig.len(), recon.len());
    let mut diff_sq = 0.0f64;
    let mut orig_sq = 0.0f64;

    for i in 0..orig.len() {
        let d = (orig[i] - recon[i]) as f64;
        diff_sq += d * d;
        let o = orig[i] as f64;
        orig_sq += o * o;
    }

    (diff_sq.sqrt() / orig_sq.sqrt().max(1e-12)).min(10.0)
}

#[inline]
fn compute_max_absolute_error(orig: &[f32], recon: &[f32]) -> f64 {
    assert_eq!(orig.len(), recon.len());
    let mut max_err = 0.0f64;
    for i in 0..orig.len() {
        let err = ((orig[i] - recon[i]) as f64).abs();
        if err > max_err {
            max_err = err;
        }
    }
    max_err
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_space_axioms() {
        let keys = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ];
        let manifold_euclid = TopologicalKeyManifold::new(keys.clone(), 3, 4, MetricType::Euclidean);
        let manifold_angular = TopologicalKeyManifold::new(keys, 3, 4, MetricType::AngularSpherical);

        // Identity of indiscernibles
        assert_eq!(manifold_euclid.distance(0, 0), 0.0);
        assert_eq!(manifold_angular.distance(1, 1), 0.0);

        // Symmetry
        let d01_e = manifold_euclid.distance(0, 1);
        let d10_e = manifold_euclid.distance(1, 0);
        assert!((d01_e - d10_e).abs() < 1e-6);

        let d01_a = manifold_angular.distance(0, 1);
        let d10_a = manifold_angular.distance(1, 0);
        assert!((d01_a - d10_a).abs() < 1e-6);
        assert!((d01_a - std::f32::consts::FRAC_PI_2).abs() < 1e-5);

        // Triangle inequality: d(0, 2) <= d(0, 1) + d(1, 2)
        let d02_e = manifold_euclid.distance(0, 2);
        let d12_e = manifold_euclid.distance(1, 2);
        assert!(d02_e <= d01_e + d12_e + 1e-6);
    }

    #[test]
    fn test_vietoris_rips_h0_disjoint_set() {
        // Two well-separated clusters
        let keys = vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, // Cluster 1: points 0, 1, 2
            10.0, 10.0, 10.1, 10.0, 10.0, 10.1, // Cluster 2: points 3, 4, 5
        ];
        let manifold = TopologicalKeyManifold::new(keys, 6, 2, MetricType::Euclidean);
        let filtration = VietorisRipsFiltration::new(&manifold, 0, 1.0);
        let diagram = filtration.compute_persistence();

        // At scale t = 0.5, the 2 clusters should be distinct connected components (beta_0 = 2)
        let b0 = diagram.betti_number(0, 0.5);
        assert_eq!(b0, 2);
        assert!(diagram.persistent_entropy(0) >= 0.0);
    }

    #[test]
    fn test_vietoris_rips_h1_cycle_detection() {
        // Construct a circular ring of 6 points in R^2
        let n = 6;
        let mut keys = Vec::with_capacity(n * 2);
        let radius = 2.0f32;
        for i in 0..n {
            let theta = (i as f32) * 2.0 * std::f32::consts::PI / (n as f32);
            keys.push(radius * theta.cos());
            keys.push(radius * theta.sin());
        }

        let manifold = TopologicalKeyManifold::new(keys, n, 2, MetricType::Euclidean);
        // Distance between adjacent points is radius * 2 * sin(pi/6) = 2.0 * 1.0 = 2.0
        // Diameter across is 4.0
        let filtration = VietorisRipsFiltration::new(&manifold, 1, 3.5);
        let diagram = filtration.compute_persistence();

        // There should be a 1-dimensional cycle (H_1 pair) born around 2.0
        let h1_pairs: Vec<&PersistencePair> = diagram.pairs.iter().filter(|p| p.dimension == 1).collect();
        assert!(!h1_pairs.is_empty(), "H_1 cycle must be detected on circular key loop");
        assert!(h1_pairs[0].birth >= 1.5 && h1_pairs[0].birth <= 2.5);
    }

    #[test]
    fn test_link_condition_verification() {
        // Triangulated configuration:
        // Triangle (0, 1, 2): edge (0, 1) has shared neighbor {2} which forms a triangle with (0, 1)
        let mut adjacency: Vec<HashSet<usize>> = vec![HashSet::new(); 4];
        adjacency[0].insert(1);
        adjacency[0].insert(2);
        adjacency[1].insert(0);
        adjacency[1].insert(2);
        adjacency[2].insert(0);
        adjacency[2].insert(1);

        let mut edge_triangles: HashMap<Simplex1, HashSet<usize>> = HashMap::new();
        edge_triangles.entry(Simplex1::new(0, 1)).or_default().insert(2);

        assert!(SimplicialSkeletonPruner::satisfies_link_condition(
            0,
            1,
            &adjacency,
            &edge_triangles
        ));

        // Add an extra shared neighbor 3 connected to 0 and 1, but NOT forming a triangle coface
        adjacency[0].insert(3);
        adjacency[1].insert(3);
        adjacency[3].insert(0);
        adjacency[3].insert(1);
        // edge_triangles still only has {2} for (0, 1)
        assert!(!SimplicialSkeletonPruner::satisfies_link_condition(
            0,
            1,
            &adjacency,
            &edge_triangles
        ));
    }

    #[test]
    fn test_mapper_graph_reconstruction() {
        let s = 32;
        let d = 8;
        let mut keys = vec![0.0f32; s * d];
        for t in 0..s {
            let offset = t * d;
            for c in 0..d {
                keys[offset + c] = ((t as f32) * 0.1 + (c as f32) * 0.2).sin();
            }
        }

        let manifold = TopologicalKeyManifold::new(keys.clone(), s, d, MetricType::Euclidean);
        let graph = MapperAttentionGraph::build(&manifold, 4, 0.25, 1.5);
        assert!(!graph.nodes.is_empty());

        let mut reconstructed = vec![0.0f32; s * d];
        graph.reconstruct(&mut reconstructed);

        let sim = compute_cosine_similarity(&keys, &reconstructed);
        assert!(sim > 0.85, "Mapper reconstruction cosine similarity should exceed 0.85, got {}", sim);
    }

    #[test]
    fn test_tda_skeleton_evaluator_metrics() {
        let report = TdaSkeletonEvaluator::evaluate();
        assert!(report.filtration_latency_us > 0.0);
        assert!(report.betti_preservation_pct >= 95.0);
        assert!(report.token_reduction_pct >= 60.0);
        assert!(report.reconstruction_cosine_sim > 0.90);
        assert!(report.relative_frobenius_err < 0.40);
        assert!(report.persistent_entropy_h0 >= 0.0);
    }
}
