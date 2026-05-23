// forest.rs — file system scanning + biological growth layout
//
// Two-pass algorithm:
//   1. Scan filesystem into a tree of `Node` (top-down).
//   2. Post-order traversal: compute subtree weights, then assign positions
//      from root outwards using phototropism-inspired growth.
//
// Layout system:
//   Each Node has DEFAULT geometry fields (position/branch_dir/branch_length)
//   computed for the global tree starting at forest.root. For drill-down
//   scenes we generate ALTERNATIVE layouts via `generate_layout(root_idx)`
//   that treat the chosen folder as the new root of its own forest — so its
//   direct subdirectories become independent trees of a new grove, files
//   become ground litter, etc. These alternative layouts are stored in a
//   `Layout` struct (not on Node), enabling many coexisting layouts per
//   forest.

use ahash::AHasher;
use glam::Vec3;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;
use walkdir::WalkDir;

/// Sentinel for `parent` field on the root node — no parent.
pub const NO_PARENT: u32 = u32::MAX;

/// Core node data — fields accessed on every render / picking pass.
/// Hot path. Kept compact so 4M of them fit in cache-friendly memory.
///
/// Geometry (position/branch_dir/branch_length) lives in `Forest::main_layout`
/// and per-scene `Layout`s, NOT here — saves 28 B per node.
///
/// Children are stored in `Forest::children_pool` as a flat Vec<u32>;
/// each node knows where its children start (`children_start`) and how many
/// (`children_count`). Eliminates 4M individual heap allocations + their
/// overhead (each Box<[u32]> needs ~16 B header + per-alloc rounding).
pub struct NodeCore {
    /// File or directory name. `Box<str>` saves the capacity word vs String.
    pub name: Box<str>,

    /// File size in bytes; for directory = aggregated subtree size.
    pub size: u64,
    /// Unix mtime in seconds. u32 — fits everything up to year 2106.
    pub mtime_secs: u32,
    /// Directory tree depth (0 = scan root). u8 — scan caps at 32.
    pub depth: u8,
    /// 1 = directory, 0 = file. u8 to keep struct tightly packed.
    pub is_dir_flag: u8,
    /// Index of parent in Forest::nodes, or NO_PARENT for the scan root.
    pub parent: u32,

    /// Start of this node's children in `Forest::children_pool`.
    pub children_start: u32,
    /// Number of children (slice length). 0 for leaf files / empty dirs.
    pub children_count: u32,

    /// Total files + dirs in this subtree (including self).
    pub subtree_count: u32,

    /// DFS preorder index. Guarantees the subtree rooted at this node
    /// occupies indices [dfs_pre, dfs_end), so the GPU can highlight a
    /// whole subtree with a single range-check uniform.
    pub dfs_pre: u32,
    pub dfs_end: u32,
}

/// Type alias for backward compatibility — the old `Node` name still works
/// throughout the rest of the codebase.
pub type Node = NodeCore;

impl NodeCore {
    pub fn is_dir(&self) -> bool { self.is_dir_flag != 0 }
    pub fn parent_opt(&self) -> Option<u32> {
        if self.parent == NO_PARENT { None } else { Some(self.parent) }
    }
}

pub struct Forest {
    pub nodes: Vec<NodeCore>,

    /// Flat pool of all children indices in the forest. Each NodeCore
    /// references its children as `pool[children_start..children_start+children_count]`.
    /// Replaces per-node Box<[u32]> heap allocs — saves ~16 B overhead and
    /// allocator rounding per node × N nodes.
    pub children_pool: Vec<u32>,

    pub root: usize,
    pub newest_mtime: u32,
    pub oldest_mtime: u32,

    /// Per-node age01 (precomputed; see `compute_age01`).
    pub age01: Vec<f32>,

    /// The "main scene" layout — global grove geometry. Shared via Arc so
    /// the main Scene can hand it to GPU builders / picker without
    /// duplicating the data. Built once during `Forest::scan`.
    pub main_layout: Arc<Layout>,
}

impl Forest {
    /// Get a slice view of `idx`'s children indices.
    pub fn children_of(&self, idx: usize) -> &[u32] {
        let n = &self.nodes[idx];
        let start = n.children_start as usize;
        let end = start + n.children_count as usize;
        &self.children_pool[start..end]
    }
}

/// Per-node positional geometry — separate from Node so we can have multiple
/// coexisting layouts of the same forest (main grove vs drill-down groves).
#[derive(Clone, Copy, Debug)]
pub struct NodeGeom {
    pub position: Vec3,
    pub branch_dir: Vec3,
    pub branch_length: f32,
}

/// An alternative layout for a subset of the forest.
///
/// Two storage modes:
///   * `Dense` — Vec indexed by node_idx. Used for the main grove (covers
///     every node in the forest). Memory: `28 * N` bytes, no HashMap overhead.
///   * `Sparse` — HashMap keyed by node_idx. Used for drill-down scenes
///     where only a subtree is laid out.
pub enum LayoutStorage {
    Dense(Vec<NodeGeom>),
    Sparse(HashMap<usize, NodeGeom>),
}

pub struct Layout {
    pub root_idx: usize,
    pub storage: LayoutStorage,
}

impl Layout {
    pub fn new_dense(root_idx: usize, n: usize) -> Self {
        Layout {
            root_idx,
            storage: LayoutStorage::Dense(vec![NodeGeom {
                position: Vec3::ZERO,
                branch_dir: Vec3::Y,
                branch_length: 0.0,
            }; n]),
        }
    }
    pub fn new_sparse(root_idx: usize) -> Self {
        Layout {
            root_idx,
            storage: LayoutStorage::Sparse(HashMap::new()),
        }
    }
    /// Look up the geometry of `idx` in this layout. Panics if not present —
    /// callers should only ask about nodes inside the subtree this layout
    /// covers.
    pub fn get(&self, idx: usize) -> NodeGeom {
        match &self.storage {
            LayoutStorage::Dense(v) => v[idx],
            LayoutStorage::Sparse(m) => *m.get(&idx).expect("layout missing node"),
        }
    }
    pub fn contains(&self, idx: usize) -> bool {
        match &self.storage {
            LayoutStorage::Dense(_) => true,  // dense covers everything
            LayoutStorage::Sparse(m) => m.contains_key(&idx),
        }
    }
    /// Insert or overwrite a node's geometry.
    pub fn insert(&mut self, idx: usize, g: NodeGeom) {
        match &mut self.storage {
            LayoutStorage::Dense(v) => v[idx] = g,
            LayoutStorage::Sparse(m) => { m.insert(idx, g); }
        }
    }
}

/// Fetch the geometry of `node_idx`. If a `layout` is supplied and contains
/// the index, that layout wins (drill-down scenes). Otherwise we fall back
/// to the forest's `main_layout` — the global grove geometry computed once
/// at scan time.
///
/// Since geometry is no longer stored on Node, a Layout MUST be available
/// for any node we want to render. The main_layout covers every node, so
/// this is always safe.
pub fn geom_of(forest: &Forest, layout: Option<&Layout>, node_idx: usize) -> NodeGeom {
    if let Some(l) = layout {
        if l.contains(node_idx) {
            return l.get(node_idx);
        }
    }
    // Fallback: main grove layout. Covers all nodes in the forest.
    forest.main_layout.get(node_idx)
}

impl Forest {
    /// Scan filesystem at `root_path` up to `max_depth`. Use a depth cap
    /// during prototyping; production version should be incremental + threaded.
    pub fn scan(root_path: &Path, max_depth: usize) -> Self {
        // ---- Phase 1: scan + build with growable Vec<u32> children. ----
        // We can't use Box<[u32]> while pushing children, so we build with
        // Vec<u32> first, then convert to Box<[u32]> after scan finishes.
        struct ScanNode {
            name: Box<str>,
            size: u64,
            mtime_secs: u32,
            depth: u8,
            is_dir_flag: u8,
            parent: u32,
            children: Vec<u32>,
        }
        let mut tmp: Vec<ScanNode> = Vec::with_capacity(1024);
        let mut path_to_idx: ahash::AHashMap<PathBuf, u32> = ahash::AHashMap::new();

        let root_canon = root_path.canonicalize().unwrap_or_else(|_| root_path.to_path_buf());

        for entry in WalkDir::new(&root_canon)
            .max_depth(max_depth)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path().to_path_buf();
            let metadata = match entry.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };
            let is_dir = metadata.is_dir();
            let size = if is_dir { 0 } else { metadata.len() };
            let mtime_secs = metadata
                .modified()
                .ok()
                .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as u32)
                .unwrap_or(0);

            let name: Box<str> = path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.to_string_lossy().into_owned())
                .into_boxed_str();

            let parent_idx = path
                .parent()
                .and_then(|p| path_to_idx.get(p).copied())
                .unwrap_or(NO_PARENT);

            let depth = entry.depth().min(255) as u8;

            let idx = tmp.len() as u32;
            tmp.push(ScanNode {
                name,
                size,
                mtime_secs,
                depth,
                is_dir_flag: if is_dir { 1 } else { 0 },
                parent: parent_idx,
                children: Vec::new(),
            });

            if parent_idx != NO_PARENT {
                tmp[parent_idx as usize].children.push(idx);
            }
            path_to_idx.insert(path, idx);
        }
        drop(path_to_idx);

        // ---- Phase 2: flatten children into a single pool, build NodeCore ----
        // Sum total children for pool capacity. Then iterate ScanNodes in
        // order, writing into pool and tracking each one's [start, end).
        let total_children: usize = tmp.iter().map(|s| s.children.len()).sum();
        let mut children_pool: Vec<u32> = Vec::with_capacity(total_children);
        let mut nodes: Vec<NodeCore> = Vec::with_capacity(tmp.len());
        for sn in tmp {
            let start = children_pool.len() as u32;
            let count = sn.children.len() as u32;
            children_pool.extend_from_slice(&sn.children);
            nodes.push(NodeCore {
                name: sn.name,
                size: sn.size,
                mtime_secs: sn.mtime_secs,
                depth: sn.depth,
                is_dir_flag: sn.is_dir_flag,
                parent: sn.parent,
                children_start: start,
                children_count: count,
                subtree_count: 1,
                dfs_pre: 0,
                dfs_end: 0,
            });
        }
        nodes.shrink_to_fit();
        children_pool.shrink_to_fit();

        // Find root: the first node we added (WalkDir yields root first).
        let root = 0;

        // Post-order: aggregate sizes and counts up the tree.
        aggregate_subtree(&mut nodes, &children_pool, root);

        // DFS preorder: each subtree becomes a contiguous range [pre, end).
        compute_dfs_indices(&mut nodes, &children_pool, root);

        // Time range for colormap.
        let (mut oldest, mut newest) = (u32::MAX, 0u32);
        for n in &nodes {
            if !n.is_dir() && n.mtime_secs > 0 {
                if n.mtime_secs < oldest { oldest = n.mtime_secs; }
                if n.mtime_secs > newest { newest = n.mtime_secs; }
            }
        }
        if oldest == u32::MAX { oldest = 0; }

        let age01 = compute_age01(&nodes, &children_pool, root);

        let placeholder = Arc::new(Layout::new_sparse(root));
        let mut forest = Forest {
            nodes,
            children_pool,
            root,
            newest_mtime: newest,
            oldest_mtime: oldest,
            age01,
            main_layout: placeholder,
        };
        let layout = generate_layout(&forest, root);
        forest.main_layout = Arc::new(layout);
        forest
    }
}

/// Compute per-node age01: for files the percentile rank of their mtime (1.0
/// oldest, 0.0 newest), for directories the mean rank over their descendants.
/// One-shot pass over the forest, ~O(N log N) from the sort.
fn compute_age01(nodes: &[Node], pool: &[u32], root: usize) -> Vec<f32> {
    let mut mtimes: Vec<u32> = nodes.iter()
        .filter(|n| !n.is_dir() && n.mtime_secs > 0)
        .map(|n| n.mtime_secs)
        .collect();
    mtimes.sort_unstable();
    let total = mtimes.len().max(1) as f32;
    let rank01 = |t: u32| -> f32 {
        if mtimes.is_empty() { return 0.5; }
        let pos = mtimes.partition_point(|&x| x < t);
        1.0 - (pos as f32 / total)
    };

    let n_nodes = nodes.len();
    let mut age_sum = vec![0.0_f32; n_nodes];
    let mut age_cnt = vec![0u32; n_nodes];
    let mut order: Vec<usize> = Vec::with_capacity(n_nodes);
    let mut stack: Vec<usize> = vec![root];
    while let Some(idx) = stack.pop() {
        order.push(idx);
        let n = &nodes[idx];
        let cs = n.children_start as usize;
        let ce = cs + n.children_count as usize;
        for &c in &pool[cs..ce] { stack.push(c as usize); }
    }
    for &idx in order.iter().rev() {
        let n = &nodes[idx];
        if !n.is_dir() {
            let r = if n.mtime_secs > 0 { rank01(n.mtime_secs) } else { 0.5 };
            age_sum[idx] = r; age_cnt[idx] = 1;
        } else {
            let mut s = 0.0; let mut c = 0u32;
            let cs = n.children_start as usize;
            let ce = cs + n.children_count as usize;
            for &ci in &pool[cs..ce] {
                let ci = ci as usize;
                s += age_sum[ci]; c += age_cnt[ci];
            }
            age_sum[idx] = s; age_cnt[idx] = c;
        }
    }
    (0..n_nodes).map(|i| {
        if age_cnt[i] > 0 { age_sum[i] / age_cnt[i] as f32 } else { 0.5 }
    }).collect()
}

// =====================================================================
// Layout generation — phototropism-inspired growth, now Layout-centric
// =====================================================================

/// Build a Layout for the subtree rooted at `root_idx`. Treats `root_idx` as
/// "the ground" — its direct subdirectories become independent trees in a
/// new grove, direct files become ground litter (scattered around origin in
/// a low disc), branches grow with phototropism as in the global layout.
///
/// Special case: if `root_idx` has NO direct subdirectories (it's a leaf
/// folder containing only files), we fall back to a single-tree layout —
/// the folder itself becomes one trunk with files in its crown.
pub fn generate_layout(forest: &Forest, root_idx: usize) -> Layout {
    // Storage mode pick: dense Vec<NodeGeom> vs sparse HashMap.
    //
    // Dense is always more compact PER ENTRY (28 B vs ~60 B with HashMap
    // overhead), but allocates room for EVERY node in the forest, even ones
    // not in this subtree. So:
    //   - If the subtree covers most of the forest (e.g. drill into Users
    //     when Users is ~75% of C:\), dense is far cheaper than sparse.
    //   - For small subtrees, sparse wins (HashMap only stores what's there).
    //
    // Threshold: if subtree covers ≥1/3 of the forest, use dense.
    // For the main grove (covers everything) this is always dense.
    let subtree_size = forest.nodes[root_idx].subtree_count as usize;
    let use_dense = subtree_size * 3 >= forest.nodes.len();
    let mut layout = if use_dense {
        Layout::new_dense(root_idx, forest.nodes.len())
    } else {
        Layout::new_sparse(root_idx)
    };

    // The root sits at origin; not rendered as a branch.
    layout.insert(root_idx, NodeGeom {
        position: Vec3::ZERO,
        branch_dir: Vec3::Y,
        branch_length: 0.0,
    });

    let top_dirs: Vec<usize> = forest.children_of(root_idx)
        .iter()
        .map(|&c| c as usize)
        .filter(|&c| forest.nodes[c].is_dir())
        .collect();

    let root_files: Vec<usize> = forest.children_of(root_idx)
        .iter()
        .map(|&c| c as usize)
        .filter(|&c| !forest.nodes[c].is_dir())
        .collect();

    // --------- Special case: leaf folder (only files) ---------
    // Treat the folder as one tree: a trunk straight up, with files as crown.
    // This matches user expectation "одно дерево как сейчас".
    if top_dirs.is_empty() && !root_files.is_empty() {
        let n_files = root_files.len() as f32;
        let trunk_len = (3.0 + n_files.sqrt() * 0.8).clamp(3.0, 12.0);
        // Folder itself: trunk straight up from origin.
        layout.insert(root_idx, NodeGeom {
            position: Vec3::ZERO,
            branch_dir: Vec3::Y,
            branch_length: trunk_len,
        });
        let tip = Vec3::Y * trunk_len;
        let cloud_radius = 0.15 + (n_files + 1.0).ln() * 0.18;
        let crown_seed = hash_node(forest, root_idx as usize);
        for &fi in &root_files {
            let leaf_seed = hash_node(forest, fi as usize);
            let offset = organic_crown_sample(leaf_seed, crown_seed, cloud_radius);
            layout.insert(fi, NodeGeom {
                position: tip + offset,
                branch_dir: Vec3::Y,
                branch_length: 0.0,
            });
        }
        return layout;
    }

    // --------- Place top-level directories as separate tree trunks ---------
    // Same algorithm as the global Forest::grow path: estimate canopy widths,
    // sort by size, place on Fibonacci spiral with adaptive spacing.
    let mut trees_with_size: Vec<(usize, f32)> = top_dirs.iter()
        .map(|&idx| {
            let sub = forest.nodes[idx].subtree_count.max(1) as f32;
            let canopy_r = (0.6 * sub.sqrt()).clamp(2.0, 30.0);
            (idx, canopy_r)
        })
        .collect();
    trees_with_size.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let breathing_space = 4.0_f32;
    let mut placed_positions: Vec<(Vec3, f32)> = Vec::with_capacity(trees_with_size.len());

    // Spatial hash to accelerate collision checks against already-placed trees.
    // Without this, the inner `placed_positions.iter().all(...)` makes the
    // outer loop O(N^2) and a folder with 30k+ top-level subdirs (e.g. WinSxS)
    // takes 25+ seconds. With a grid lookup we touch only ~10 nearby cells
    // per check → near-linear scaling, hundreds of ms for the same input.
    //
    // Cell size: we pick a value comfortably larger than the typical canopy
    // radius (3..30 range) so a tree intersects at most 2-3 cells along each
    // axis. 16 units works well for the (-300..300)-ish placement range we
    // see in practice; index range is then ~37x37 = small HashMap.
    let cell_size: f32 = 16.0;
    let cell_of = |p: Vec3| -> (i32, i32) {
        ((p.x / cell_size).floor() as i32, (p.z / cell_size).floor() as i32)
    };
    let mut grid: std::collections::HashMap<(i32, i32), Vec<u32>> =
        std::collections::HashMap::with_capacity(trees_with_size.len());
    let mut used_area: f32 = 0.0;

    for (i, &(tree_idx, canopy_r)) in trees_with_size.iter().enumerate() {
        let golden_angle = 2.399963_f32;
        let mut try_r = (used_area / std::f32::consts::PI).sqrt() + canopy_r + breathing_space;

        let mut placed = false;
        let mut chosen = Vec3::ZERO;
        for attempt in 0..32 {
            let theta = (i as f32) * golden_angle + (attempt as f32) * 0.5;
            let candidate = Vec3::new(try_r * theta.cos(), 0.0, try_r * theta.sin());

            // Only check trees in nearby grid cells. Reach = max possible
            // sum of radii + breathing_space, rounded up in cells.
            // We expand a 3x3 cell neighbourhood — enough since cell_size
            // (16) > typical canopy_r (3..30 with 30 rare).
            let (cx, cz) = cell_of(candidate);
            let max_other_r = 30.0_f32; // worst case canopy
            let reach_cells = ((canopy_r + max_other_r + breathing_space) / cell_size).ceil() as i32 + 1;
            let mut ok = true;
            'outer: for dz in -reach_cells..=reach_cells {
                for dx in -reach_cells..=reach_cells {
                    if let Some(bucket) = grid.get(&(cx + dx, cz + dz)) {
                        for &pi in bucket {
                            let (p, other_r) = placed_positions[pi as usize];
                            let d = (candidate - p).length();
                            if d <= canopy_r + other_r + breathing_space {
                                ok = false;
                                break 'outer;
                            }
                        }
                    }
                }
            }

            if ok {
                chosen = candidate;
                placed = true;
                break;
            }
            try_r += canopy_r * 0.3;
        }
        if !placed {
            let fallback_r = try_r + canopy_r * 2.0;
            let theta = (i as f32) * golden_angle;
            chosen = Vec3::new(fallback_r * theta.cos(), 0.0, fallback_r * theta.sin());
        }
        let new_idx = placed_positions.len() as u32;
        placed_positions.push((chosen, canopy_r));
        grid.entry(cell_of(chosen)).or_default().push(new_idx);
        used_area += std::f32::consts::PI * canopy_r * canopy_r;

        // Trunk length & lean (deterministic from path).
        // Reverted to canopy-based formula. The earlier sqrt(subtree) variant
        // exploded for huge subtrees (110+ units, putting crowns off-screen).
        // Stem visibility under crowns is now solved by render order — branches
        // draw AFTER points so they sit on top of leaf clouds instead of
        // being buried under them. See renderer.rs render().
        let trunk_len = (canopy_r * 1.3).max(4.0).min(40.0);
        let seed = hash_node(forest, tree_idx as usize);
        let (lu1, lu2, _) = split_seed(seed);
        let lean_angle = (lu1 * 0.25).max(0.0);
        let lean_dir_rad = lu2 * std::f32::consts::TAU;
        let trunk_dir = Vec3::new(
            lean_angle.sin() * lean_dir_rad.cos(),
            lean_angle.cos(),
            lean_angle.sin() * lean_dir_rad.sin(),
        ).normalize_or(Vec3::Y);

        layout.insert(tree_idx, NodeGeom {
            position: chosen,
            branch_dir: trunk_dir,
            branch_length: trunk_len,
        });
    }

    // --------- BFS-expand each tree using phototropic algorithm ---------
    let mut queue: Vec<usize> = top_dirs.clone();
    let mut head = 0;
    while head < queue.len() {
        let parent_idx = queue[head];
        head += 1;
        layout_children_into(forest, &mut layout, parent_idx);
        for &c in forest.children_of(parent_idx).iter() {
            let c = c as usize;
            if forest.nodes[c].is_dir() {
                queue.push(c);
            }
        }
    }

    // --------- Root files: scatter around as "почва" (ground litter) ---------
    // Direct files of the chosen folder become a low scattered disc around
    // origin, between the trees. They're displayed but don't cluster on any
    // single trunk — they belong to the folder itself, not a subtree.
    if !root_files.is_empty() {
        let crown_seed = hash_node(forest, root_idx as usize);
        // Scatter disc radius: tied to the grove size so files spread between
        // the trees rather than piling up at origin or escaping the grove.
        let grove_extent: f32 = placed_positions.iter()
            .map(|(p, r)| (p.length() + r))
            .fold(2.0_f32, f32::max);
        let n_files = root_files.len() as f32;
        let disc_radius = (grove_extent * 0.7).max(1.0 + n_files.sqrt() * 0.3);

        for &fi in &root_files {
            let leaf_seed = hash_node(forest, fi as usize);
            let (u1, u2, u3) = split_seed(leaf_seed);
            // Uniform-in-disc: r = R * sqrt(u), theta uniform.
            let r = disc_radius * u1.sqrt();
            let theta = u2 * std::f32::consts::TAU;
            // Small vertical jitter so files don't all sit at y=0 in a perfect plane.
            let y_jitter = (u3 - 0.5) * 0.3;
            let pos = Vec3::new(r * theta.cos(), y_jitter, r * theta.sin());
            let _ = crown_seed; // crown_seed unused in disc layout; kept for future shaping
            layout.insert(fi, NodeGeom {
                position: pos,
                branch_dir: Vec3::Y,
                branch_length: 0.0,
            });
        }
    }

    layout
}

/// Layout the children of `parent_idx` into the given Layout. Reads the
/// parent's geometry from the Layout (must already be placed). Same
/// phototropism algorithm as the global Forest::grow — only the storage
/// is different.
fn layout_children_into(forest: &Forest, layout: &mut Layout, parent_idx: usize) {
    let parent_g = layout.get(parent_idx);
    let parent_pos = parent_g.position;
    let parent_dir = parent_g.branch_dir;
    let parent_len = parent_g.branch_length;
    let parent_depth = forest.nodes[parent_idx].depth;

    let tip = parent_pos + parent_dir * parent_len;

    let (dir_children, file_children): (Vec<usize>, Vec<usize>) = forest.children_of(parent_idx)
        .iter()
        .map(|&c| c as usize)
        .partition(|&c| forest.nodes[c].is_dir());

    // ---- subdirectories as branches ----
    let base_angle = (85.0_f32 - (parent_depth as f32) * 5.0).max(30.0).to_radians();
    let mut placed_dirs: Vec<Vec3> = Vec::with_capacity(dir_children.len());

    for (_i, &child_idx) in dir_children.iter().enumerate() {
        const BASE_LEN: f32 = 0.3;
        const FILE_W: f32 = 0.8;
        const DIR_W: f32 = 1.5;
        const SUBTREE_W: f32 = 0.4;
        let n_direct_files = forest.children_of(child_idx).iter()
            .filter(|&&c| !forest.nodes[c as usize].is_dir()).count() as f32;
        let n_direct_dirs = forest.children_of(child_idx).iter()
            .filter(|&&c| forest.nodes[c as usize].is_dir()).count() as f32;
        let subtree = forest.nodes[child_idx].subtree_count.max(1) as f32;
        let child_len = BASE_LEN
            + FILE_W    * (n_direct_files + 1.0).ln()
            + DIR_W     * (n_direct_dirs  + 1.0).ln()
            + SUBTREE_W * subtree.ln();
        let parent_cap = parent_len * 0.85;
        let child_len = child_len.clamp(BASE_LEN, parent_cap.max(BASE_LEN * 2.0));

        let seed = hash_node(forest, child_idx as usize);
        let rand_dir = deterministic_dir_in_cone(seed, parent_dir, base_angle);

        // Phototropism: push away from recently-placed siblings.
        // Capped to last 16 — full O(N^2) sweep would crawl on huge fanout
        // folders (e.g. node_modules subfolders with 10k+ direct children).
        // Recency window gives essentially the same visual result.
        let mut dir = rand_dir;
        let start = placed_dirs.len().saturating_sub(16);
        for sib_dir in &placed_dirs[start..] {
            let dot = dir.dot(*sib_dir);
            if dot > 0.0 {
                dir -= *sib_dir * (dot * 0.6);
            }
        }
        dir = dir.normalize_or(parent_dir);
        dir = (dir + Vec3::Y * 0.1).normalize_or(dir);
        placed_dirs.push(dir);

        layout.insert(child_idx, NodeGeom {
            position: tip,
            branch_dir: dir,
            branch_length: child_len,
        });
    }

    // ---- file leaves as cloud around tip ----
    let n_files = file_children.len() as f32;
    let cloud_radius = if n_files > 0.0 {
        0.15 + (n_files + 1.0).ln() * 0.18
    } else {
        0.0
    };
    let crown_seed = hash_node(forest, parent_idx as usize);

    for &file_idx in &file_children {
        let leaf_seed = hash_node(forest, file_idx as usize);
        let offset = organic_crown_sample(leaf_seed, crown_seed, cloud_radius);
        layout.insert(file_idx, NodeGeom {
            position: tip + offset,
            branch_dir: Vec3::Y,
            branch_length: 0.0,
        });
    }
}

/// Iterative post-order: compute subtree_count and aggregated size.
fn aggregate_subtree(nodes: &mut [Node], pool: &[u32], root: usize) {
    let mut order: Vec<usize> = Vec::with_capacity(nodes.len());
    let mut stack: Vec<usize> = vec![root];
    while let Some(idx) = stack.pop() {
        order.push(idx);
        let n = &nodes[idx];
        let cs = n.children_start as usize;
        let ce = cs + n.children_count as usize;
        for &c in &pool[cs..ce] {
            stack.push(c as usize);
        }
    }
    for &idx in order.iter().rev() {
        let mut count = 1u32;
        let mut size = nodes[idx].size;
        let cs = nodes[idx].children_start as usize;
        let ce = cs + nodes[idx].children_count as usize;
        for &cid in &pool[cs..ce] {
            let c = cid as usize;
            count = count.saturating_add(nodes[c].subtree_count);
            if nodes[idx].is_dir() {
                size = size.saturating_add(nodes[c].size);
            }
        }
        nodes[idx].subtree_count = count;
        if nodes[idx].is_dir() {
            nodes[idx].size = size;
        }
    }
}

/// Compute DFS preorder index for every node. After this call, the subtree
/// rooted at any node N occupies the contiguous range
/// [nodes[N].dfs_pre, nodes[N].dfs_end) of preorder numbers.
///
/// This is critical for GPU-side subtree highlighting: instead of uploading
/// a bitmap of "selected" flags on every hover (1 byte × N nodes), we just
/// send 2 u32s in a uniform, and the shader checks `dfs_pre ∈ [start, end)`.
///
/// Iterative DFS — no stack overflow on deep filesystems (Windows %TMP% can
/// be 30+ levels deep with recursion).
fn compute_dfs_indices(nodes: &mut [Node], pool: &[u32], root: usize) {
    // Stack entry: (node_idx, has_been_entered).
    let mut stack: Vec<(usize, bool)> = Vec::with_capacity(64);
    stack.push((root, false));
    let mut counter: u32 = 0;
    while let Some((idx, entered)) = stack.pop() {
        if entered {
            nodes[idx].dfs_end = counter;
        } else {
            nodes[idx].dfs_pre = counter;
            counter += 1;
            stack.push((idx, true));
            let n = &nodes[idx];
            let cs = n.children_start as usize;
            let ce = cs + n.children_count as usize;
            // Reverse order so natural ordering of children is visited next.
            for &c in pool[cs..ce].iter().rev() {
                stack.push((c as usize, false));
            }
        }
    }
}

/// Deterministic 64-bit hash for a node — used as RNG seed for organic
/// placement (crown shape, branch direction). Replaces the path-based hash
/// since we no longer store full paths.
///
/// We mix the node's name + parent's name + grandparent's name (up to a
/// few levels) so siblings get distinct seeds even when their names are
/// short or numeric. Walking the whole chain would be overkill — three
/// levels is enough variability for any real filesystem.
fn hash_node(forest: &Forest, idx: usize) -> u64 {
    let mut h = AHasher::default();
    let mut cur = idx;
    for _ in 0..3 {
        forest.nodes[cur].name.as_ref().hash(&mut h);
        let p = forest.nodes[cur].parent;
        if p == NO_PARENT { break; }
        cur = p as usize;
    }
    h.finish()
}

/// Two independent floats in [0,1) from a single u64 seed.
fn split_seed(seed: u64) -> (f32, f32, f32) {
    let a = (seed & 0xFFFF) as f32 / 65536.0;
    let b = ((seed >> 16) & 0xFFFF) as f32 / 65536.0;
    let c = ((seed >> 32) & 0xFFFF) as f32 / 65536.0;
    (a, b, c)
}

/// Sample a direction inside a cone around `axis` with half-angle `half_angle`.
fn deterministic_dir_in_cone(seed: u64, axis: Vec3, half_angle: f32) -> Vec3 {
    let (u1, u2, _) = split_seed(seed);
    // cos(theta) uniform in [cos(half_angle), 1]
    let cos_t = 1.0 - u1 * (1.0 - half_angle.cos());
    let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
    let phi = u2 * std::f32::consts::TAU;
    // local-space sample (axis = +Z locally)
    let local = Vec3::new(sin_t * phi.cos(), sin_t * phi.sin(), cos_t);
    // rotate so local +Z aligns with axis
    align_z_to(axis) * local
}

/// Sample a point inside an organic, irregular crown shape.
///
/// We start with a uniform-in-ball sample, then:
///   1. Deform into an ellipsoid (per-crown anisotropy: vertical, flat, tilted)
///   2. Perturb the radius by a noise function (pushes some directions out,
///      pulls others in — gives "ragged" silhouette instead of perfect sphere)
///   3. Bias density toward center via density_exp (more concentrated)
///
/// `crown_seed` controls the SHAPE of this particular crown (anisotropy axes,
/// noise pattern). It should be the parent folder's hash, not per-leaf, so all
/// leaves in one folder share the same crown shape.
///
/// `leaf_seed` controls the POSITION of this individual leaf within the crown.
fn organic_crown_sample(leaf_seed: u64, crown_seed: u64, radius: f32) -> Vec3 {
    let (u1, u2, u3) = split_seed(leaf_seed);

    // 1. Base uniform-in-ball sample.
    let theta = u1 * std::f32::consts::TAU;
    let cos_phi = 1.0 - 2.0 * u2;
    let sin_phi = (1.0 - cos_phi * cos_phi).max(0.0).sqrt();
    // Density: r ~ u^(1/2.4) is more concentrated toward center than uniform (1/3).
    // Gives a denser core, sparser edges — looks more like real foliage.
    let r_base = radius * u3.powf(1.0 / 2.4);

    let dir = Vec3::new(
        sin_phi * theta.cos(),
        sin_phi * theta.sin(),
        cos_phi,
    );

    // 2. Per-crown anisotropy: stretch along a deterministic axis.
    // From crown_seed, derive an axis direction and a stretch factor.
    let (cs1, cs2, cs3) = split_seed(crown_seed);
    let crown_axis = Vec3::new(
        cs1 * 2.0 - 1.0,
        cs2 * 2.0 - 1.0,
        cs3 * 2.0 - 1.0,
    ).normalize_or(Vec3::Y);
    let stretch = 0.6 + cs1 * 1.0; // 0.6 .. 1.6 stretch along crown_axis
    let parallel = dir.dot(crown_axis);
    let parallel_part = crown_axis * parallel;
    let perpendicular_part = dir - parallel_part;
    let stretched_dir = perpendicular_part + parallel_part * stretch;

    // 3. Noise perturbation: vary the effective radius by direction.
    // Cheap pseudo-noise: sum of three sinusoids with crown-seeded phases.
    let phase1 = (crown_seed & 0xFFFF) as f32 * 0.001;
    let phase2 = ((crown_seed >> 16) & 0xFFFF) as f32 * 0.001;
    let phase3 = ((crown_seed >> 32) & 0xFFFF) as f32 * 0.001;
    let n1 = (stretched_dir.x * 3.0 + phase1).sin();
    let n2 = (stretched_dir.y * 3.7 + phase2).sin();
    let n3 = (stretched_dir.z * 4.1 + phase3).sin();
    let noise = (n1 + n2 + n3) / 3.0; // in [-1, 1]
    // Perturb radius by ±25%. Some directions push out, others recess.
    let r_noisy = r_base * (1.0 + noise * 0.25);

    stretched_dir * r_noisy
}

/// 3x3 rotation that sends +Z to `target` (assumed unit).
fn align_z_to(target: Vec3) -> glam::Mat3 {
    let z = Vec3::Z;
    let t = target.normalize_or(Vec3::Z);
    let dot = z.dot(t);
    if dot > 0.9999 {
        return glam::Mat3::IDENTITY;
    }
    if dot < -0.9999 {
        return glam::Mat3::from_axis_angle(Vec3::X, std::f32::consts::PI);
    }
    let axis = z.cross(t).normalize();
    let angle = dot.acos();
    glam::Mat3::from_axis_angle(axis, angle)
}

// Small extension trait: glam doesn't ship normalize_or in all versions.
trait NormalizeOr {
    fn normalize_or(self, fallback: Vec3) -> Vec3;
}
impl NormalizeOr for Vec3 {
    fn normalize_or(self, fallback: Vec3) -> Vec3 {
        let len = self.length();
        if len > 1e-6 { self / len } else { fallback }
    }
}

// =====================================================================
// Path reconstruction helpers
// =====================================================================

/// Reconstruct a node's full path by walking the parent chain. Since we no
/// longer store paths on every node (memory win), we build them on demand.
/// Called only at hover-change frequency (~tens per second at most), not
/// per-frame — performance is fine even for 32-deep paths.
///
/// Cleans up the Windows `\\?\` extended-length prefix that
/// `Path::canonicalize` adds — users find raw paths more readable.
pub fn build_full_path(forest: &Forest, idx: usize) -> String {
    // Collect names from leaf up to root.
    let mut parts: Vec<&str> = Vec::with_capacity(forest.nodes[idx].depth as usize + 1);
    let mut cur = idx;
    loop {
        parts.push(forest.nodes[cur].name.as_ref());
        let p = forest.nodes[cur].parent;
        if p == NO_PARENT { break; }
        cur = p as usize;
    }
    parts.reverse();

    // Root name is something like `\\?\C:\` on Windows after canonicalize.
    // Strip the `\\?\` prefix if present so users see plain `C:\...`.
    if let Some(first) = parts.first_mut() {
        if let Some(rest) = first.strip_prefix(r"\\?\") {
            *first = rest;
        }
    }

    let mut out = String::with_capacity(parts.iter().map(|s| s.len() + 1).sum::<usize>());
    for (i, n) in parts.iter().enumerate() {
        if i == 0 {
            out.push_str(n);
            if !out.ends_with(std::path::MAIN_SEPARATOR) {
                out.push(std::path::MAIN_SEPARATOR);
            }
        } else {
            out.push_str(n);
            if i + 1 < parts.len() { out.push(std::path::MAIN_SEPARATOR); }
        }
    }
    out
}

/// Like `build_full_path` but returns (parent_path_with_sep, leaf_name).
/// Useful for tooltips that colour the leaf differently from the rest.
pub fn build_full_path_split(forest: &Forest, idx: usize) -> (String, String) {
    let full = build_full_path(forest, idx);
    let name: String = forest.nodes[idx].name.as_ref().to_string();
    let parent_part = full.strip_suffix(&name).unwrap_or(&full).to_string();
    (parent_part, name)
}
