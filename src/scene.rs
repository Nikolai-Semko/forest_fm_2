// scene.rs — bundle of data describing one renderable "world view".
//
// A Scene holds:
//   - which subset of the global Forest is shown (root_node + descendants)
//   - per-scene camera state
//   - per-scene picking index (octree built from that subset)
//   - optional alternative `Layout` (drill-down scenes re-grow the subtree
//     as its own grove; main scene uses the default geometry stored on
//     `Forest::nodes`).
//   - GPU buffers (built lazily by Renderer when scene becomes active)
//
// Two scenes coexist when split-screen is active:
//   - main scene: the whole forest (root = forest.root), no layout override
//   - detail scene: drill-down on a clicked subtree, with a fresh Layout
//     where the subtree's direct subdirs become a new grove of trees
//
// Smooth morph: when a new detail scene opens, the right-side camera starts
// equal to the left-side camera (same yaw/pitch/distance/target) and
// interpolates toward an auto-framed pose over ~0.5s.

use glam::Vec3;
use std::ops::Range;
use std::sync::Arc;

use crate::forest::{Forest, Layout, NO_PARENT, build_full_path, geom_of};
use crate::picking::{Octree, Pickable, build_pickables};

/// Camera state — duplicated from renderer for plain-data ownership.
/// We keep the live `OrbitCamera` inside the renderer, but each Scene
/// remembers its camera *desired state* and the renderer mirrors it.
#[derive(Clone, Copy, Debug)]
pub struct CameraState {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl CameraState {
    pub fn lerp(&self, other: &CameraState, t: f32) -> CameraState {
        let t = t.clamp(0.0, 1.0);
        // Yaw is angular — wrap to nearest path.
        let mut dy = other.yaw - self.yaw;
        while dy > std::f32::consts::PI { dy -= std::f32::consts::TAU; }
        while dy < -std::f32::consts::PI { dy += std::f32::consts::TAU; }
        CameraState {
            target: self.target.lerp(other.target, t),
            // Distance: exponential interp so feel is uniform across scales.
            distance: self.distance.powf(1.0 - t) * other.distance.powf(t),
            yaw: self.yaw + dy * t,
            pitch: self.pitch + (other.pitch - self.pitch) * t,
        }
    }
}

/// A single renderable subset of the forest.
pub struct Scene {
    /// Root node of this scene (index into Forest::nodes).
    /// For the main scene: forest.root. For detail scenes: any directory node.
    pub root_idx: usize,

    /// Pre-flattened DFS list of node indices visible in this scene.
    /// Used to filter what gets uploaded to GPU and indexed for picking.
    pub visible_nodes: Vec<usize>,

    /// Picking acceleration structure for this subset.
    pub octree: Octree,

    pub camera: CameraState,

    /// Optional title for display in the viewport header.
    pub title: String,

    /// Optional alternative layout. `None` for main scene (use defaults from
    /// forest.nodes). `Some` for drill-down scenes (the subtree was re-grown
    /// as its own grove). Shared via Arc — a cache may keep multiple drill-in
    /// layouts alive at once.
    pub layout: Option<Arc<Layout>>,
}

impl Scene {
    /// Build a Scene that shows the entire forest (everything under
    /// forest.root). Uses the default layout stored on Forest::nodes.
    pub fn from_forest_root(forest: &Forest) -> Self {
        let visible_nodes: Vec<usize> = (0..forest.nodes.len()).collect();
        let pickables = build_pickables(forest);
        let octree = Octree::build(pickables);
        let (center, radius) = compute_bounds(forest, &visible_nodes, None);
        let camera = CameraState {
            target: Vec3::new(center.x, center.y + radius * 0.3, center.z),
            distance: (radius * 2.2).max(15.0),
            yaw: 0.6,
            pitch: 0.2,
        };
        let title = build_full_path(forest, forest.root);
        Scene { root_idx: forest.root, visible_nodes, octree, camera, title, layout: None }
    }

    /// Build a Scene that drills into the subtree rooted at `root_idx`, using
    /// the supplied alternative `Layout` (which should have been generated
    /// for this same `root_idx`).
    pub fn from_subtree(forest: &Forest, root_idx: usize, layout: Arc<Layout>) -> Self {
        let mut visible_nodes = Vec::new();
        let mut stack: Vec<usize> = vec![root_idx];
        while let Some(idx) = stack.pop() {
            visible_nodes.push(idx);
            for &c in forest.children_of(idx).iter() {
                stack.push(c as usize);
            }
        }
        let pickables = build_pickables_subset(forest, &visible_nodes, &*layout);
        let octree = Octree::build(pickables);
        let (center, radius) = compute_bounds(forest, &visible_nodes, Some(&*layout));
        let camera = CameraState {
            target: Vec3::new(center.x, center.y + radius * 0.3, center.z),
            distance: (radius * 2.2).max(5.0),
            yaw: 0.6,
            pitch: 0.2,
        };
        let title = forest.nodes[root_idx].name.as_ref().to_string();
        Scene {
            root_idx, visible_nodes, octree, camera, title,
            layout: Some(layout),
        }
    }
}

/// Same logic as picking::build_pickables but only for a subset of nodes,
/// reading positions from an alternative `Layout`.
fn build_pickables_subset(forest: &Forest, visible: &[usize], layout: &Layout) -> Vec<Pickable> {
    use crate::picking::PickKind;
    let mut out = Vec::with_capacity(visible.len());

    // Max subtree count *within this subset* — gives proper scale for the
    // drill-down view so branches re-rank visually.
    let max_subtree = visible.iter()
        .map(|&i| forest.nodes[i].subtree_count)
        .max()
        .unwrap_or(1) as f32;
    let log_max = max_subtree.max(2.0).ln();

    for &idx in visible {
        let n = &forest.nodes[idx];
        let g = geom_of(forest, Some(layout), idx);
        if n.is_dir() {
            if g.branch_length < 1e-4 { continue; }
            let tip = g.position + g.branch_dir * g.branch_length;
            let sub = n.subtree_count.max(1) as f32;
            let mass01 = (sub.ln() / log_max).clamp(0.0, 1.0);
            let thickness_px = 0.7 + mass01 * 2.8;
            // See picking::build_pickables for explanation of this AABB radius.
            let world_radius = (g.branch_length * 0.05).max(0.15).min(2.0);
            out.push(Pickable {
                node_idx: idx as u32,
                kind: PickKind::Branch,
                p0: g.position,
                p1: tip,
                world_radius,
                thickness_px,
            });
        } else {
            let log_size = ((n.size as f32) + 1.0).log2();
            let size01 = (log_size / 30.0).clamp(0.0, 1.0);
            let radius = 0.012 + size01 * 0.06;
            out.push(Pickable {
                node_idx: idx as u32,
                kind: PickKind::Leaf,
                p0: g.position,
                p1: g.position,
                world_radius: radius,
                thickness_px: 0.0,
            });
        }
    }
    out
}

/// Compute world-space bounds of the visible subset, reading positions from
/// `layout` if supplied, else from `forest.nodes`.
fn compute_bounds(forest: &Forest, visible: &[usize], layout: Option<&Layout>) -> (Vec3, f32) {
    if visible.is_empty() { return (Vec3::ZERO, 1.0); }
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for &i in visible {
        let g = geom_of(forest, layout, i);
        lo = lo.min(g.position);
        hi = hi.max(g.position);
        if g.branch_length > 0.0 {
            let tip = g.position + g.branch_dir * g.branch_length;
            lo = lo.min(tip);
            hi = hi.max(tip);
        }
    }
    let center = (lo + hi) * 0.5;
    let radius = (hi - lo).length() * 0.5;
    (center, radius.max(1.0))
}

// ---------- Selection & hover state ----------

/// What's currently highlighted in a scene (hover or click).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selection {
    pub node_idx: u32,
    /// True if the user wants the entire subtree highlighted (default on hover).
    /// False would be self-only — currently unused but reserved.
    pub include_subtree: bool,
}

/// Compute the index range of subtree descendants (in DFS preorder) of a node.
/// Pre-computed once per scene (currently we just walk on demand — TODO cache).
pub fn subtree_range(forest: &Forest, root_idx: usize) -> Range<usize> {
    let _ = forest;
    let _ = root_idx;
    // Placeholder for future packed-index optimization. Subtree membership
    // is checked on the fly via ancestor walk for now.
    0..0
}

/// Is `descendant_idx` inside the subtree rooted at `ancestor_idx`?
/// Walks parent chain. O(depth).
pub fn is_descendant_of(forest: &Forest, descendant_idx: usize, ancestor_idx: usize) -> bool {
    if descendant_idx == ancestor_idx { return true; }
    let mut cur = forest.nodes[descendant_idx].parent;
    while cur != NO_PARENT {
        let p = cur as usize;
        if p == ancestor_idx { return true; }
        cur = forest.nodes[p].parent;
    }
    false
}

// ---------- Split-screen animation ----------

/// Smooth open/close of the second viewport. Drives both layout (split ratio)
/// and the right-viewport camera morph.
pub struct SplitAnimation {
    /// 0 = closed (no right viewport). 1 = fully open.
    pub t: f32,
    /// Target value (0 or 1). t animates toward it.
    pub target_t: f32,
    /// Camera state of the right viewport at animation start. Lerps from this
    /// to scene.camera as t goes 0→1.
    pub right_camera_start: CameraState,
    /// Was animation just started this frame? (Used to seed start state.)
    pub started: bool,
}

impl SplitAnimation {
    pub fn closed() -> Self {
        Self {
            t: 0.0,
            target_t: 0.0,
            right_camera_start: CameraState {
                target: Vec3::ZERO, distance: 1.0, yaw: 0.0, pitch: 0.0,
            },
            started: false,
        }
    }
    pub fn open(&mut self, from_camera: CameraState) {
        self.target_t = 1.0;
        self.right_camera_start = from_camera;
        self.started = true;
    }
    pub fn close(&mut self) {
        self.target_t = 0.0;
        self.started = false;
    }
    /// Advance animation by `dt` seconds. Returns true if still animating.
    pub fn step(&mut self, dt: f32) -> bool {
        const DUR: f32 = 0.45; // seconds for full open/close
        let speed = 1.0 / DUR;
        let delta = speed * dt;
        let prev = self.t;
        if self.t < self.target_t {
            self.t = (self.t + delta).min(self.target_t);
        } else if self.t > self.target_t {
            self.t = (self.t - delta).max(self.target_t);
        }
        // Smoothstep on raw t for nicer easing.
        prev != self.t
    }
    /// Eased value in [0,1] for visual interpolation.
    pub fn eased(&self) -> f32 {
        let x = self.t.clamp(0.0, 1.0);
        x * x * (3.0 - 2.0 * x)
    }
    pub fn is_open(&self) -> bool {
        self.t > 0.001 || self.target_t > 0.001
    }
}
