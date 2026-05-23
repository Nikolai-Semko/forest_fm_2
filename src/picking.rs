// picking.rs — spatial index (loose octree) + ray-cast picking.
//
// Two kinds of pickable objects in the forest:
//   1. Leaves (files): represented as spheres at `node.position` with
//      world-space radius derived from log-size (same formula renderer uses).
//   2. Branches (directories with branch_length > 0): represented as
//      capsules from `node.position` to `node.position + branch_dir * branch_length`.
//
// We pick the object whose surface (not center!) is *closest in screen space*
// to the cursor, because that matches user intent:
//   - a tiny leaf right under the cursor beats a huge branch the ray
//     happens to pass near in world space
//   - a fat trunk slightly off-screen-axis loses to a thin twig the cursor
//     is actually on
//
// Algorithm:
//   1. Octree-query candidates whose AABB intersects the ray.
//   2. For each candidate, compute the WORLD-space closest approach of ray
//      to surface, then PROJECT that point to screen and measure pixel distance.
//   3. Reject if pixel distance > object's screen radius + hover slack.
//   4. Among accepted candidates, prefer (smaller t, smaller screen_dist).
//
// The octree is built ONCE at construction (filesystem snapshot is static
// during a session). Rebuild is needed only on rescan.

use glam::{Mat4, Vec3};
use crate::forest::Forest;

// ---------- AABB helpers ----------

#[derive(Clone, Copy, Debug)]
struct Aabb {
    min: Vec3,
    max: Vec3,
}

impl Aabb {
    fn empty() -> Self {
        Self { min: Vec3::splat(f32::INFINITY), max: Vec3::splat(f32::NEG_INFINITY) }
    }
    fn from_point(p: Vec3) -> Self {
        Self { min: p, max: p }
    }
    fn union_point(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }
    fn union(&mut self, other: &Aabb) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }
    fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
    fn expand(&mut self, r: f32) {
        self.min -= Vec3::splat(r);
        self.max += Vec3::splat(r);
    }
    /// Slab-method ray vs AABB; returns (t_enter, t_exit) clipped to ≥0, or None.
    fn ray_hit(&self, origin: Vec3, inv_dir: Vec3) -> Option<(f32, f32)> {
        let t1 = (self.min - origin) * inv_dir;
        let t2 = (self.max - origin) * inv_dir;
        let tmin = t1.min(t2);
        let tmax = t1.max(t2);
        let t_enter = tmin.x.max(tmin.y).max(tmin.z).max(0.0);
        let t_exit = tmax.x.min(tmax.y).min(tmax.z);
        if t_enter <= t_exit { Some((t_enter, t_exit)) } else { None }
    }
}

// ---------- Picking entities ----------

/// What's stored in each octree leaf. Keep it cache-friendly (24 bytes).
/// For leaves: p1 == p0, radius = sphere radius.
/// For branches: p0..p1 capsule axis, radius = world-space half-thickness
/// (approx — branches have *pixel* thickness, not world; we use a generous
/// fallback radius and rely on screen-space test for accuracy).
#[derive(Clone, Copy, Debug)]
pub struct Pickable {
    pub node_idx: u32,
    pub kind: PickKind,
    pub p0: Vec3,
    pub p1: Vec3,
    /// World-space radius used for the AABB and as an initial broad filter.
    /// For branches this is set to a screen-space-equivalent radius at average depth.
    pub world_radius: f32,
    /// Screen radius (in pixels) used for the precise hover test. For leaves
    /// this is recomputed per frame from world_radius. For branches it's the
    /// fixed pixel thickness from the renderer.
    pub thickness_px: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PickKind {
    Leaf,
    Branch,
}

// ---------- Octree ----------

/// Maximum entities per leaf node before we try to subdivide. For huge
/// scenes (3M+ entities) the octree itself takes significant memory
/// from per-node Box allocations and Vec<u32> items. Higher capacity =
/// fewer leaves = less alloc overhead. Pick at the boundary where linear
/// scan in `ray_query` is still cheap (~256 items × 50 ns each = 13 µs,
/// imperceptible at typical mouse-move rate).
const LEAF_CAPACITY: usize = 256;
/// Maximum depth — prevents pathological subdivision on coincident points.
const MAX_DEPTH: u32 = 16;

enum OctNode {
    Leaf { bounds: Aabb, items: Vec<u32> },
    Internal { bounds: Aabb, children: [Box<OctNode>; 8] },
}

impl OctNode {
    fn bounds(&self) -> Aabb {
        match self {
            OctNode::Leaf { bounds, .. } | OctNode::Internal { bounds, .. } => *bounds,
        }
    }
}

pub struct Octree {
    pub entities: Vec<Pickable>,
    root: OctNode,
    root_bounds: Aabb,
}

impl Octree {
    pub fn build(entities: Vec<Pickable>) -> Self {
        if entities.is_empty() {
            let empty_bounds = Aabb { min: Vec3::ZERO, max: Vec3::splat(1.0) };
            return Self {
                entities,
                root: OctNode::Leaf { bounds: empty_bounds, items: Vec::new() },
                root_bounds: empty_bounds,
            };
        }
        // Compute overall bounds (expanded by each entity's radius).
        let mut bounds = Aabb::empty();
        for e in &entities {
            let mut b = Aabb::from_point(e.p0);
            b.union_point(e.p1);
            b.expand(e.world_radius);
            bounds.union(&b);
        }
        // Make slightly cubic to avoid degenerate splits; just inflate by 1%.
        let diag = (bounds.max - bounds.min) * 0.01 + Vec3::splat(0.01);
        bounds.min -= diag;
        bounds.max += diag;

        let all_idx: Vec<u32> = (0..entities.len() as u32).collect();
        let root = build_node(&entities, all_idx, bounds, 0);
        Self { entities, root, root_bounds: bounds }
    }

    /// Visit every entity whose AABB the ray crosses. Visitor returns true to
    /// keep going. Implementation uses an explicit stack (deep trees, no recursion).
    pub fn ray_query<F: FnMut(u32)>(&self, origin: Vec3, dir: Vec3, mut visit: F) {
        // Pre-compute inverse direction (componentwise reciprocal). Where a
        // direction component is zero, slab method needs +/-inf — glam Vec3
        // handles 1.0/0.0 = inf correctly.
        let inv_dir = Vec3::new(
            if dir.x.abs() > 1e-9 { 1.0 / dir.x } else { f32::INFINITY },
            if dir.y.abs() > 1e-9 { 1.0 / dir.y } else { f32::INFINITY },
            if dir.z.abs() > 1e-9 { 1.0 / dir.z } else { f32::INFINITY },
        );
        if self.root_bounds.ray_hit(origin, inv_dir).is_none() {
            return;
        }
        let mut stack: Vec<&OctNode> = Vec::with_capacity(64);
        stack.push(&self.root);
        while let Some(node) = stack.pop() {
            match node {
                OctNode::Leaf { items, .. } => {
                    for &i in items { visit(i); }
                }
                OctNode::Internal { children, .. } => {
                    for c in children.iter() {
                        if c.bounds().ray_hit(origin, inv_dir).is_some() {
                            stack.push(c.as_ref());
                        }
                    }
                }
            }
        }
    }
}

fn build_node(entities: &[Pickable], items: Vec<u32>, bounds: Aabb, depth: u32) -> OctNode {
    if items.len() <= LEAF_CAPACITY || depth >= MAX_DEPTH {
        return OctNode::Leaf { bounds, items };
    }
    // Subdivide into 8 octants around bounds.center().
    let c = bounds.center();
    let mut buckets: [Vec<u32>; 8] = Default::default();
    let mut child_bounds: [Aabb; 8] = [Aabb::empty(); 8];
    for (oi, b) in child_bounds.iter_mut().enumerate() {
        let mx = if oi & 1 != 0 { c.x } else { bounds.min.x };
        let xx = if oi & 1 != 0 { bounds.max.x } else { c.x };
        let my = if oi & 2 != 0 { c.y } else { bounds.min.y };
        let yy = if oi & 2 != 0 { bounds.max.y } else { c.y };
        let mz = if oi & 4 != 0 { c.z } else { bounds.min.z };
        let zz = if oi & 4 != 0 { bounds.max.z } else { c.z };
        *b = Aabb { min: Vec3::new(mx, my, mz), max: Vec3::new(xx, yy, zz) };
    }
    for &i in &items {
        let e = &entities[i as usize];
        // Use entity center to pick octant. For capsules: midpoint.
        let ec = (e.p0 + e.p1) * 0.5;
        let mut oi = 0;
        if ec.x > c.x { oi |= 1; }
        if ec.y > c.y { oi |= 2; }
        if ec.z > c.z { oi |= 4; }
        buckets[oi].push(i);
    }
    // Avoid pathological case where all entities land in one bucket — then
    // recursion won't shrink the problem. Stop subdividing.
    if buckets.iter().filter(|b| !b.is_empty()).count() <= 1 {
        return OctNode::Leaf { bounds, items };
    }
    // Build children. Empty buckets become empty leaves.
    let mut bucket_iter = buckets.into_iter();
    let mut bound_iter = child_bounds.iter();
    let make_child = |b: Vec<u32>, cb: Aabb| -> Box<OctNode> {
        if b.is_empty() {
            Box::new(OctNode::Leaf { bounds: cb, items: Vec::new() })
        } else {
            Box::new(build_node(entities, b, cb, depth + 1))
        }
    };
    let children: [Box<OctNode>; 8] = [
        make_child(bucket_iter.next().unwrap(), *bound_iter.next().unwrap()),
        make_child(bucket_iter.next().unwrap(), *bound_iter.next().unwrap()),
        make_child(bucket_iter.next().unwrap(), *bound_iter.next().unwrap()),
        make_child(bucket_iter.next().unwrap(), *bound_iter.next().unwrap()),
        make_child(bucket_iter.next().unwrap(), *bound_iter.next().unwrap()),
        make_child(bucket_iter.next().unwrap(), *bound_iter.next().unwrap()),
        make_child(bucket_iter.next().unwrap(), *bound_iter.next().unwrap()),
        make_child(bucket_iter.next().unwrap(), *bound_iter.next().unwrap()),
    ];
    OctNode::Internal { bounds, children }
}

// ---------- Ray vs primitives ----------

/// Closest pair of points between two lines (not segments) p1=o+t*d and
/// segment a..b. Returns (t_on_ray, point_on_segment).
/// Reference: Real-Time Collision Detection §5.1.9.
fn closest_ray_segment(origin: Vec3, dir: Vec3, a: Vec3, b: Vec3) -> (f32, Vec3) {
    let d1 = dir;           // direction of ray (not necessarily unit, but is for us)
    let d2 = b - a;         // direction of segment
    let r = origin - a;
    let a_dd = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);
    if e <= 1e-9 {
        // Segment degenerates into point a.
        let t_ray = (-d1.dot(r) / a_dd).max(0.0);
        return (t_ray, a);
    }
    let c = d1.dot(r);
    let b_dot = d1.dot(d2);
    let denom = a_dd * e - b_dot * b_dot;
    let t_ray;
    let s_seg;
    if denom != 0.0 {
        s_seg = ((a_dd * f - b_dot * c) / denom).clamp(0.0, 1.0);
    } else {
        s_seg = 0.0;
    }
    t_ray = ((b_dot * s_seg - c) / a_dd).max(0.0);
    // Re-clamp segment after t_ray clamp, but for our purposes a single pass
    // is good enough (we measure pixel distance afterwards anyway).
    (t_ray, a + d2 * s_seg)
}

// ---------- Public picking API ----------

#[derive(Clone, Copy, Debug)]
pub struct PickResult {
    pub node_idx: u32,
    pub kind: PickKind,
    pub world_hit: Vec3,    // closest point on the picked object (for visualization, focus)
    pub t: f32,             // ray parameter at hit
    pub screen_dist_px: f32, // pixel distance from cursor to surface (for tie-break / quality)
}

pub struct Picker<'a> {
    pub octree: &'a Octree,
}

impl<'a> Picker<'a> {
    /// Cast a ray and find the best pick under the cursor.
    ///
    /// * `view_proj` — to project candidate world points to screen for the
    ///   final pixel-distance test.
    /// * `(cursor_px, cursor_py)` — cursor in window pixels.
    /// * `(vw, vh)` — viewport size in pixels.
    /// * `slack_px` — extra hover radius (cursor can be N pixels off-surface
    ///   and still hit). Typical: 6 px.
    pub fn pick(
        &self,
        origin: Vec3,
        dir: Vec3,
        view_proj: Mat4,
        cursor_px: f32,
        cursor_py: f32,
        vw: f32,
        vh: f32,
        slack_px: f32,
    ) -> Option<PickResult> {
        // Collect candidate indices (octree returns each one ≥ once, possibly
        // duplicates due to AABB straddling — dedupe).
        let mut seen: ahash::AHashSet<u32> = ahash::AHashSet::with_capacity(64);
        let mut best: Option<PickResult> = None;

        self.octree.ray_query(origin, dir, |i| {
            if !seen.insert(i) { return; }
            let e = &self.octree.entities[i as usize];
            let (t, hit_world) = match e.kind {
                PickKind::Leaf => {
                    // Ray to point distance, find foot of perpendicular.
                    let to_p = e.p0 - origin;
                    let t = to_p.dot(dir).max(0.0);
                    (t, e.p0)
                }
                PickKind::Branch => {
                    closest_ray_segment(origin, dir, e.p0, e.p1)
                }
            };

            // Project hit_world to screen.
            let clip = view_proj * glam::Vec4::new(hit_world.x, hit_world.y, hit_world.z, 1.0);
            if clip.w <= 0.001 { return; } // behind camera
            let ndc = glam::Vec2::new(clip.x / clip.w, clip.y / clip.w);
            // NDC → pixel
            let sx = (ndc.x * 0.5 + 0.5) * vw;
            let sy = (1.0 - (ndc.y * 0.5 + 0.5)) * vh;
            let dx = sx - cursor_px;
            let dy = sy - cursor_py;
            let pix_dist = (dx * dx + dy * dy).sqrt();

            // Screen-space radius:
            //  - Branches use fixed pixel thickness.
            //  - Leaves project world_radius into pixels at this depth.
            let screen_radius_px = match e.kind {
                PickKind::Branch => (e.thickness_px * 0.5).max(2.0),
                PickKind::Leaf => {
                    // Same projection math as in shader: world_per_px = w * 2 / vh
                    let world_per_px = clip.w * 2.0 / vh;
                    // Take effective radius = max(natural, MIN_PX) to match
                    // how leaf is actually drawn (small files inflated to MIN_PX).
                    let natural_px = e.world_radius / world_per_px.max(1e-6);
                    natural_px.max(0.7)
                }
            };

            if pix_dist > screen_radius_px + slack_px {
                return;
            }

            // Quality: pixel distance INSIDE the surface counts as 0;
            // distance OUTSIDE (in slack zone) counts proportionally.
            let inside_dist = (pix_dist - screen_radius_px).max(0.0);

            let candidate = PickResult {
                node_idx: e.node_idx,
                kind: e.kind,
                world_hit: hit_world,
                t,
                screen_dist_px: inside_dist,
            };

            // Selection rule: prefer smaller t, but if two objects are
            // overlapping/close in depth, prefer one the cursor is more
            // clearly *on* (smaller inside_dist).
            //
            // Practical formula: a leaf clearly under the cursor (inside_dist=0)
            // beats a branch the ray happens to graze (inside_dist>0). Among
            // objects with same inside_dist=0, smaller t wins (closer object).
            let take = match best {
                None => true,
                Some(b) => {
                    // Tier 1: both inside their surface → smaller t wins.
                    // Tier 2: any in slack zone → smaller inside_dist wins.
                    if candidate.screen_dist_px == 0.0 && b.screen_dist_px == 0.0 {
                        candidate.t < b.t
                    } else {
                        candidate.screen_dist_px < b.screen_dist_px
                    }
                }
            };
            if take { best = Some(candidate); }
        });

        best
    }
}

// ---------- Build pickables from a Forest ----------

/// Walk the forest and emit one Pickable per visible object (leaves + branches
/// with non-zero length). Matches what the renderer actually draws.
///
/// Geometry is read from `forest.main_layout` — the global grove layout
/// computed at scan time. Drill-down scenes have their own pickables built
/// via `scene::build_pickables_subset` passing their per-scene Layout.
pub fn build_pickables(forest: &Forest) -> Vec<Pickable> {
    let mut out = Vec::with_capacity(forest.nodes.len());
    let layout = &forest.main_layout;

    let max_subtree = forest.nodes.iter().map(|n| n.subtree_count).max().unwrap_or(1) as f32;
    let log_max = max_subtree.max(2.0).ln();

    for (idx, n) in forest.nodes.iter().enumerate() {
        let g = layout.get(idx);
        if n.is_dir() {
            if g.branch_length < 1e-4 { continue; }
            let tip = g.position + g.branch_dir * g.branch_length;
            let sub = n.subtree_count.max(1) as f32;
            let mass01 = (sub.ln() / log_max).clamp(0.0, 1.0);
            let thickness_px = 0.7 + mass01 * 2.8;
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
