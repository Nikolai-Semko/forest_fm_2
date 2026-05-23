// main.rs — entry point: parse args, scan, build forest, run window loop.
//
// Responsibilities:
//   * CLI parsing & scan stats printout.
//   * Window/event loop bootstrap.
//   * Mouse/keyboard routing: dispatch input to whichever viewport (main or
//     detail) is "owning" the gesture, with rules:
//       - Drag is sticky: the viewport where LMB/RMB went DOWN owns the drag
//         until the button is released. This avoids glitches when the cursor
//         crosses the split boundary mid-gesture.
//       - Wheel/hover follows the cursor's CURRENT viewport.
//   * Click vs drag discrimination: an LMB press → small movement → release
//     (under CLICK_PIXEL_THRESHOLD) is treated as a click; we then run picking
//     in the active viewport and, if hit, open/replace the detail scene.
//   * Hover picking each cursor move: cheap (octree is O(log N) + small leaf).
//   * Esc semantics: first Esc closes the detail panel, second Esc exits.
//   * Theme toggle on `T`. Reset main camera on `R`.

mod forest;
mod labels;
mod picking;
mod renderer;
mod scene;
mod tooltip;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use forest::{Forest, Layout, generate_layout};
use picking::{Octree, Picker};
use renderer::{OrbitCamera, Renderer};
use scene::Scene;

const MAX_DEPTH_DEFAULT: usize = 32;

/// A press that moves less than this many pixels before release is a click.
const CLICK_PIXEL_THRESHOLD: f32 = 4.0;

/// Slack hover radius in pixels — cursor can be this far off-surface and
/// still register a pick.
const PICK_SLACK_PX: f32 = 6.0;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    // arg parsing: forest_fm [path] [max_depth]
    let mut args = std::env::args().skip(1);
    let root: PathBuf = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap());
    let max_depth: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(MAX_DEPTH_DEFAULT);

    println!("Scanning {} (max depth {})...", root.display(), max_depth);
    let t0 = std::time::Instant::now();
    let forest = Forest::scan(&root, max_depth);
    let dt = t0.elapsed();

    // ----- gather stats -----
    let dirs = forest.nodes.iter().filter(|n| n.is_dir()).count();
    let files = forest.nodes.len() - dirs;
    let total_bytes: u64 = forest.nodes.iter()
        .filter(|n| !n.is_dir())
        .map(|n| n.size)
        .sum();
    let max_depth_seen: u8 = forest.nodes.iter().map(|n| n.depth).max().unwrap_or(0);
    let top_level_dirs = forest.children_of(forest.root)
        .iter()
        .filter(|&&c| forest.nodes[c as usize].is_dir())
        .count();

    let size_str = format_bytes(total_bytes);

    println!();
    println!("=== Scan complete in {:.2}s ===", dt.as_secs_f32());
    println!("  Total dirs:        {}", format_count(dirs));
    println!("  Total files:       {}", format_count(files));
    println!("  Total size:        {}", size_str);
    println!("  Top-level trees:   {}", top_level_dirs);
    println!("  Max depth reached: {}", max_depth_seen);
    if (max_depth_seen as usize) >= max_depth {
        println!("  ⚠ Depth limit hit — some folders may be incomplete.");
        println!("    Re-run with a higher limit if needed: forest_fm <path> <depth>");
    }

    // Top 5 heaviest top-level directories.
    let mut tops: Vec<(String, u64, u32)> = forest.children_of(forest.root)
        .iter()
        .filter(|&&c| forest.nodes[c as usize].is_dir())
        .map(|&c| (
            forest.nodes[c as usize].name.as_ref().to_string(),
            forest.nodes[c as usize].size,
            forest.nodes[c as usize].subtree_count,
        ))
        .collect();
    tops.sort_by(|a, b| b.1.cmp(&a.1));
    if !tops.is_empty() {
        println!();
        println!("  Largest top-level folders:");
        for (name, size, count) in tops.iter().take(5) {
            println!("    {:>10}  ({} items)  {}",
                format_bytes(*size),
                format_count(*count as usize),
                name,
            );
        }
    }

    println!();
    println!("Controls:");
    println!("  LMB-drag       orbit   (viewport where pressed owns the drag)");
    println!("  LMB-click      drill into the subtree under cursor (opens right panel)");
    println!("  RMB-drag       pan");
    println!("  Wheel          zoom to cursor (in viewport under cursor)");
    println!("  T              toggle dark / light theme");
    println!("  R              reset main camera");
    println!("  F              toggle folder name labels");
    println!("  Esc            close detail panel  →  quit");
    println!();

    let event_loop = EventLoop::new().expect("event loop");

    // Wrap forest in Arc so Renderer and main can both reference it cheaply.
    let forest = Arc::new(forest);
    let main_scene = Scene::from_forest_root(&forest);

    let mut app = App {
        forest: Some(forest),
        pending_main_scene: Some(main_scene),
        main_scene: None,
        detail_scene: None,
        renderer: None,
        cursor_pos: (0.0, 0.0),
        last_mouse: None,
        lmb_state: ButtonGesture::Up,
        rmb_state: ButtonGesture::Up,
        layout_cache: HashMap::new(),
        layout_lru: Vec::new(),
    };
    event_loop.run_app(&mut app).expect("run");
}

// =====================================================================
// App state & input routing
// =====================================================================

/// Which viewport a gesture started in. We commit to it for the whole gesture.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Viewport { Main, Detail }

/// State of a single mouse button.
#[derive(Clone, Copy, Debug)]
enum ButtonGesture {
    Up,
    /// Button just pressed; remember origin (window coords), which viewport
    /// owns the gesture, and the largest movement since press.
    Down { press_pos: (f64, f64), active: Viewport, max_move: f32 },
}

struct App {
    /// Temporarily owned until `resumed()` hands it to the renderer.
    forest: Option<Arc<Forest>>,
    /// Same: the main scene built in `main()`, consumed by `resumed()`.
    pending_main_scene: Option<Scene>,

    /// Once `resumed()` runs, these are populated. The renderer also keeps
    /// per-scene GPU buffers; here on the App we keep the picking octrees
    /// and the camera *targets*. (Current cameras live on the renderer for
    /// continuous-animation reasons.)
    main_scene: Option<Scene>,
    detail_scene: Option<Scene>,

    renderer: Option<Renderer>,

    /// Last cursor position in window-pixel coords. Tracked every frame —
    /// needed for wheel zoom even when no button is down.
    cursor_pos: (f64, f64),

    /// Previous cursor position for drag delta computation.
    last_mouse: Option<(f64, f64)>,

    lmb_state: ButtonGesture,
    rmb_state: ButtonGesture,

    /// Cache of Layouts keyed by their root node index. Once we generate a
    /// drill-down layout for a folder, we keep it around so re-clicking the
    /// same folder is instant. Layouts are wrapped in Arc so the active
    /// Scene can hold a cheap shared reference while the cache also keeps
    /// its own copy. Capped by LAYOUT_CACHE_CAPACITY via the LRU list below.
    layout_cache: HashMap<usize, Arc<Layout>>,
    /// LRU order for layout_cache. Front = oldest, back = most recent.
    /// On overflow, oldest entries are evicted.
    layout_lru: Vec<usize>,
}

impl App {
    /// Map a window-coords cursor position to whichever viewport contains it,
    /// plus local coords inside that viewport (origin at the viewport's
    /// top-left). Returns (Viewport, local_x, local_y, vp_w, vp_h).
    fn viewport_at(renderer: &Renderer, cursor: (f64, f64)) -> (Viewport, f32, f32, f32, f32) {
        let (main, detail) = renderer.viewport_rects();
        if let Some((dx, dy, dw, dh)) = detail {
            if (cursor.0 as f32) >= dx {
                return (
                    Viewport::Detail,
                    cursor.0 as f32 - dx,
                    cursor.1 as f32 - dy,
                    dw, dh,
                );
            }
        }
        let (mx, my, mw, mh) = main;
        (Viewport::Main, cursor.0 as f32 - mx, cursor.1 as f32 - my, mw, mh)
    }

    /// Borrow the octree for the given viewport (immutable). None if detail
    /// viewport requested but no detail scene exists.
    fn octree_for(&self, vp: Viewport) -> Option<&Octree> {
        match vp {
            Viewport::Main => self.main_scene.as_ref().map(|s| &s.octree),
            Viewport::Detail => self.detail_scene.as_ref().map(|s| &s.octree),
        }
    }

    /// Get the scene's root_idx for the given viewport.
    fn scene_root_idx(&self, vp: Viewport) -> Option<usize> {
        match vp {
            Viewport::Main => self.main_scene.as_ref().map(|s| s.root_idx),
            Viewport::Detail => self.detail_scene.as_ref().map(|s| s.root_idx),
        }
    }

    /// Run picking in the given viewport at the window-coords `cursor`.
    /// Returns the picked node_idx, or None.
    ///
    /// Hover granularity: raw pick can land on any node (leaf, twig, mid
    /// branch, trunk). We then climb the parent chain up to the *tree-level* —
    /// the direct child of scene.root_idx that the hit belongs to. This means
    /// hovering anywhere on a tree (leaf, trunk, anywhere) highlights the
    /// WHOLE tree as one unit, not a sub-branch. UX-wise: clearer, more
    /// predictable, much easier to target visually.
    ///
    /// Safe pattern: snapshot camera + viewport params out of the renderer
    /// (releases &mut renderer borrow), then borrow self.octree, then call
    /// the picker.
    fn pick_in(&mut self, vp: Viewport, cursor: (f64, f64)) -> Option<u32> {
        // Snapshot what we need from the renderer first (no overlap with the
        // immutable self.octree borrow below).
        let snapshot = {
            let r = self.renderer.as_mut()?;
            let (which, lx, ly, vw, vh) = Self::viewport_at(r, cursor);
            if which != vp { return None; }
            let cam: &mut OrbitCamera = match vp {
                Viewport::Main => r.main_camera(),
                Viewport::Detail => r.detail_camera()?,
            };
            let (vp_mat, _, _) = cam.view_proj(vw / vh.max(1.0));
            let (origin, dir) = cam.ray_from_cursor(lx, ly, vw, vh);
            (origin, dir, vp_mat, lx, ly, vw, vh)
        };
        let (origin, dir, vp_mat, lx, ly, vw, vh) = snapshot;

        let raw_hit = {
            let octree = self.octree_for(vp)?;
            let picker = Picker { octree };
            picker.pick(origin, dir, vp_mat, lx, ly, vw, vh, PICK_SLACK_PX)?
        };

        // Climb parent chain up to the direct child of scene.root_idx
        // (= "the tree" this hit belongs to).
        let scene_root = self.scene_root_idx(vp)?;
        let forest = self.renderer.as_ref()?.forest.clone();
        let tree_idx = climb_to_tree(&forest, raw_hit.node_idx as usize, scene_root);
        Some(tree_idx as u32)
    }

    /// Handle a click on a node in the given viewport.
    ///   - Click in MAIN: open (or replace) detail with that node's subtree.
    ///   - Click in DETAIL: drill deeper — replace detail with that subtree.
    /// File clicks (non-dir leaves) are no-ops for now.
    fn handle_click(&mut self, node_idx: u32) {
        // Phase 1: grab a cheap Arc<Forest> clone (no &mut self overlap).
        let forest = self.renderer.as_ref().unwrap().forest.clone();
        if !forest.nodes[node_idx as usize].is_dir() { return; }

        // Re-clicking the same subtree is a no-op.
        if let Some(existing) = &self.detail_scene {
            if existing.root_idx == node_idx as usize { return; }
        }

        // Phase 2: get-or-build a Layout for this subtree.
        //
        // We keep a small LRU cache (capacity LAYOUT_CACHE_CAPACITY) so that
        // ping-ponging between a handful of folders is instant, but exploring
        // many large folders doesn't accumulate unbounded memory. For huge
        // forests (3M+ nodes) each Layout can be 100+ MB, so the cache must
        // be small.
        const LAYOUT_CACHE_CAPACITY: usize = 3;
        let key = node_idx as usize;
        let layout = if let Some(arc) = self.layout_cache.get(&key) {
            // Hit: bump to most-recent.
            let arc = arc.clone();
            self.layout_lru.retain(|&k| k != key);
            self.layout_lru.push(key);
            arc
        } else {
            let new_layout = Arc::new(generate_layout(&forest, key));
            // Evict oldest while over capacity.
            while self.layout_lru.len() >= LAYOUT_CACHE_CAPACITY {
                if let Some(oldest) = self.layout_lru.first().copied() {
                    self.layout_lru.remove(0);
                    self.layout_cache.remove(&oldest);
                } else { break; }
            }
            self.layout_cache.insert(key, new_layout.clone());
            self.layout_lru.push(key);
            new_layout
        };

        // Phase 3: build the scene (re-grown grove of subdirs as trees, files
        // as ground litter — see forest::generate_layout for layout rules).
        let new_scene = Scene::from_subtree(&forest, key, layout);

        // Phase 4: hand it to the renderer, then stash on App.
        self.renderer.as_mut().unwrap().open_detail(&new_scene);
        self.detail_scene = Some(new_scene);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_some() { return; }
        let attrs = Window::default_attributes()
            .with_title("forest_fm — prototype")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 800));
        let window = Arc::new(event_loop.create_window(attrs).expect("window"));
        let forest = self.forest.take().expect("forest");
        let main_scene = self.pending_main_scene.take().expect("main scene");
        let renderer = pollster::block_on(Renderer::new(window.clone(), forest, &main_scene));
        self.renderer = Some(renderer);
        self.main_scene = Some(main_scene);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Cheap early-out if the window isn't up yet.
        if self.renderer.is_none() { return; }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                self.renderer.as_mut().unwrap().resize(size.width, size.height);
            }

            WindowEvent::RedrawRequested => {
                let has_detail_after = {
                    let r = self.renderer.as_mut().unwrap();
                    r.render();
                    r.window.request_redraw();
                    r.has_detail()
                };
                // If detail finished closing this frame, drop our scene copy too.
                if !has_detail_after && self.detail_scene.is_some() {
                    self.detail_scene = None;
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                let cursor = self.cursor_pos;
                let active_vp = {
                    let r = self.renderer.as_ref().unwrap();
                    Self::viewport_at(r, cursor).0
                };
                match button {
                    MouseButton::Left => {
                        if pressed {
                            self.lmb_state = ButtonGesture::Down {
                                press_pos: cursor,
                                active: active_vp,
                                max_move: 0.0,
                            };
                        } else {
                            // Release — was it a click?
                            if let ButtonGesture::Down { active, max_move, .. } = self.lmb_state {
                                if max_move < CLICK_PIXEL_THRESHOLD {
                                    // Tiny defensive check: cursor still in the same vp.
                                    let r = self.renderer.as_ref().unwrap();
                                    let still_in = Self::viewport_at(r, cursor).0 == active;
                                    if still_in {
                                        if let Some(node_idx) = self.pick_in(active, cursor) {
                                            self.handle_click(node_idx);
                                        }
                                    }
                                }
                            }
                            self.lmb_state = ButtonGesture::Up;
                            self.last_mouse = None;
                        }
                    }
                    MouseButton::Right => {
                        if pressed {
                            self.rmb_state = ButtonGesture::Down {
                                press_pos: cursor,
                                active: active_vp,
                                max_move: 0.0,
                            };
                        } else {
                            self.rmb_state = ButtonGesture::Up;
                            self.last_mouse = None;
                        }
                    }
                    _ => {}
                }
            }

            WindowEvent::CursorLeft { .. } => {
                let r = self.renderer.as_mut().unwrap();
                r.set_main_hover(None);
                r.set_detail_hover(None);
                r.set_tooltip(None, (0.0, 0.0));
            }

            WindowEvent::CursorMoved { position, .. } => {
                let cur = (position.x, position.y);
                self.cursor_pos = cur;

                // Update max_move on any held buttons.
                bump_move(&mut self.lmb_state, cur);
                bump_move(&mut self.rmb_state, cur);

                // ---- DRAG: route to whichever viewport OWNS the gesture ----
                let lmb_active = match self.lmb_state {
                    ButtonGesture::Down { active, .. } => Some(active),
                    _ => None,
                };
                let rmb_active = match self.rmb_state {
                    ButtonGesture::Down { active, .. } => Some(active),
                    _ => None,
                };

                if let Some(vp) = lmb_active {
                    if let Some(prev) = self.last_mouse {
                        let dx = (cur.0 - prev.0) as f32 * 0.005;
                        let dy = (cur.1 - prev.1) as f32 * 0.005;
                        let r = self.renderer.as_mut().unwrap();
                        if let Some(cam) = camera_for(r, vp) {
                            cam.yaw -= dx;
                            cam.pitch = (cam.pitch + dy).clamp(-1.5, 1.5);
                        }
                    }
                } else if let Some(vp) = rmb_active {
                    if let Some(prev) = self.last_mouse {
                        let dx = (cur.0 - prev.0) as f32;
                        let dy = (cur.1 - prev.1) as f32;
                        let r = self.renderer.as_mut().unwrap();
                        let (_, _, _, _, vh) = Self::viewport_at(r, cur);
                        if let Some(cam) = camera_for(r, vp) {
                            cam.pan(dx, dy, vh);
                        }
                    }
                } else {
                    // No drag → hover-pick in whichever viewport contains the cursor.
                    let which = {
                        let r = self.renderer.as_ref().unwrap();
                        Self::viewport_at(r, cur).0
                    };
                    let hit = self.pick_in(which, cur);

                    // Build tooltip text from the hit (if any).
                    let tooltip_runs = hit.and_then(|node_idx| {
                        let forest = self.renderer.as_ref()?.forest.clone();
                        Some(format_tooltip_rich(&forest, node_idx as usize))
                    });

                    let r = self.renderer.as_mut().unwrap();
                    match which {
                        Viewport::Main => {
                            r.set_main_hover(hit);
                            r.set_detail_hover(None);
                        }
                        Viewport::Detail => {
                            r.set_main_hover(None);
                            r.set_detail_hover(hit);
                        }
                    }
                    r.set_tooltip_rich(tooltip_runs, (cur.0 as f32, cur.1 as f32));
                }

                self.last_mouse = Some(cur);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => (p.y as f32) / 60.0,
                };
                let factor = (-scroll * 0.15).exp();
                let cursor = self.cursor_pos; // copy out before &mut renderer
                let r = self.renderer.as_mut().unwrap();
                let (which, lx, ly, vw, vh) = Self::viewport_at(r, cursor);
                if let Some(cam) = camera_for(r, which) {
                    let (origin, dir) = cam.ray_from_cursor(lx, ly, vw, vh);
                    // Focus on ground-plane intersection if reasonable, else
                    // a point along the ray at the current camera distance.
                    let focus = if dir.y.abs() > 1e-3 {
                        let t = -origin.y / dir.y;
                        if t > 0.0 && t < 5000.0 { origin + dir * t }
                        else                     { origin + dir * cam.distance }
                    } else {
                        origin + dir * cam.distance
                    };
                    cam.zoom_to_world_point(focus, factor);
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    if let PhysicalKey::Code(code) = event.physical_key {
                        let r = self.renderer.as_mut().unwrap();
                        match code {
                            KeyCode::Escape => {
                                // First Esc closes detail; second Esc exits.
                                if r.has_detail() {
                                    r.close_detail();
                                    // Note: we don't clear self.detail_scene here;
                                    // that happens on the next RedrawRequested
                                    // once the close animation completes.
                                } else {
                                    event_loop.exit();
                                }
                            }
                            KeyCode::KeyT => r.toggle_theme(),
                            KeyCode::KeyR => {
                                let cam = r.main_camera();
                                cam.yaw = 0.6;
                                cam.pitch = 0.2;
                            }
                            KeyCode::KeyF => {
                                // Toggle folder-name labels at trunk bases.
                                let new_state = !r.main_labels.enabled;
                                r.main_labels.enabled = new_state;
                                r.detail_labels.enabled = new_state;
                            }
                            _ => {}
                        }
                    }
                }
            }

            _ => {}
        }
    }
}

// ---- free helpers (kept out of impl to sidestep borrow gymnastics) ----

fn bump_move(g: &mut ButtonGesture, cur: (f64, f64)) {
    if let ButtonGesture::Down { press_pos, max_move, .. } = g {
        let dx = (cur.0 - press_pos.0) as f32;
        let dy = (cur.1 - press_pos.1) as f32;
        let d = (dx * dx + dy * dy).sqrt();
        if d > *max_move { *max_move = d; }
    }
}

fn camera_for<'r>(r: &'r mut Renderer, vp: Viewport) -> Option<&'r mut OrbitCamera> {
    match vp {
        Viewport::Main => Some(r.main_camera()),
        Viewport::Detail => r.detail_camera(),
    }
}

/// Walk up the parent chain from `node_idx` until we reach a node whose
/// Walk up the parent chain from `node_idx` until we reach a node whose
/// parent is `scene_root`. That node IS "the tree" this hit belongs to.
/// Fallback: if we hit scene_root itself (or have no parent), return node_idx
/// unchanged — corner case for files directly under scene_root (ground litter).
fn climb_to_tree(forest: &Forest, node_idx: usize, scene_root: usize) -> usize {
    if node_idx == scene_root { return node_idx; }
    let mut cur = node_idx;
    loop {
        let p = forest.nodes[cur].parent;
        if p == forest::NO_PARENT { return cur; }      // reached forest root
        let p = p as usize;
        if p == scene_root { return cur; }
        if cur == p { return cur; }                    // self-loop guard
        cur = p;
    }
}

/// Build a 3-line tooltip for the node at `idx`:
///   Line 1: full path (with leaf name marked between {{ and }})
///   Line 2: "N подпапок" if N > 0
///   Line 3: "M файлов" if M > 0
///
/// The {{ }} markers are stripped before display; the renderer uses them to
/// know where to colour the leaf name brighter. Currently we just display
/// the plain path — colour styling via glyphon attribute ranges is a
/// follow-up.
fn format_tooltip(forest: &Forest, idx: usize) -> String {
    let n = &forest.nodes[idx];
    let (n_dirs, n_files) = forest.children_of(idx).iter().fold((0u32, 0u32), |(d, f), &c| {
        if forest.nodes[c as usize].is_dir() { (d + 1, f) } else { (d, f + 1) }
    });
    let (parent_part, leaf_name) = forest::build_full_path_split(forest, idx);

    let mut out = String::with_capacity(64);
    out.push_str(&parent_part);
    out.push_str(&leaf_name);

    if n.is_dir() {
        if n_dirs > 0 {
            out.push('\n');
            out.push_str(&format!("{} подпапок", n_dirs));
        }
        if n_files > 0 {
            out.push('\n');
            out.push_str(&format!("{} файлов", n_files));
        }
        out.push('\n');
        out.push_str(&format_bytes(n.size));
    } else {
        out.push('\n');
        out.push_str(&format_bytes(n.size));
    }
    out
}

/// Build a tooltip as coloured spans. The leaf name is rendered in a brighter
/// colour than the parent path so the user can quickly see WHAT is being
/// hovered, with the path as context.
///
/// Layout:
///   Line 1: <dim>parent path with trailing separator</dim><bright>leaf name</bright>
///   Line 2: <normal>N подпапок</normal>           (if dir and N>0)
///   Line 3: <normal>M файлов</normal>             (if dir and M>0)
///   (file case: just size)
fn format_tooltip_rich(forest: &Forest, idx: usize) -> Vec<(String, glyphon::Color)> {
    let n = &forest.nodes[idx];
    let (n_dirs, n_files) = forest.children_of(idx).iter().fold((0u32, 0u32), |(d, f), &c| {
        if forest.nodes[c as usize].is_dir() { (d + 1, f) } else { (d, f + 1) }
    });
    let (parent_part, leaf_name) = forest::build_full_path_split(forest, idx);

    let dim    = glyphon::Color::rgb(170, 170, 165);
    let bright = glyphon::Color::rgb(255, 255, 245);
    let counts = glyphon::Color::rgb(200, 200, 192);

    let mut out: Vec<(String, glyphon::Color)> = Vec::new();
    out.push((parent_part, dim));
    out.push((leaf_name, bright));

    if n.is_dir() {
        if n_dirs > 0 {
            out.push((format!("\n{} подпапок", n_dirs), counts));
        }
        if n_files > 0 {
            out.push((format!("\n{} файлов", n_files), counts));
        }
        out.push((format!("\n{}", format_bytes(n.size)), counts));
    } else {
        out.push((format!("\n{}", format_bytes(n.size)), counts));
    }
    out
}

// =====================================================================
// Formatting helpers
// =====================================================================

/// Format a byte count as "1.23 GB" / "456 MB" / "789 KB" / "12 B".
fn format_bytes(b: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = b as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{} {}", b, UNITS[u])
    } else {
        format!("{:.2} {}", v, UNITS[u])
    }
}

/// Format an integer with thousand separators: 1234567 → "1,234,567".
fn format_count(n: usize) -> String {
    let s = n.to_string();
    let chars: Vec<char> = s.chars().rev().collect();
    let mut out: Vec<char> = Vec::with_capacity(chars.len() + chars.len() / 3);
    for (i, c) in chars.iter().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(*c);
    }
    out.iter().rev().collect()
}
