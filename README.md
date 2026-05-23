# forest_fm — 3D filesystem visualizer

[Watch the demo video](https://youtu.be/z95tXfgzGaE)

A "forest" view of any directory tree, written in Rust with wgpu. Folders grow
as parabolic trees in 3D using a phototropism-inspired layout; files appear as
anti-aliased point clouds around the branches that contain them. Two
side-by-side viewports let you drill into a subtree without losing context.

## Build

Requires Rust 1.75+ (any recent stable). On Linux you also need the usual
graphics dev libraries (Vulkan loader, libxkbcommon, libwayland-client, etc.).
On Windows / macOS there are no extra system prerequisites.

```bash
cd forest_fm
cargo build --release
```

First build pulls a lot of dependencies (wgpu, glyphon, glam, …) and can take
a few minutes. Subsequent builds are seconds.

## Run

The program accepts a scan path as the first command-line argument and an
optional max-depth as the second. If no path is given, it scans the current
working directory.

```bash
# scan a specific directory
./target/release/forest_fm /path/to/scan

# scan with a custom max depth (default is 32)
./target/release/forest_fm /path/to/scan 16

# Windows
forest_fm.exe "C:\Program Files"
forest_fm.exe C:\ 16
```

You can also **drag a folder onto the executable** in your file manager — the
folder path is passed in as argv[1].

### Adding to the Windows context menu

Save the following as `register.reg`, edit the path, double-click. Then any
folder gets a "View as forest" option in its right-click menu:

```reg
Windows Registry Editor Version 5.00

[HKEY_CLASSES_ROOT\Directory\shell\forest_fm]
@="View as forest"
"Icon"="C:\\Tools\\forest_fm.exe"

[HKEY_CLASSES_ROOT\Directory\shell\forest_fm\command]
@="\"C:\\Tools\\forest_fm.exe\" \"%1\""

[HKEY_CLASSES_ROOT\Directory\Background\shell\forest_fm]
@="View as forest"
"Icon"="C:\\Tools\\forest_fm.exe"

[HKEY_CLASSES_ROOT\Directory\Background\shell\forest_fm\command]
@="\"C:\\Tools\\forest_fm.exe\" \"%V\""
```

## Controls

| Input               | Action                                                                                |
|---------------------|---------------------------------------------------------------------------------------|
| **LMB drag**        | Orbit camera (in the viewport where the drag started)                                 |
| **LMB click**       | Drill into the hovered subtree → opens the right viewport with that folder as root   |
| **RMB drag**        | Pan camera                                                                            |
| **Wheel**           | Zoom to cursor (in the viewport under the cursor)                                     |
| **T**               | Toggle dark / light theme                                                             |
| **R**               | Reset main camera                                                                     |
| **F**               | Toggle folder-name labels at trunk bases                                              |
| **Esc**             | Close the detail panel; press again to quit                                           |

Hovering over a tree highlights its entire subtree and shows a tooltip with
the full path and folder statistics. Clicking on the trunk drills in.

## What you see

- Each folder becomes a tree with a trunk, branches for its subfolders, and
  files as leaves in the crown.
- **Trunk length / branch thickness** scale with `log(subtree_count)`, so a
  folder containing a million files stands out among neighbours that hold
  only a few hundred.
- **Leaf size** scales with file size (logarithmic). Clamped to 1–8 px on
  screen so giants don't fill the viewport at close zoom and tiny files
  don't disappear at far zoom.
- **Leaf color** encodes file age: bright green = recently modified, yellow
  = medium, red = old.
- **Branch color** is a muted blend of brown and the average-age tint of the
  files underneath it.
- Subtle **wind sway** animates leaves so overlapping crowns parallax
  slightly — the motion gives your visual system the depth cue that a still
  point cloud lacks.
- **Folder labels** at trunk bases show the names of the largest folders
  (up to 200 per viewport). Press `F` to toggle them.

The right viewport, when open, shows a drill-down "forest" of the clicked
folder: each of its immediate subfolders becomes its own tree.

## Memory and performance

Memory footprint scales linearly with the node count. Typical numbers for a
Windows `C:\` scan with ~4M nodes:

- **Base** (just the main forest): ~1.0 GB
- **With one drill-down active**: ~1.3–1.5 GB

Several memory optimizations are in place:

- **Compact node representation**: 32 bytes per node (was 100+) — `Box<str>`
  for names, flat children-pool index ranges instead of per-node `Vec`s,
  `u32` parent index with sentinel for root, age stored as `u8`.
- **Geometry separated from nodes**: positions live only in `Layout`, not on
  the `Node` struct, so a single forest can have multiple layouts in flight
  (main + drill-down).
- **Flat-arena octree** for picking — no per-node `Box` allocations.
- **Split pickables**: `PickableLeaf` (20 B) and `PickableBranch` (44 B) are
  packed into separate pools; the octree stores a packed u32 handle (high
  bit = kind, low 31 = index) instead of one 48-byte combined struct.
- **LRU layout cache** for drill-downs (capacity = 1), so re-clicking the
  same folder is instant but old layouts are freed quickly.
- **Layout dropped after main scene is built** — saves ~112 MB once the GPU
  buffers + picker have taken what they need.

## Drill-down layout algorithm

The slow part of a drill-in is computing the new layout. On dense folders
(e.g. `C:\Windows\WinSxS`, 31k top-level subdirectories) the original
algorithm was O(N²) and took ~25 seconds. Two fixes brought it to ~3 s,
mostly bounded by layout I/O:

- **Spatial hash grid** for trunk placement. Each tree only collision-checks
  against neighbours in the same 16-unit grid cell and the immediate ring
  around it, instead of every previously-placed trunk.
- **Incremental `used_area`** accumulator — was being recomputed from
  scratch each iteration (O(N) inside O(N) loop).
- **Phototropism windowed** to the last 16 siblings when expanding a
  fan-out branch, instead of every sibling placed so far.

## File layout

```
src/
  main.rs       — window + event loop + key/mouse handling
  forest.rs     — filesystem scan, growth layout, geometry helpers
  scene.rs      — scenes (visible-node sets, camera state, split animation)
  picking.rs    — pickable pool, flat-arena octree, ray query
  renderer.rs   — wgpu pipelines, vertex buffers, camera, render loop
  shader.wgsl   — point-billboard with analytic AA, line shader
  tooltip.rs    — hover tooltip with text rendering (glyphon)
  labels.rs     — thin sans-serif folder labels at trunk bases (glyphon)
```

## What's NOT here yet

- **Disk cache** of scan results — every launch re-walks the filesystem.
  Planned: `%APPDATA%/forest_fm/cache/<hash-of-path>.bin` with bincode.
- **Filesystem watcher** for live updates.
- **In-app folder picker** (open-folder dialog).
- **Search**.
- **Smooth morph animation** when drilling in — currently the right
  viewport just snaps to the new layout. Several attempts were prototyped
  and reverted; the architecture for a faithful "tree falls apart into a
  grove" morph turned out larger than the current code can absorb cleanly.

## Tuning knobs

`forest.rs::generate_layout`:
- `breathing_space` (4.0) — padding between trees during placement
- `golden_angle` (2.399…) — base spiral angle for trunk distribution
- `MAX_DEPTH_DEFAULT` (32) — scan recursion cap

`forest.rs::layout_children_into`:
- `base_angle` formula — how widely sub-branches splay (decreases with
  depth so deep branches stay tighter)
- `BASE_LEN`, `FILE_W`, `DIR_W`, `SUBTREE_W` — branch length weights
- the recency window for phototropism (currently `last_16`)

`renderer.rs::build_gpu_data`:
- `radius = 0.012 + size01 * 0.06` — world-space leaf radius range
- `thickness_px = 0.7 + mass01 * 2.8` — branch thickness range in pixels
- `alpha` formulas for leaves and branches
- `age_colormap` — color gradient by file age

`shader.wgsl::vs_point`:
- `min_px = 1.0`, `max_px = 8.0` — leaf size clamp on screen
- the `sway` block — wind animation amplitude and frequency

`labels.rs`:
- `FONT_SIZE` (12.0)
- `LABEL_ALPHA_DARK` / `LABEL_ALPHA_LIGHT` — label opacity per theme
- `MAX_LABELS` (200) — cap on labels per scene
- `MAX_NAME_CHARS` (32) — truncation length for long folder names
