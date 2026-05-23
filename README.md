# forest_fm — 3D file manager prototype

A minimal "forest visualization" of a directory tree, written in Rust with wgpu.
Folders grow as branches in 3D using a phototropism-inspired layout; files
appear as anti-aliased point clouds around their parent directory.

## Build

You need Rust 1.75+ (any recent stable). On Linux you also need the usual
graphics dev libraries (Vulkan loader, libxkbcommon, libwayland-client, etc.).

```bash
cd forest_fm
cargo build --release
```

First build will take a few minutes (wgpu pulls in a lot). Subsequent builds
are seconds.

## Run

```bash
# scan current directory, depth 8
cargo run --release

# scan a specific directory
cargo run --release -- /path/to/scan

# scan deeper (warning: tree explodes fast)
cargo run --release -- /path/to/scan 12
```

## Controls

- **Left-mouse drag** — orbit camera
- **Mouse wheel** — zoom
- **R** — reset orientation
- **Esc** — quit

## What you should see

- A central trunk (the root folder).
- Subdirectories branching outward, with longer branches for bigger subtrees.
- Files as little dots clustered around the tip of each branch.
- Colors: bright green = recently modified, yellow = medium, red = old.
- Size: larger files render as bigger, more opaque dots.
- A subtle "sway" animation so overlapping leaves parallax slightly — your
  visual system uses that motion cue to reconstruct depth.

## What's NOT in this prototype (intentionally)

- No database / persistent index — full rescan on every launch.
- No filesystem watcher — static snapshot.
- No picking / interaction with individual files.
- No LOD — every node is rendered every frame.
- No search.
- No labels / text rendering.

Each of those is its own next step. The current code is a foundation to feel
out the visual concept and tune the growth algorithm.

## File layout

- `src/main.rs` — window/event loop, input.
- `src/forest.rs` — filesystem scanning + biological growth layout.
- `src/renderer.rs` — wgpu pipelines, instance buffers, camera.
- `src/shader.wgsl` — point billboard with analytical AA + line shader.

## Tuning knobs to play with

In `forest.rs::grow`:
- `ROOT_LEN` — initial trunk length.
- `base_angle` formula — how wide branches splay vs depth.
- `child_len` clamp — branch length range.
- The `0.3` repulsion factor — how strongly siblings push each other apart.
- The `Vec3::Y * 0.15` bias — how much branches "seek light" upward.

In `renderer.rs::build_gpu_data`:
- `radius = 0.012 + size01 * 0.06` — world-space leaf size range.
- `alpha = 0.35 + size01 * 0.55` — leaf opacity range.
- `age_colormap` function — color gradient by file age.

In `shader.wgsl::vs_point`:
- The `sway` block — animation amplitude and frequency.
