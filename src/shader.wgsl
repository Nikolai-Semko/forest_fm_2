// shader.wgsl — two pipelines (point billboards + capsule lines).
//
// Highlight model:
//   The uniform carries `hl_range = (start, end, _, _)`. Any instance whose
//   DFS preorder index satisfies start <= dfs_pre < end is "highlighted".
//   This range encodes "this subtree is hovered/selected" in O(1) GPU work
//   per instance — no per-frame buffer rewrites.
//
//   Special case: when start == end, no instance matches. We always exclude
//   start == 0 && end == 0 (interpreted as "nothing highlighted").

struct Camera {
    view_proj: mat4x4<f32>,
    cam_right: vec4<f32>,
    cam_up:    vec4<f32>,
    viewport:  vec4<f32>,   // (w, h, time, theme_flag) theme: 0=dark, 1=light
    hl_range:  vec4<u32>,   // (dfs_start, dfs_end, _, _)
};

@group(0) @binding(0) var<uniform> camera: Camera;

fn is_highlighted(dfs_pre: u32) -> bool {
    // Empty range guard: when both 0, nothing is highlighted.
    if (camera.hl_range.x == 0u && camera.hl_range.y == 0u) { return false; }
    return dfs_pre >= camera.hl_range.x && dfs_pre < camera.hl_range.y;
}

/// True if ANY highlight is currently active (cursor hovering some subtree).
/// When false, no dimming is applied — scene looks normal.
fn any_highlight_active() -> bool {
    return !(camera.hl_range.x == 0u && camera.hl_range.y == 0u);
}

fn theme_is_light() -> bool {
    return camera.viewport.w > 0.5;
}

// -------------- POINT PIPELINE --------------

struct PointInstance {
    @location(0) position: vec3<f32>,
    @location(1) radius:   f32,
    @location(2) color:    vec4<f32>,
    @location(3) age01:    f32,
    @location(4) dfs_pre:  u32,
};

struct PointVsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) radius_px: f32,
    @location(3) alpha_scale: f32,
    @location(4) highlight: f32,    // 0 or 1
};

@vertex
fn vs_point(@builtin(vertex_index) vid: u32, inst: PointInstance) -> PointVsOut {
    var corners = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0,  1.0),
    );
    let c = corners[vid];

    let t = camera.viewport.z;
    let phase = inst.position.x * 1.7 + inst.position.y * 2.3 + inst.position.z * 1.1;
    let sway = vec3<f32>(
        sin(t * 0.6 + phase) * 0.015,
        cos(t * 0.5 + phase) * 0.012,
        sin(t * 0.7 + phase * 0.5) * 0.015,
    );
    let world = inst.position + sway;

    let clip_center = camera.view_proj * vec4<f32>(world, 1.0);
    let w = max(clip_center.w, 0.001);
    let world_per_px = w * 2.0 / camera.viewport.y;
    let natural_px = inst.radius / world_per_px;

    let highlighted = is_highlighted(inst.dfs_pre);
    // INVERTED HIGHLIGHT MODEL:
    //   Previously: highlighted leaves grew bigger (3x area → 10x fill rate).
    //   That destroyed fps when hovering a subtree with millions of leaves.
    //   Now: highlighted leaves stay at natural size, NON-highlighted leaves
    //   are dimmed instead (alpha reduction, no size change). Fill rate at
    //   hover time is identical or LOWER than no-hover state.
    let min_px = 0.7;
    let effective_px = max(natural_px, min_px);
    let area_natural = natural_px * natural_px;
    let area_effective = effective_px * effective_px;
    let alpha_scale = clamp(area_natural / area_effective, 0.0, 1.0);

    let r = effective_px * world_per_px;
    let world_quad = world + camera.cam_right.xyz * (c.x * r) + camera.cam_up.xyz * (c.y * r);

    var out: PointVsOut;
    out.clip = camera.view_proj * vec4<f32>(world_quad, 1.0);
    out.uv = c;
    out.color = inst.color;
    out.radius_px = effective_px;
    out.alpha_scale = alpha_scale;
    // Pass 2 bits of state through one float: 0.0 = normal (no highlight active),
    // 0.5 = dimmed (highlight active, this instance is NOT highlighted),
    // 1.0 = focused (this instance IS highlighted).
    if (highlighted) {
        out.highlight = 1.0;
    } else if (any_highlight_active()) {
        out.highlight = 0.5;
    } else {
        out.highlight = 0.0;
    }
    return out;
}

@fragment
fn fs_point(in: PointVsOut) -> @location(0) vec4<f32> {
    let d = length(in.uv);
    let aa = clamp(1.0 / max(in.radius_px, 1.0), 0.02, 0.5);
    let alpha = 1.0 - smoothstep(1.0 - aa, 1.0, d);

    var rgb = in.color.rgb;
    var alpha_final = alpha * in.color.a * in.alpha_scale;

    if (in.highlight > 0.75) {
        // FOCUSED: this leaf is inside the hovered subtree. Boost color a bit
        // for clear distinction, but DO NOT grow its size — fill rate stays
        // bounded even on huge subtrees.
        rgb = rgb * 1.2;
        alpha_final = alpha_final * 1.15;
    } else if (in.highlight > 0.25) {
        // DIMMED: a hover is active somewhere else. Reduce alpha so the
        // surrounding forest fades into the background and the focused
        // subtree stands out by contrast.
        alpha_final = alpha_final * 0.12;
    } else if (theme_is_light()) {
        // Light theme without hover: darken leaf colors slightly so they're
        // visible on the pale background.
        rgb = rgb * 0.75;
    }

    if (alpha_final < 0.002) { discard; }
    return vec4<f32>(rgb, alpha_final);
}

// -------------- LINE (CAPSULE) PIPELINE --------------

struct LineInstance {
    @location(0) p0:            vec3<f32>,
    @location(1) thickness_px:  f32,
    @location(2) p1:            vec3<f32>,
    @location(3) dfs_pre:       u32,
    @location(4) color:         vec4<f32>,
};

struct LineVsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) thickness_px: f32,
    @location(3) highlight: f32,
};

@vertex
fn vs_line(@builtin(vertex_index) vid: u32, inst: LineInstance) -> LineVsOut {
    var corners = array<vec2<f32>, 4>(
        vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0), vec2<f32>(-1.0,  1.0),
    );
    let c = corners[vid];

    var clip0 = camera.view_proj * vec4<f32>(inst.p0, 1.0);
    var clip1 = camera.view_proj * vec4<f32>(inst.p1, 1.0);

    let NEAR_W: f32 = 0.01;
    let w0_bad = clip0.w < NEAR_W;
    let w1_bad = clip1.w < NEAR_W;

    if (w0_bad && w1_bad) {
        var out_degen: LineVsOut;
        out_degen.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out_degen.color = inst.color;
        out_degen.uv = c;
        out_degen.thickness_px = 0.0;
        out_degen.highlight = 0.0;
        return out_degen;
    }
    if (w0_bad) {
        let t = (NEAR_W - clip0.w) / (clip1.w - clip0.w);
        clip0 = mix(clip0, clip1, t);
    } else if (w1_bad) {
        let t = (NEAR_W - clip1.w) / (clip0.w - clip1.w);
        clip1 = mix(clip1, clip0, t);
    }

    let highlighted = is_highlighted(inst.dfs_pre);
    // INVERTED HIGHLIGHT (see vs_point): branches stay at natural thickness,
    // non-highlighted branches are dimmed in the fragment shader instead.
    let thickness_eff = inst.thickness_px;

    let endpoint_clip = select(clip1, clip0, c.x < 0.0);
    let ndc0 = clip0.xy / clip0.w;
    let ndc1 = clip1.xy / clip1.w;
    var dir = ndc1 - ndc0;
    let dir_len = length(dir);
    if (dir_len < 1e-5) { dir = vec2<f32>(1.0, 0.0); } else { dir = dir / dir_len; }
    let perp = vec2<f32>(-dir.y, dir.x);

    let half_thick_ndc_y = (thickness_eff * 0.5) / camera.viewport.y * 2.0;
    let half_thick_ndc_x = (thickness_eff * 0.5) / camera.viewport.x * 2.0;
    let perp_offset = vec2<f32>(perp.x * half_thick_ndc_x, perp.y * half_thick_ndc_y);

    let ndc_xy = endpoint_clip.xy / endpoint_clip.w + perp_offset * c.y;
    var out: LineVsOut;
    out.clip = vec4<f32>(
        ndc_xy.x * endpoint_clip.w,
        ndc_xy.y * endpoint_clip.w,
        endpoint_clip.z,
        endpoint_clip.w,
    );
    out.color = inst.color;
    out.uv = c;
    out.thickness_px = thickness_eff;
    // 3-state encoding: 0.0 = normal, 0.5 = dimmed, 1.0 = focused.
    if (highlighted) {
        out.highlight = 1.0;
    } else if (any_highlight_active()) {
        out.highlight = 0.5;
    } else {
        out.highlight = 0.0;
    }
    return out;
}

@fragment
fn fs_line(in: LineVsOut) -> @location(0) vec4<f32> {
    let d = abs(in.uv.y);
    let aa = clamp(1.0 / max(in.thickness_px * 0.5, 1.0), 0.05, 0.5);
    let alpha = 1.0 - smoothstep(1.0 - aa, 1.0, d);
    let thin = clamp(in.thickness_px, 0.0, 1.0);
    var alpha_final = alpha * in.color.a * mix(thin * thin, 1.0, thin);

    var rgb = in.color.rgb;
    if (in.highlight > 0.75) {
        // FOCUSED branch (in hovered subtree): brighten + boost alpha.
        rgb = rgb * 1.4 + vec3<f32>(0.1, 0.1, 0.05);
        alpha_final = max(alpha_final, alpha * 0.6);
    } else if (in.highlight > 0.25) {
        // DIMMED branch (hover active elsewhere): fade into background.
        alpha_final = alpha_final * 0.15;
    } else if (theme_is_light()) {
        rgb = rgb * 0.55;
        alpha_final = alpha_final * 1.4;
    }

    if (alpha_final < 0.002) { discard; }
    return vec4<f32>(rgb, alpha_final);
}
