// labels.rs — draws folder names at the base of each top-level tree.
//
// Each top-level folder (direct child of `forest.root`) gets a thin
// sans-serif label rendered just below where its trunk meets the ground
// plane. Labels are projected from 3D world space onto screen each frame,
// so they follow the camera naturally.
//
// Rendering uses glyphon (the same crate as the Tooltip), but a separate
// TextRenderer instance so the two don't fight for the atlas. One Buffer
// is created per label and cached — folder names don't change after scan.

use glyphon::{
    Attrs, Buffer, Cache, Color, Family, FontSystem, Metrics, Resolution, Shaping,
    SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};
use wgpu::{Device, MultisampleState, Queue, RenderPass, TextureFormat};
use glam::{Mat4, Vec3, Vec4};

use crate::forest::Forest;
use crate::scene::Scene;

const FONT_SIZE: f32 = 12.0;
const LINE_HEIGHT: f32 = 14.0;
/// Y-offset in screen pixels from the projected trunk base down to the label
/// top-left. Positive = below the trunk.
const LABEL_DROP_PX: f32 = 6.0;
/// Maximum label width before truncation. Long folder names get cut with `…`.
const MAX_NAME_CHARS: usize = 32;
/// Alpha (0–255) for the label fill. Low for an unobtrusive look — about 10%.
const LABEL_ALPHA_DARK: u8 = 25;
const LABEL_ALPHA_LIGHT: u8 = 40;

/// One label per top-level folder. Buffers are built once at scene
/// construction, screen positions are recomputed every frame.
pub struct Labels {
    font_system: FontSystem,
    swash_cache: SwashCache,
    viewport: Viewport,
    atlas: TextAtlas,
    text_renderer: TextRenderer,
    /// One Buffer per label. Index aligns with `world_positions`.
    buffers: Vec<Buffer>,
    /// World position of each label anchor (base of the trunk).
    world_positions: Vec<Vec3>,
    /// True when labels list is non-empty.
    visible: bool,
    /// User toggle (F key). When false, prepare/render skip work entirely.
    pub enabled: bool,
}

impl Labels {
    pub fn new(device: &Device, queue: &Queue, format: TextureFormat) -> Self {
        let font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let cache = Cache::new(device);
        let viewport = Viewport::new(device, &cache);
        let mut atlas = TextAtlas::new(device, queue, &cache, format);
        let text_renderer = TextRenderer::new(
            &mut atlas, device, MultisampleState::default(), None,
        );
        Self {
            font_system, swash_cache, viewport, atlas, text_renderer,
            buffers: Vec::new(),
            world_positions: Vec::new(),
            visible: false,
            enabled: true,
        }
    }

    /// Rebuild label set from a scene. Call this when the scene changes
    /// (open detail, close detail, main scene first load). Each top-level
    /// folder of `scene.root_idx` becomes one label, capped to the largest
    /// `MAX_LABELS` so we don't spend seconds shaping text for tens of
    /// thousands of trees the user can barely see anyway.
    pub fn rebuild_from_scene(&mut self, forest: &Forest, scene: &Scene) {
        self.buffers.clear();
        self.world_positions.clear();

        const MAX_LABELS: usize = 200;

        // Collect candidate top-level folders (dirs only) with subtree size.
        let root_idx = scene.root_idx;
        let children = forest.children_of(root_idx);
        let layout = scene.layout.as_deref();
        let mut candidates: Vec<(usize, u32)> = children.iter()
            .filter_map(|&c| {
                let idx = c as usize;
                let n = &forest.nodes[idx];
                if !n.is_dir() { return None; }
                let g = crate::forest::geom_of(forest, layout, idx);
                if g.branch_length < 0.5 { return None; }
                Some((idx, n.subtree_count))
            })
            .collect();
        // Sort by subtree size descending — biggest folders get labels first.
        candidates.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        candidates.truncate(MAX_LABELS);

        for (idx, _) in candidates {
            let n = &forest.nodes[idx];
            let g = crate::forest::geom_of(forest, layout, idx);

            let mut name = n.name.as_ref().to_string();
            if name.chars().count() > MAX_NAME_CHARS {
                let truncated: String = name.chars().take(MAX_NAME_CHARS - 1).collect();
                name = format!("{truncated}…");
            }

            let mut buf = Buffer::new(
                &mut self.font_system,
                Metrics::new(FONT_SIZE, LINE_HEIGHT),
            );
            buf.set_size(&mut self.font_system, Some(220.0), Some(LINE_HEIGHT * 1.5));
            buf.set_text(
                &mut self.font_system,
                &name,
                Attrs::new().family(Family::SansSerif),
                Shaping::Advanced,
            );
            buf.shape_until_scroll(&mut self.font_system, false);

            self.buffers.push(buf);
            // Anchor at the trunk base.
            self.world_positions.push(g.position);
        }

        self.visible = !self.buffers.is_empty();
    }

    /// Clear labels (e.g. when the right viewport closes).
    pub fn clear(&mut self) {
        self.buffers.clear();
        self.world_positions.clear();
        self.visible = false;
    }

    /// Update per-frame state: project each anchor to screen, build TextAreas,
    /// upload glyphs. `view_proj`, `viewport_w/h` describe the target rect
    /// (sub-viewport for split-screen). `theme_is_light` selects the color.
    pub fn prepare(
        &mut self,
        device: &Device,
        queue: &Queue,
        view_proj: &Mat4,
        viewport_w: u32,
        viewport_h: u32,
        theme_is_light: bool,
    ) {
        self.viewport.update(queue, Resolution {
            width: viewport_w,
            height: viewport_h,
        });

        if !self.visible || !self.enabled {
            let _ = self.text_renderer.prepare(
                device, queue,
                &mut self.font_system,
                &mut self.atlas,
                &self.viewport,
                std::iter::empty::<TextArea>(),
                &mut self.swash_cache,
            );
            return;
        }

        // Project each anchor; cull anything behind camera or off-screen.
        let vw = viewport_w as f32;
        let vh = viewport_h as f32;
        let alpha = if theme_is_light { LABEL_ALPHA_LIGHT } else { LABEL_ALPHA_DARK };
        let (r, g, b) = if theme_is_light { (30u8, 30u8, 30u8) } else { (235u8, 235u8, 235u8) };

        // We need a stable holding place for screen coords (TextArea borrows the
        // buffer). Compute first, then build TextArea iterator from them.
        let mut screen_pos = Vec::with_capacity(self.buffers.len());
        for &p in &self.world_positions {
            let clip = *view_proj * Vec4::new(p.x, p.y, p.z, 1.0);
            if clip.w <= 0.05 { screen_pos.push(None); continue; }
            let ndc_x = clip.x / clip.w;
            let ndc_y = clip.y / clip.w;
            // To pixel coords (top-left origin).
            let sx = (ndc_x * 0.5 + 0.5) * vw;
            let sy = (1.0 - (ndc_y * 0.5 + 0.5)) * vh;
            // Off-screen with generous margin → cull.
            if sx < -100.0 || sx > vw + 100.0 || sy < -50.0 || sy > vh + 50.0 {
                screen_pos.push(None);
            } else {
                screen_pos.push(Some((sx, sy)));
            }
        }

        let areas = self.buffers.iter().enumerate().filter_map(|(i, buf)| {
            let (sx, sy) = screen_pos[i]?;
            Some(TextArea {
                buffer: buf,
                // Centre the label horizontally under the trunk: shift left
                // by ~half the text width. We don't have exact measured width
                // here without re-shaping; an approximation by name length
                // is close enough for unobtrusive labels.
                left: sx - 4.0 * FONT_SIZE * 0.25,
                top: sy + LABEL_DROP_PX,
                scale: 1.0,
                bounds: TextBounds {
                    left: 0,
                    top: 0,
                    right: vw as i32,
                    bottom: vh as i32,
                },
                default_color: Color::rgba(r, g, b, alpha),
                custom_glyphs: &[],
            })
        });

        let _ = self.text_renderer.prepare(
            device, queue,
            &mut self.font_system,
            &mut self.atlas,
            &self.viewport,
            areas,
            &mut self.swash_cache,
        );
    }

    /// Draw labels inside the active render pass.
    pub fn render<'pass>(&'pass self, pass: &mut RenderPass<'pass>) {
        if !self.visible || !self.enabled { return; }
        let _ = self.text_renderer.render(&self.atlas, &self.viewport, pass);
    }
}
