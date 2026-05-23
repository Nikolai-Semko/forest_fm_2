// tooltip.rs — text overlay for displaying hover-info near the cursor.
//
// Uses glyphon (cosmic-text + wgpu glyph atlas) to render multiline text on
// top of the main scene without disturbing its rendering pipelines.
//
// API:
//   Tooltip::new(device, queue, format) — sets up glyphon atlas/renderer/font.
//   tooltip.set_text(lines: &[&str]) — replace what's drawn.
//   tooltip.set_position(x_px, y_px) — move the tooltip block.
//   tooltip.clear() — hide it.
//   tooltip.prepare(device, queue, viewport_w, viewport_h)
//     — call once per frame BEFORE the main render pass.
//   tooltip.render(pass) — call inside the main render pass to draw.
//
// Notes for keeping this working across glyphon versions:
//   * If glyphon API changes between minor versions, the call sites in
//     `prepare()` and `render()` may need small tweaks. Atlas/cache/renderer
//     setup is the most version-sensitive area.

use glyphon::{
    Attrs, Buffer, Cache, Color, Family, FontSystem, Metrics, Resolution, Shaping,
    SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};
use wgpu::{Device, MultisampleState, Queue, RenderPass, TextureFormat};

pub struct Tooltip {
    font_system: FontSystem,
    swash_cache: SwashCache,
    viewport: Viewport,
    atlas: TextAtlas,
    text_renderer: TextRenderer,
    buffer: Buffer,

    text: String,
    pos: (f32, f32),
    visible: bool,
}

const FONT_SIZE: f32 = 13.0;
const LINE_HEIGHT: f32 = 17.0;
const TOOLTIP_PAD: f32 = 6.0;
/// Pixel offset from cursor to top-left of tooltip block.
const CURSOR_OFFSET: (f32, f32) = (14.0, 14.0);
/// Maximum tooltip width in pixels. Long paths wrap to next line.
const MAX_WIDTH_PX: f32 = 460.0;

impl Tooltip {
    pub fn new(device: &Device, queue: &Queue, format: TextureFormat) -> Self {
        let mut font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let cache = Cache::new(device);
        let viewport = Viewport::new(device, &cache);
        let mut atlas = TextAtlas::new(device, queue, &cache, format);
        let text_renderer = TextRenderer::new(&mut atlas, device, MultisampleState::default(), None);

        // Pre-create the buffer. We mutate its text on demand. Wide buffer
        // so long paths fit without rewrapping cost.
        let mut buffer = Buffer::new(&mut font_system, Metrics::new(FONT_SIZE, LINE_HEIGHT));
        buffer.set_size(&mut font_system, Some(MAX_WIDTH_PX), Some(200.0));

        Self {
            font_system, swash_cache, viewport, atlas, text_renderer, buffer,
            text: String::new(),
            pos: (0.0, 0.0),
            visible: false,
        }
    }

    pub fn set_text(&mut self, text: impl Into<String>) {
        let s = text.into();
        if s == self.text { return; }
        self.text = s;
        self.buffer.set_text(
            &mut self.font_system,
            &self.text,
            Attrs::new().family(Family::SansSerif),
            Shaping::Advanced,
        );
        // Re-shape: walk runs once so width/height are correct in `prepare`.
        self.buffer.shape_until_scroll(&mut self.font_system, false);
        self.visible = !self.text.is_empty();
    }

    /// Set text from multiple coloured runs. Each `(text, color)` tuple is
    /// drawn with its own colour; the runs are concatenated without
    /// separator. Useful for tooltips where part of the text (e.g. the leaf
    /// name) should stand out from the rest of the path.
    ///
    /// Uses glyphon's `set_rich_text` API (cosmic-text spans). If the API
    /// surface differs in your glyphon version, fall back to `set_text` —
    /// the runs are simply concatenated without colour distinction.
    pub fn set_rich_text(&mut self, runs: &[(String, Color)]) {
        // Compose cache key: concat all run texts. Used to avoid re-shaping.
        let composed: String = runs.iter().map(|(t, _)| t.as_str()).collect();
        if composed == self.text {
            return;
        }
        self.text = composed;

        // Build the iterator of (str, Attrs) for set_rich_text.
        // We borrow each &str from the input slice — lifetime safe because
        // glyphon copies the data internally during set_rich_text.
        let default_attrs = Attrs::new().family(Family::SansSerif);
        let spans: Vec<(&str, Attrs)> = runs.iter()
            .map(|(t, color)| (
                t.as_str(),
                Attrs::new().family(Family::SansSerif).color(*color),
            ))
            .collect();

        self.buffer.set_rich_text(
            &mut self.font_system,
            spans.iter().cloned(),
            default_attrs,
            Shaping::Advanced,
        );
        self.buffer.shape_until_scroll(&mut self.font_system, false);
        self.visible = !self.text.is_empty();
    }

    pub fn set_position(&mut self, x_px: f32, y_px: f32) {
        self.pos = (x_px, y_px);
    }

    pub fn clear(&mut self) {
        self.visible = false;
        self.text.clear();
    }

    pub fn is_visible(&self) -> bool { self.visible }

    /// Compute the bounding box of the laid-out text. Used to draw the
    /// background plate and to clamp the tooltip on-screen.
    fn text_bounds(&self) -> (f32, f32) {
        // Sum of run widths × line count, approximate.
        let mut w = 0.0_f32;
        let mut lines = 0;
        for run in self.buffer.layout_runs() {
            if run.line_w > w { w = run.line_w; }
            lines += 1;
        }
        let h = (lines as f32) * LINE_HEIGHT;
        (w.min(MAX_WIDTH_PX), h)
    }

    /// Tooltip top-left, clamped to keep the box on-screen.
    fn clamped_pos(&self, vw: f32, vh: f32) -> (f32, f32) {
        let (tw, th) = self.text_bounds();
        let total_w = tw + TOOLTIP_PAD * 2.0;
        let total_h = th + TOOLTIP_PAD * 2.0;
        let mut x = self.pos.0 + CURSOR_OFFSET.0;
        let mut y = self.pos.1 + CURSOR_OFFSET.1;
        if x + total_w > vw { x = (self.pos.0 - CURSOR_OFFSET.0 - total_w).max(2.0); }
        if y + total_h > vh { y = (self.pos.1 - CURSOR_OFFSET.1 - total_h).max(2.0); }
        (x.max(0.0), y.max(0.0))
    }

    /// Called once per frame BEFORE the main render pass. Updates the glyph
    /// atlas if text or position changed.
    pub fn prepare(
        &mut self,
        device: &Device,
        queue: &Queue,
        viewport_w: u32,
        viewport_h: u32,
    ) {
        self.viewport.update(queue, Resolution {
            width: viewport_w,
            height: viewport_h,
        });

        if !self.visible {
            // Still call prepare with zero areas so the renderer state stays consistent.
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

        let (tx, ty) = self.clamped_pos(viewport_w as f32, viewport_h as f32);

        let area = TextArea {
            buffer: &self.buffer,
            left: tx + TOOLTIP_PAD,
            top: ty + TOOLTIP_PAD,
            scale: 1.0,
            bounds: TextBounds {
                left: tx as i32,
                top: ty as i32,
                right: (tx + MAX_WIDTH_PX + TOOLTIP_PAD * 2.0) as i32,
                bottom: (ty + 400.0) as i32,
            },
            // White on dark scenes, near-black on light scenes is hard to switch
            // from inside glyphon; we settle on a high-contrast off-white that
            // looks OK on both themes (just slightly less crisp on light).
            default_color: Color::rgb(240, 240, 232),
            custom_glyphs: &[],
        };

        let _ = self.text_renderer.prepare(
            device, queue,
            &mut self.font_system,
            &mut self.atlas,
            &self.viewport,
            std::iter::once(area),
            &mut self.swash_cache,
        );
    }

    /// Called inside the active render pass to draw the tooltip glyphs.
    /// Background plate is handled by a separate sprite pipeline elsewhere
    /// (see renderer.rs::tooltip_plate_pipeline).
    pub fn render<'pass>(&'pass self, pass: &mut RenderPass<'pass>) {
        if !self.visible { return; }
        let _ = self.text_renderer.render(&self.atlas, &self.viewport, pass);
    }

    /// Get computed plate rect (x, y, w, h) for the background drawer.
    /// Returns None if not visible.
    pub fn plate_rect(&self, vw: f32, vh: f32) -> Option<(f32, f32, f32, f32)> {
        if !self.visible { return None; }
        let (tw, th) = self.text_bounds();
        let (x, y) = self.clamped_pos(vw, vh);
        Some((x, y, tw + TOOLTIP_PAD * 2.0, th + TOOLTIP_PAD * 2.0))
    }
}
