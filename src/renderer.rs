// renderer.rs — wgpu rendering with two pipelines (points, lines).
//
// Two-viewport (split-screen) support:
// - We allocate ONE set of GPU buffers per Scene (point/line instances), built
//   from that scene's `visible_nodes`. The main scene contains the whole forest;
//   the optional detail scene contains a subtree.
// - During render we issue separate passes per viewport using set_viewport()
//   + set_scissor_rect(). Each viewport gets its own CameraUniform with its
//   own view-proj. The shader code is shared — it just reads a different
//   uniform binding.
// - Highlight is per-viewport: each scene maintains its own hover/select
//   state independently. We pass `hl_range: vec2<u32>` in the uniform — any
//   instance whose dfs_pre falls in [start, end) gets a glow boost. This
//   gives O(1) subtree highlighting on GPU without per-frame buffer rebuilds.
//
// Split-screen animation:
// - SplitAnimation::t ∈ [0,1] drives the layout split AND the right-side
//   camera morph. At t=0 the detail viewport doesn't render. At t=1 it
//   occupies 40% of the window.
// - When opening, the detail camera is seeded with the main camera's pose
//   and exponentially eased toward its natural framing — this creates the
//   visual effect of one tree "popping out" of the forest into its own view.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::forest::Forest;
use crate::scene::{CameraState, Scene, SplitAnimation};
use crate::tooltip::Tooltip;

// ---------- GPU types ----------

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    cam_right: [f32; 4],
    cam_up: [f32; 4],
    /// (w, h, time, theme_flag) where theme_flag is 0 for dark, 1 for light.
    viewport: [f32; 4],
    /// Highlight range in DFS preorder indices: instances with dfs_pre in
    /// [x, y) get an emphasis boost. (0,0) means no hover. z, w reserved.
    hl_range: [u32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PointInstance {
    pub position: [f32; 3],
    pub radius: f32,
    pub color: [f32; 4],
    pub age01: f32,
    pub dfs_pre: u32,     // DFS preorder index for highlight test
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LineInstance {
    pub p0: [f32; 3],
    pub thickness_px: f32,
    pub p1: [f32; 3],
    pub dfs_pre: u32,     // DFS preorder index for highlight test
    pub color: [f32; 4],
}

// ---------- Camera (orbit) ----------

pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    inv_vp: Mat4,
    last_aspect: f32,
}

impl OrbitCamera {
    pub fn new(target: Vec3, distance: f32, yaw: f32, pitch: f32) -> Self {
        Self { target, distance, yaw, pitch, inv_vp: Mat4::IDENTITY, last_aspect: 1.0 }
    }
    pub fn from_state(s: CameraState) -> Self {
        Self::new(s.target, s.distance, s.yaw, s.pitch)
    }
    pub fn to_state(&self) -> CameraState {
        CameraState { target: self.target, distance: self.distance, yaw: self.yaw, pitch: self.pitch }
    }
    pub fn eye(&self) -> Vec3 {
        let cp = self.pitch.cos();
        let dir = Vec3::new(self.yaw.sin() * cp, self.pitch.sin(), self.yaw.cos() * cp);
        self.target + dir * self.distance
    }
    pub fn view_proj(&mut self, aspect: f32) -> (Mat4, Vec3, Vec3) {
        let eye = self.eye();
        let view = Mat4::look_at_rh(eye, self.target, Vec3::Y);
        let proj = Mat4::perspective_rh(45f32.to_radians(), aspect, 0.05, 5000.0);
        let vp = proj * view;
        self.inv_vp = vp.inverse();
        self.last_aspect = aspect;
        let f = (self.target - eye).normalize();
        let r = f.cross(Vec3::Y).normalize();
        let u = r.cross(f).normalize();
        (vp, r, u)
    }
    /// Recompute view-proj for a given aspect and return the matrix without
    /// any side effect borrow gymnastics. Camera caches inv_vp internally.
    pub fn cached_view_proj(&self) -> Mat4 {
        // After view_proj() has been called this frame, inv_vp is set;
        // we reconstruct the forward matrix from its inverse.
        self.inv_vp.inverse()
    }
    fn unproject(&self, ndc_x: f32, ndc_y: f32, ndc_z: f32) -> Vec3 {
        self.inv_vp.project_point3(glam::Vec3::new(ndc_x, ndc_y, ndc_z))
    }
    /// Cast a ray from the camera eye through (mouse_x, mouse_y) in **viewport-local**
    /// pixel coordinates (not window coordinates — caller must subtract the
    /// viewport origin first). Returns (origin, direction). Direction is unit.
    pub fn ray_from_cursor(&self, mouse_x: f32, mouse_y: f32, vp_w: f32, vp_h: f32) -> (Vec3, Vec3) {
        let nx = (mouse_x / vp_w) * 2.0 - 1.0;
        let ny = 1.0 - (mouse_y / vp_h) * 2.0;
        let near = self.unproject(nx, ny, 0.0);
        let far = self.unproject(nx, ny, 1.0);
        let dir = (far - near).normalize_or_zero();
        (near, dir)
    }
    pub fn zoom_to_world_point(&mut self, world_pt: Vec3, factor: f32) {
        let eye = self.eye();
        let to_eye = eye - world_pt;
        let to_target = self.target - world_pt;
        let new_eye = world_pt + to_eye * factor;
        let new_target = world_pt + to_target * factor;
        self.target = new_target;
        self.distance = (new_eye - new_target).length().clamp(0.5, 5000.0);
    }
    pub fn zoom(&mut self, factor: f32) {
        self.distance = (self.distance * factor).clamp(0.5, 5000.0);
    }
    pub fn pan(&mut self, dx: f32, dy: f32, viewport_h: f32) {
        let eye = self.eye();
        let forward = (self.target - eye).normalize_or_zero();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        let world_per_pixel = 2.0 * (22.5_f32.to_radians()).tan() * self.distance / viewport_h;
        self.target -= right * dx * world_per_pixel;
        self.target += up * dy * world_per_pixel;
    }
}

// ---------- Per-scene GPU resources ----------

struct SceneGpu {
    point_buf: wgpu::Buffer,
    point_count: u32,
    line_buf: wgpu::Buffer,
    line_count: u32,
    camera: OrbitCamera,
    /// Hovered subtree root (node index). None = no hover.
    hover: Option<u32>,
    /// Selected subtree root (persistent across hover changes). None = no
    /// click selection. Used by the LEFT viewport to remember which subtree
    /// is currently expanded into the right panel.
    selected: Option<u32>,
}

// ---------- Theme ----------

#[derive(Clone, Copy, Debug)]
pub enum Theme { Dark, Light }

impl Theme {
    fn bg(self) -> wgpu::Color {
        match self {
            Theme::Dark => wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            Theme::Light => wgpu::Color { r: 0.94, g: 0.94, b: 0.92, a: 1.0 },
        }
    }
    fn as_uniform_flag(self) -> f32 {
        match self { Theme::Dark => 0.0, Theme::Light => 1.0 }
    }
    pub fn toggle(self) -> Theme {
        match self { Theme::Dark => Theme::Light, Theme::Light => Theme::Dark }
    }
}

// ---------- Renderer ----------

pub struct Renderer {
    pub window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    main_uniform_buf: wgpu::Buffer,
    main_bind_group: wgpu::BindGroup,
    detail_uniform_buf: wgpu::Buffer,
    detail_bind_group: wgpu::BindGroup,
    /// Layout kept so we could rebuild bind groups on demand (currently unused).
    _bgl: wgpu::BindGroupLayout,

    point_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,

    depth_view: wgpu::TextureView,

    main_gpu: SceneGpu,
    detail_gpu: Option<SceneGpu>,

    /// Auto-framed target camera for the currently-opening detail scene.
    /// While split.target_t == 1, the detail camera approaches this each frame.
    detail_target_camera: Option<CameraState>,

    pub forest: Arc<Forest>,

    pub split: SplitAnimation,
    pub theme: Theme,
    /// Hover tooltip (shared between viewports — only one is hovered at a time).
    pub tooltip: Tooltip,
    /// Folder-name labels at the base of each top-level tree, one set per
    /// viewport (so the right panel can have labels for its drill-down trees).
    pub main_labels: crate::labels::Labels,
    pub detail_labels: crate::labels::Labels,
    start_time: std::time::Instant,
    last_frame_time: std::time::Instant,
}

impl Renderer {
    pub async fn new(window: Arc<Window>, forest: Arc<Forest>, main_scene: &Scene) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await.expect("adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                }, None,
            )
            .await.expect("device");

        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps.formats.iter().copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // ---- shader & bind group layout ----
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Two independent uniform buffers (one per viewport). Two bind groups
        // pointing at them. We cannot reuse one buffer because the render pass
        // is recorded once and submitted — write_buffer mid-pass isn't a thing.
        let make_uniform = |label: &str| -> (wgpu::Buffer, wgpu::BindGroup) {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<CameraUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &bgl,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
            });
            (buf, bg)
        };
        let (main_uniform_buf, main_bind_group) = make_uniform("uniform-main");
        let (detail_uniform_buf, detail_bind_group) = make_uniform("uniform-detail");

        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        // ---- point pipeline ----
        let point_attrs = [
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0,  shader_location: 0 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32,   offset: 12, shader_location: 1 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 2 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32,   offset: 32, shader_location: 3 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32,    offset: 36, shader_location: 4 },
        ];
        let point_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<PointInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &point_attrs,
        };
        let blend_alpha = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::OVER,
        };
        let depth_format = wgpu::TextureFormat::Depth32Float;

        let point_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pp"),
            layout: Some(&pl_layout),
            vertex: wgpu::VertexState {
                module: &shader, entry_point: "vs_point",
                buffers: &[point_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader, entry_point: "fs_point",
                targets: &[Some(wgpu::ColorTargetState {
                    format, blend: Some(blend_alpha), write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format, depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(), bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None, cache: None,
        });

        // ---- line pipeline ----
        let line_attrs = [
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0,  shader_location: 0 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32,   offset: 12, shader_location: 1 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 16, shader_location: 2 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32,    offset: 28, shader_location: 3 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 4 },
        ];
        let line_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LineInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &line_attrs,
        };
        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lp"),
            layout: Some(&pl_layout),
            vertex: wgpu::VertexState {
                module: &shader, entry_point: "vs_line",
                buffers: &[line_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader, entry_point: "fs_line",
                targets: &[Some(wgpu::ColorTargetState {
                    format, blend: Some(blend_alpha), write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format, depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(), bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None, cache: None,
        });

        let depth_view = make_depth(&device, config.width, config.height, depth_format);
        let main_gpu = build_scene_gpu(&device, &forest, main_scene);
        let tooltip = Tooltip::new(&device, &queue, format);
        let mut main_labels = crate::labels::Labels::new(&device, &queue, format);
        main_labels.rebuild_from_scene(&forest, main_scene);
        let detail_labels = crate::labels::Labels::new(&device, &queue, format);

        Self {
            window,
            surface,
            device,
            queue,
            config,
            main_uniform_buf, main_bind_group,
            detail_uniform_buf, detail_bind_group,
            _bgl: bgl,
            point_pipeline, line_pipeline,
            depth_view,
            main_gpu,
            detail_gpu: None,
            detail_target_camera: None,
            forest,
            split: SplitAnimation::closed(),
            theme: Theme::Dark,
            tooltip,
            main_labels,
            detail_labels,
            start_time: std::time::Instant::now(),
            last_frame_time: std::time::Instant::now(),
        }
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 { return; }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = make_depth(&self.device, w, h, wgpu::TextureFormat::Depth32Float);
    }

    pub fn window_size(&self) -> (f32, f32) {
        (self.config.width as f32, self.config.height as f32)
    }

    /// Compute viewport rects in window-pixel coords. Returns
    /// (main_rect, optional detail_rect) as (x, y, w, h).
    pub fn viewport_rects(&self) -> ((f32, f32, f32, f32), Option<(f32, f32, f32, f32)>) {
        let (w, h) = self.window_size();
        let s = self.split.eased();
        if s < 0.001 {
            return ((0.0, 0.0, w, h), None);
        }
        // Detail panel max width = 40% of window; main fills the rest with a small gap.
        const GAP: f32 = 2.0;
        let max_detail_w = w * 0.4;
        let detail_w = (max_detail_w * s).max(1.0);
        let main_w = (w - detail_w - GAP).max(1.0);
        let main = (0.0, 0.0, main_w, h);
        let detail = (main_w + GAP, 0.0, detail_w, h);
        (main, Some(detail))
    }

    pub fn main_camera(&mut self) -> &mut OrbitCamera { &mut self.main_gpu.camera }
    pub fn detail_camera(&mut self) -> Option<&mut OrbitCamera> {
        self.detail_gpu.as_mut().map(|g| &mut g.camera)
    }

    pub fn set_main_hover(&mut self, node_idx: Option<u32>) { self.main_gpu.hover = node_idx; }
    pub fn set_detail_hover(&mut self, node_idx: Option<u32>) {
        if let Some(g) = &mut self.detail_gpu { g.hover = node_idx; }
    }

    /// Update tooltip text + position. None text hides the tooltip.
    pub fn set_tooltip(&mut self, text: Option<String>, cursor: (f32, f32)) {
        match text {
            Some(t) if !t.is_empty() => {
                self.tooltip.set_text(t);
                self.tooltip.set_position(cursor.0, cursor.1);
            }
            _ => self.tooltip.clear(),
        }
    }

    /// Multi-coloured tooltip. `runs` is a list of (text, color) spans drawn
    /// without separator. Use empty/None to hide.
    pub fn set_tooltip_rich(&mut self, runs: Option<Vec<(String, glyphon::Color)>>, cursor: (f32, f32)) {
        match runs {
            Some(r) if !r.is_empty() && r.iter().any(|(t, _)| !t.is_empty()) => {
                self.tooltip.set_rich_text(&r);
                self.tooltip.set_position(cursor.0, cursor.1);
            }
            _ => self.tooltip.clear(),
        }
    }

    /// Toggle theme.
    pub fn toggle_theme(&mut self) {
        self.theme = self.theme.toggle();
    }

    /// Open (or replace) the detail viewport.
    pub fn open_detail(&mut self, detail_scene: &Scene) {
        let mut gpu = build_scene_gpu(&self.device, &self.forest, detail_scene);
        self.main_gpu.selected = Some(detail_scene.root_idx as u32);

        let start_cam = match &self.detail_gpu {
            Some(g) => g.camera.to_state(),
            None => self.main_gpu.camera.to_state(),
        };
        gpu.camera = OrbitCamera::from_state(start_cam);
        self.detail_gpu = Some(gpu);
        self.split.open(start_cam);
        self.detail_target_camera = Some(detail_scene.camera);

        // Refresh folder labels for the new drill-down scene.
        self.detail_labels.rebuild_from_scene(&self.forest, detail_scene);
    }

    pub fn close_detail(&mut self) {
        self.split.close();
        self.main_gpu.selected = None;
        self.detail_labels.clear();
    }

    pub fn has_detail(&self) -> bool {
        self.detail_gpu.is_some()
    }

    pub fn detail_visible_size(&self) -> Option<(f32, f32)> {
        let (_, d) = self.viewport_rects();
        d.map(|(_, _, w, h)| (w, h))
    }

    /// Render one frame.
    pub fn render(&mut self) {
        let now = std::time::Instant::now();
        let dt = (now - self.last_frame_time).as_secs_f32().min(0.1);
        self.last_frame_time = now;

        let was_open = self.split.is_open();
        self.split.step(dt);

        // While the split is fully open and we have a target camera, ease
        // toward it. This is independent of the open/close animation t — it
        // continues every frame so the user can still grab the camera and
        // reframe (the lerp speed is low enough not to fight a hold-and-drag).
        if self.split.target_t > 0.5 {
            if let (Some(g), Some(target)) = (self.detail_gpu.as_mut(), self.detail_target_camera) {
                let cur = g.camera.to_state();
                let interp_amount = (dt * 4.5).min(1.0);
                let interp = cur.lerp(&target, interp_amount);
                g.camera = OrbitCamera::from_state(interp);
                // Once close enough, drop the target so user-camera-control
                // doesn't fight us.
                let close = (interp.distance - target.distance).abs() < target.distance * 0.02
                    && (interp.target - target.target).length() < target.distance * 0.02;
                if close { self.detail_target_camera = None; }
            }
        }
        if !self.split.is_open() && was_open {
            self.detail_gpu = None;
            self.detail_target_camera = None;
        }

        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => { self.surface.configure(&self.device, &self.config); return; }
        };
        let view = frame.texture.create_view(&Default::default());
        let (main_rect, detail_rect) = self.viewport_rects();
        let t = self.start_time.elapsed().as_secs_f32();
        let theme_flag = self.theme.as_uniform_flag();

        // Build & write main uniform.
        let main_uni = build_camera_uniform(
            &mut self.main_gpu.camera,
            main_rect.2, main_rect.3,
            t, theme_flag,
            self.main_gpu.hover.or(self.main_gpu.selected),
            &self.forest,
        );
        self.queue.write_buffer(&self.main_uniform_buf, 0, bytemuck::bytes_of(&main_uni));

        // Build & write detail uniform.
        if let (Some(rect), Some(g)) = (detail_rect, self.detail_gpu.as_mut()) {
            let uni = build_camera_uniform(
                &mut g.camera,
                rect.2, rect.3,
                t, theme_flag,
                g.hover,
                &self.forest,
            );
            self.queue.write_buffer(&self.detail_uniform_buf, 0, bytemuck::bytes_of(&uni));
        }

        let mut enc = self.device.create_command_encoder(&Default::default());

        // Prepare tooltip glyph atlas BEFORE we enter the render pass.
        // glyphon::TextRenderer::prepare uploads to its internal atlas
        // texture via queue.write_texture, which must be done outside any
        // active render pass.
        self.tooltip.prepare(&self.device, &self.queue, self.config.width, self.config.height);

        // Prepare folder-name labels for both viewports. Same constraint as
        // tooltip: glyph atlas uploads must happen before any render pass.
        let main_aspect = main_rect.2 / main_rect.3.max(1.0);
        let (main_vp, _, _) = self.main_gpu.camera.view_proj(main_aspect);
        let theme_light = self.theme.as_uniform_flag() > 0.5;
        self.main_labels.prepare(
            &self.device, &self.queue,
            &main_vp,
            main_rect.2 as u32, main_rect.3 as u32,
            theme_light,
        );
        if let (Some(rect), Some(g)) = (detail_rect, self.detail_gpu.as_mut()) {
            let aspect = rect.2 / rect.3.max(1.0);
            let (vp, _, _) = g.camera.view_proj(aspect);
            self.detail_labels.prepare(
                &self.device, &self.queue,
                &vp,
                rect.2 as u32, rect.3 as u32,
                theme_light,
            );
        }

        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.theme.bg()),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None, occlusion_query_set: None,
            });

            // --- Main viewport ---
            pass.set_viewport(main_rect.0, main_rect.1, main_rect.2, main_rect.3, 0.0, 1.0);
            pass.set_scissor_rect(
                main_rect.0 as u32, main_rect.1 as u32,
                main_rect.2 as u32, main_rect.3 as u32,
            );
            // Draw order: points (leaves) FIRST, then lines (branches) on top.
            // Reason: branches are visually thin and easily obscured by dense
            // leaf clouds. Drawing them last means trunks always remain visible
            // and clickable, even under huge crowns. Solves the "trunk vanishes
            // under crown" UX issue without distorting the geometry layout.
            pass.set_pipeline(&self.point_pipeline);
            pass.set_bind_group(0, &self.main_bind_group, &[]);
            pass.set_vertex_buffer(0, self.main_gpu.point_buf.slice(..));
            pass.draw(0..4, 0..self.main_gpu.point_count);
            pass.set_pipeline(&self.line_pipeline);
            pass.set_bind_group(0, &self.main_bind_group, &[]);
            pass.set_vertex_buffer(0, self.main_gpu.line_buf.slice(..));
            pass.draw(0..4, 0..self.main_gpu.line_count);

            // --- Detail viewport ---
            if let (Some(rect), Some(g)) = (detail_rect, self.detail_gpu.as_ref()) {
                pass.set_viewport(rect.0, rect.1, rect.2, rect.3, 0.0, 1.0);
                pass.set_scissor_rect(
                    rect.0 as u32, rect.1 as u32,
                    rect.2.max(1.0) as u32, rect.3.max(1.0) as u32,
                );
                pass.set_pipeline(&self.point_pipeline);
                pass.set_bind_group(0, &self.detail_bind_group, &[]);
                pass.set_vertex_buffer(0, g.point_buf.slice(..));
                pass.draw(0..4, 0..g.point_count);
                pass.set_pipeline(&self.line_pipeline);
                pass.set_bind_group(0, &self.detail_bind_group, &[]);
                pass.set_vertex_buffer(0, g.line_buf.slice(..));
                pass.draw(0..4, 0..g.line_count);
            }
            // (main pass ends here — note no tooltip rendering inside it)
        }

        // --- Tooltip + Labels pass: separate, no depth attachment ---
        // glyphon's internal pipeline is built with `depth_stencil: None`, so
        // it cannot run inside a pass that has a depth attachment (wgpu validation
        // rejects format mismatch). We open a second short pass that LOADs the
        // already-rendered color and only adds the text on top.
        {
            let mut tip_pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("text"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None, occlusion_query_set: None,
            });

            // Main viewport labels (within main_rect).
            tip_pass.set_viewport(main_rect.0, main_rect.1, main_rect.2, main_rect.3, 0.0, 1.0);
            tip_pass.set_scissor_rect(
                main_rect.0 as u32, main_rect.1 as u32,
                main_rect.2 as u32, main_rect.3 as u32,
            );
            self.main_labels.render(&mut tip_pass);

            // Detail viewport labels.
            if let Some(rect) = detail_rect {
                tip_pass.set_viewport(rect.0, rect.1, rect.2, rect.3, 0.0, 1.0);
                tip_pass.set_scissor_rect(
                    rect.0 as u32, rect.1 as u32,
                    rect.2.max(1.0) as u32, rect.3.max(1.0) as u32,
                );
                self.detail_labels.render(&mut tip_pass);
            }

            // Tooltip is fullscreen-anchored (cursor coords are in window space).
            tip_pass.set_viewport(0.0, 0.0, self.config.width as f32, self.config.height as f32, 0.0, 1.0);
            tip_pass.set_scissor_rect(0, 0, self.config.width, self.config.height);
            self.tooltip.render(&mut tip_pass);
        }

        self.queue.submit(Some(enc.finish()));
        frame.present();
    }
}

// ---------- Helpers ----------

fn make_depth(device: &wgpu::Device, w: u32, h: u32, fmt: wgpu::TextureFormat) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&Default::default())
}

fn build_camera_uniform(
    camera: &mut OrbitCamera,
    vw: f32, vh: f32,
    time: f32, theme_flag: f32,
    highlight_node: Option<u32>,
    forest: &Forest,
) -> CameraUniform {
    let aspect = vw / vh.max(1.0);
    let (vp, right, up) = camera.view_proj(aspect);
    let (hl_start, hl_end) = match highlight_node {
        Some(idx) => {
            let n = &forest.nodes[idx as usize];
            (n.dfs_pre, n.dfs_end)
        }
        None => (0, 0),
    };
    CameraUniform {
        view_proj: vp.to_cols_array_2d(),
        cam_right: Vec4::new(right.x, right.y, right.z, 0.0).to_array(),
        cam_up: Vec4::new(up.x, up.y, up.z, 0.0).to_array(),
        viewport: [vw, vh, time, theme_flag],
        hl_range: [hl_start, hl_end, 0, 0],
    }
}

fn build_scene_gpu(device: &wgpu::Device, forest: &Forest, scene: &Scene) -> SceneGpu {
    let (point_data, line_data) = build_gpu_data(forest, &scene.visible_nodes, scene.layout.as_deref());
    let point_count = point_data.len() as u32;
    let line_count = line_data.len() as u32;
    let point_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("points"),
        contents: bytemuck::cast_slice(&point_data),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let line_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("lines"),
        contents: bytemuck::cast_slice(&line_data),
        usage: wgpu::BufferUsages::VERTEX,
    });
    SceneGpu {
        point_buf, point_count, line_buf, line_count,
        camera: OrbitCamera::from_state(scene.camera),
        hover: None, selected: None,
    }
}

/// Build (point, line) instance buffers from a subset of forest nodes.
/// Identical color/size logic to the previous monolithic builder; the only
/// new pieces are (a) `dfs_pre` per instance and (b) the loop iterates over
/// `visible` instead of all nodes. If `layout` is `Some`, positions for nodes
/// in its subtree are taken from there (drill-down regrown layout); otherwise
/// fallback to the default geometry on `forest.nodes`.
fn build_gpu_data(
    forest: &Forest,
    visible: &[usize],
    layout: Option<&crate::forest::Layout>,
) -> (Vec<PointInstance>, Vec<LineInstance>) {
    use crate::forest::geom_of;
    let mut points = Vec::with_capacity(visible.len());
    let mut lines: Vec<LineInstance> = Vec::with_capacity(visible.len());

    // age01 was precomputed once during Forest::scan — just index it now.
    // Was ~50-100ms per drill-in for 3M-node forests; now ~zero cost.
    let age = &forest.age01;

    // max_subtree restricted to the visible subset — branches in a drill-down
    // view re-rank visually (the new root is "the biggest thing in this view").
    let max_subtree = visible.iter()
        .map(|&i| forest.nodes[i].subtree_count)
        .max().unwrap_or(1) as f32;
    let log_max = max_subtree.max(2.0).ln();

    for &idx in visible {
        let n = &forest.nodes[idx];
        let g = geom_of(forest, layout, idx);
        if n.is_dir() {
            if g.branch_length < 1e-4 { continue; }
            let tip = g.position + g.branch_dir * g.branch_length;
            let sub = n.subtree_count.max(1) as f32;
            let mass01 = (sub.ln() / log_max).clamp(0.0, 1.0);
            let thickness_px = 0.7 + mass01 * 2.8;
            let alpha = 0.03 + mass01 * 0.25;

            let avg_age01 = age[idx];
            let tint = age_colormap(avg_age01);
            let brown = Vec3::new(0.55, 0.42, 0.30);
            let mixed = brown * 0.30 + tint * 0.70;

            lines.push(LineInstance {
                p0: g.position.to_array(),
                thickness_px,
                p1: tip.to_array(),
                dfs_pre: n.dfs_pre,
                color: [mixed.x, mixed.y, mixed.z, alpha],
            });
            continue;
        }

        let log_size = ((n.size as f32) + 1.0).log2();
        let size01 = (log_size / 30.0).clamp(0.0, 1.0);
        let radius = 0.012 + size01 * 0.06;
        let age01 = age[idx];
        let color = age_colormap(age01);
        let alpha = 0.08 + size01 * 0.77;
        points.push(PointInstance {
            position: g.position.to_array(),
            radius,
            color: [color.x, color.y, color.z, alpha],
            age01,
            dfs_pre: n.dfs_pre,
            _pad: [0.0; 2],
        });
    }
    (points, lines)
}

fn age_colormap(age01: f32) -> Vec3 {
    let a = age01.clamp(0.0, 1.0);
    if a < 0.5 {
        let t = a * 2.0;
        Vec3::new(t, 1.0, 0.15 * (1.0 - t))
    } else {
        let t = (a - 0.5) * 2.0;
        Vec3::new(1.0, 1.0 - t, 0.0)
    }
}
