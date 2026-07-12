//! The `three-d` renderer for an [`SseqScene`].
//!
//! This is the only module that depends on a graphics library. It consumes the renderer-agnostic
//! [`SseqScene`] produced by [`super::scene`] and draws it as a 3D chart: one small sphere per
//! class, thin cylinders for differentials/products/axes.
//!
//! Two entry points:
//! - [`run`] / [`show`] open an interactive, orbitable window with arrow-key page switching.
//! - [`render_to_png`] / [`render_to_png_from`] render a single page offscreen and write a PNG,
//!   without ever entering the interactive loop (useful for scripted captures and CI).

use three_d::*;

use super::scene::{Edge, EdgeKind, Node, SseqScene};
use crate::{Product, Sseq, SseqProfile};

/// Radius of a class node, in world units.
const NODE_RADIUS: f32 = 0.08;
/// Radius of a differential/product line, in world units.
const EDGE_RADIUS: f32 = 0.015;
/// Radius of a grid/axis line, in world units.
const GRID_RADIUS: f32 = 0.006;

/// Shown when a window / GL context cannot be created — almost always missing runtime libraries.
const WINDOW_HINT: &str = "failed to open a window or GL context.\n\
    On Linux the visualizer needs X11/Wayland and OpenGL libraries available at runtime.\n\
    - Inside the Nix dev shell: the `ext` devShell exports LD_LIBRARY_PATH with libX11, \
      libxkbcommon, wayland and libGL (see ext/flake.nix); enter it with `nix develop ./ext`.\n\
    - On a headless machine: run under `xvfb-run`, or capture offscreen with \
      `SSEQ_VIZ3D_SCREENSHOT=out.png`.";

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Extract the geometry of `sseq` (generic over the profile) and open the interactive window.
pub fn show<P: SseqProfile<3>>(sseq: &Sseq<3, P>, products: &[(String, Product<3>)]) {
    let max_r = super::scene::max_page(sseq);
    run(super::scene::extract_scene(sseq, products, max_r));
}

/// Extract the geometry of `sseq` and render a single `page` offscreen to a PNG at `path`.
pub fn render_to_png_from<P: SseqProfile<3>>(
    sseq: &Sseq<3, P>,
    products: &[(String, Product<3>)],
    page: usize,
    width: u32,
    height: u32,
    path: &str,
) {
    let max_r = super::scene::max_page(sseq);
    render_to_png(
        &super::scene::extract_scene(sseq, products, max_r),
        page,
        width,
        height,
        path,
    );
}

/// Open a window and render `scene` interactively. Blocks until the window is closed.
pub fn run(scene: SseqScene) {
    let window = Window::new(WindowSettings {
        title: "Sseq<3> — 3D visualizer".to_string(),
        max_size: Some((1280, 800)),
        ..Default::default()
    })
    .expect(WINDOW_HINT);
    let context = window.gl();

    let mut camera = frame_camera(&scene, window.viewport());
    let mut control = OrbitControl::new(scene_center(&scene), 0.5 * scene_span(&scene), 20.0 * scene_span(&scene));

    let mut nodes = instanced(&context, CpuMesh::sphere(16), Srgba::new(30, 30, 30, 255));
    let mut edges = instanced(&context, CpuMesh::cylinder(12), Srgba::WHITE);
    let mut grid = instanced(&context, CpuMesh::cylinder(8), Srgba::WHITE);
    grid.set_instances(&grid_instances(scene.min, scene.max));

    let (light, ambient) = lights(&context);

    // Precompute the per-page instance data so the render loop (which must be `'static`) owns it.
    let node_instances: Vec<Instances> =
        scene.pages.iter().map(|p| node_instances(&p.nodes)).collect();
    let edge_instances_per_page: Vec<Instances> =
        scene.pages.iter().map(|p| edge_instances(&p.edges)).collect();
    let page_rs: Vec<i32> = scene.pages.iter().map(|p| p.r).collect();
    let n_pages = scene.pages.len();

    let mut current: usize = 0;
    if let Some(i) = node_instances.get(current) {
        nodes.set_instances(i);
    }
    if let Some(i) = edge_instances_per_page.get(current) {
        edges.set_instances(i);
    }
    if let Some(r) = page_rs.get(current) {
        println!("Showing page E_{r}  (←/→ to change, {n_pages} pages)");
    }

    window.render_loop(move |mut frame_input| {
        control.handle_events(&mut camera, &mut frame_input.events);
        camera.set_viewport(frame_input.viewport);

        let mut changed = false;
        for event in &frame_input.events {
            if let Event::KeyPress { kind, .. } = event {
                match kind {
                    Key::ArrowRight if current + 1 < n_pages => {
                        current += 1;
                        changed = true;
                    }
                    Key::ArrowLeft if current > 0 => {
                        current -= 1;
                        changed = true;
                    }
                    _ => {}
                }
            }
        }
        if changed {
            if let Some(i) = node_instances.get(current) {
                nodes.set_instances(i);
            }
            if let Some(i) = edge_instances_per_page.get(current) {
                edges.set_instances(i);
            }
            if let Some(r) = page_rs.get(current) {
                println!("Showing page E_{r}");
            }
        }

        frame_input
            .screen()
            .clear(ClearState::color_and_depth(0.96, 0.96, 0.96, 1.0, 1.0))
            .render(
                &camera,
                nodes.into_iter().chain(&edges).chain(&grid),
                &[&light, &ambient],
            );

        FrameOutput::default()
    });
}

/// Render a single `page` of `scene` offscreen and write it to `path` as a PNG.
///
/// This does not enter the interactive loop and shows no visible window beyond the momentary GL
/// context needed to render. three-d 0.19 has no surfaceless context, so a GL-capable display is
/// still required — on a headless box run this under `xvfb-run`.
pub fn render_to_png(scene: &SseqScene, page: usize, width: u32, height: u32, path: &str) {
    // A window is the only way to obtain a native GL context in three-d; we render to an offscreen
    // target rather than to its surface, and never run the event loop.
    let window = Window::new(WindowSettings {
        title: "sseq viz3d (offscreen)".to_string(),
        max_size: Some((width, height)),
        ..Default::default()
    })
    .expect(WINDOW_HINT);
    let context = window.gl();

    let viewport = Viewport::new_at_origo(width, height);
    let camera = frame_camera(scene, viewport);
    let (light, ambient) = lights(&context);

    let mut nodes = instanced(&context, CpuMesh::sphere(16), Srgba::new(30, 30, 30, 255));
    let mut edges = instanced(&context, CpuMesh::cylinder(12), Srgba::WHITE);
    let mut grid = instanced(&context, CpuMesh::cylinder(8), Srgba::WHITE);
    grid.set_instances(&grid_instances(scene.min, scene.max));
    if let Some(p) = scene.pages.get(page) {
        nodes.set_instances(&node_instances(&p.nodes));
        edges.set_instances(&edge_instances(&p.edges));
    }

    let color = Texture2D::new_empty::<[u8; 4]>(
        &context,
        width,
        height,
        Interpolation::Linear,
        Interpolation::Linear,
        None,
        Wrapping::ClampToEdge,
        Wrapping::ClampToEdge,
    );
    let depth = DepthTexture2D::new::<f32>(
        &context,
        width,
        height,
        Wrapping::ClampToEdge,
        Wrapping::ClampToEdge,
    );
    let target = RenderTarget::new(color.as_color_target(None), depth.as_depth_target());
    target
        .clear(ClearState::color_and_depth(0.96, 0.96, 0.96, 1.0, 1.0))
        .render(
            &camera,
            nodes.into_iter().chain(&edges).chain(&grid),
            &[&light, &ambient],
        );

    let pixels: Vec<[u8; 4]> = target.read_color();
    write_png(path, width, height, &pixels);
    println!("Wrote screenshot to {path}");
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Centre of the scene's bounding box.
fn scene_center(scene: &SseqScene) -> Vec3 {
    vec3(
        (scene.min[0] + scene.max[0]) as f32 / 2.0,
        (scene.min[1] + scene.max[1]) as f32 / 2.0,
        (scene.min[2] + scene.max[2]) as f32 / 2.0,
    )
}

/// Largest extent of the scene's bounding box (at least 1, to frame a single point).
fn scene_span(scene: &SseqScene) -> f32 {
    ((scene.max[0] - scene.min[0]).max(scene.max[1] - scene.min[1]))
        .max(scene.max[2] - scene.min[2])
        .max(1) as f32
}

/// A perspective camera framing the whole scene from a fixed offset.
fn frame_camera(scene: &SseqScene, viewport: Viewport) -> Camera {
    let center = scene_center(scene);
    let span = scene_span(scene);
    Camera::new_perspective(
        viewport,
        center + vec3(span * 1.5, span * 1.2, span * 2.0),
        center,
        vec3(0.0, 1.0, 0.0),
        degrees(45.0),
        0.1,
        1000.0,
    )
}

fn lights(context: &Context) -> (DirectionalLight, AmbientLight) {
    (
        DirectionalLight::new(context, 1.0, Srgba::WHITE, vec3(-0.5, -1.0, -0.7)),
        AmbientLight::new(context, 0.6, Srgba::WHITE),
    )
}

/// A fresh instanced object (no instances yet) with the given base mesh and albedo.
fn instanced(context: &Context, mesh: CpuMesh, albedo: Srgba) -> Gm<InstancedMesh, PhysicalMaterial> {
    Gm::new(
        InstancedMesh::new(context, &Instances::default(), &mesh),
        PhysicalMaterial::new(
            context,
            &CpuMaterial {
                albedo,
                ..Default::default()
            },
        ),
    )
}

/// The transformation placing a unit-length, x-aligned cylinder onto the segment `a`–`b`.
fn segment_transform(a: Vec3, b: Vec3, radius: f32) -> Mat4 {
    let dir = b - a;
    let len = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z).sqrt();
    if len < f32::EPSILON {
        // Degenerate segment; hide it by scaling to nothing.
        return Mat4::from_scale(0.0);
    }
    let rot = Quat::from_arc(vec3(1.0, 0.0, 0.0), dir / len, None);
    Mat4::from_translation(a) * Mat4::from(rot) * Mat4::from_nonuniform_scale(len, radius, radius)
}

fn color_for(kind: &EdgeKind) -> Srgba {
    match kind {
        EdgeKind::Differential(_) => Srgba::new(40, 90, 220, 255), // blue
        EdgeKind::Structline(_) => Srgba::new(30, 150, 70, 255),   // green
    }
}

fn node_instances(nodes: &[Node]) -> Instances {
    Instances {
        transformations: nodes
            .iter()
            .map(|n| Mat4::from_translation(n.pos.into()) * Mat4::from_scale(NODE_RADIUS))
            .collect(),
        ..Default::default()
    }
}

fn edge_instances(edges: &[Edge]) -> Instances {
    Instances {
        transformations: edges
            .iter()
            .map(|e| segment_transform(e.src.into(), e.dst.into(), EDGE_RADIUS))
            .collect(),
        colors: Some(edges.iter().map(|e| color_for(&e.kind)).collect()),
        ..Default::default()
    }
}

/// The static grid: faint lines parallel to the `n` and `s` axes on the base plane, plus the three
/// coloured axes emanating from the minimum corner.
fn grid_instances(min: [i32; 3], max: [i32; 3]) -> Instances {
    let (x0, y0, z0) = (min[0] as f32, min[1] as f32, min[2] as f32);
    let (x1, y1) = (max[0] as f32, max[1] as f32);
    let grid = Srgba::new(150, 150, 150, 255);

    let mut transformations = Vec::new();
    let mut colors = Vec::new();

    for n in min[0]..=max[0] {
        transformations.push(segment_transform(
            vec3(n as f32, y0, z0),
            vec3(n as f32, y1, z0),
            GRID_RADIUS,
        ));
        colors.push(grid);
    }
    for s in min[1]..=max[1] {
        transformations.push(segment_transform(
            vec3(x0, s as f32, z0),
            vec3(x1, s as f32, z0),
            GRID_RADIUS,
        ));
        colors.push(grid);
    }

    // Coloured axes: n (red), s (green), w (blue).
    let origin = vec3(x0, y0, z0);
    for (end, color) in [
        (vec3(max[0] as f32, y0, z0), Srgba::new(200, 60, 60, 255)),
        (vec3(x0, max[1] as f32, z0), Srgba::new(60, 160, 60, 255)),
        (vec3(x0, y0, max[2] as f32), Srgba::new(60, 60, 200, 255)),
    ] {
        transformations.push(segment_transform(origin, end, GRID_RADIUS * 1.6));
        colors.push(color);
    }

    Instances {
        transformations,
        colors: Some(colors),
        ..Default::default()
    }
}

/// Write RGBA pixels (bottom-left origin, as returned by GL) to a PNG, flipping to top-left origin.
fn write_png(path: &str, width: u32, height: u32, pixels: &[[u8; 4]]) {
    let (w, h) = (width as usize, height as usize);
    let mut data = Vec::with_capacity(w * h * 4);
    for y in (0..h).rev() {
        for x in 0..w {
            data.extend_from_slice(&pixels[y * w + x]);
        }
    }

    let file = match std::fs::File::create(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("failed to create screenshot {path}: {e}");
            return;
        }
    };
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    match encoder.write_header().and_then(|mut w| w.write_image_data(&data)) {
        Ok(()) => {}
        Err(e) => eprintln!("failed to encode screenshot {path}: {e}"),
    }
}
