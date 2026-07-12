//! The `three-d` renderer for an [`SseqScene`].
//!
//! This is the only module that depends on a graphics library. It consumes the renderer-agnostic
//! [`SseqScene`] produced by [`super::scene`] and draws it as an orbitable 3D chart: one small
//! sphere per class, thin cylinders for differentials/products/axes, and arrow-key page switching.

use three_d::*;

use super::scene::{Edge, EdgeKind, SseqScene};
use crate::{Product, Sseq, SseqProfile};

/// Radius of a class node, in world units.
const NODE_RADIUS: f32 = 0.08;
/// Radius of a differential/product line, in world units.
const EDGE_RADIUS: f32 = 0.015;
/// Radius of a grid/axis line, in world units.
const GRID_RADIUS: f32 = 0.006;

/// Convenience wrapper: extract the geometry of `sseq` (generic over the profile) and render it.
pub fn show<P: SseqProfile<3>>(sseq: &Sseq<3, P>, products: &[(String, Product<3>)]) {
    let max_r = super::scene::max_page(sseq);
    run(super::scene::extract_scene(sseq, products, max_r));
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

/// Build the static grid: faint lines parallel to the `n` and `s` axes on the base plane, plus the
/// three coloured axes emanating from the minimum corner.
fn grid_instances(min: [i32; 3], max: [i32; 3]) -> Instances {
    let (x0, y0, z0) = (min[0] as f32, min[1] as f32, min[2] as f32);
    let (x1, y1) = (max[0] as f32, max[1] as f32);
    let grid = Srgba::new(150, 150, 150, 255);

    let mut transformations = Vec::new();
    let mut colors = Vec::new();

    // Gridlines on the base plane z = z0.
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

/// Open a window and render `scene` interactively. Blocks until the window is closed.
pub fn run(scene: SseqScene) {
    let window = Window::new(WindowSettings {
        title: "Sseq<3> — 3D visualizer".to_string(),
        max_size: Some((1280, 800)),
        ..Default::default()
    })
    .unwrap();
    let context = window.gl();

    // Frame the scene: look at its centre from an offset proportional to its size.
    let center = vec3(
        (scene.min[0] + scene.max[0]) as f32 / 2.0,
        (scene.min[1] + scene.max[1]) as f32 / 2.0,
        (scene.min[2] + scene.max[2]) as f32 / 2.0,
    );
    let span = ((scene.max[0] - scene.min[0]).max(scene.max[1] - scene.min[1]))
        .max(scene.max[2] - scene.min[2])
        .max(1) as f32;
    let mut camera = Camera::new_perspective(
        window.viewport(),
        center + vec3(span * 1.5, span * 1.2, span * 2.0),
        center,
        vec3(0.0, 1.0, 0.0),
        degrees(45.0),
        0.1,
        1000.0,
    );
    let mut control = OrbitControl::new(center, 0.5 * span, 20.0 * span);

    // Nodes: instanced spheres (per-page instances are set below).
    let mut nodes = Gm::new(
        InstancedMesh::new(&context, &Instances::default(), &CpuMesh::sphere(16)),
        PhysicalMaterial::new(
            &context,
            &CpuMaterial {
                albedo: Srgba::new(30, 30, 30, 255),
                ..Default::default()
            },
        ),
    );

    // Edges: instanced cylinders, coloured per instance.
    let mut edges = Gm::new(
        InstancedMesh::new(&context, &Instances::default(), &CpuMesh::cylinder(12)),
        PhysicalMaterial::new(
            &context,
            &CpuMaterial {
                albedo: Srgba::WHITE,
                ..Default::default()
            },
        ),
    );

    // Static grid/axes.
    let grid = Gm::new(
        InstancedMesh::new(
            &context,
            &grid_instances(scene.min, scene.max),
            &CpuMesh::cylinder(8),
        ),
        PhysicalMaterial::new(
            &context,
            &CpuMaterial {
                albedo: Srgba::WHITE,
                ..Default::default()
            },
        ),
    );

    let light = DirectionalLight::new(&context, 1.0, Srgba::WHITE, vec3(-0.5, -1.0, -0.7));
    let ambient = AmbientLight::new(&context, 0.6, Srgba::WHITE);

    // Precompute the per-page instance data up front, so the render loop owns it (the closure must
    // be `'static`) rather than borrowing `scene`.
    let node_instances: Vec<Instances> = scene
        .pages
        .iter()
        .map(|p| Instances {
            transformations: p
                .nodes
                .iter()
                .map(|n| Mat4::from_translation(n.pos.into()) * Mat4::from_scale(NODE_RADIUS))
                .collect(),
            ..Default::default()
        })
        .collect();
    let edge_instances_per_page: Vec<Instances> =
        scene.pages.iter().map(|p| edge_instances(&p.edges)).collect();
    let page_rs: Vec<i32> = scene.pages.iter().map(|p| p.r).collect();
    let n_pages = scene.pages.len();

    // Optional headless capture: if `SSEQ_VIZ3D_SCREENSHOT` points at a path, render a few frames,
    // dump the framebuffer as a PPM there, and exit. Lets the visualizer be smoke-tested without a
    // human at the keyboard (e.g. under Xvfb). Interactive use is unaffected when it's unset.
    let screenshot_path = std::env::var("SSEQ_VIZ3D_SCREENSHOT").ok();
    let mut frame: u32 = 0;

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

        let screen = frame_input.screen();
        screen
            .clear(ClearState::color_and_depth(0.96, 0.96, 0.96, 1.0, 1.0))
            .render(
                &camera,
                nodes.into_iter().chain(&edges).chain(&grid),
                &[&light, &ambient],
            );

        frame += 1;
        if let Some(path) = &screenshot_path {
            // Wait a few frames for the swapchain to settle, then capture and exit.
            if frame >= 3 {
                write_ppm(path, &screen, frame_input.viewport);
                println!("Wrote screenshot to {path}");
                return FrameOutput {
                    exit: true,
                    ..Default::default()
                };
            }
        }

        FrameOutput::default()
    });
}

/// Dump the framebuffer as a binary PPM (P6). Dependency-free; used by the headless capture path.
fn write_ppm(path: &str, screen: &RenderTarget, viewport: Viewport) {
    let (w, h) = (viewport.width as usize, viewport.height as usize);
    let pixels: Vec<[u8; 4]> = screen.read_color();
    let mut buf = format!("P6\n{w} {h}\n255\n").into_bytes();
    // OpenGL reads bottom-to-top; flip vertically so the image is upright.
    for y in (0..h).rev() {
        for x in 0..w {
            let px = pixels[y * w + x];
            buf.extend_from_slice(&px[0..3]);
        }
    }
    if let Err(e) = std::fs::write(path, buf) {
        eprintln!("failed to write screenshot {path}: {e}");
    }
}
