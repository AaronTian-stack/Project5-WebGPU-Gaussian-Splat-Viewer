struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) pixel_center: vec2<f32>,
    @location(2) conic: vec3<f32>,
};

struct Splat {
    position: u32, // 2 f16
    color: array<u32, 2>, // 4 f16
    conic: array<u32, 2>, // 4 f16. cov + radius
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> render_settings: RenderSettings;

@group(1) @binding(0) var<storage, read> splats: array<Splat>;
@group(1) @binding(1) var<storage, read> sort_indices: array<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    let splat_idx = sort_indices[instance_index];
    let splat = splats[splat_idx];

    let center = unpack2x16float(splat.position);
    let conic_0 = unpack2x16float(splat.conic[0]);
    let conic_1 = unpack2x16float(splat.conic[1]);
    let quad_size = vec2<f32>(conic_1.y, conic_1.y) / camera.viewport;

    let offsets = array<vec2<f32>, 6>(
        vec2<f32>(-quad_size.x, -quad_size.y),
        vec2<f32>(quad_size.x, -quad_size.y),
        vec2<f32>(-quad_size.x, quad_size.y),
        vec2<f32>(-quad_size.x, quad_size.y),
        vec2<f32>(quad_size.x, -quad_size.y),
        vec2<f32>(quad_size.x, quad_size.y)
    );
    out.position = vec4<f32>(center + offsets[vertex_index], 0.0, 1.0);
    out.pixel_center = (center * vec2f(0.5, -0.5) + 0.5) * camera.viewport;
    out.conic = vec3<f32>(conic_0.x, conic_0.y, conic_1.x);

    let color = array<f32,4>(
        f32(unpack2x16float(splat.color[0]).x),
        f32(unpack2x16float(splat.color[0]).y),
        f32(unpack2x16float(splat.color[1]).x),
        f32(unpack2x16float(splat.color[1]).y)
    );
    out.color = vec4<f32>(color[0], color[1], color[2], color[3]);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = in.position.xy - in.pixel_center;

    let power = -0.5 * (in.conic.x * d.x * d.x + in.conic.z * d.y * d.y) - in.conic.y * d.x * d.y;

    if (power > 0.0) {
        return vec4<f32>(0.0);
    }

    let alpha = min(0.99, in.color.a * exp(power));

    return vec4<f32>(in.color.rgb, alpha);
}