const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortKeyPerThread: u32;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

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
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    position: u32, // 2 f16
    color: array<u32, 2>, // 4 f16
    conic: array<u32, 2>, // 4 f16. cov + radius
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> render_settings: RenderSettings;

@group(1) @binding(0) var<storage, read> gaussians: array<Gaussian>;
@group(1) @binding(1) var<storage, read_write> splats: array<Splat>;
@group(1) @binding(2) var<storage, read> sh_coeffs: array<u32>;

@group(2) @binding(0) var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1) var<storage, read_write> sort_depths: array<u32>;
@group(2) @binding(2) var<storage, read_write> sort_indices: array<u32>;
@group(2) @binding(3) var<storage, read_write> sort_dispatch: DispatchIndirect;

fn fast_modulo(a: u32, b: u32) -> u32 {
    return a - (a / b) * b;
}

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    // 16 max coefficients, each is f16 so 8*3 for 3 channels
    // Offset by coefficient, divide by 2 for f16. Then get color in that coefficient
    let channel = fast_modulo(c_idx, 2);
    let base_idx = splat_idx * 24 + (c_idx / 2) * 3 + channel;

    let color01 = unpack2x16float(sh_coeffs[base_idx + 0]);
    let color23 = unpack2x16float(sh_coeffs[base_idx + 1]);

    if (fast_modulo(c_idx, 2) == 0u) {
        return vec3f(color01.x, color01.y, color23.x);
    }

    return vec3f(color01.y, color23.x, color23.y);
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    // Transform Gaussian position to NDC for culling
    let gaussian = gaussians[idx];
    
    // Unpack position and opacity
    let pos_XY = unpack2x16float(gaussian.pos_opacity[0]); // x, y
    let pos_Y_opacity = unpack2x16float(gaussian.pos_opacity[1]); // z, opacity
    let pos = vec3<f32>(pos_XY.x, pos_XY.y, pos_Y_opacity.x);
    let opacity = 1.0 / (1.0 + exp(-pos_Y_opacity.y)); // Sigmoid

    let view_pos = camera.view * vec4<f32>(pos, 1.0);
    var pos_ndc = camera.proj * view_pos;
    pos_ndc /= pos_ndc.w;
    if (abs(pos_ndc.x) > 1.2 || abs(pos_ndc.y) > 1.2 || view_pos.z < 0.0) {
        return;
    }

    // Unpack rotation and scale
    let rot_WX = unpack2x16float(gaussian.rot[0]);
    let rot_YZ = unpack2x16float(gaussian.rot[1]);
    let rot = vec4<f32>(rot_WX.y, rot_YZ.x, rot_YZ.y, rot_WX.x); // XYZW quaternion
    
    let scale_0 = exp(unpack2x16float(gaussian.scale[0])); // scale_0, scale_1
    let scale_1 = exp(unpack2x16float(gaussian.scale[1])); // scale_2, padding
    let scale = vec3<f32>(scale_0.x, scale_0.y, scale_1.x);

    let R = mat3x3<f32>(
        1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z), 2.0 * (rot.x * rot.y - rot.w * rot.z), 2.0 * (rot.x * rot.z + rot.w * rot.y),
        2.0 * (rot.x * rot.y + rot.w * rot.z), 1.0 - 2.0 * (rot.x * rot.x + rot.z * rot.z), 2.0 * (rot.y * rot.z - rot.w * rot.x),
        2.0 * (rot.x * rot.z - rot.w * rot.y), 2.0 * (rot.y * rot.z + rot.w * rot.x), 1.0 - 2.0 * (rot.x * rot.x + rot.y * rot.y)
    );
    let S = mat3x3<f32>(
        render_settings.gaussian_scaling * scale.x, 0.0, 0.0,
        0.0, render_settings.gaussian_scaling * scale.y, 0.0,
        0.0, 0.0, render_settings.gaussian_scaling * scale.z
    );

    let cov3D = transpose(S * R) * S * R;

    let t = view_pos.xyz;
    let J = mat3x3<f32>(
        camera.focal.x / t.z, 0.0, -(camera.focal.x * t.x) / (t.z * t.z),
        0.0, camera.focal.y / t.z, -(camera.focal.y * t.y) / (t.z * t.z),
        0.0, 0.0, 0.0
    );

    let W = transpose(mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz));

    let T = W * J;

    let Vrk = mat3x3<f32>(
        cov3D[0][0], cov3D[0][1], cov3D[0][2],
        cov3D[0][1], cov3D[1][1], cov3D[1][2],
        cov3D[0][2], cov3D[1][2], cov3D[2][2]
    );

    var cov2D = transpose(T) * transpose(Vrk) * T;
    cov2D[0][0] += 0.3;
    cov2D[1][1] += 0.3;

    let cov2D_flat = vec3<f32>(
        cov2D[0][0],
        cov2D[0][1],
        cov2D[1][1]
    );

    let det = cov2D_flat.x * cov2D_flat.z - cov2D_flat.y * cov2D_flat.y;
    if (det == 0.0) {
        return;
    }
    let det_inv = 1.0 / det;
    let conic = vec3<f32>(cov2D_flat.z * det_inv, -cov2D_flat.y * det_inv, cov2D_flat.x * det_inv); // Upper triangle of matrix

    let mid = 0.5 * (cov2D_flat.x + cov2D_flat.z);
    let lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    let lambda2 = mid - sqrt(max(0.1, mid * mid - det));
    let radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));

    let cam_pos = -camera.view[3].xyz;
    let view_dir = normalize(pos - cam_pos);
    let color = computeColorFromSH(view_dir, idx, u32(render_settings.sh_deg));

    // Append the splat
    let splat_idx = atomicAdd(&sort_infos.keys_size, 1u);
    splats[splat_idx].position = pack2x16float(pos_ndc.xy);
    splats[splat_idx].color[0] = pack2x16float(color.xy);
    splats[splat_idx].color[1] = pack2x16float(vec2<f32>(color.z, opacity));
    splats[splat_idx].conic[0] = pack2x16float(conic.xy);
    splats[splat_idx].conic[1] = pack2x16float(vec2<f32>(conic.z, radius));

    sort_indices[splat_idx] = splat_idx;

    var depth_uint = bitcast<u32>(-view_pos.z);
    // Flip bits for sorting in radix
    // If sign bit is set (negative), flip all bits
    // If sign bit is not set (positive), flip only sign bit
    let mask = select(0x80000000u, 0xFFFFFFFFu, (depth_uint & 0x80000000u) != 0u);
    sort_depths[splat_idx] = depth_uint ^ mask;

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    if (splat_idx % keys_per_dispatch == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}