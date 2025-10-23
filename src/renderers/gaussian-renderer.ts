import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  update_scaling: (scaling: number) => void;
}

// Utility to create GPU buffers
const createBuffer = (
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBuffer | ArrayBufferView
) => {
  const buffer = device.createBuffer({ label, size, usage });
  if (data) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
): GaussianRenderer {

  const sorter = get_sorter(pc.num_points, device);
  
  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  const nulling_data = new Uint32Array([0]);
  // To clear number of points to 0 before preprocess
  const null_buffer = createBuffer(
    device,
    'null buffer',
    4,
    GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    nulling_data
  );
  device.queue.writeBuffer(null_buffer, 0, nulling_data);

  // vertexCount: u32, instanceCount: u32, firstVertex: u32, firstInstance: u32
  const indirect_draw_buffer = createBuffer(
    device,
    'indirect draw',
    4 * 4,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    new Uint32Array([6, pc.num_points, 0, 0])
  );

  const splat_buffer = createBuffer(
    device,
    'splat buffer',
    pc.num_points * 5 * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  );

  const render_settings_buffer = createBuffer(
    device,
    'render settings',
    8, // 2 f32
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    new Float32Array([1.0, pc.sh_deg]) // default scaling 1.0
  );

  // 'auto' seemingly removes unused bindings!!! So make them manually
  
  const camera_bind_group_layout = device.createBindGroupLayout({
    label: 'camera bind group layout',
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE | GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' }
      }
    ]
  });

  const gaussian_bind_group_layout = device.createBindGroupLayout({
    label: 'gaussian bind group layout',
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'read-only-storage' }
      }
    ]
  });

  const sort_bind_group_layout = device.createBindGroupLayout({
    label: 'sort bind group layout',
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' }
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' }
      }
    ]
  });

  const preprocess_pipeline_layout = device.createPipelineLayout({
    label: 'preprocess pipeline layout',
    bindGroupLayouts: [
      camera_bind_group_layout,
      gaussian_bind_group_layout,
      sort_bind_group_layout
    ]
  });

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const preprocess_pipeline = device.createComputePipeline({
    label: 'preprocess',
    layout: preprocess_pipeline_layout,
    compute: {
      module: device.createShaderModule({ code: preprocessWGSL }),
      entryPoint: 'preprocess',
      constants: {
        workgroupSize: C.histogram_wg_size,
        sortKeyPerThread: c_histogram_block_rows,
      },
    },
  });

  const sort_bind_group = device.createBindGroup({
    label: 'sort',
    layout: sort_bind_group_layout,
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
    ],
  });


  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  
  const render_splat_bind_group_layout = device.createBindGroupLayout({
    label: 'render splat bind group layout',
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'read-only-storage' }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'read-only-storage' }
      }
    ]
  });

  const render_pipeline_layout = device.createPipelineLayout({
    label: 'render pipeline layout',
    bindGroupLayouts: [
      camera_bind_group_layout,
      render_splat_bind_group_layout
    ]
  });
  
  const render_shader = device.createShaderModule({code: renderWGSL});
  const render_pipeline = device.createRenderPipeline({
    label: 'gaussian render',
    layout: render_pipeline_layout,
    vertex: {
      module: render_shader,
      entryPoint: 'vs_main',
    },
    fragment: {
      module: render_shader,
      entryPoint: 'fs_main',
      targets: [{
        format: presentation_format,
        blend: {
          color: {
            srcFactor: 'src-alpha',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add',
          },
          alpha: {
            srcFactor: 'one',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add',
          },
        },
      }],
    },
    primitive: {
      topology: 'triangle-list',
      cullMode: 'back',
      frontFace: 'ccw',
    }
  });

  const camera_settings_bind_group = device.createBindGroup({
    label: 'gaussian camera',
    layout: camera_bind_group_layout,
    entries: [
      {binding: 0, resource: { buffer: camera_buffer }},
      {binding: 1, resource: { buffer: render_settings_buffer }},
    ],
  });

  const gaussian_splat_bind_group = device.createBindGroup({
    label: 'gaussian splats',
    layout: gaussian_bind_group_layout,
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 1, resource: { buffer: splat_buffer } },
      { binding: 2, resource: { buffer: pc.sh_buffer } },
    ],
  });

  const splat_bind_group = device.createBindGroup({
    label: 'splat bind group',
    layout: render_splat_bind_group_layout,
    entries: [
      { binding: 0, resource: { buffer: splat_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
    ],
  });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  let preprocess_func = (encoder: GPUCommandEncoder) => {
    let preprocess_pass = encoder.beginComputePass({
        label: 'preprocess',
    });
    preprocess_pass.setPipeline(preprocess_pipeline);
    preprocess_pass.setBindGroup(0, camera_settings_bind_group);
    preprocess_pass.setBindGroup(1, gaussian_splat_bind_group);
    preprocess_pass.setBindGroup(2, sort_bind_group);
    const workgroups_needed = Math.ceil(pc.num_points / C.histogram_wg_size);
    preprocess_pass.dispatchWorkgroups(workgroups_needed);
    preprocess_pass.end();
  };

  let render_func = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    const pass = encoder.beginRenderPass({
      label: 'gaussian render',
      colorAttachments: [
        {
          view: texture_view,
          loadOp: 'clear',
          storeOp: 'store',
        }
      ],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, camera_settings_bind_group);
    pass.setBindGroup(1, splat_bind_group);

    pass.drawIndirect(indirect_draw_buffer, 0);
    pass.end();
  };

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {

      encoder.copyBufferToBuffer(null_buffer, 0, sorter.sort_info_buffer, 0, 4); // Clear keys_size to 0
      encoder.copyBufferToBuffer(null_buffer, 0, sorter.sort_dispatch_indirect_buffer, 0, 4); // Clear dispatch_x to 0
      // encoder.copyBufferToBuffer(null_buffer, 0, indirect_draw_buffer, 4, 4); // Set instance count to 0

      preprocess_func(encoder);
      
      sorter.sort(encoder);

      encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, indirect_draw_buffer, 4, 4); // Copy key size to indirect draw buffer

      render_func(encoder, texture_view);
    },
    camera_buffer,
    update_scaling: (scaling: number) => {
      device.queue.writeBuffer(render_settings_buffer, 0, new Float32Array([scaling, pc.sh_deg]));
    },
  };
}
