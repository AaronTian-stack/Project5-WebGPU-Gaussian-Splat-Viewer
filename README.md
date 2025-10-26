# Project5-WebGPU-Gaussian-Splat-Viewer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 5**

* Aaron Tian
  * [LinkedIn](https://www.linkedin.com/in/aaron-c-tian/), [personal website](https://aarontian-stack.github.io/)
* Tested on: Windows 22H2 (26100.6584), Intel Core Ultra 7 265k @ 3.90GHz, 32GB RAM, RTX 5070 12GB (release driver 581.15)

## Live Demo

[Link](https://aarontian-stack.github.io/Project5-WebGPU-Gaussian-Splat-Viewer/)

[![live demo](images/demo.gif)](https://aarontian-stack.github.io/Project5-WebGPU-Gaussian-Splat-Viewer/)

## Summary

A WebGPU app for rendering Gaussian splats as described in [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). This allows for real time rendering of high quality radiance fields.

## Implementation

The app renders pretrained Gaussian splat models in the `.ply` format. A preprocessing compute shader is used to perform tasks such as transforming the Gaussians' positions, calculating the covariance matrix, and calculating size and color (this is the "splatting"). The Gaussians are then sorted back-to-front using radix sort. Finally the Gaussians (now splats) are rendered as quads using a single indirect draw call that had its instance count set in the preprocessing step.

### Preprocessing Compute Shader

Gaussian positions are transformed into NDC space using the camera's view-projection matrix. The covariance matrix is calculated from the Gaussian's rotation and scale. Each Gaussian stores spherical harmonic coefficients for calculating color based off the view direction. This information is appended to a buffer of 2D splats to be rendered along with a separate buffer for sorting those splats. This also keeps track of the number of splats that survive view-frustum culling, allowing the number of splats to be sorted to be copied to the indirect draw buffer later (the workgroup size for the sorting pass is also calculated here).

### Rendering

Each splat is rendered as a quad. The size simply comes from the result in the preprocessing step. For each fragment of the quad we determine if it is inside the splat (ellipse) using the centered matrix equation. If the test passes, we output the color of the splat. The opacity of the color decays exponentially based off the distance of the center of the splat.

## Performance Analysis

With either the Bicycle or Bonsai splat scenes, the average frame rate is the same and never drops below my max refresh rate of 100 FPS, regardless of settings, so I discuss theoretical performance implications of different settings below.

### Point Cloud vs Gaussian Splatting

The point cloud renders each Gaussian using point primitives, so no triangles are involved compared to splatting which uses quads (2 triangles per Gaussian). There is also no shading work in the fragment shader for point clouds (outputs constant color). Thus the Gaussian splatting incurs the additional costs of:
* Triangle rasterization
* Shading/Blending

### Workgroup Size

A larger workgroup size in the preprocessing step may offer benefits such as better memory coalescing, especially considering our work is a 1D dispatch. We do not use shared memory in the shader, so that is not relevant here. If the preprocessing step uses too many registers on a CU it might be better to reduce the workgroup size to mitigate spilling to shared memory. However to my knowledge it is not possible to determine this kind of information in WebGPU.


### View-frustum Culling

By culling splats that are outside the view-frustum, we can reduce the number of splats that need to be sorted and rendered. This should reduce compute pressure (from sorting step) and graphics pipeline pressure (less rasterization, shading, blending). The check for culling in the preprocessing shader is very simple, but it causes an early return. Depending on the order of the Gaussians in memory, this could cause some divergence in the preprocessing step.

### Number of Gaussians

A larger amount of Gaussians should increase the workload in all parts of the application. Loading from disk will take longer, preprocessing will take longer, sorting will take longer, and rendering will take longer due to more primitives.

## Bloopers

My Gaussians got flattened into a line...

![wtf](images/wtf.png)

## Credits

- [Vite](https://vitejs.dev/)
- [tweakpane](https://tweakpane.github.io/docs//v3/monitor-bindings/)
- [stats.js](https://github.com/mrdoob/stats.js)
- [wgpu-matrix](https://github.com/greggman/wgpu-matrix)
- Special Thanks to: Shrek Shao (Google WebGPU team) & [Differential Gaussian Renderer](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
