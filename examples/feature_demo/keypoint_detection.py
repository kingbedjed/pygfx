"""
Image with Points Overlaid
==========================

Show an image with points overlaid.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import imageio.v3 as iio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import numpy as np
import numpy as np
from wgpu.utils.compute import compute_with_buffers

shader_src = """
@group(0) @binding(0)
var<storage, read> input_image: array<u32>;
@group(0) @binding(1)
var<storage, read_write> filtered: array<u32>;
@group(0) @binding(2)
var<storage, read> image_shape: array<u32>;

struct Inputs {
  @builtin(workgroup_id) wgid : vec3<u32>,
  @builtin(local_invocation_id) local_invocation_id : vec3<u32>,
  @builtin(local_invocation_index) lid : u32,
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
}

@compute @workgroup_size(1)
fn main(inputs : Inputs) {
  let workgroup_index =  
     inputs.wgid.x +
     inputs.wgid.y * inputs.num_workgroups.x +
     inputs.wgid.z * inputs.num_workgroups.x * inputs.num_workgroups.y;

  let sobel_x = input_image[workgroup_index - 1] -
                 input_image[workgroup_index + 1];

  let sobel_y = input_image[workgroup_index - image_shape[1]] -
                 input_image[workgroup_index + image_shape[1]];

  let sobel_mag = sqrt(f32(sobel_x * sobel_x) + f32(sobel_y * sobel_y));

  if (sobel_mag > 100.0) {
    filtered[workgroup_index] = u32(sobel_mag);
  } else if (sobel_mag < 0.0) {
    filtered[workgroup_index] = 0;
  }
}
"""


im = np.ascontiguousarray(iio.imread("imageio:astronaut.png")[..., 0]).astype(np.uint32)
# im = np.zeros((10, 10), dtype=np.uint32)
# im[:5] = 128
# im[:, :5] = 255

# define bindings
bindings = {
    0: im,
    2: np.array(im.shape, dtype=np.uint32),
}

# run shader
out = compute_with_buffers(
    input_arrays=bindings,
    output_arrays={1: (np.prod(im.shape), "I")},
    shader=shader_src,
    n=im.shape if len(im.shape) == 3 else im.shape + (1,),
)
print(np.frombuffer(out[1], dtype=np.uint32))
filtered = np.frombuffer(out[1], dtype=np.uint32).reshape(im.shape)

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# add image

image = gfx.Image(
    gfx.Geometry(grid=gfx.Texture(filtered, dim=2)),
    gfx.ImageBasicMaterial(clim=(0, 255)),
)
scene.add(image)

# add points

# xx = [182, 180, 161, 153, 191, 237, 293, 300, 272, 267, 254]
# yy = [145, 131, 112, 59, 29, 14, 48, 91, 136, 137, 172]
# sizes = np.arange(1, len(xx) + 1, dtype=np.float32)

# points = gfx.Points(
#     gfx.Geometry(
#         positions=[(x, y, 1) for x, y in zip(xx, yy)],
#         sizes=sizes,
#     ),
#     gfx.PointsMaterial(
#         color=(0, 1, 1, 1),
#         size=10,
#         size_space="world",
#         size_mode="vertex",
#     ),
# )
# scene.add(points)

camera = gfx.PerspectiveCamera(0)
camera.local.scale_y = -1
camera.show_object(scene, match_aspect=True, scale=1.05)

controller = gfx.PanZoomController(camera, register_events=renderer)

if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
