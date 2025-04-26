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
import wgpu
from pygfx.utils.compute import ComputeShader

# About workgroup/invocation id's:
#
# * local_invocation_id: the vec3 indicating the current invocation into the
#   workgroup, as specified using @workgroup_size,  i.e. its position in the
#   workgroup grid.
# * local_invocation_index: the u32 represening the 'flat' local_invocation_id.
# * workgroup_id: the vec3 indicating the position of the workgroup in overall
#   compute shader grid, as specified by dispatch_workgroups().
# * global_invocation_id: workgroup_id * workgroup_size + local_invocation_id.

shader_src = """
@group(0) @binding(0) var imageTexture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> points_buffer: array<f32>;

@compute @workgroup_size(10, 10)
fn calc_points(
    @builtin(local_invocation_index) local_invocation_index: u32,
    @builtin(local_invocation_id) local_invocation_id: vec3u
) {
    let size = textureDimensions(imageTexture, 0);
    points_buffer[local_invocation_index * 3] = f32(local_invocation_id.x) * 10.0;
    points_buffer[local_invocation_index * 3 + 1] = f32(local_invocation_id.y) * 10.0;
    points_buffer[local_invocation_index * 3 + 2] = f32(1);
}
"""

im = np.ascontiguousarray(iio.imread("imageio:astronaut.png"))

image_texture = gfx.Texture(
    im,
    dim=2,
    usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
)

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# add image

image = gfx.Image(
    gfx.Geometry(grid=image_texture),
    gfx.ImageBasicMaterial(clim=(0, 255)),
)
scene.add(image)

num_points = 100

point_coords_buffer = gfx.Buffer(
    nbytes=num_points * 3 * 4,
    nitems=num_points,
    format="3xf4",
    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
)

sizes = np.ones(shape=num_points, dtype=np.float32) * 10

points = gfx.Points(
    gfx.Geometry(
        positions=point_coords_buffer,
        sizes=sizes,
    ),
    gfx.PointsMaterial(
        color=(0, 1, 1, 1),
        size=10,
        size_space="world",
        size_mode="vertex",
    ),
)
scene.add(points)

camera = gfx.PerspectiveCamera(0)
camera.local.scale_y = -1
camera.show_object(scene, match_aspect=True, scale=1.05)

controller = gfx.PanZoomController(camera, register_events=renderer)

calc_shader = ComputeShader(
    shader_src,
    entry_point="calc_points",
    # report_time=True,
)
calc_shader.set_resource(0, image.geometry.grid)
calc_shader.set_resource(1, point_coords_buffer, clear=True)

calc_shader.dispatch(1)

if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
