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

num_workgroups_x = 5
num_workgroups_y = 5

shader_src = """
@group(0) @binding(0) var imageTexture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> points_coords_buffer: array<f32>;
@group(0) @binding(2) var<storage, read_write> points_color_buffer: array<f32>;

const chunkWidth = 10u;
const chunkHeight = 10u;

@compute @workgroup_size(chunkWidth, chunkHeight, 1)
fn calc_points(
    @builtin(local_invocation_index) local_invocation_index: u32,
    @builtin(local_invocation_id) local_invocation_id: vec3u,
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(global_invocation_id) global_invocation_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
) {
    let size = textureDimensions(imageTexture, 0);
    let vector_index = workgroup_id.y * chunkWidth * chunkHeight * num_workgroups.x +
                       workgroup_id.x * chunkWidth * chunkHeight +
                       local_invocation_index;
    points_coords_buffer[vector_index * 3] = f32(global_invocation_id.x) * 10.0;
    points_coords_buffer[vector_index * 3 + 1] = f32(global_invocation_id.y) * 10.0;
    points_coords_buffer[vector_index * 3 + 2] = f32(1);

    points_color_buffer[vector_index * 4] = f32(local_invocation_index) / f32(chunkWidth * chunkHeight);
    points_color_buffer[vector_index * 4 + 1] = f32(workgroup_id.x) / f32(num_workgroups.x);
    points_color_buffer[vector_index * 4 + 2] = f32(workgroup_id.y) / f32(num_workgroups.y);
    points_color_buffer[vector_index * 4 + 3] = f32(1);
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

num_points = 100 * num_workgroups_x * num_workgroups_y

point_coords_buffer = gfx.Buffer(
    nbytes=num_points * 3 * 4,
    nitems=num_points,
    format="3xf4",
    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
)

point_colors_buffer = gfx.Buffer(
    nbytes=num_points * 4 * 4,
    nitems=num_points,
    format="4xf4",
    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
)

sizes = np.ones(shape=num_points, dtype=np.float32) * 10

points = gfx.Points(
    gfx.Geometry(
        positions=point_coords_buffer,
        sizes=sizes,
        colors=point_colors_buffer,
    ),
    gfx.PointsMaterial(
        # color=(0, 1, 1, 1),
        # size=1,
        size_space="world",
        size_mode="vertex",
        color_mode="vertex",
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
calc_shader.set_resource(2, point_colors_buffer, clear=True)

calc_shader.dispatch(num_workgroups_x, num_workgroups_y, 1)

if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
