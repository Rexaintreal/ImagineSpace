"""
generate_mesh.py  (patched — fixes blank screenshots)

Merges CUT3R's per-frame output (camera/*.npz pose+intrinsics,
depth/*.npy, conf/*.npy, color/*.png) into one colored point cloud,
then runs Poisson surface reconstruction to get a mesh.

FIX IN THIS VERSION:
Your screenshots were blank because Open3D's offscreen renderer
(`vis.create_window(visible=False)`) still needs a working GPU/EGL
context, which fails under WSL (confirmed by the ZINK/EGL errors in
your run log) exactly like the live window does. It fails *silently*
-- no crash, no error, just an empty framebuffer saved as a "valid"
PNG. This version renders screenshots with matplotlib instead, which
is pure CPU and has no GPU/EGL dependency at all, so it can't hit the
same failure mode.

Everything else (CLI args, Poisson mesh, point cloud merge logic) is
UNCHANGED from your original script.

Assumes CUT3R's output_dir layout:
  output_dir/camera/000000.npz  -> pose (4,4), intrinsics (3,3)
  output_dir/depth/000000.npy   -> (H, W) depth map
  output_dir/conf/000000.npy    -> (H, W) confidence map
  output_dir/color/000000.png   -> (H, W, 3) RGB frame, same res as depth
"""

import argparse
import glob
import os

import numpy as np
import open3d as o3d
from PIL import Image

import matplotlib
matplotlib.use("Agg")  # no display / no GPU needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_frame(output_dir, frame_id):
    cam = np.load(os.path.join(output_dir, "camera", f"{frame_id}.npz"))
    pose = cam["pose"].astype(np.float64)              # (4,4)
    intrinsics = cam["intrinsics"].astype(np.float64)  # (3,3)

    depth = np.load(os.path.join(output_dir, "depth", f"{frame_id}.npy")).astype(np.float64)
    conf = np.load(os.path.join(output_dir, "conf", f"{frame_id}.npy")).astype(np.float64)

    color_path = os.path.join(output_dir, "color", f"{frame_id}.png")
    color = np.asarray(Image.open(color_path).convert("RGB")).astype(np.float64) / 255.0

    return pose, intrinsics, depth, conf, color


def backproject_frame(pose, intrinsics, depth, conf, color, conf_thresh):
    h, w = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    us = us.astype(np.float64)
    vs = vs.astype(np.float64)

    mask = conf > conf_thresh
    if not np.any(mask):
        return np.empty((0, 3)), np.empty((0, 3))

    z = depth[mask]
    u = us[mask]
    v = vs[mask]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=1)  # (N, 3)

    # ADJUST IF NEEDED: if `pose` turns out to be world-to-camera instead of
    # camera-to-world, invert it first: pose = np.linalg.inv(pose)
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (pose @ pts_cam_h.T).T[:, :3]

    cols = color[mask]  # (N, 3), already 0-1 range

    return pts_world, cols


def merge_point_cloud(output_dir, conf_thresh):
    camera_files = sorted(glob.glob(os.path.join(output_dir, "camera", "*.npz")))
    all_pts, all_cols = [], []

    for f in camera_files:
        frame_id = os.path.splitext(os.path.basename(f))[0]
        pose, intrinsics, depth, conf, color = load_frame(output_dir, frame_id)
        pts, cols = backproject_frame(pose, intrinsics, depth, conf, color, conf_thresh)
        if pts.shape[0] == 0:
            print(f"  frame {frame_id}: 0 points (below conf_thresh)")
            continue
        print(f"  frame {frame_id}: {pts.shape[0]} points")
        all_pts.append(pts)
        all_cols.append(cols)

    pts = np.concatenate(all_pts, axis=0)
    cols = np.concatenate(all_cols, axis=0)
    print(f"Total merged points: {pts.shape[0]}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd


def mesh_from_point_cloud(pcd, voxel_size, poisson_depth, density_trim_quantile):
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 4, max_nn=30)
    )
    pcd_down.orient_normals_consistent_tangent_plane(30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_down, depth=poisson_depth
    )
    densities = np.asarray(densities)

    low_conf = densities < np.quantile(densities, density_trim_quantile)
    mesh.remove_vertices_by_mask(low_conf)

    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles "
          f"(after trimming bottom {density_trim_quantile:.0%} density)")
    return mesh, pcd_down


def _set_equal_aspect(ax, pts):
    """matplotlib 3D doesn't auto-equalize axes; do it manually so the
    model doesn't look stretched/squashed."""
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    centers = (mins + maxs) / 2
    radius = max((maxs - mins).max() / 2, 1e-6)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def save_pointcloud_screenshot(pcd, out_path, width=1280, height=800, max_points=150000):
    """CPU-only point cloud screenshot via matplotlib. No GPU/EGL needed,
    so it can't hit the WSL Open3D offscreen-render failure."""
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if pcd.has_colors() else None

    if pts.shape[0] == 0:
        print(f"  WARNING: point cloud is empty, skipping screenshot for {out_path}")
        return

    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
        if cols is not None:
            cols = cols[idx]

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor("#111318")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#111318")

    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c=cols if cols is not None else "#8ca0c8",
        s=1.2, linewidths=0, depthshade=True,
    )
    _set_equal_aspect(ax, pts)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=-60)
    fig.tight_layout(pad=0)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved screenshot: {out_path}")


def save_mesh_screenshot(mesh, out_path, width=1280, height=800, max_tris=120000):
    """CPU-only mesh screenshot via matplotlib. No GPU/EGL needed."""
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)

    if verts.shape[0] == 0 or tris.shape[0] == 0:
        print(f"  WARNING: mesh is empty, skipping screenshot for {out_path}")
        return

    if tris.shape[0] > max_tris:
        idx = np.random.choice(tris.shape[0], max_tris, replace=False)
        tris = tris[idx]

    if mesh.has_vertex_colors():
        vcols = np.asarray(mesh.vertex_colors)
        face_colors = vcols[tris].mean(axis=1)
    else:
        face_colors = np.tile([0.55, 0.63, 0.78], (tris.shape[0], 1))

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor("#111318")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#111318")

    polys = verts[tris]  # (n_tris, 3, 3)
    pc = Poly3DCollection(polys, facecolor=face_colors, edgecolor="none", linewidths=0)
    ax.add_collection3d(pc)

    _set_equal_aspect(ax, verts)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=-60)
    fig.tight_layout(pad=0)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved screenshot: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True,
                         help="CUT3R output dir containing camera/ depth/ conf/ color/")
    parser.add_argument("--conf_thresh", type=float, default=3.0)
    parser.add_argument("--voxel_size", type=float, default=0.5)
    parser.add_argument("--poisson_depth", type=int, default=9)
    parser.add_argument("--density_trim_quantile", type=float, default=0.15)
    parser.add_argument("--out_pointcloud", default="mesh_out/pointcloud.ply")
    parser.add_argument("--out_mesh", default="mesh_out/mesh.ply")
    parser.add_argument("--screenshot_pointcloud", default="mesh_out/pointcloud_view.png")
    parser.add_argument("--screenshot_mesh", default="mesh_out/mesh_view.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_pointcloud), exist_ok=True)

    print("Merging point cloud from all frames...")
    pcd = merge_point_cloud(args.output_dir, args.conf_thresh)
    o3d.io.write_point_cloud(args.out_pointcloud, pcd)
    save_pointcloud_screenshot(pcd, args.screenshot_pointcloud)

    print("Running Poisson mesh reconstruction...")
    mesh, pcd_down = mesh_from_point_cloud(
        pcd, args.voxel_size, args.poisson_depth, args.density_trim_quantile
    )
    o3d.io.write_triangle_mesh(args.out_mesh, mesh)
    save_mesh_screenshot(mesh, args.screenshot_mesh)

    print("Done.")
    print(f"  point cloud: {args.out_pointcloud}")
    print(f"  mesh:        {args.out_mesh}")


if __name__ == "__main__":
    main()
