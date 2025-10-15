import argparse
import numpy as np
import open3d as o3d


def load_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise RuntimeError(f"No points found in {path}")
    return pcd


def maybe_downsample(pcd: o3d.geometry.PointCloud, voxel_size: float | None):
    if voxel_size and voxel_size > 0:
        return pcd.voxel_down_sample(voxel_size)
    return pcd


def main():
    parser = argparse.ArgumentParser(description="View a PLY point cloud using Open3D")
    parser.add_argument("ply", nargs="?", default="point_cloud.ply", help="Path to PLY file (default: point_cloud.ply)")
    parser.add_argument("--voxel", type=float, default=0.0, help="Voxel size for optional downsampling (e.g., 0.01)")
    args = parser.parse_args()

    pcd = load_ply(args.ply)
    pcd = maybe_downsample(pcd, args.voxel)

    # Estimate normals for better shading (optional)
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

    o3d.visualization.draw_geometries([pcd], window_name=f"Viewer - {args.ply}")


if __name__ == "__main__":
    main()
