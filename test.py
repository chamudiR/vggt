import torch
import numpy as np
from pathlib import Path
import argparse
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# Select device and a safe dtype. Only query CUDA capability if CUDA is available.
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    major, _ = torch.cuda.get_device_capability()
    # bfloat16 on Ampere (CC >= 8.0), else use float16
    dtype = torch.bfloat16 if major >= 8 else torch.float16
else:
    # On CPU, stick to float32 for widest compatibility
    dtype = torch.float32

# CLI: allow passing a folder with images
parser = argparse.ArgumentParser(description="Run VGGT on a set of images and export a point cloud")
parser.add_argument("--images_dir", type=str, default=None, help=r"D:\5th sem\Image proccessing\vggt\examples\llff_flower")
args = parser.parse_args()

def list_images_in_dir(folder: str) -> list[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"images_dir not found or not a directory: {folder}")
    files = [fp for fp in p.iterdir() if fp.suffix.lower() in exts]
    files.sort(key=lambda x: x.name)
    if not files:
        raise FileNotFoundError(f"No images with extensions {sorted(exts)} found in {folder}")
    return [str(fp) for fp in files]

# Either load from a folder, or fallback to an explicit list
if args.images_dir:
    image_names = list_images_in_dir(args.images_dir)
else:
    # Replace with your actual image paths (fallback example)
    image_names = [
        r"D:\\5th sem\\Image proccessing\\vggt\\examples\\llff_fern\\images\\004.png",
        r"D:\\5th sem\\Image proccessing\\vggt\\examples\\llff_fern\\images\\005.png",
        r"D:\\5th sem\\Image proccessing\\vggt\\examples\\llff_fern\\images\\006.png",
        r"D:\\5th sem\\Image proccessing\\vggt\\examples\\llff_fern\\images\\007.png",
        r"D:\\5th sem\\Image proccessing\\vggt\\examples\\llff_fern\\images\\008.png",
        r"D:\\5th sem\\Image proccessing\\vggt\\examples\\llff_fern\\images\\009.png",
        r"D:\\5th sem\\Image proccessing\\vggt\\examples\\llff_fern\\images\\010.png",
        r"D:\\5th sem\\Image proccessing\\vggt\\examples\\llff_fern\\images\\011.png",
        r"D:\\5th sem\\Image proccessing\\vggt\\examples\\llff_fern\\images\\012.png",
        r"D:\\5th sem\\Image proccessing\\vggt\\examples\\llff_fern\\images\\013.png",
        r"D:\\5th sem\\Image proccessing\\vggt\\examples\\llff_fern\\images\\014.png",
        r"D:\\5th sem\\Image proccessing\\vggt\\examples\\llff_fern\\images\\015.png",
        r"D:\\5th sem\\Image proccessing\\vggt\\examples\\llff_fern\\images\\016.png",
        r"D:\\5th sem\\Image proccessing\\vggt\\examples\\llff_fern\\images\\017.png",
        r"D:\\5th sem\\Image proccessing\\vggt\\examples\\llff_fern\\images\\018.png",
    ]
images = load_and_preprocess_images(image_names).to(device)
print(f"Device: {device}, dtype: {dtype}")
print(f"Images tensor shape: {tuple(images.shape)}")

# Keep it snappy on CPU by reducing the number of images
if device == "cpu" and images.shape[0] > 2:
    images = images[:2]
    print("Running with the first 2 images on CPU for speed...")

# Toggle to True if you want to download and use pretrained weights (may be large and slow on first run)
USE_PRETRAINED = True
if USE_PRETRAINED:
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
else:
    model = VGGT().to(device)
print("Model ready (pretrained:" , USE_PRETRAINED, ")")

with torch.no_grad():
    if device == "cuda":
        # Use CUDA autocast with chosen dtype
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    else:
        # No autocast on CPU by default; run in float32
        predictions = model(images)

# Print a compact summary of outputs
def _describe(name, val):
    try:
        import torch as _t
        if isinstance(val, _t.Tensor):
            return f"Tensor{tuple(val.shape)} {val.dtype}"
        if isinstance(val, (list, tuple)):
            parts = []
            for i, v in enumerate(val[:5]):
                if isinstance(v, _t.Tensor):
                    parts.append(f"[{i}] Tensor{tuple(v.shape)} {v.dtype}")
                else:
                    parts.append(f"[{i}] {type(v).__name__}")
            more = " …" if len(val) > 5 else ""
            return f"{type(val).__name__}[{len(val)}]: " + ", ".join(parts) + more
        if isinstance(val, dict):
            keys = list(val.keys())
            return f"dict(keys={keys[:10]}{' …' if len(keys)>10 else ''})"
        return type(val).__name__
    except Exception as e:
        return f"<error describing: {e}>"

if isinstance(predictions, dict):
    print("Predictions (dict) keys:", list(predictions.keys()))
    for k, v in list(predictions.items())[:10]:
        print(f"- {k}: {_describe(k, v)}")
else:
    print("Predictions:", _describe("predictions", predictions))

print("Done.")

# ------------------------
# Export a colored point cloud (PLY)
# ------------------------
def save_point_cloud_ply(path, points_xyz, colors_rgb):
    """Save a point cloud to ASCII PLY.
    points_xyz: (N,3) float32/float64
    colors_rgb: (N,3) uint8 in [0,255]
    """
    points_xyz = np.asarray(points_xyz)
    colors_rgb = np.asarray(colors_rgb)
    assert points_xyz.shape[0] == colors_rgb.shape[0], "points and colors size mismatch"
    N = points_xyz.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for p, c in zip(points_xyz, colors_rgb):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

# Use predicted world_points if present; otherwise compute from depth and cameras
try:
    wp = predictions.get("world_points", None)
    conf = predictions.get("world_points_conf", None)
    if wp is None:
        # Optional fallback using depth + pose encoding
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        pose_enc = predictions["pose_enc"]  # (B, S, 9)
        B, S = pose_enc.shape[:2]
        H, W = images.shape[-2:]
        extri, intri = pose_encoding_to_extri_intri(pose_enc, (H, W))
        depth = predictions["depth"]  # (B, S, H, W, 1)
        # Use first batch
        wp_np = unproject_depth_map_to_point_map(depth[0], extri[0], intri[0])  # (S,H,W,3)
        wp = torch.from_numpy(wp_np)

    # Shapes: wp (B?, S, H, W, 3) or (S,H,W,3)
    if wp.dim() == 5:
        wp = wp[0]  # take first batch -> (S,H,W,3)
    if conf is not None and conf.dim() == 4:
        conf = conf[0]  # (S,H,W)

    S, H, W, _ = wp.shape
    # Colors from input images: images is (S,3,H,W) in [0,1]
    img = images[:S].detach().cpu().permute(0, 2, 3, 1).contiguous().numpy()
    img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)  # (S,H,W,3)

    # Flatten
    pts = wp.detach().cpu().contiguous().view(-1, 3).numpy()
    cols = img_uint8.reshape(-1, 3)
    if conf is not None:
        mask = (conf.detach().cpu().numpy().reshape(-1) > 0.0)
    else:
        # Basic validity: finite and non-zero depth (z)
        valid = np.isfinite(pts).all(axis=1)
        mask = valid & (np.abs(pts).sum(axis=1) > 0)

    pts = pts[mask]
    cols = cols[mask]

    # Subsample if too many points (to keep file small). Keep ~1M max.
    max_pts = 1_000_000
    if pts.shape[0] > max_pts:
        idx = np.linspace(0, pts.shape[0] - 1, max_pts).astype(np.int64)
        pts = pts[idx]
        cols = cols[idx]

    out_path = "point_cloud.ply"
    save_point_cloud_ply(out_path, pts, cols)
    print(f"Saved point cloud: {out_path} with {pts.shape[0]} points")
except Exception as e:
    print(f"Point cloud export skipped due to error: {e}")
