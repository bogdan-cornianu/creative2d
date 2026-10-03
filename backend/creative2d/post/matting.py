"""Background removal: BiRefNet (ML) with a border-color key fallback."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from PIL import Image, ImageFilter

from .. import config
from ..gen import downloads


def has_alpha(img: Image.Image, min_transparent: float = 0.02) -> bool:
    """True when the image already has a meaningful transparent area."""
    if img.mode != "RGBA":
        return False
    a = np.asarray(img.getchannel("A"))
    return float((a < 250).mean()) >= min_transparent


def border_key(img: Image.Image, tolerance: float = 40.0, softness: float = 20.0) -> Image.Image:
    """Remove a flat background by keying out the median border color.

    Good enough for prompts that ask for a plain solid background, and needs
    no model. Only pixels connected to the border are removed, so interior
    areas with the background color survive.
    """
    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.sqrt(((rgb - bg) ** 2).sum(-1))
    candidate = dist < tolerance + softness

    # Keep only background-colored regions connected to the border.
    labels, _ = ndimage.label(candidate)
    edge_labels = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    reach = np.isin(labels, edge_labels[edge_labels > 0])

    alpha = np.full(dist.shape, 255.0)
    soft = np.clip((dist - tolerance) / max(softness, 1e-3), 0, 1) * 255
    alpha[reach] = soft[reach]
    out = img.convert("RGBA")
    out.putalpha(Image.fromarray(alpha.astype(np.uint8), "L"))
    return out


def _load_birefnet(device: str):
    import torch
    from transformers import AutoModelForImageSegmentation

    repo = config.load_models_config().get("matting", {}).get("repo", "ZhengPeng7/BiRefNet")
    model = AutoModelForImageSegmentation.from_pretrained(repo, trust_remote_code=True, local_files_only=True)
    model.to(device).eval()
    model.to(torch.float32)
    return model


def birefnet(img: Image.Image) -> Image.Image:
    import torch
    from torchvision import transforms

    from ..gen import model_cache

    device = config.detect_device()
    model = model_cache.get(f"matting:{device}", lambda: _load_birefnet(device), heavy=False)
    res = int(config.load_models_config().get("matting", {}).get("resolution", 1024))
    tf = transforms.Compose(
        [
            transforms.Resize((res, res)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    rgb = img.convert("RGB")
    x = tf(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x)[-1].sigmoid().cpu()[0].squeeze()
    mask = transforms.functional.to_pil_image(pred).resize(rgb.size, Image.BILINEAR)
    out = rgb.convert("RGBA")
    out.putalpha(mask)
    return out


def matte(img: Image.Image, method: str = "auto") -> Image.Image:
    """Return RGBA with background removed.

    auto: keep existing alpha if present, else BiRefNet when ML deps exist
    and the model is downloaded, else border key.
    """
    if method in ("auto", "keep") and has_alpha(img):
        return img.convert("RGBA")
    if method == "birefnet" or (method == "auto" and config.ml_available() and downloads.is_ready("matting")):
        downloads.require("matting")
        return birefnet(img)
    return border_key(img)


def keep_main_subject(img: Image.Image, reach: float = 0.03) -> Image.Image:
    """Drop disconnected blobs far from the largest one (stray duplicates or
    background debris). Parts within `reach` (fraction of image size) of the
    main subject, like a held weapon, are kept."""
    a = np.asarray(img.getchannel("A"))
    solid = a > 32
    labels, n = ndimage.label(solid)
    if n <= 1:
        return img
    sizes = ndimage.sum(solid, labels, range(1, n + 1))
    main = labels == (int(np.argmax(sizes)) + 1)
    grow = max(1, int(reach * max(img.size)))
    near = ndimage.binary_dilation(main, iterations=grow)
    keep_labels = np.unique(labels[near & solid])
    keep = np.isin(labels, keep_labels[keep_labels > 0])
    # Keep soft edge pixels next to kept regions.
    keep = ndimage.binary_dilation(keep, iterations=2)
    out = np.array(img.convert("RGBA"))
    out[..., 3] = np.where(keep, out[..., 3], 0)
    return Image.fromarray(out, "RGBA")


def subject_share(img: Image.Image) -> float:
    """How much of the foreground belongs to one blob (0..1).

    Uses the model-free border key on a downscaled copy, so it is cheap.
    A single centered subject scores near 1; a sheet or pattern of many
    small subjects scores low. Blobs touching the image edge count against
    the score (cropped subjects, patterns filling the frame).
    """
    small = img.copy()
    small.thumbnail((256, 256))
    # Cloud models may already return alpha; otherwise key out the flat background.
    keyed = small if has_alpha(small) else border_key(small.convert("RGB"))
    a = np.asarray(keyed.getchannel("A")) > 128
    total = a.sum()
    if total < 0.01 * a.size:
        return 0.0
    labels, n = ndimage.label(a)
    sizes = ndimage.sum(a, labels, range(1, n + 1))
    main = int(np.argmax(sizes)) + 1
    share = float(sizes[main - 1] / total)
    mask = labels == main
    edge = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]]).mean()
    return float(share * (1.0 - min(1.0, edge * 4)))


def clean_alpha(img: Image.Image, min_alpha: int = 8) -> Image.Image:
    """Drop near-invisible haze and clear RGB under fully transparent pixels
    (prevents colored halos when textures are filtered)."""
    arr = np.array(img.convert("RGBA"))
    arr[arr[..., 3] < min_alpha] = 0
    return Image.fromarray(arr, "RGBA")


def erode_edge(img: Image.Image, px: int = 1) -> Image.Image:
    """Shrink alpha by px to remove background-colored fringe."""
    if px <= 0:
        return img
    a = img.getchannel("A").filter(ImageFilter.MinFilter(2 * px + 1))
    out = img.copy()
    out.putalpha(a)
    return out
