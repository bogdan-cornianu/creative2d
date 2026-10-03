import numpy as np
from PIL import Image

from creative2d.post.isometric import to_iso
from creative2d.post.pixelate import build_palette, hard_alpha, pixelize, quantize
from creative2d.post.seamless import blend_seams, seam_error
from creative2d.post.trim import fit_frames_to_cell, fit_to_cell, trim


def _noise(w, h, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 4), dtype=np.uint8), "RGBA")


def test_hard_alpha_only_0_or_255():
    img = _noise(32, 32)
    a = np.asarray(hard_alpha(img).getchannel("A"))
    assert set(np.unique(a)) <= {0, 255}


def test_quantize_limits_colors_and_palette_lock():
    img = _noise(64, 64, 1)
    img.putalpha(255)
    q = quantize(img, colors=8)
    assert len(np.unique(np.asarray(q.convert("RGB")).reshape(-1, 3), axis=0)) <= 8
    pal = build_palette(img, 8)
    q2 = quantize(_noise(64, 64, 2), palette=pal)
    c1 = {tuple(c) for c in np.asarray(q.convert("RGB")).reshape(-1, 3)}
    c2 = {tuple(c) for c in np.asarray(q2.convert("RGB")).reshape(-1, 3)}
    palette_colors = {tuple(pal.getpalette()[i : i + 3]) for i in range(0, 8 * 3, 3)}
    assert c1 <= palette_colors and c2 <= palette_colors


def test_pixelize_size_and_alpha():
    img = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    img.paste(Image.new("RGBA", (256, 256), (200, 50, 50, 255)), (128, 128))
    out = pixelize(img, (32, 32), colors=4)
    assert out.size == (32, 32)
    a = np.asarray(out.getchannel("A"))
    assert set(np.unique(a)) <= {0, 255}
    assert out.getpixel((16, 16))[3] == 255 and out.getpixel((1, 1))[3] == 0
    # No dark fringe from transparent pixels.
    r, g, b, _ = out.getpixel((8, 16))
    assert r > 150


def test_trim_and_fit_bottom_anchor():
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    img.paste(Image.new("RGBA", (20, 40), (0, 0, 255, 255)), (10, 10))
    t = trim(img)
    assert t.size == (20, 40)
    cell = fit_to_cell(t, (64, 64), anchor="bottom")
    assert cell.size == (64, 64)
    assert cell.getchannel("A").getbbox()[3] == 64  # touches bottom
    assert cell.getchannel("A").getbbox()[1] == 0  # full height used


def test_fit_frames_shared_scale():
    a = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    a.paste(Image.new("RGBA", (20, 40), (255, 0, 0, 255)), (40, 60))
    b = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    b.paste(Image.new("RGBA", (20, 20), (255, 0, 0, 255)), (40, 80))
    fa, fb = fit_frames_to_cell([a, b], (32, 32))
    ha = fa.getchannel("A").getbbox()
    hb = fb.getchannel("A").getbbox()
    assert ha[3] == hb[3] == 32  # same baseline
    assert (hb[3] - hb[1]) * 2 == ha[3] - ha[1]


def test_blend_seams_reduces_seam_error():
    x = np.linspace(0, 255, 64)
    grad = np.tile(x, (64, 1))
    arr = np.stack([grad, grad, grad, np.full_like(grad, 255)], -1).astype(np.uint8)
    img = Image.fromarray(arr, "RGBA")
    assert seam_error(img) > 10
    assert seam_error(blend_seams(img)) < seam_error(img) / 5


def test_iso_projection_dims_and_shape():
    tile = Image.new("RGBA", (32, 32), (0, 200, 0, 255))
    iso = to_iso(tile, 64)
    assert iso.size == (64, 32)
    assert iso.getpixel((32, 16))[3] == 255  # center filled
    assert iso.getpixel((1, 1))[3] == 0  # corner empty
    block = to_iso(tile, 64, depth=16, pixel=True)
    assert block.size == (64, 48)
    assert block.getpixel((16, 36))[3] == 255  # left face
    l, r = block.getpixel((16, 36)), block.getpixel((48, 36))
    assert l[1] > r[1]  # right face darker


def test_keep_main_subject_drops_far_blobs():
    from creative2d.post.matting import keep_main_subject

    img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    img.paste(Image.new("RGBA", (60, 80), (255, 0, 0, 255)), (70, 60))  # main
    img.paste(Image.new("RGBA", (10, 10), (0, 255, 0, 255)), (132, 100))  # touching-ish (2px gap): kept
    img.paste(Image.new("RGBA", (20, 20), (0, 0, 255, 255)), (5, 5))  # far duplicate: dropped
    out = keep_main_subject(img)
    assert out.getpixel((100, 100))[3] == 255
    assert out.getpixel((135, 105))[3] == 255
    assert out.getpixel((10, 10))[3] == 0


def test_subject_share_single_vs_pattern():
    from PIL import ImageDraw

    from creative2d.post.matting import subject_share

    single = Image.new("RGB", (256, 256), "white")
    ImageDraw.Draw(single).ellipse((70, 60, 190, 220), fill=(200, 30, 30))
    pattern = Image.new("RGB", (256, 256), "white")
    d = ImageDraw.Draw(pattern)
    for x in range(0, 256, 40):
        for y in range(0, 256, 40):
            d.ellipse((x, y, x + 25, y + 25), fill=(200, 30, 30))
    assert subject_share(single) > 0.9
    assert subject_share(pattern) < 0.3
