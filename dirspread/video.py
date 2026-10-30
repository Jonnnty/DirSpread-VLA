import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W = 672
HEADER = 40
ROW_H = 280
BOT_H = 306
IMAGE_TOP = 82
FONT = ImageFont.load_default(22)
FONT_SM = ImageFont.load_default(16)


def projection(place):
    bx, by, px, py, ux, uy, uz, bowl_r, plate_r = [float(v) for v in place]
    canvas = Image.new("RGB", (224, 224), (22, 22, 22))
    draw = ImageDraw.Draw(canvas)
    origin = np.array([112.0, 96.0])
    scale = 78.0 / max(plate_r, 1e-3)
    plate_xy = np.array([px, py])

    def panel(xy):
        rel = np.asarray(xy, dtype=np.float64) - plate_xy
        return (float(origin[0] + rel[0] * scale), float(origin[1] - rel[1] * scale))

    center = panel(plate_xy)
    rad = plate_r * scale
    draw.ellipse((center[0] - rad, center[1] - rad, center[0] + rad, center[1] + rad), outline=(230, 230, 230), width=2)
    draw.line((center[0] - 10, center[1], center[0] + 10, center[1]), fill=(190, 190, 190))
    draw.line((center[0], center[1] - 10, center[0], center[1] + 10), fill=(190, 190, 190))
    up = np.array([ux, uy, uz], dtype=np.float64)
    norm = np.linalg.norm(up)
    up = up / norm if norm else np.array([0.0, 0.0, 1.0])
    helper = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(up, helper)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(up, x_axis)
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    rim = np.array([bx, by, 0.0]) + bowl_r * (np.cos(angles)[:, None] * x_axis + np.sin(angles)[:, None] * y_axis)
    draw.polygon([panel(point[:2]) for point in rim], fill=(64, 44, 16), outline=(255, 176, 46))
    bowl_px = panel([bx, by])
    draw.line((center[0], center[1], bowl_px[0], bowl_px[1]), fill=(255, 90, 70), width=2)
    draw.ellipse((bowl_px[0] - 3, bowl_px[1] - 3, bowl_px[0] + 3, bowl_px[1] + 3), fill=(255, 176, 46))
    lean = np.array([ux, uy], dtype=np.float64)
    if np.linalg.norm(lean) > 1e-4:
        lean = lean / np.linalg.norm(lean) * 18
        draw.line((bowl_px[0], bowl_px[1], bowl_px[0] + lean[0], bowl_px[1] - lean[1]), fill=(120, 210, 255), width=2)
    bar = 0.05 * scale
    draw.line((12, 16, 12 + bar, 16), fill=(180, 180, 180), width=2)
    draw.text((12, 18), "5 cm", font=FONT_SM, fill=(180, 180, 180))
    offset_cm = float(np.hypot(place[0] - place[2], place[1] - place[3]) * 100)
    draw.text((8, 186), f"offset {offset_cm:.1f} cm", font=FONT_SM, fill=(255, 220, 160))
    return np.asarray(canvas)


def row_image(fixed, wrist, place, banner):
    canvas = Image.new("RGB", (W, ROW_H), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), banner, font=FONT, fill=(255, 255, 255))
    panels = ((fixed, "fixed"), (wrist, "wrist"), (projection(place), "top view"))
    for index, (image, label) in enumerate(panels):
        x0 = index * 224
        draw.text((x0 + 8, 32), label, font=FONT_SM, fill=(210, 210, 210))
        canvas.paste(Image.fromarray(image), (x0, 56))
    return np.asarray(canvas)


def method_row(fixed, wrist, place, step_no, target, view):
    canvas = Image.new("RGB", (W, BOT_H), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), f"adjusted  step {step_no}", font=FONT, fill=(255, 255, 255))
    draw.text((8, 32), f"target: {target}", font=FONT, fill=(255, 228, 170))
    panels = ((fixed, "fixed"), (wrist, "wrist"), (projection(place), "top view"))
    for index, (image, label) in enumerate(panels):
        x0 = index * 224
        draw.text((x0 + 8, IMAGE_TOP - 24), label, font=FONT_SM, fill=(210, 210, 210))
        canvas.paste(Image.fromarray(image), (x0, IMAGE_TOP))
    x0 = 0 if view == "固定机位" else 224
    draw.rectangle((x0 + 2, IMAGE_TOP - 2, x0 + 221, IMAGE_TOP + 221), outline=(255, 210, 40), width=4)
    return np.asarray(canvas)


def header_bar(seg_id, phrase, note):
    bar = Image.new("RGB", (W, HEADER), (10, 10, 10))
    draw = ImageDraw.Draw(bar)
    draw.text((8, 8), f"seg {seg_id}  {phrase} ({note})", font=FONT, fill=(255, 228, 170))
    return np.asarray(bar)


def write_video(path, seg_id, phrase, note, limit, orig_f, orig_w, orig_places, photos_f, photos_w, places, phases, mains):
    height = HEADER + ROW_H + BOT_H
    raw_path = str(path) + ".raw"
    with open(raw_path, "wb") as handle:
        for index in range(limit):
            step_no = index + 1
            watching = "bowl" if phases[index] == "看碗" else "plate"
            frame = np.concatenate(
                [
                    header_bar(seg_id, phrase, note),
                    row_image(orig_f[index], orig_w[index], orig_places[index], f"original  step {step_no}"),
                    method_row(photos_f[index], photos_w[index], places[index], step_no, watching, mains[index]),
                ],
                axis=0,
            )
            handle.write(frame.tobytes())
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{W}x{height}",
            "-r",
            "10",
            "-i",
            raw_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(path),
        ],
        check=True,
    )
    os.remove(raw_path)
