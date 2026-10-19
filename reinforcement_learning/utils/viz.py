"""Overlays for eval videos (drawn on rendered frames only, never on policy observations)."""
import numpy as np
import torch
from PIL import Image, ImageDraw

from mani_skill.utils.structs import Pose


def ballast_pixels(u, camera="render_camera"):
    """(num_envs, 2) pixel coords in the human render camera of the PushT ballast, on the tee's top face (where
    tee_show_ballast puts its marker); None if the env has no ballast."""
    if getattr(u, "tee_ballast_mass_ratio", 0) <= 0:
        return None
    x, y = u.tee_ballast_pos
    local = Pose.create_from_pq(p=torch.tensor([[float(x), float(y), 0.02]], device=u.device))
    world = (u.tee.pose * local).p
    params = u._human_render_cameras[camera].get_params()
    ext, K = params["extrinsic_cv"].to(world.device), params["intrinsic_cv"].to(world.device)
    cam = torch.einsum("bij,bj->bi", ext[..., :3], world) + ext[..., 3]
    uv = torch.einsum("bij,bj->bi", K, cam)
    return (uv[:, :2] / uv[:, 2:3]).cpu().numpy()


def draw_marker(img, xy, radius=7):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    x, y = float(xy[0]), float(xy[1])
    d.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(25, 90, 255), outline=(255, 255, 255), width=2)
    d.text((x + radius + 3, y - 6), "ballast", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(im)
