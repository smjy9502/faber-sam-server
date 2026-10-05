"""Respect uploaded alpha independently of inference RGB and model confidence."""
from PIL import Image, ImageChops


def preserve_alpha(source: Image.Image, model_alpha: Image.Image) -> tuple[Image.Image, Image.Image]:
    if source.size != model_alpha.size:
        raise ValueError("Alpha dimensions must match the source")
    rgba = source.convert("RGBA")
    # min retains designer antialiasing without multiplying two soft edges.
    # Opaque inputs are the identity: min(255, model) == model.
    alpha = ImageChops.darker(rgba.getchannel("A"), model_alpha.convert("L"))
    rgba.putalpha(alpha)
    return alpha, rgba
