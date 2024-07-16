from PIL import Image


def get_concat_h(im1: Image, im2: Image) -> Image:
    # concat images horizontally
    dst = Image.new("RGB", (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst
