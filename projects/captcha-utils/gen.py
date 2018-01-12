import string

from PIL import Image, ImageFont, ImageDraw


def font_setup(fontpath, fontsize):
    """
    Args:
        fontpath: path to font.tff file
        fontsize: font size

    Returns:
        ImageFont.truetype font
    """
    return ImageFont.truetype(fontpath, fontsize)

def generate(mode, text, font, imgsize, position, bgcolor, fgcolor):
    """
    Args:
        mode: image mode, RGB, L, P, RGBA etc.
        text: text render on the image
        font: the font render the text, ImageFont.truetype object
        imgsize: image size
        position: where the text be placed
        bgcolor: background color
        fgcolor: foreground color

    Returns:
        A captcha image, PIL.Image object.
    """
    image = Image.new(mode, imgsize, bgcolor)
    draw_handle = ImageDraw.Draw(image)
    text = " ".join(text)
    draw_handle.text(xy=position, text=text, fill=fgcolor, font=font)
    return image