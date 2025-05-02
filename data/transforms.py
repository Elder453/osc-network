"""
Data transformation classes for preprocessing icon images.

Includes custom transformation for resizing and padding icons 
to create a consistent format for model input.
"""

import random
from PIL import Image, ImageOps
from torchvision import transforms


class RandomResizeAndPad(object):
    """
    Randomly resizes an image and pastes it onto a fixed-size canvas at a random position.

    Parameters
    ----------
    canvas_size : tuple
        Output canvas size as (width, height)
    scale_range : tuple
        Range of scale factors (min, max)
    fill : int or tuple
        Background fill color (default: 0 for black)
    """
    def __init__(self, canvas_size=(32, 32), scale_range=(0.7, 1.0), fill=(0)):
        self.canvas_size = canvas_size
        self.scale_range = scale_range
        self.fill = fill

    def __call__(self, img):
        # sample a random scale factor from the provided range
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        orig_w, orig_h = img.size
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))
        
        # resize the icon image
        resized_img = img.resize((new_w, new_h), resample=Image.BICUBIC)

        # invert the colors: black becomes white and vice versa
        inverted_img = ImageOps.invert(resized_img.convert("L"))
        
        # create a blank canvas with the given size and fill color
        canvas = Image.new("L", self.canvas_size, self.fill)
        
        # calculate maximum offsets such that the icon fits within the canvas
        max_x = self.canvas_size[0] - new_w
        max_y = self.canvas_size[1] - new_h
        
        # randomly select the top-left coordinates
        x_offset = random.randint(0, max_x) if max_x > 0 else 0
        y_offset = random.randint(0, max_y) if max_y > 0 else 0
        
        # paste the resized image onto the canvas at the random location
        canvas.paste(inverted_img, (x_offset, y_offset))
        return canvas


# Default transform used across all datasets
COMMON_TRANSFORM = transforms.Compose([
    RandomResizeAndPad(canvas_size=(32, 32), scale_range=(0.7, 1.0), fill=(0)),
    transforms.ToTensor(),
])