from PIL import Image
import numpy as np
import os
import skimage
from skimage import morphology

palette = os.path.abspath(os.path.join("/opt/project", 'data/palette.txt'))

def read_palette():
  with open(palette) as f:
    palette_read = f.readlines()
    palette_read = [x.rstrip() for x in palette_read]
    palette_read = [[int(y) for y in x.split(" ")] for x in palette_read]
    return np.array(palette_read)

palette_mem = read_palette()


def imread_indexed(filename, transform=None):
  """ Load image given filename."""

  try:
    im = Image.open(filename)
    if transform is not None:
      im = transform(im)
  except:
    return None

  annotation = np.atleast_3d(im)[...,0]
  # return annotation,np.array(im.getpalette()).reshape((-1,3))
  return annotation
def imwrite_indexed(filename,array,color_palette=palette_mem):
  """ Save indexed png."""

  if np.atleast_3d(array).shape[2] != 1:
    raise Exception("Saving indexed PNGs requires 2D array.")

  im = Image.fromarray(array)
  im = im.convert('L')
  im.putpalette(color_palette.ravel().tolist())
  im.save(filename, format='PNG')

def overlay(image, mask, colors=[255, 0, 0], cscale=2, alpha=0.4):
  """ Overlay segmentation on top of RGB image. """

  colors = np.atleast_2d(colors) * cscale

  im_overlay = image.copy()
  object_ids = np.unique(mask)

  for object_id in object_ids[1:]:
    # Overlay color on  binary mask

    foreground = image * alpha + np.ones(image.shape) * (1 - alpha) * np.array(colors[object_id])
    binary_mask = mask == object_id

    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]

    countours = skimage.morphology.binary.binary_dilation(binary_mask) ^ binary_mask
    im_overlay[countours, :] = 0

  return im_overlay.astype(image.dtype)