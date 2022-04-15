import numpy as np
from PIL import ImageFont, Image, ImageDraw
import os
import glob
import tqdm
import pickle
import torch
from pdb import set_trace as breakpoint

def read_font(fn, size=28):
    """
    Read a font file and generate all letters as images.
 
    :param fn: path to font file as TTF
    :rtype: str
    :return: images
    :rtype: np.ndarray
    """
 
    # a bit smaller than image size to take transformations into account
    # points = size - size/4
    points = size
    font = ImageFont.truetype(fn, int(points))
 
    # some fonts do not like this
    # https://github.com/googlefonts/fontbakery/issues/703
    try:
        # https://stackoverflow.com/questions/43060479/how-to-get-the-font-pixel-height-using-pil-imagefont
        ascent, descent = font.getmetrics()
        (width, baseline), (offset_x, offset_y) = font.font.getsize('A')
    except IOError:
        return None
 
    data = []
    characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-*/=()[]\{\}'
    for char in characters:
        img = Image.new('L', (size, size), 255)
        draw = ImageDraw.Draw(img)
        textsize = draw.textsize(char, font=font)
        draw.text(((-offset_x + size - textsize[0])//2, (-offset_y + size - textsize[1])//2), char, font=font)
 
        matrix = np.array(img).astype(np.float32)
        matrix = 255 - matrix
        matrix /= 255.
        data.append(matrix)
 
    return np.array(data), np.arange(len(characters))


# get list of all files ending in .ttf in the fonts-main directory (recursive)
fonts = glob.glob(os.path.join('fonts', '**', '*.ttf'), recursive=True)

size = 256
# size = 28
# dataset to keep font_index, character_index, and path to image (saved as torch Tensor and filetype is .pt)
dataset = []
loop = tqdm.tqdm(total=len(fonts))
# mkdir font_images
root_path = f'font_images_{size}'
if not os.path.exists(root_path):
    os.mkdir(root_path)
for font_index, font in enumerate(fonts):
    try:
        data, chars = read_font(font, size)
        # mkdir font_images/font{font_index}
        font_path = os.path.join(root_path, 'font{}'.format(font_index))
        if not os.path.exists(font_path):
            os.mkdir(font_path)
        for char_index, char in enumerate(chars):
            # save to font_path/char{char_index}.pt
            path = os.path.join(font_path, 'char{}.pt'.format(char_index))
            # save as torch Tensor
            torch.save(torch.from_numpy(data[char]), path)
            # append to dataset
            dataset.append((font_index, char_index, path))
        loop.update(1)
    except:
        print(f'Error reading font {font}')

# pickle dataset
dataset_path = f'{root_path}/dataset.pkl'
with open(dataset_path, 'wb') as f:
    pickle.dump(dataset, f)