from PIL import Image
import os
from multiprocessing import Pool

input_dir = '../data/RAISE_HR/'
output_dir = '../data/RAISE_LR/'
scale = 4.

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

image_list = os.listdir(input_dir)
image_list = [os.path.join(input_dir, _) for _ in image_list]

# Define the pool function
def downscale(name):
    print(name)
    with Image.open(name) as im:
        w, h = im.size
        w_new = int(w / scale)
        h_new = int(h / scale)
        im_new = im.resize((w_new, h_new), Image.ANTIALIAS)

        save_name = os.path.join(output_dir, name.split('/')[-1])
        im_new.save(save_name)

p = Pool(5)
p.map(downscale, image_list)