import glob
import imageio
import os
img_save_dir = './LS_1/image/'
images = []
for file_name in sorted(os.listdir(img_save_dir)):
    if file_name.endswith('.png'):
        file_path = os.path.join(img_save_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave(img_save_dir+'movie.gif', images, format='GIF', fps=10) 

