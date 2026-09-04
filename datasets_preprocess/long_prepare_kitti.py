from PIL import Image
import numpy as np


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    return depth


import glob
import os
import shutil

for TARGET_FRAMES in [50,100,150,200,250,300,350,400,450,500]:
    depth_dirs = glob.glob("./data/kitti/val/*/proj_depth/groundtruth/image_02")
    for dir in depth_dirs:
        # new depth dir
        new_depth_dir = f"./data/long_kitti_s1/depth_selection/val_selection_cropped/groundtruth_depth_gathered_{TARGET_FRAMES}/" + dir.split("/")[-4]+"_02"
        # print(new_depth_dir)
        new_image_dir = f"./data/long_kitti_s1/depth_selection/val_selection_cropped/image_gathered_{TARGET_FRAMES}/" + dir.split("/")[-4]+"_02"
        os.makedirs(new_depth_dir, exist_ok=True)
        os.makedirs(new_image_dir, exist_ok=True)
        
        # get all depth files and calculate the actual frame count
        all_depth_files = sorted(glob.glob(dir + "/*.png"))
        actual_frames = min(len(all_depth_files), TARGET_FRAMES)
        print(f"directory {dir.split('/')[-4]}: target frames {TARGET_FRAMES}, actual available frames {len(all_depth_files)}, actual processed frames {actual_frames}")

        for depth_file in all_depth_files[:TARGET_FRAMES]:
            new_path = new_depth_dir + "/" + depth_file.split("/")[-1]
            shutil.copy(depth_file, new_path)
            # get the path of the corresponding image
            mid = "_".join(depth_file.split("/")[-5].split("_")[:3])
            image_file = depth_file.replace('val', mid).replace('proj_depth/groundtruth/image_02', 'image_02/data')
            print(image_file)
            # check if the image file exists
            if os.path.exists(image_file):
                new_path = new_image_dir + "/" + image_file.split("/")[-1]
                shutil.copy(image_file, new_path)
            else:
                print("Image file does not exist: ", image_file)