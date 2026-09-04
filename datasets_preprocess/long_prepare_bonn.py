import glob
import os
import shutil
import numpy as np

START_FRAME = 30  # inital frame
for TARGET_FRAMES in [50,100,150,200,250,300,350,400,450,500]:
    END_FRAME = START_FRAME + TARGET_FRAMES   # end frame

    dirs = glob.glob("./data/bonn/rgbd_bonn_dataset/*/")
    dirs = sorted(dirs)

    # create new base directory
    base_new_dir = "./data/long_bonn_s1/rgbd_bonn_dataset/"
    os.makedirs(base_new_dir, exist_ok=True)

    print(f"specified frame range: {START_FRAME} to {END_FRAME}, target frames: {TARGET_FRAMES}")

    # extract frames
    for dir in dirs:
        # get original directory name
        dir_name = os.path.basename(os.path.dirname(dir))
        # build new directory path
        new_base_dir = base_new_dir + dir_name + '/'
        
        # pre-calculate the actual available frames for each modality
        rgb_frames = glob.glob(dir + 'rgb/*.png')
        rgb_frames = sorted(rgb_frames)
        available_rgb_frames = len(rgb_frames)
        
        depth_frames = glob.glob(dir + 'depth/*.png')
        depth_frames = sorted(depth_frames)
        available_depth_frames = len(depth_frames)
        
        gt_path = dir + "groundtruth.txt"
        gt = np.loadtxt(gt_path)
        available_gt_frames = len(gt)
        
        # calculate the actual frames for each modality in the specified range
        actual_rgb_frames = min(available_rgb_frames - START_FRAME, TARGET_FRAMES)
        actual_depth_frames = min(available_depth_frames - START_FRAME, TARGET_FRAMES)
        actual_gt_frames = min(available_gt_frames - START_FRAME, TARGET_FRAMES)
        
        # take the minimum value of the three modalities to ensure consistency
        final_frame_count = min(actual_rgb_frames, actual_depth_frames, actual_gt_frames)
        print(f"  final unified frame count: {final_frame_count}")
        
        # process RGB frames
        rgb_frames = rgb_frames[START_FRAME:START_FRAME + final_frame_count]
        new_dir = new_base_dir + f'rgb_{TARGET_FRAMES}/'
        if os.path.exists(new_dir):
            shutil.rmtree(new_dir)
        for frame in rgb_frames:
            os.makedirs(new_dir, exist_ok=True)
            shutil.copy(frame, new_dir)

        # process Depth frames
        depth_frames = depth_frames[START_FRAME:START_FRAME + final_frame_count]
        new_dir = new_base_dir + f'depth_{TARGET_FRAMES}/'
        if os.path.exists(new_dir):
            shutil.rmtree(new_dir)
        for frame in depth_frames:
            os.makedirs(new_dir, exist_ok=True)
            shutil.copy(frame, new_dir)

        # process Groundtruth
        gt_final = gt[START_FRAME:START_FRAME + final_frame_count]
        gt_file = new_base_dir + f'groundtruth_{TARGET_FRAMES}.txt'
        if os.path.exists(gt_file):
            os.remove(gt_file)
        np.savetxt(gt_file, gt_final)