# save_img_dir = '/projects/kosecka/hongrui/dataset/youtubeASL/youtubeASL_frames'
# save_anno_dir = '/projects/kosecka/hongrui/dataset/youtubeASL/youtubeASL_anno'
存放第一批的处理数据

# save_img_dir = '/scratch/rhong5/dataset/youtubeASL_frame_pose_0602/youtubeASL_frames'
# save_anno_dir = '/scratch/rhong5/dataset/youtubeASL_frame_pose_0602/youtubeASL_anno'
因为projects目录下没有空间了, 所以把剩下的数据放在这里

save_img_dir = '/scratch/rhong5/dataset/youtubeASL_frame_pose_0614/youtubeASL_frames'
save_anno_dir = '/scratch/rhong5/dataset/youtubeASL_frame_pose_0614/youtubeASL_anno'
因为预处理的条件, 就是人的bbox检测, 第一帧没有人或者多人就跳过该视频,导致处理完之后, 只有40W+的视频. 以为是漏了. 所以建了上面的文件夹.

---------------------------------------------the below is written on July 26, 2025-----------------------------------
以上的数据全部都被删掉了 因为之前的数据有问题.
目前新的数据处理过的关键帧和pose信息 放在/projects/kosecka/hongrui/dataset/youtubeASL/processed_0722. 大概有60w个clip.