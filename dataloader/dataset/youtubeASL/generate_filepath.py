import os, sys


def get_filepath(root_dir):

    names = os.listdir(root_dir)

    filepaths = []
    for name in names:
        if name.endswith('.json'):
            filepaths.append(os.path.join(root_dir, name))
    return filepaths


def write_filepaths_to_txt(filepaths, output_txt):
    with open(output_txt, 'w') as f:
        for filepath in filepaths:
            f.write(filepath + '\n')


if __name__ == '__main__':
    dir1 = '/projects/kosecka/hongrui/dataset/youtubeASL/youtubeASL_anno'

    filepaths1 = get_filepath(dir1)
    print(f"Number of JSON files in {dir1}: {len(filepaths1)}")

    dirs = '/scratch/rhong5/dataset/youtubeASL_frame_pose_0602/youtubeASL_anno/'

    filepaths2 = get_filepath(dirs)
    print(f"Number of JSON files in {dirs}: {len(filepaths2)}")
    filepaths = filepaths1 + filepaths2
    

    output_txt = '/projects/kosecka/hongrui/dataset/youtubeASL/youtubeASL_anno_all_filepaths.txt'

    write_filepaths_to_txt(filepaths, output_txt)
    print(f"Filepaths written to {output_txt}")