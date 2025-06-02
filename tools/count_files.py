import os
import subprocess

def count_files(directory):
    """
    Efficient and SLURM-safe file count using shell, with fallback to Python.
    """
    try:
        # Fast shell method
        output = subprocess.check_output(f'find {directory} -type f | wc -l', shell=True)
        return int(output.strip())
    except Exception as e:
        print(f"[WARN] Shell method failed for {directory}: {e}")
        # Fallback to scandir
        try:
            return sum(1 for entry in os.scandir(directory) if entry.is_file())
        except Exception as e2:
            print(f"[ERROR] Scandir also failed for {directory}: {e2}")
            return -1

origin_img_path = '/scratch/rhong5/dataset/youtubeASL_frames/'
origin_anno_path = '/scratch/rhong5/dataset/youtubeASL_anno/'
target_img_path = '/projects/kosecka/hongrui/dataset/youtubeASL/youtubeASL_frames'
target_anno_path = '/projects/kosecka/hongrui/dataset/youtubeASL/youtubeASL_anno'

paths = {
    "origin_img_path": origin_img_path,
    "origin_anno_path": origin_anno_path,
    "target_img_path": target_img_path,
    "target_anno_path": target_anno_path
}

results = {}
for key, path in paths.items():
    results[key] = path
    results[key.replace("path", "nums")] = count_files(path)

# Save to output file
with open("youtubeASL.txt", "w") as f:
    for key, val in results.items():
        f.write(f"{key}: {val}\n")
