import zipfile

zip_path = '/scratch/rhong5/dataset/how2sign/video_level/train/train_raw_videos_all.zip'

output_dir = "/scratch/rhong5/dataset/how2sign/video_level/train/rgb_front"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    bad_file = zip_ref.testzip()
    if bad_file:
        print(f"⚠️ Corrupted entry: {bad_file}")
    else:
        zip_ref.extractall(output_dir)
        print("✅ Successfully extracted all files")
