import os, sys



def create_filepath(input_dir, output_dir, split):    
    """
    Create a file path for the current script.
    """
    
    # get all json files in the input directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    json_filepaths = [os.path.join(input_dir, f) for f in json_files]

    # create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    ## write the file paths to a text file
    output_filepath = os.path.join(output_dir, f'{split}_annos_filepath.txt')
    with open(output_filepath, 'w') as f:
        for filepath in json_filepaths:
            f.write(f"{filepath}\n")

def read_filepaths():
    """
    Read file paths from a text file.
    """
    filepath = '/projects/kosecka/hongrui/dataset/youtubeASL/processed_0722/train_annos_filepath.txt'
    new_filepaths = []
    with open(filepath, 'r') as f:
        filepaths = [line.strip() for line in f.readlines()]
        for json_path in filepaths:
            video_path = json_path.replace('annos', 'frames').replace('_anno.json', '_frames.mp4')
            if not os.path.exists(video_path):
                print(f"Warning: {video_path} does not exist.")
                continue
            new_filepaths.append(video_path)
    print(f"Read {len(new_filepaths)} file paths from {filepath}")
            

def count_filepaths():
    """
    Count the number of file paths in a text file.
    """
    filepath = '/projects/kosecka/hongrui/dataset/youtubeASL/processed_0722/all_annos_filepath.txt'
    with open(filepath, 'r') as f:
        filepaths = [line.strip() for line in f.readlines()]
        print(f"Number of file paths {filepath}: {len(filepaths)}")

    filepath = '/projects/kosecka/hongrui/dataset/youtubeASL/processed_0722/train_annos_filepath.txt'
    with open(filepath, 'r') as f:
        filepaths = [line.strip() for line in f.readlines()]
        print(f"Number of file paths {filepath}: {len(filepaths)}")

    filepath = '/projects/kosecka/hongrui/dataset/youtubeASL/processed_0722/test_annos_filepath.txt'
    with open(filepath, 'r') as f:
        filepaths = [line.strip() for line in f.readlines()]
        print(f"Number of file paths {filepath}: {len(filepaths)}")
        
        
def rewrite_filepaths_text():
    """
    Rewrite file paths in a text file.
    """
    filepath = '/projects/kosecka/hongrui/dataset/youtubeASL/processed_0722/all_annos_filepath.txt'
    with open(filepath, 'r') as f:
        filepaths = [line.strip() for line in f.readlines()]
        
    train_filepaths = filepaths[:550000]
    test_filepaths = filepaths[550000:]
    
    train_filepath_txt = '/projects/kosecka/hongrui/dataset/youtubeASL/processed_0722/train_annos_filepath.txt'
    with open(train_filepath_txt, 'w') as f:
        for path in train_filepaths:
            f.write(f"{path}\n")

    test_filepath_txt = '/projects/kosecka/hongrui/dataset/youtubeASL/processed_0722/test_annos_filepath.txt'
    with open(test_filepath_txt, 'w') as f:
        for path in test_filepaths:
            f.write(f"{path}\n")

if __name__ == "__main__":
    # Example usage
    input_dir = '/projects/kosecka/hongrui/dataset/youtubeASL/processed_0722/annos'
    output_dir = '/projects/kosecka/hongrui/dataset/youtubeASL/processed_0722'
    split = 'all'
    
    # create_filepath(input_dir, output_dir, split)

    # read_filepaths()
    count_filepaths()
    # rewrite_filepaths_text()

    print(f"File paths for {split} split have been created in {output_dir}.")