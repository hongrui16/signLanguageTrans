import fiftyone as fo
import fiftyone.utils.huggingface as fouh

fo.config.database_uri = "mongodb://localhost:27017"  # 确保与此处端口一致

# Load the dataset
# Note: other available arguments include 'max_samples', etc
dataset = fouh.load_from_hub("Voxel51/WLASL")

# Launch the App
session = fo.launch_app(dataset)
