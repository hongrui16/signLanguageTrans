import os
import sys
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

if __name__ == '__main__':
    parent_dir = os.path.join(os.path.dirname(__file__), './..')
    sys.path.insert(0, parent_dir)

from dataloader.dataset.youtubeASL.youtubeASL import YouTubeASL
from dataloader.dataset.youtubeASL.youtubeASLFrames import YouTubeASLFrames
from dataloader.dataset.youtubeASL.youTubeASLOnlineDet import YouTubeASLOnlineDet
from dataloader.dataset.youtubeASL.youtubeASLFramesNaive import YouTubeASLFramesNaive
from dataloader.dataset.youtubeASL.youtubeASLFramesComposed import YouTubeASLFramesComposed

from dataloader.dataset.how2sign.how2sign_openpose import How2SignOpenPose
from dataloader.dataset.how2sign.how2signNaive import How2SignNaive


def get_dataloader(
    dataset_name = None,
    logger=None,
    debug=False,
    distributed=False,
    world_size=1,
    rank=0,
    split = 'train',
    **kwargs,
):
    batch_size = kwargs.get('batch_size', 32)
    modality = kwargs.get('modality', 'pose')
    img_size = kwargs.get('img_size', (224, 224))
    # num_pose_seq=self.num_pose_seq,
    # num_frame_seq=self.num_frame_seq,
    num_pose_seq = kwargs.get('num_pose_seq', 90)  # Default for YouTubeASL
    num_frame_seq = kwargs.get('num_frame_seq', 30)  # Default for
    delete_blury_frames = kwargs.get('delete_blury_frames', False)  # Default for YouTubeASL
    use_mini_dataset = kwargs.get('use_mini_dataset', False)  # Default for debugging
    
    # Log distributed training configuration
    if logger is not None:
        logger.info(f"Dataset: {dataset_name}, Distributed {split}: {distributed}, World size: {world_size}, Rank: {rank}", main_process_only=True)
    else:
        print(f"Dataset: {dataset_name}, Distributed {split}: {distributed}, World size: {world_size}, Rank: {rank}")

    # Adjust num_workers based on debug mode and distributed setup
    if debug:
        num_workers = 0
    else:
        num_workers = min(5, os.cpu_count() // world_size)  # Scale workers with available CPUs

    if dataset_name in ['YouTubeASLFrames', 'YouTubeASLFramesNaive', 'YouTubeASLOnlineDet', 'YouTubeASLFramesComposed']:
        # if not split == 'train':
        #     if logger is not None:
        #         logger.warning(f"Dataset {dataset_name} only supports 'train' split. Returning None for {split}.", main_process_only=True)
        #     else:
        #         print(f"Dataset {dataset_name} only supports 'train' split. Returning None for {split}.")
        #     return None, None, None
        

        data_dir = '/scratch/rhong5/dataset/youtube_ASL/'
        
        if dataset_name == 'YouTubeASLFramesNaive':
            dataset = YouTubeASLFramesNaive(
                split = split,
                debug=debug,
                logger=logger,
                modality=modality,
                img_size=img_size,
                num_pose_seq = num_pose_seq,
                num_frame_seq = num_frame_seq,
                delete_blury_frames = delete_blury_frames,
            )
        elif dataset_name == 'YouTubeASLFrames':
            dataset = YouTubeASLFrames(
                clip_frame_dir='/scratch/rhong5/dataset/youtubeASL_frames',
                clip_anno_dir='/scratch/rhong5/dataset/youtubeASL_anno',
                debug=debug,
                logger=logger,
                modality = modality,
                img_size=img_size,
                num_pose_seq = num_pose_seq,
                num_frame_seq = num_frame_seq,
                delete_blury_frames = delete_blury_frames,
            )
        elif dataset_name == 'YouTubeASLOnlineDet':
            video_dir = '/projects/kosecka/hongrui/dataset/youtubeASL/youtube_ASL/'
            dataset = YouTubeASLOnlineDet(
                video_dir = video_dir,
                debug=debug,
                modality = modality,
                img_size=img_size,
                logger=logger,
                num_pose_seq = num_pose_seq,
                num_frame_seq = num_frame_seq,
                delete_blury_frames = delete_blury_frames,
            )
        elif dataset_name == 'YouTubeASLFramesComposed':
            dataset = YouTubeASLFramesComposed(
                debug=debug,
                modality = modality,
                img_size=img_size,
                logger=logger,
                num_pose_seq = num_pose_seq,
                num_frame_seq = num_frame_seq,
                delete_blury_frames = delete_blury_frames,
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    elif dataset_name in ['How2SignOpenPose', 'How2SignNaive']:

        if dataset_name == 'How2SignOpenPose':
            data_dir = '/projects/kosecka/hongrui/dataset/how2sign'
            sentence_csv_path = os.path.join(data_dir, f'how2sign_{split}.csv')
            video_dir = f'/projects/kosecka/hongrui/dataset/how2sign/video_level/{split}/rgb_front/raw_videos'
            kpts_json_dir = f'/projects/kosecka/hongrui/dataset/how2sign/sentence_level/{split}/rgb_front/features/openpose_output/json'
            dataset = How2SignOpenPose(
                sentence_csv_path,
                kpts_json_dir,
                video_dir,
                debug=debug,
                modality=modality,
                img_size=img_size,
                logger=logger,
                pose_seq_len = num_pose_seq,
                frame_seq_len = num_frame_seq,
                delete_blury_frames = delete_blury_frames,

            )
        elif dataset_name == 'How2SignNaive':
            dataset = How2SignNaive(
                split,
                debug=debug,
                modality=modality,
                logger=logger,
                img_size=img_size,
                pose_seq_len = num_pose_seq,
                frame_seq_len = num_frame_seq,
                delete_blury_frames = delete_blury_frames,
                use_mini_dataset = use_mini_dataset,
            )

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Use DistributedSampler for training if distributed=True
    data_sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle= (split == 'train'),
    ) if distributed else None

    data_loader = DataLoader(
        dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = ((data_sampler is None) and (split == 'train')),
        sampler = data_sampler,
        drop_last = (split == 'train'),
        pin_memory = True
    )

    # Log sampler type
    if not data_sampler is None:
        sampler_type = {type(data_sampler).__name__}
    else:
        sampler_type = 'None'
    if logger is not None:
        logger.info(f"{dataset_name} {split}; sample num: {len(dataset)}", main_process_only=True)
        logger.info(f"{dataset_name} {split} sampler: {sampler_type}", main_process_only=True)
    else:
        print(f"{dataset_name} {split}; sample num: {len(dataset)}")
        print(f"{dataset_name} {split} sampler: {sampler_type}")


    return data_loader, dataset, data_sampler

if __name__ == '__main__':
    dataset_name = 'how2sign'
    dataset_name = 'YouTubeASLFramesNaive'
    data_loader, dataset, data_sampler = get_dataloader(dataset_name, split = 'train', debug=True, n_frames=30, distributed=False, batch_size=2)
    for i, data in enumerate(data_loader):
        print(i)
        frames_tensor, text, keypoints_dict = data
        print(frames_tensor.shape)
        print(text)
        print(keypoints_dict['hand'].shape)
        print(keypoints_dict['body'].shape)
        print(keypoints_dict['face'].shape)
        break
