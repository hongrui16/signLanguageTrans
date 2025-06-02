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
from dataloader.dataset.how2sign.how2sign_openpose import How2SignDataset

def get_dataloader(
    dataset_name,
    logger=None,
    debug=False,
    n_frames=30,
    distributed=False,
    world_size=1,
    rank=0,
    **kwargs,
):
    train_batch_size = kwargs.get('train_batch_size', 32)
    val_batch_size = kwargs.get('val_batch_size', 32)
    test_batch_size = kwargs.get('test_batch_size', 32)

    # Log distributed training configuration
    if logger is not None:
        logger.info(f"Dataset: {dataset_name}, Distributed training: {distributed}, World size: {world_size}, Rank: {rank}", main_process_only=True)
    else:
        print(f"Dataset: {dataset_name}, Distributed training: {distributed}, World size: {world_size}, Rank: {rank}")

    # Adjust num_workers based on debug mode and distributed setup
    if debug:
        num_workers = 0
    else:
        num_workers = min(5, os.cpu_count() // world_size)  # Scale workers with available CPUs

    if dataset_name == 'YouTubeASLFrames':
        data_dir = '/scratch/rhong5/dataset/youtube_ASL/'
        train_dataset = YouTubeASLFrames(
            clip_frame_dir='/scratch/rhong5/dataset/youtubeASL_frames',
            clip_anno_dir='/scratch/rhong5/dataset/youtubeASL_anno',
            num_frames_per_clip=n_frames,
            debug=debug
        )

        if logger is not None:
            logger.info(f"Train dataset dir: {data_dir}; sample num: {len(train_dataset)}", main_process_only=True)
        else:
            print(f"Train dataset dir: {data_dir}; sample num: {len(train_dataset)}")

        # Use DistributedSampler for training if distributed=True
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        ) if distributed else None

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            num_workers=num_workers,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            drop_last=True,
            pin_memory=True
        )

        # Log sampler type
        if logger is not None:
            logger.info(f"YouTubeASLFrames train sampler: {type(train_loader.sampler).__name__}", main_process_only=True)
        else:
            print(f"YouTubeASLFrames train sampler: {type(train_loader.sampler).__name__}")

        val_loader = None
        val_dataset = None
        val_sampler = None
        
        test_loader = None
        test_dataset = None
        test_sampler = None

    
    if dataset_name == 'YouTubeASLOnlineDet':
        video_dir = '/projects/kosecka/hongrui/dataset/youtubeASL/youtube_ASL/'
        train_dataset = YouTubeASLOnlineDet(
            video_dir = video_dir,
            num_frames_per_clip=n_frames,
            debug=debug
        )

        if logger is not None:
            logger.info(f"Train dataset dir: {video_dir}; sample num: {len(train_dataset)}", main_process_only=True)
        else:
            print(f"Train dataset dir: {video_dir}; sample num: {len(train_dataset)}")

        # Use DistributedSampler for training if distributed=True
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        ) if distributed else None

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            num_workers=num_workers,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            drop_last=True,
            pin_memory=True
        )

        # Log sampler type
        if logger is not None:
            logger.info(f"YouTubeASLOnlineDet train sampler: {type(train_loader.sampler).__name__}", main_process_only=True)
        else:
            print(f"YouTubeASLOnlineDet train sampler: {type(train_loader.sampler).__name__}")

        val_loader = None
        val_dataset = None
        val_sampler = None
        test_loader = None
        test_dataset = None
        test_sampler = None

    elif dataset_name == 'how2sign':
        splits = ['train', 'val', 'test']
        data_dir = '/projects/kosecka/hongrui/dataset/how2sign'
        data_sets = []
        data_loaders = []

        for split in splits:
            sentence_csv_path = os.path.join(data_dir, f'how2sign_{split}.csv')
            video_dir = f'/projects/kosecka/hongrui/dataset/how2sign/video_level/{split}/rgb_front/raw_videos'
            kpts_json_dir = f'/projects/kosecka/hongrui/dataset/how2sign/sentence_level/{split}/rgb_front/features/openpose_output/json'
            data_set = How2SignDataset(
                sentence_csv_path,
                kpts_json_dir,
                video_dir,
                n_frames=n_frames,
                debug=debug
            )
            data_sets.append(data_set)

            if split == 'train':
                train_dataset = data_set
                # Use DistributedSampler for training if distributed=True
                train_sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True
                ) if distributed else None

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=train_batch_size,
                    num_workers=num_workers,
                    shuffle=(train_sampler is None),
                    sampler=train_sampler,
                    drop_last=True,
                    pin_memory=True
                )

                # Log sampler type
                if logger is not None:
                    logger.info(f"how2sign train sampler: {type(train_loader.sampler).__name__}", main_process_only=True)
                else:
                    print(f"how2sign train sampler: {type(train_loader.sampler).__name__}")

            else:
                sampler = DistributedSampler(
                    data_set,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False
                ) if distributed else None
                data_loader = DataLoader(
                    data_set,
                    batch_size=val_batch_size,
                    num_workers=num_workers,
                    shuffle=(sampler is None),
                    sampler=sampler,
                    drop_last=False,
                    pin_memory=True
                )
                if split == 'val':
                    val_loader = data_loader
                    val_dataset = data_set
                    val_sampler = sampler
                else:
                    test_loader = data_loader
                    test_dataset = data_set
                    test_sampler = sampler

        train_dataset, val_dataset, test_dataset = data_sets

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, train_sampler, val_sampler, test_sampler

if __name__ == '__main__':
    dataset_name = 'how2sign'
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_dataloader(dataset_name)
    for i, data in enumerate(val_loader):
        print(i)
        frames_tensor, text, keypoints_dict = data
        print(frames_tensor.shape)
        print(text)
        print(keypoints_dict['hand'].shape)
        print(keypoints_dict['body'].shape)
        print(keypoints_dict['face'].shape)
        break
