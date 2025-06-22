# -*- coding: utf-8 -*-


arg_settings = {

    # "resume": None, 
    'resume': '/scratch/rhong5/weights/temp_training_weights/signLanguageTrans/YouTubeASLFramesNaive/20250611-055348_JID-3719511/uniSign_pose_best.pth',
    'dataset_name': 'How2SignNaive',
    # "dataset_name": 'YouTubeASLFramesNaive',
    "feature_encoder": None,
    "modality": 'pose', # 'pose', 'rgb', 'pose_rgb',
    'finetune': True,
    'n_frames': 200,
    'train_batch_size': 45,
    'eval_batch_size': 45,
    'debug': False,
    'max_epochs': 55,
    'eval_log_dir': '',
    'img_size': (224, 224),
    'eval_batch_size': 45,
}
