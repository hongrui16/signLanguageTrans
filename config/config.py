# -*- coding: utf-8 -*-


arg_settings = {

    "resume": None, 
    # 'resume': '/scratch/rhong5/weights/temp_training_weights/signLanguageTrans/YouTubeASLFramesNaive/20250611-055348_JID-3719511/uniSign_pose_best.pth',
    'dataset_name': 'How2SignNaive',# 'YouTubeASLFramesNaive', 'YouTubeASLFrames', 'YouTubeASLOnlineDet', 'YouTubeASLFramesComposed', 'How2SignOpenPose', 'How2SignNaive'
    "modality": 'rgb', # 'pose', 'rgb', 'pose_rgb',
    'finetune': True,
    'train_batch_size': 20,
    'eval_batch_size': 20,
    'debug': False,
    'max_epochs': 55,
    'eval_log_dir': '',
    'img_size': (224, 224),
    'freeze_llm': True, #True, # False for finetuning the LLM, True for freezing the LLM
    'llm_name': 'mbart-large-50', # 't5-base', 't5-large', 't5-xl', 't5-xxl', 'google/flan-t5-base', 'google/flan-t5-large', 'google/flan-t5-xl', 'google/flan-t5-xxl', 'facebook/mbart-large-50'
    'pose_set': 'hand_body_face', # 'hand_body_face', 'body', 'hand', 
    'model_name': 'YouTubeASLBaseline', # 'google/flan-t5-base', 'google/flan-t5-large', 'google/flan-t5-xl', 'google/flan-t5-xxl', 'facebook/mbart-large-50'
    'img_encoder_name': 'resnet34', # resnet50, resnet34, dinov2_vits14
    'num_pose_seq': 60, # 90 for YouTubeASL, 60 for How2Sign
    'num_frame_seq': 60, # 90 for YouTubeASL,
    'delete_blury_frames': True, # True for YouTubeASL, False for How2Sign
    'freeze_llm_at_early_epochs': 15, # 0 for finetuning the LLM, > 0 for freezing the LLM at early epochs
    'use_mini_dataset': True, # True for debugging, False for full training
}
