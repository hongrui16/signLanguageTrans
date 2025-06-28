# -*- coding: utf-8 -*-


arg_settings = {

    "resume": None, 
    # 'resume': '/scratch/rhong5/weights/temp_training_weights/signLanguageTrans/YouTubeASLFramesNaive/20250611-055348_JID-3719511/uniSign_pose_best.pth',
    'dataset_name': 'How2SignNaive',# 'YouTubeASLFramesNaive', 'YouTubeASLFrames', 'YouTubeASLOnlineDet', 'YouTubeASLFramesComposed', 'How2SignOpenPose', 'How2SignNaive'
    "img_encoder": None,
    "modality": 'pose', # 'pose', 'rgb', 'pose_rgb',
    'finetune': True,
    'n_frames': 200,
    'train_batch_size': 60,
    'eval_batch_size': 60,
    'debug': False,
    'max_epochs': 55,
    'eval_log_dir': '',
    'img_size': (224, 224),
    'freeze_llm': False, #True, # False for finetuning the LLM, True for freezing the LLM
    'llm_name': 'mbart-large-50', # 't5-base', 't5-large', 't5-xl', 't5-xxl', 'google/flan-t5-base', 'google/flan-t5-large', 'google/flan-t5-xl', 'google/flan-t5-xxl', 'facebook/mbart-large-50'
    'pose_set': 'hand_body', # 'hand_body_face', 'body', 'hand', 
    'model_name': 'YouTubeASLBaseline', # 'google/flan-t5-base', 'google/flan-t5-large', 'google/flan-t5-xl', 'google/flan-t5-xxl', 'facebook/mbart-large-50'
}
