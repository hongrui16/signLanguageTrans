
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import argparse
import datetime
import time
import cv2
import shutil
import logging
from accelerate.logging import get_logger
from accelerate import Accelerator
from accelerate.state import PartialState

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# import list and tube
from typing import List, Tuple

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from network.uni_sign_network import UniSignNetwork
from network.feature_encoder import get_encoder
from dataloader.dataset.youtubeASL.youtubeASL import YouTubeASL, YouTubeASLClip
from transformers import MBartTokenizer, MBartForConditionalGeneration


class ManoParaUnetDiffusion():
    def __init__(self, args):
        self.args = args
        self.debug = args.debug
        self.model_name = args.model_name
        self.resume_path = args.resume

        time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        slurm_job_id = os.environ.get('SLURM_JOB_ID') if os.environ.get('SLURM_JOB_ID') is not None else '0'
        log_dir = f'Zlog/{time_stamp}_ID-{slurm_job_id}'
        self.log_dir = log_dir  

        os.makedirs(log_dir, exist_ok=True)
        shutil.copy(__file__, log_dir)

        self.ckpt_dir = '/scratch/rhong5/weights/temp_training_weights/singLangTran'
        os.makedirs(self.ckpt_dir, exist_ok=True)

        
        log_path = os.path.join(self.log_dir, "info.log")
        shutil.copyfile("config/config.py", os.path.join(self.log_dir, f"config_{slurm_job_id}.py"))

        self.accelerator = Accelerator(
        # gradient_accumulation_steps=gradient_accumulation_steps,
        # mixed_precision=mixed_precision,
        # log_with=report_to,
        # project_config=accelerator_project_config,
        )


        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
        )

        # ✅ Get logger manually (since `get_logger(__name__)` no longer exists in `accelerate`)
        self.logger = logging.getLogger(__name__)

        # ✅ Ensure `main_process_only=True` works
        def patched_log_method(original_method):
            def new_method(msg, *args, main_process_only=False, **kwargs):
                if not main_process_only or self.accelerator.is_main_process:
                    original_method(msg, *args, **kwargs)
            return new_method

        # Patch `info`, `warning`, `error`, `debug` methods
        self.logger.info = patched_log_method(self.logger.info)
        self.logger.warning = patched_log_method(self.logger.warning)
        self.logger.error = patched_log_method(self.logger.error)
        self.logger.debug = patched_log_method(self.logger.debug)


        
        self.logger.info(f"\n{os.path.basename(__file__)}\n", main_process_only=True)
        self.logger.info(f'logging_dir: {self.log_dir}', main_process_only=True)
        self.logger.info(f'output_ckpts_dir: {self.ckpt_dir}\n', main_process_only=True)



        self.epochs = 45
        self.logger.info(f"Training epochs: {self.epochs}", main_process_only=True)
        
        # use_condition = False        
        
        self.feature_encoder_name = 'dino_v2'
        self.logger.info(f"feature encoder: {self.use_feature_encoder}", main_process_only=True)
        
        self.use_feature_encoder = False
        self.logger.info(f"Use use_feature_encoder: {self.use_feature_encoder}", main_process_only=True)

        # dataset_name = 'MNIST'
        dataset_name = 'youtubeASL'
        self.logger.info(f"Dataset name: {dataset_name}", main_process_only=True)





        self.train_batch_size = 50
        if self.debug:
            self.train_batch_size = 2
        
        self.logger.info(f"Train batch size: {self.train_batch_size}", main_process_only=True)

        self.name_prefix = f'{time_stamp}_ID-{slurm_job_id}_{dataset_name}_{self.model_name}'
        self.logger.info(f"Name prefix: {self.name_prefix}", main_process_only=True)


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}", main_process_only=True)

        if self.feature_encoder:
            self.feature_encoder, encoder_output_size = get_encoder(self.encoder_name, self.device)
        else:
            encoder_output_size = 0

        num_keypoints = {}
        num_keypoints["lh"] = 21
        num_keypoints["rh"] = 21
        num_keypoints["body"] = 9
        num_keypoints["face"] = 18

        self.tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="en_XX")
        self.logger.info("Tokenizer loaded.", main_process_only=True)

        if self.use_feature_encoder:
            # DINOv2（低学习率）优化器
            self.optimizer_encoder = torch.optim.Adam(self.feature_encoder.parameters(), lr=1e-4)  # 比较小的学习率

        self.UniSignModel = UniSignNetwork(num_keypoints, hidden_dim=256, LLM_name="facebook/mbart-large-50", device = 'cpu', tokenizer = self.tokenizer)

        # UniSignModel 的优化器
        self.optimizer_UniSignModel = torch.optim.Adam(self.UniSignModel.parameters(), lr=1e-4)

        
        if self.resume_path is not None:
            self.load_ckpt(self.resume_path)
            self.logger.info(f"Resuming from checkpoint: {self.resume_path}", main_process_only=True)
        

        self.UniSignModel.to(self.device)
        self.feature_encoder.to(self.device)
        self.tokenizer.to(self.device)


        self.logger.info("Dataloader configuration.", main_process_only=True)
        # transform = transforms.Compose([
        #     transforms.Resize((224, 224)),  # 调整大小到 224x224
        #     transforms.ToTensor(),  # 转换为 Tensor，自动归一化到 [0,1]
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        #     ])

        data_dir = '/scratch/rhong5/datasets/youtubeASL'
        train_dataset = YouTubeASL(data_dir)
        self.logger.info(f"Train dataset dir: {data_dir}; sample num: {len(train_dataset)}", main_process_only=True)
        

        self.train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, num_workers = 5, shuffle=True, drop_last=True, pin_memory=True)
        # self.val_loader = DataLoader(val_dataset, batch_size=self.train_batch_size, num_workers = 5, shuffle=False, drop_last=False, pin_memory=True)
        # self.test_loader = DataLoader(test_dataset, batch_size=self.train_batch_size, num_workers = 5, shuffle=False, drop_last=False, pin_memory=True)



        for batch in self.train_loader:
            x_0 = batch[0]
            if isinstance(x_0, torch.Tensor):
                self.logger.info("Batch Shape:", x_0.shape, main_process_only=True)  #([64, 50])
            break
    

    def training(self, dataloader, epoch, mode='train'):

        self.UniSignModel.train()
        self.feature_encoder.train()
        
        epoch_loss = 0

        dataloader = self.train_loader
        total_steps = len(dataloader)  # Total steps in the epoch

        update_steps = 20  # Number of updates per epoch
        step_interval = max(total_steps // update_steps, 1)  # Calculate interval for updates


        progress_bar = tqdm(
            iterable=dataloader,
            total=total_steps,
            desc=f"{mode} Epoch {epoch + 1}/{self.epochs}",
        )

        torch.autograd.set_detect_anomaly(True)
        for step, batch in enumerate(dataloader):
            if self.debug and step >= 5:
                break

            if step % step_interval == 0 or step == total_steps - 1:  # Update progress bar 50 times
                progress_bar.update(step_interval)

            frames_tensor, text, keypoints_dict = batch
            if self.use_feature_encoder:
                pass
            
            hand_keypoints = keypoints_dict['hand']
            body_keypoints = keypoints_dict['body'].to(self.device)
            face_keypoints = keypoints_dict['face'].to(self.device)
                
            left_hand_keypoints = hand_keypoints[:, :, :21, :].to(self.device)
            right_hand_keypoints = hand_keypoints[:, :, 21:, :].to(self.device)

            decoder_input_ids, targets = self.encode_text(text)

            logits, encoder_hidden = self.UniSignModel(left_hand_keypoints, right_hand_keypoints, body_keypoints, face_keypoints, decoder_input_ids = decoder_input_ids)

        
            loss = self.compute_translation_loss(targets, logits)
            self.optimizer_UniSignModel.zero_grad()
            if self.use_feature_encoder:
                self.optimizer_encoder.zero_grad()  # 让 DINOv2 计算梯度
            
            loss.backward()
            self.optimizer_UniSignModel.step()
            
            if self.use_feature_encoder:
                self.optimizer_encoder.step()  # 仅在 pose 任务更新 DINOv2
            epoch_loss += loss.item()

            if step == 0:
                ## print the gpu usage 
                self.logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB at batch size: {self.batch_size} ", main_process_only=True)
                self.logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB", main_process_only=True)

        # print('epoch_noise_loss:', epoch_noise_loss)        
        avg_loss = epoch_loss / len(dataloader)
        
        self.logger.info(f"{mode:<5} Epoch {epoch + 1}, Loss: {avg_loss:.4f}", main_process_only=True)

        return avg_loss
    
    
    def tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize and encode text into numerical IDs using MBART tokenizer."""
        encoded = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128, add_special_tokens=True)
        return encoded['input_ids'].squeeze().to(self.device)

    def encode_text(self, text: str) -> torch.Tensor:
                # Tokenize and encode ground-truth text
        try:
            batch_texts = [self.tokenize_text(t) for t in text]  # List of [seq_len] tensors
        except Exception as e:
            print(f"Error tokenizing text: {e}")
            raise
        
        max_len = max(t.size(0) for t in batch_texts)
        padded_texts = torch.stack([torch.cat([t, torch.tensor([self.tokenizer.pad_token_id] * (max_len - t.size(0)), device=self.device)]) if t.size(0) < max_len else t for t in batch_texts])
        
        
        # Shift for teacher forcing
        targets = padded_texts[:, 1:]  # Remove <s>, shift for next token

        decoder_input_ids = padded_texts[:, :-1] # Input for teacher forcing
        return decoder_input_ids, targets

    def compute_translation_loss(self, targets: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute the translation loss between predicted logits and ground-truth text using mbart-large-50.
        
        Args:
            targets: Ground-truth token IDs, shape [batch_size, sequence_length]
            logits: Predicted logits from the model, shape [batch_size, sequence_length, vocab_size]
        
        Returns:
            Tensor representing the loss.
        """
        
        # Verify inputs
        if logits.dim() != 3 or logits.shape[-1] != self.tokenizer.vocab_size:
            raise ValueError(f"Unexpected logits shape {logits.shape}. Expected [batch_size, sequence_length, {self.tokenizer.vocab_size}]")

        
        # Process logits
        # logits_probs = F.softmax(logits, dim=-1)
        # predicted_ids = torch.argmax(logits_probs, dim=-1)


        # Adjust lengths if necessary
        if logits.size(1) != targets.size(1):
            min_len = min(logits.size(1), targets.size(1))
            logits = logits[:, :min_len, :]
            targets = targets[:, :min_len]
        
        # Compute loss
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)
        loss = criterion(logits_flat, targets_flat)
        
        return loss

    # Add this method to your class`

    def run_all(self):
        
        train_loss_history = []
        
        val_loss_history = []

        
        self.best_loss = float('inf')
        loss_curve_filepath = f'{self.log_dir}/training_loss_{self.name_prefix}.jpg'

        for epoch in range(self.epochs):
            train_loss = self.training(self.train_loader, epoch, 'train', self.debug)
            train_loss_history.append(train_loss)


            if train_loss < self.best_loss:
                self.best_loss = train_loss
                ckpt_path = f"{self.ckpt_dir}/{self.name_prefix}_best.pth"
                self.best_ckpt_path = ckpt_path
                self.logger.info(f"Best epoch {epoch}: {self.best_loss:.4f}", main_process_only=True)
            else:
                ckpt_path = f"{self.ckpt_dir}/{self.name_prefix}.pth"
            self.save_ckpt(epoch, ckpt_path)

            
            if os.path.exists(loss_curve_filepath):
                os.remove(loss_curve_filepath)

            if len(val_loss_history) > 0:
                num_subplots = 2
            else:
                num_subplots = 1
               
            fig, axes = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 4))
            # Flatten the axes array for easier indexing
            axes = axes.flatten()

            # 第一张子图：train_noise_loss & val_noise_loss
            axes[0].plot(train_loss_history, label='Train Loss', marker='o', linestyle='-')
            axes[0].set_xlabel("Epochs")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Train Loss")
            axes[0].legend()
            axes[0].grid(True)

            if num_subplots == 2:
                # 第2张子图：val_noise_loss
                axes[1].plot(val_loss_history, label='Validation Loss', marker='o', linestyle='-')
                axes[1].set_xlabel("Epochs")
                axes[1].set_ylabel("Loss")
                axes[1].set_title("Validation Loss")
                axes[1].legend()
                axes[1].grid(True)


            # 调整子图布局，使标题和标签不重叠
            plt.tight_layout()
            
            plt.savefig(loss_curve_filepath)
            plt.close()

            self.logger.info(f"saving loss curve to: {loss_curve_filepath}\n", main_process_only=True)

            
            if self.debug and epoch > 2:
                break
        

        self.logger.info("Done!")
    
    def save_ckpt(self, epoch, ckpt_path):
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        # save checkpoint

        model_dict = {
            'UniSignModel': self.UniSignModel.state_dict(),
            'encoder': self.feature_encoder.state_dict() if self.use_feature_encoder else None,
            'optimizer_uniSign': self.optimizer_UniSignModel.state_dict(),
            'optimizer_encoder': self.optimizer_encoder.state_dict() if self.use_feature_encoder else None,
            'epoch': epoch,
        }

        # save the model
        torch.save(model_dict, ckpt_path)
        self.logger.info(f'Saving model to: {ckpt_path}', main_process_only=True)

    def load_ckpt(self, ckpt_path):
        model_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
        self.UniSignModel.load_state_dict(model_dict['UniSignModel'], strict=False)
        self.optimizer_UniSignModel.load_state_dict(model_dict['optimizer_uniSign'])
        if self.use_feature_encoder:
            self.feature_encoder.load_state_dict(model_dict['encoder'], strict=False)
            self.optimizer_encoder.load_state_dict(model_dict['optimizer_encoder'])
        self.best_epoch = model_dict['epoch']
        self.logger.info(f'Loading model from: {ckpt_path}', main_process_only=True)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--debug", action="store_true")
    args.add_argument('--model_name', type=str, default='Unet2D_CRS_Cond', help='Model name, e.g., Unet2D, Unet2D_CRS_Cond, Unet2D_Linear_Cond')
    args.add_argument('--resume', type=str, default=None, help='Resume training from a checkpoint')
    args = args.parse_args()


    # test_forward()
    # test_vec_diffusion()
    # interhand_diffusion_on_mano_para(args)
    # xD_pose_2dunet_diffusion_on_interhand(args)
    manoDiffusion = ManoParaUnetDiffusion(args)
    manoDiffusion.run_all()
