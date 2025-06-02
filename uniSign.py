import os
import absl.logging
import logging

# ✅ 强制 `absl` 只输出 ERROR 及以上的日志
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
absl.logging.set_verbosity(absl.logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl.logging").setLevel(logging.ERROR)
logging.getLogger("absl.initialize_log").setLevel(logging.ERROR)
os.environ["EGL_LOG_LEVEL"] = "error"  # 降低 EGL 日志级别
os.environ["GLOG_minloglevel"] = "3"  # 只允许 ERROR 级别
os.environ["MESA_DEBUG"] = "0"
os.environ["MESA_LOG_LEVEL"] = "error"


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
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# # 3️⃣ 释放 GPU 显存，防止 OOM
# torch.cuda.empty_cache()

# # 屏蔽 OpenGL/EGL 的 INFO 级别日志
# os.environ["GLOG_minloglevel"] = "3"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["EGL_LOG_LEVEL"] = "error"
# os.environ["MESA_LOG_LEVEL"] = "error"

# class NullWriter:
#     def write(self, message):
#         pass
#     def flush(self):
#         pass

# sys.stdout = NullWriter()  # 屏蔽标准输出
# sys.stderr = NullWriter()  # 屏蔽错误输出

# # 继续导入其他库
import torch.multiprocessing as mp


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
from transformers.modeling_outputs import BaseModelOutput

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from network.uni_sign_network import UniSignNetwork
from network.feature_encoder import get_encoder
from transformers import MBartTokenizer, MBartForConditionalGeneration
from dataloader.dataloader import get_dataloader
from utils.mediapipe_kpts_mapping import MediapipeKptsMapping
from config.config import arg_settings


class UniSignTrans():
    def __init__(self, args):
        self.args = args
        self.debug = args.debug
        self.model_name = args.model_name
        self.resume_path = args.resume
        self.dataset_name = args.dataset_name
        self.feature_encoder_name = args.feature_encoder
        self.modality = args.modality
        self.finetune = args.finetune
        self.n_frames = args.n_frames
        self.batch_size = args.batch_size

        time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        slurm_job_id = os.environ.get('SLURM_JOB_ID') if os.environ.get('SLURM_JOB_ID') is not None else '0'

        self.weight_name_prefix = f'{time_stamp}_JID-{slurm_job_id}_{self.model_name}_{self.dataset_name}_{self.modality}'

        log_dir = f'Zlog/{time_stamp}_JID-{slurm_job_id}'
        self.log_dir = log_dir  

        os.makedirs(log_dir, exist_ok=True)
        shutil.copy(__file__, os.path.join(log_dir, f"{__file__.split('/')[-1].split('.')[0]}_{time_stamp}.py"))
        config_filename = os.path.join(log_dir, f"config_{time_stamp}.py")
        shutil.copy("config/config.py", config_filename)
        # shutil.copy("config/config.py", os.path.join(self.log_dir, f"config_{slurm_job_id}.py"))

        self.ckpt_dir = '/scratch/rhong5/weights/temp_training_weights/singLangTran'
        os.makedirs(self.ckpt_dir, exist_ok=True)

        
        log_path = os.path.join(self.log_dir, "info.log")
        # shutil.copyfile("config/config.py", os.path.join(self.log_dir, f"config_{slurm_job_id}.py"))

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
        # self.logger.warning = patched_log_method(self.logger.warning)
        self.logger.error = patched_log_method(self.logger.error)
        # self.logger.debug = patched_log_method(self.logger.debug)
        logging.getLogger("gl_context").setLevel(logging.ERROR)
        logging.getLogger("gl_context_egl").setLevel(logging.ERROR)


        self.logger.info(f"\n{os.path.basename(__file__)}\n", main_process_only=True)
        self.logger.info(f'logging_dir: {self.log_dir}', main_process_only=True)
        self.logger.info(f'output_ckpts_dir: {self.ckpt_dir}\n', main_process_only=True)

        self.start_epoch = 0
        self.best_loss = float('inf')
        self.max_epochs = 45
        self.logger.info(f"Training epochs: {self.max_epochs}", main_process_only=True)
        
        # use_condition = False        
        

        self.logger.info(f"feature encoder: {self.feature_encoder_name}", main_process_only=True)
        
        self.use_feature_encoder = False
        self.logger.info(f"Use use_feature_encoder: {self.use_feature_encoder}", main_process_only=True)

        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}", main_process_only=True)

        # dataset_name = 'MNIST'
        self.logger.info(f"Dataset name: {self.dataset_name}", main_process_only=True)

        self.train_batch_size = self.batch_size
        if self.debug:
            self.train_batch_size = 5
        
        self.logger.info(f"Train batch size: {self.train_batch_size}", main_process_only=True)

        
        self.logger.info("Dataloader configuration.", main_process_only=True)
        # transform = transforms.Compose([
        #     transforms.Resize((224, 224)),  # 调整大小到 224x224
        #     transforms.ToTensor(),  # 转换为 Tensor，自动归一化到 [0,1]
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        #     ])

        train_loader, val_loader, test_loader, \
        train_dataset,val_dataset, test_dataset, \
        self.train_sampler, self.val_sampler, self.test_sampler = get_dataloader(
            dataset_name=self.dataset_name,
            logger=self.logger,
            debug=self.debug,
            train_batch_size=self.train_batch_size,
            val_batch_size=self.train_batch_size,
            test_batch_size=self.train_batch_size,
            modality = self.modality,
            n_frames = self.n_frames,
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        self.logger.info(f"Train dataset size: {len(self.train_dataset)}", main_process_only=True)
        self.logger.info(f"Val dataset size: {len(self.val_dataset)}", main_process_only=True)  
        self.logger.info(f"Test dataset size: {len(self.test_dataset)}", main_process_only=True)
        self.logger.info(f"Train batch size: {self.train_batch_size}", main_process_only=True)
        self.logger.info(f"Val batch size: {self.train_batch_size}", main_process_only=True)
        self.logger.info(f"Test batch size: {self.train_batch_size}", main_process_only=True)

        
        for batch in self.train_loader:
            frames_tensor, text, keypoints_dict = batch
            face_keypoints = keypoints_dict['face'].to(self.device).float()
            body_keypoints = keypoints_dict['body'].to(self.device).float()
            hand_keypoints = keypoints_dict['hand'][:, :, :21, :].to(self.device).float()
            self.logger.info(f"hand_keypoints Shape: {hand_keypoints.shape}", main_process_only=True)  
            self.logger.info(f"body_keypoints Shape: {body_keypoints.shape}", main_process_only=True)  
            self.logger.info(f"face_keypoints Shape: {face_keypoints.shape}", main_process_only=True)  
            if isinstance(frames_tensor, torch.Tensor):
                self.logger.info(f"frames_tensor Shape: {frames_tensor.shape}", main_process_only=True)  
            break


        if 'rgb' in self.modality.lower() and self.feature_encoder_name:
            self.feature_encoder, encoder_output_size = get_encoder(self.feature_encoder_name, self.device)
        else:
            encoder_output_size = 0
            self.feature_encoder = None

        
        
        self.hand_mapping = MediapipeKptsMapping.hand_keypoints_mapping
        self.face_mapping = MediapipeKptsMapping.face_keypoints_mapping
        self.body_mapping = MediapipeKptsMapping.body_keypoints_mapping

        # Define keypoint indices mapping to MediaPipe landmarks
        self.hand_indices = [value for key, value in self.hand_mapping.items()]  # Map to MediaPipe hand landmarks (0–20)
        self.body_indices = [value for key, value in self.body_mapping.items()]  # Map to MediaPipe body landmarks (0–8)
        self.face_indices = [value for key, value in self.face_mapping.items()]
        

        num_keypoints = {}
        num_keypoints["lh"] = len(self.hand_indices)  # 21
        num_keypoints["rh"] = len(self.hand_indices)  # 21
        num_keypoints["body"] = len(self.body_indices)  # 
        num_keypoints["face"] = len(self.face_indices)  # 

        self.tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="en_XX")
        self.logger.info("Tokenizer loaded.", main_process_only=True)

        if self.use_feature_encoder:
            # DINOv2（低学习率）优化器
            self.optimizer_encoder = torch.optim.Adam(self.feature_encoder.parameters(), lr=1e-4)  # 比较小的学习率

        self.UniSignModel = UniSignNetwork(hidden_dim=256, LLM_name="facebook/mbart-large-50", device = self.device, tokenizer = self.tokenizer)
        self.UniSignModel.float()

        
        self.UniSignModel.to(self.device)

        # UniSignModel 的优化器
        self.optimizer_UniSignModel = torch.optim.Adam(self.UniSignModel.parameters(), lr=1e-4)


        if self.resume_path is not None:
            self.load_ckpt(self.resume_path)
            self.logger.info(f"Resuming from checkpoint: {self.resume_path}", main_process_only=True)
        


        
        if self.use_feature_encoder:
            self.feature_encoder.to(self.device)

    

    def training(self, dataloader, epoch, mode='train'):

        self.UniSignModel.train()
        if self.use_feature_encoder:
            self.feature_encoder.train()
        
        epoch_loss = 0

        dataloader = self.train_loader
        total_steps = len(dataloader)  # Total steps in the epoch

        update_steps = 20  # Number of updates per epoch
        step_interval = max(total_steps // update_steps, 1)  # Calculate interval for updates


        progress_bar = tqdm(
            iterable=dataloader,
            total=total_steps,
            desc=f"{mode} Epoch {epoch}/{ self.max_epochs}",
        )

        torch.autograd.set_detect_anomaly(True)
        self.logger.info(f"Start {mode} Epoch {epoch}", main_process_only=True)
        for step, batch in enumerate(dataloader):
            if self.debug and step >= 5:
                break

            if step % step_interval == 0 or step == total_steps - 1:  # Update progress bar 50 times
                progress_bar.update(step_interval)

            frames_tensor, text, keypoints_dict = batch
            if self.use_feature_encoder:
                pass
            
            hand_keypoints = keypoints_dict['hand']
            body_keypoints = keypoints_dict['body'].to(self.device).float()
            face_keypoints = keypoints_dict['face'].to(self.device).float()
                
            right_hand_keypoints = hand_keypoints[:, :, :21, :].to(self.device).float()
            left_hand_keypoints = hand_keypoints[:, :, 21:, :].to(self.device).float()

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
                self.logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB at batch size: {self.train_batch_size} ", main_process_only=True)
                self.logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB", main_process_only=True)

        # print('epoch_noise_loss:', epoch_noise_loss)        
        avg_loss = epoch_loss / len(dataloader)
        
        self.logger.info(f"{mode:<5} Epoch {epoch + 1}, Loss: {avg_loss:.4f}", main_process_only=True)

        return avg_loss
    
    def evaluate(self, dataloader, epoch, mode='val'):
        """
        Evaluate the UniSignNetwork model on the validation or test set, computing BLEU scores and loss.
        
        Args:
            dataloader: DataLoader for validation or test data
            epoch: Current epoch number
            mode: Mode of evaluation ('val' or 'test')
        
        Returns:
            dict: Contains average loss and BLEU scores (1-4)
        """
        if mode not in ['val', 'test']:
            raise ValueError("Mode must be 'val' or 'test'")

        self.UniSignModel.eval()
        if self.use_feature_encoder:
            self.feature_encoder.eval()
        
        epoch_loss = 0
        bleu_scores = {1: [], 2: [], 3: [], 4: []}  # Store BLEU scores for each n-gram
        smoothing = SmoothingFunction().method1  # Smoothing for BLEU score
        
        total_steps = len(dataloader)
        update_steps = 20  # Number of updates per epoch
        step_interval = max(total_steps // update_steps, 1)
        
        progress_bar = tqdm(
            iterable=dataloader,
            total=total_steps,
            desc=f"{mode} Epoch {epoch}/{self.max_epochs}",
        )
        
        self.logger.info(f"Start {mode} Epoch {epoch}", main_process_only=True)
        
        with torch.no_grad():  # Disable gradient computation for evaluation
            for step, batch in enumerate(dataloader):
                if self.debug and step >= 5:
                    break
                
                if step % step_interval == 0 or step == total_steps - 1:
                    progress_bar.update(step_interval)
                
                frames_tensor, text, keypoints_dict = batch
                
                hand_keypoints = keypoints_dict['hand']
                body_keypoints = keypoints_dict['body'].to(self.device).float()
                face_keypoints = keypoints_dict['face'].to(self.device).float()
                
                right_hand_keypoints = hand_keypoints[:, :, :21, :].to(self.device).float()
                left_hand_keypoints = hand_keypoints[:, :, 21:, :].to(self.device).float()
                
                decoder_input_ids, targets = self.encode_text(text)
                
                # Get model output (translated_text, encoder_hidden)
                translated_text, encoder_hidden = self.UniSignModel(
                    left_hand_keypoints,
                    right_hand_keypoints,
                    body_keypoints,
                    face_keypoints,
                    split=mode,
                    decoder_input_ids=decoder_input_ids
                )
                
                # Compute logits for loss by re-running the model with generated_ids
                # Encode translated_text back to token IDs
                generated_ids = self.tokenizer(
                    translated_text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=128
                )['input_ids'].to(self.device)
                    
                # Wrap encoder_hidden in BaseModelOutput
                # encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)
                encoder_outputs = (encoder_hidden,)

                # Run LLM to get logits
                outputs = self.UniSignModel.llm_trans.model(
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=generated_ids,
                    return_dict=True
                )
                logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
            

                # Compute translation loss
                loss = self.compute_translation_loss(targets, logits)
                epoch_loss += loss.item()
                
                # Use translated_text for BLEU score
                for i, pred_text in enumerate(translated_text):
                    ref_text = self.tokenizer.decode(targets[i], skip_special_tokens=True)
                    
                    # Log sample translations for the first batch
                    if step == 0 and i < 2:
                        self.logger.info(
                            f"Sample {i} - Predicted: {pred_text[:100]}... | Reference: {ref_text[:100]}...",
                            main_process_only=True
                        )
                    
                    # Tokenize for BLEU score
                    pred_tokens = pred_text.split()
                    ref_tokens = ref_text.split()
                    
                    # Calculate BLEU scores for n-grams 1 to 4
                    for n in range(1, 5):
                        try:
                            score = sentence_bleu(
                                [ref_tokens],
                                pred_tokens,
                                weights=[1.0/n] * n + [0.0] * (4-n),
                                smoothing_function=smoothing
                            )
                            bleu_scores[n].append(score)
                        except Exception as e:
                            self.logger.warning(f"BLEU-{n} computation failed for sample {i}: {e}")
                            bleu_scores[n].append(0.0)
                
                if step == 0:
                    self.logger.info(
                        f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB at batch size: {self.train_batch_size}",
                        main_process_only=True
                    )
                    self.logger.info(
                        f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB",
                        main_process_only=True
                    )
        
        # Compute average metrics
        avg_loss = epoch_loss / len(dataloader)
        avg_bleu = {n: np.mean(scores) for n, scores in bleu_scores.items()}
        
        # Log results
        self.logger.info(
            f"{mode:<5} Epoch {epoch + 1}, Loss: {avg_loss:.4f}, "
            f"BLEU-1: {avg_bleu[1]:.4f}, BLEU-2: {avg_bleu[2]:.4f}, "
            f"BLEU-3: {avg_bleu[3]:.4f}, BLEU-4: {avg_bleu[4]:.4f}",
            main_process_only=True
        )
        
        return {
            'loss': avg_loss,
            'bleu_1': avg_bleu[1],
            'bleu_2': avg_bleu[2],
            'bleu_3': avg_bleu[3],
            'bleu_4': avg_bleu[4]
        }
    
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

    def run_training_only(self):
        
        train_loss_history = []
        
        val_loss_history = []

        
        loss_curve_filepath = f'{self.log_dir}/training_loss_curve.jpg'

        self.logger.info("Start training...", main_process_only=True)
        for epoch in range(self.start_epoch, self.max_epochs):
            train_loss = self.training(self.train_loader, epoch, 'train')
            train_loss_history.append(train_loss)


            if train_loss < self.best_loss:
                self.best_loss = train_loss
                ckpt_path = f"{self.ckpt_dir}/{self.weight_name_prefix}_best.pth"
                self.best_ckpt_path = ckpt_path
                self.logger.info(f"Best epoch {epoch}: {self.best_loss:.4f}", main_process_only=True)
            else:
                ckpt_path = f"{self.ckpt_dir}/{self.weight_name_prefix}.pth"
            self.save_ckpt(epoch, ckpt_path)

            
            if os.path.exists(loss_curve_filepath):
                os.remove(loss_curve_filepath)

            if len(val_loss_history) > 0:
                num_subplots = 2
            else:
                num_subplots = 1
               
            fig, axes = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 4))
            # Flatten the axes array for easier indexing

            if num_subplots == 1:
                axes = [axes]

            # 第一张子图：train_noise_loss & val_noise_loss
            epochs = range(epoch, epoch + len(train_loss_history))
            axes[0].plot(epochs, train_loss_history, label='Train Loss', marker='o', linestyle='-')
            axes[0].set_xlabel("Epochs")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Train Loss")
            axes[0].legend()
            axes[0].grid(True)

            if num_subplots == 2:
                # 第2张子图：val_noise_loss
                val_epochs = range(epoch, epoch + len(val_loss_history))
                axes[1].plot(val_epochs, val_loss_history, label='Validation Loss', marker='o', linestyle='-')
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

    def run_all(self):
        train_loss_history = []
        val_loss_history = []
        bleu_history = {1: [], 2: [], 3: [], 4: []}  # Store BLEU scores for each n-gram
        
        loss_curve_filepath = f'{self.log_dir}/loss_curve.jpg'
        bleu_curve_filepath = f'{self.log_dir}/bleu_scores.jpg'
        
        self.logger.info("Start training...", main_process_only=True)
        
        for epoch in range(self.start_epoch, self.max_epochs):
            # Training
            train_loss = self.training(self.train_loader, epoch, 'train')
            train_loss_history.append(train_loss)
            
            # Validation
            val_metrics = self.evaluate(self.val_loader, epoch, 'val')
            val_loss_history.append(val_metrics['loss'])
            for n in range(1, 5):
                bleu_history[n].append(val_metrics[f'bleu_{n}'])
            
            # Save best model checkpoint
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                ckpt_path = f"{self.ckpt_dir}/{self.weight_name_prefix}_best.pth"
                self.best_ckpt_path = ckpt_path
                self.logger.info(f"Best epoch {epoch}: {self.best_loss:.4f}", main_process_only=True)
            else:
                ckpt_path = f"{self.ckpt_dir}/{self.weight_name_prefix}.pth"
            self.save_ckpt(epoch, ckpt_path)
            
            # Plot and save loss curve (train and val in one figure)
            if os.path.exists(loss_curve_filepath):
                os.remove(loss_curve_filepath)
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            epochs = range(self.start_epoch, self.start_epoch + len(train_loss_history))
            
            # Plot train and validation loss
            ax.plot(epochs, train_loss_history, label='Train Loss', marker='o', linestyle='-')
            ax.plot(epochs, val_loss_history, label='Validation Loss', marker='s', linestyle='--')
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.set_title("Training and Validation Loss")
            ax.legend()
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(loss_curve_filepath)
            plt.close()
            
            self.logger.info(f"Saving loss curve to: {loss_curve_filepath}", main_process_only=True)
            
            # Plot and save BLEU scores
            if os.path.exists(bleu_curve_filepath):
                os.remove(bleu_curve_filepath)
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            for n in range(1, 5):
                ax.plot(epochs, bleu_history[n], label=f'BLEU-{n}', marker='o', linestyle='-')
            
            ax.set_xlabel("Epochs")
            ax.set_ylabel("BLEU Score")
            ax.set_title("Validation BLEU Scores")
            ax.legend()
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(bleu_curve_filepath)
            plt.close()
            
            self.logger.info(f"Saving BLEU scores plot to: {bleu_curve_filepath}", main_process_only=True)
            
            if self.debug and epoch > 2:
                break
        
        # Test the best model on the test set
        self.logger.info("Loading best model for testing...", main_process_only=True)
        self.load_ckpt(self.best_ckpt_path)
        
        test_metrics = self.evaluate(self.test_loader, epoch=0, mode='test')
        
        # Save test results
        test_results_filepath = f'{self.log_dir}/test_results.txt'
        with open(test_results_filepath, 'w') as f:
            f.write(f"Test Results:\n")
            f.write(f"Loss: {test_metrics['loss']:.4f}\n")
            f.write(f"BLEU-1: {test_metrics['bleu_1']:.4f}\n")
            f.write(f"BLEU-2: {test_metrics['bleu_2']:.4f}\n")
            f.write(f"BLEU-3: {test_metrics['bleu_3']:.4f}\n")
            f.write(f"BLEU-4: {test_metrics['bleu_4']:.4f}\n")
        
        self.logger.info(f"Test results saved to: {test_results_filepath}", main_process_only=True)
        self.logger.info(
            f"Test Loss: {test_metrics['loss']:.4f}, "
            f"BLEU-1: {test_metrics['bleu_1']:.4f}, "
            f"BLEU-2: {test_metrics['bleu_2']:.4f}, "
            f"BLEU-3: {test_metrics['bleu_3']:.4f}, "
            f"BLEU-4: {test_metrics['bleu_4']:.4f}",
            main_process_only=True
        )
        
        self.logger.info("Done!", main_process_only=True)
    
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
            'best_loss': self.best_loss,
            # 'best_epoch': epoch,
        }

        # save the model
        torch.save(model_dict, ckpt_path)
        self.logger.info(f'Saving model to: {ckpt_path}', main_process_only=True)

    def load_ckpt(self, ckpt_path):
        model_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
        self.UniSignModel.load_state_dict(model_dict['UniSignModel'], strict=False)
        if not self.finetune:    
            self.optimizer_UniSignModel.load_state_dict(model_dict['optimizer_uniSign'])
        if self.use_feature_encoder:
            self.feature_encoder.load_state_dict(model_dict['encoder'], strict=False)
            if not self.finetune:
                self.optimizer_encoder.load_state_dict(model_dict['optimizer_encoder'])

        if not self.finetune:
            self.start_epoch = model_dict['epoch']
        self.logger.info(f"Start epoch: {self.start_epoch}", main_process_only=True)
        if 'best_loss' in model_dict and not self.finetune:
            self.best_loss = model_dict['best_loss']
            self.logger.info(f"Best loss: {self.best_loss}", main_process_only=True)
        else:
            self.best_loss = float('inf')

        self.logger.info(f'Loading model from: {ckpt_path}', main_process_only=True)





def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--model_name", type=str, default='uniSign', help="Model name")

    parser.add_argument("--resume", type=str, default=arg_settings["resume"], help="Resume training from a checkpoint")
    parser.add_argument("--dataset_name", type=str, default=arg_settings["dataset_name"], help="Dataset name")
    parser.add_argument("--feature_encoder", type=str, default=arg_settings["feature_encoder"], help="Feature encoder name")
    parser.add_argument("--modality", type=str, default=arg_settings["modality"], help="Modality, e.g., rgb, pose, rgb_pose")
    parser.add_argument("--finetune", default=arg_settings["finetune"], action="store_true", help="Fine-tune the model")
    parser.add_argument("--n_frames", type=int, default=arg_settings["n_frames"], help="Number of frames")
    parser.add_argument("--batch_size", type=int, default=arg_settings["batch_size"], help="Batch size")
    return parser.parse_args()


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)

    args = get_args()

    # test_forward()
    # test_vec_diffusion()
    # interhand_diffusion_on_mano_para(args)
    # xD_pose_2dunet_diffusion_on_interhand(args)
    uniSignTrans = UniSignTrans(args)
    uniSignTrans.run_all()
