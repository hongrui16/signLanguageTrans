import os
import absl.logging
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime
import time
import cv2
import shutil
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from accelerate.logging import get_logger
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from typing import List, Tuple
from transformers.modeling_outputs import BaseModelOutput
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from network.uni_sign_network import UniSignNetwork
from network.feature_encoder import get_encoder
from transformers import MBartTokenizer, MBartForConditionalGeneration
from dataloader.dataloader import get_dataloader
from utils.mediapipe_kpts_mapping import MediapipeKptsMapping
from config.config import arg_settings

# Set up logging to suppress unnecessary outputs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
absl.logging.set_verbosity(absl.logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl.logging").setLevel(logging.ERROR)
logging.getLogger("absl.initialize_log").setLevel(logging.ERROR)
os.environ["EGL_LOG_LEVEL"] = "error"
os.environ["GLOG_minloglevel"] = "3"
os.environ["MESA_DEBUG"] = "0"
os.environ["MESA_LOG_LEVEL"] = "error"

class UniSignTrans:
    def __init__(self, args, **kwargs):
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
        self.use_feature_encoder = args.use_feature_encoder

        time_stamp = kwargs.get('time_stamp', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        slurm_job_id = kwargs.get('slurm_job_id', os.environ.get('SLURM_JOB_ID', '0'))


        # Initialize distributed environment
        self.init_distributed()

        
        self.weight_name_prefix = f'{time_stamp}_JID-{slurm_job_id}_{self.model_name}_{self.dataset_name}_{self.modality}'
        dir_str = f'{time_stamp}_JID-{slurm_job_id}'
        log_dir = f'Zlog/{dir_str}'
        self.log_dir = log_dir

        if self.accelerator.is_main_process:
            os.makedirs(log_dir, exist_ok=True)
            shutil.copy(__file__, os.path.join(log_dir, f"{__file__.split('/')[-1].split('.')[0]}_{time_stamp}.py"))
            config_filename = os.path.join(log_dir, f"config_{time_stamp}.py")
            shutil.copy("config/config.py", config_filename)

        # Synchronize all processes to ensure directory exists
        if dist.is_initialized():
            dist.barrier()
            
        self.ckpt_dir = f'/scratch/rhong5/weights/temp_training_weights/signLanguageTrans/{dir_str}'
        if self.accelerator.is_main_process:
            os.makedirs(self.ckpt_dir, exist_ok=True)

        # Synchronize all processes to ensure checkpoint directory exists
        if dist.is_initialized():
            dist.barrier()

        log_path = os.path.join(self.log_dir, "info.log")
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

        # Patch logger methods for main process only
        def patched_log_method(original_method):
            def new_method(msg, *args, main_process_only=False, **kwargs):
                if not main_process_only or self.accelerator.is_main_process:
                    original_method(msg, *args, **kwargs)
            return new_method

        self.logger.info = patched_log_method(self.logger.info)
        self.logger.error = patched_log_method(self.logger.error)
        self.logger.warning = patched_log_method(self.logger.warning)  # <- 添加这一行

        logging.getLogger("gl_context").setLevel(logging.ERROR)
        logging.getLogger("gl_context_egl").setLevel(logging.ERROR)

        self.logger.info(f"\n{os.path.basename(__file__)}\n", main_process_only=True)
        self.logger.info(f'logging_dir: {self.log_dir}', main_process_only=True)
        self.logger.info(f'output_ckpts_dir: {self.ckpt_dir}\n', main_process_only=True)

        self.start_epoch = 0
        self.best_loss = float('inf')
        self.max_epochs = 45
        self.logger.info(f"Training epochs: {self.max_epochs}", main_process_only=True)

        self.logger.info(f"feature encoder: {self.feature_encoder_name}", main_process_only=True)

        self.logger.info(f"Use use_feature_encoder: {self.use_feature_encoder}", main_process_only=True)

        self.device = self.accelerator.device
        self.logger.info(f"Using device: {self.device}", main_process_only=True)

        self.logger.info(f"Dataset name: {self.dataset_name}", main_process_only=True)

        self.train_batch_size = self.batch_size
        if self.debug:
            self.train_batch_size = 5
        self.logger.info(f"Train batch size: {self.train_batch_size}", main_process_only=True)

        self.logger.info("Dataloader configuration.", main_process_only=True)
        train_loader, val_loader, test_loader, \
        train_dataset,val_dataset, test_dataset, \
        self.train_sampler, self.val_sampler, self.test_sampler  = get_dataloader(
            dataset_name=self.dataset_name,
            logger=self.logger,
            debug=self.debug,
            train_batch_size=self.train_batch_size,
            val_batch_size=self.train_batch_size,
            test_batch_size=self.train_batch_size,
            modality=self.modality,
            n_frames=self.n_frames,
            distributed=True,
            world_size=self.accelerator.num_processes,
            rank=self.accelerator.process_index
        )

        if isinstance(train_loader.sampler, DistributedSampler):
            self.logger.info("Train loader correctly uses DistributedSampler", main_process_only=True)
        else:
            self.logger.warning("Train loader does NOT use DistributedSampler", main_process_only=True)

        if not val_loader is None and isinstance(val_loader.sampler, DistributedSampler):
            self.logger.info("Val loader correctly uses DistributedSampler", main_process_only=True)
        else:
            self.logger.warning("Val loader does NOT use DistributedSampler", main_process_only=True)
        
        if not test_loader is None and isinstance(test_loader.sampler, DistributedSampler):
            self.logger.info("Test loader correctly uses DistributedSampler", main_process_only=True)
        else:
            self.logger.warning("Test loader does NOT use DistributedSampler", main_process_only=True)

        self.train_loader = self.accelerator.prepare(train_loader)
        self.val_loader = self.accelerator.prepare(val_loader) if val_loader else None
        self.test_loader = self.accelerator.prepare(test_loader) if test_loader else None
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.logger.info(f"Train dataset size: {len(self.train_dataset)}", main_process_only=True)
        self.logger.info(f"Val dataset size: {len(self.val_dataset) if self.val_dataset else 0}", main_process_only=True)
        self.logger.info(f"Test dataset size: {len(self.test_dataset) if self.test_dataset else 0}", main_process_only=True)
        self.logger.info(f"Train batch size: {self.train_batch_size}", main_process_only=True)
        


        # Log sampler type after accelerator.prepare for debugging
        self.logger.info(f"Train loader sampler after prepare: {type(self.train_loader.sampler).__name__}", main_process_only=True)

        # Log sample batch shapes
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

        if 'rgb' in self.modality.lower() and self.use_feature_encoder:
            self.feature_encoder, encoder_output_size = get_encoder(self.feature_encoder_name, self.device)
            
        else:
            encoder_output_size = 0
            self.feature_encoder = None

        self.hand_mapping = MediapipeKptsMapping.hand_keypoints_mapping
        self.face_mapping = MediapipeKptsMapping.face_keypoints_mapping
        self.body_mapping = MediapipeKptsMapping.body_keypoints_mapping

        self.hand_indices = [value for key, value in self.hand_mapping.items()]
        self.body_indices = [value for key, value in self.body_mapping.items()]
        self.face_indices = [value for key, value in self.face_mapping.items()]

        num_keypoints = {
            "lh": len(self.hand_indices),
            "rh": len(self.hand_indices),
            "body": len(self.body_indices),
            "face": len(self.face_indices)
        }

        self.tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="en_XX")
        self.logger.info("Tokenizer loaded.", main_process_only=True)

        self.UniSignModel = UniSignNetwork(hidden_dim=256, LLM_name="facebook/mbart-large-50", device=self.device, tokenizer=self.tokenizer)
        self.UniSignModel.float().to(self.device)
        # Wrap model with DDP
        self.UniSignModel = self.accelerator.prepare(self.UniSignModel)

        
        self.optimizer_UniSignModel = torch.optim.Adam(self.UniSignModel.parameters(), lr=2e-4)  # Scaled for effective batch size
        self.optimizer_UniSignModel = self.accelerator.prepare(self.optimizer_UniSignModel)

        

        if self.use_feature_encoder:
            self.feature_encoder.to(self.device)
            self.optimizer_encoder = torch.optim.Adam(self.feature_encoder.parameters(), lr=2e-4)
            self.optimizer_encoder = self.accelerator.prepare(self.optimizer_encoder)
            self.accelerator.prepare(self.feature_encoder)

        if self.resume_path is not None:
            self.load_ckpt_after_acceleratorPrepare(self.resume_path)
            self.logger.info(f"Resuming from checkpoint: {self.resume_path}", main_process_only=True)
        
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.logger.info(f"[AFTER PREPARE] Distributed training initialized: {dist.is_initialized()}", main_process_only=True)
            self.logger.info(f"[AFTER PREPARE] World size: {dist.get_world_size()}, Rank: {dist.get_rank()}")
        else:
            self.logger.info("Distributed training not initialized.")


        self.best_ckpt_path = None

    def init_distributed(self):
        """Initialize distributed training environment."""
        self.accelerator = Accelerator(gradient_accumulation_steps=1)  # Explicitly set for clarity
        print(f"Rank {self.accelerator.process_index}/{self.accelerator.num_processes} initialized")
        # print(f"Distributed training initialized: {dist.is_initialized()}")
        print(f"Accelerator num_processes: {self.accelerator.num_processes}")
        # print(f"World size: {dist.get_world_size() if dist.is_initialized() else 1}, Rank: {dist.get_rank() if dist.is_initialized() else 0}")


    def training(self, dataloader, epoch, mode='train'):
        self.UniSignModel.train()
        if self.use_feature_encoder:
            self.feature_encoder.train()

        epoch_loss = 0
        total_steps = len(dataloader)
        update_steps = 20
        step_interval = max(total_steps // update_steps, 1)

        progress_bar = tqdm(
            iterable=dataloader,
            total=total_steps,
            desc=f"{mode} Epoch {epoch}/{self.max_epochs}",
            disable=not self.accelerator.is_main_process
        )

        self.logger.info(f"Start {mode} Epoch {epoch}", main_process_only=True)
        for step, batch in enumerate(dataloader):
            if self.debug and step >= 5:
                break

            if step % step_interval == 0 or step == total_steps - 1:
                progress_bar.update(step_interval)

            frames_tensor, text, keypoints_dict = batch
            # Move all tensors to the correct device
            hand_keypoints = keypoints_dict['hand'].to(self.device).float()
            body_keypoints = keypoints_dict['body'].to(self.device).float()
            face_keypoints = keypoints_dict['face'].to(self.device).float()
            right_hand_keypoints = hand_keypoints[:, :, :21, :].float()
            left_hand_keypoints = hand_keypoints[:, :, 21:, :].float()
            if self.use_feature_encoder:
                frames_tensor = frames_tensor.to(self.device).float()

            decoder_input_ids, targets = self.encode_text(text)

            with self.accelerator.accumulate(self.UniSignModel):  # Support gradient accumulation
                logits, encoder_hidden = self.UniSignModel(
                    left_hand_keypoints,
                    right_hand_keypoints,
                    body_keypoints,
                    face_keypoints,
                    decoder_input_ids=decoder_input_ids
                )
                loss = self.compute_translation_loss(targets, logits)
                self.accelerator.backward(loss)

                self.optimizer_UniSignModel.step()
                self.optimizer_UniSignModel.zero_grad()
                if self.use_feature_encoder:
                    self.optimizer_encoder.step()
                    self.optimizer_encoder.zero_grad()

                # Reduce loss across GPUs for each batch
                loss_tensor = torch.tensor(loss.item(), device=self.device)
                if dist.is_initialized():
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                epoch_loss += loss_tensor.item() / self.accelerator.num_processes

            if epoch == 0 and step == 0 and self.accelerator.is_main_process:
                self.logger.info(
                    f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB at batch size: {self.train_batch_size}",
                    main_process_only=True
                )
                self.logger.info(
                    f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB",
                    main_process_only=True
                )

        avg_loss = epoch_loss / len(dataloader)
        self.logger.info(f"{mode:<5} Epoch {epoch + 1}, Loss: {avg_loss:.4f}", main_process_only=True)
        return avg_loss

    def evaluate(self, dataloader, epoch, mode='val'):
        if mode not in ['val', 'test']:
            raise ValueError("Mode must be 'val' or 'test'")

        self.UniSignModel.eval()
        if self.use_feature_encoder:
            self.feature_encoder.eval()

        epoch_loss = 0
        bleu_scores = {1: [], 2: [], 3: [], 4: []}
        smoothing = SmoothingFunction().method1

        total_steps = len(dataloader)
        update_steps = 20
        step_interval = max(total_steps // update_steps, 1)

        progress_bar = tqdm(
            iterable=dataloader,
            total=total_steps,
            desc=f"{mode} Epoch {epoch}/{self.max_epochs}",
            disable=not self.accelerator.is_main_process
        )

        self.logger.info(f"Start {mode} Epoch {epoch}", main_process_only=True)

        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                if self.debug and step >= 5:
                    break

                if step % step_interval == 0 or step == total_steps - 1:
                    progress_bar.update(step_interval)

                frames_tensor, text, keypoints_dict = batch
                # Move all tensors to the correct device
                hand_keypoints = keypoints_dict['hand'].to(self.device).float()
                body_keypoints = keypoints_dict['body'].to(self.device).float()
                face_keypoints = keypoints_dict['face'].to(self.device).float()
                right_hand_keypoints = hand_keypoints[:, :, :21, :].float()
                left_hand_keypoints = hand_keypoints[:, :, 21:, :].float()
                if self.use_feature_encoder:
                    frames_tensor = frames_tensor.to(self.device).float()

                decoder_input_ids, targets = self.encode_text(text)
                translated_text, encoder_hidden = self.UniSignModel(
                    left_hand_keypoints,
                    right_hand_keypoints,
                    body_keypoints,
                    face_keypoints,
                    split=mode,
                    decoder_input_ids=decoder_input_ids
                )

                generated_ids = self.tokenizer(
                    translated_text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=128
                )['input_ids'].to(self.device)

                encoder_outputs = (encoder_hidden,)
                outputs = self.UniSignModel.module.llm_trans.model(
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=generated_ids,
                    return_dict=True
                )
                logits = outputs.logits

                loss = self.compute_translation_loss(targets, logits)
                # Reduce loss across GPUs for each batch
                loss_tensor = torch.tensor(loss.item(), device=self.device)
                if dist.is_initialized():
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                epoch_loss += loss_tensor.item() / self.accelerator.num_processes

                for i, pred_text in enumerate(translated_text):
                    ref_text = self.tokenizer.decode(targets[i], skip_special_tokens=True)
                    if step == 0 and i < 2 and self.accelerator.is_main_process:
                        self.logger.info(
                            f"Sample {i} - Predicted: {pred_text[:100]}... | Reference: {ref_text[:100]}...",
                            main_process_only=True
                        )

                    pred_tokens = pred_text.split()
                    ref_tokens = ref_text.split()

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

                if step == 0 and self.accelerator.is_main_process:
                    self.logger.info(
                        f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB at batch size: {self.train_batch_size}",
                        main_process_only=True
                    )
                    self.logger.info(
                        f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB",
                        main_process_only=True
                    )

        local_sample_count = len(bleu_scores[1]) if 1 in bleu_scores and bleu_scores[1] else 0  # 假设 bleu_scores[n] 长度一致
        sample_count_tensor = torch.tensor(local_sample_count, device=self.device, dtype=torch.float32)
        if dist.is_initialized():
            dist.all_reduce(sample_count_tensor, op=dist.ReduceOp.SUM)
        total_samples = sample_count_tensor.item()

        # 聚合 BLEU 分数
        avg_bleu = {}
        for n in range(1, 5):
            # 计算本地 BLEU 分数总和
            local_bleu_sum = np.sum(bleu_scores[n]) if n in bleu_scores and bleu_scores[n] else 0.0
            local_bleu_sum_tensor = torch.tensor(local_bleu_sum, device=self.device, dtype=torch.float32)
            
            # 聚合所有 GPU 的 BLEU 分数总和
            if dist.is_initialized():
                dist.all_reduce(local_bleu_sum_tensor, op=dist.ReduceOp.SUM)
            
            # 计算全局平均 BLEU 分数（加权平均）
            avg_bleu[n] = local_bleu_sum_tensor.item() / total_samples if total_samples > 0 else 0.0

        # 计算平均损失
        avg_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0.0

        # 记录日志
        self.logger.info(
            f"{mode:<5} Epoch {epoch + 1}, Loss: {avg_loss:.4f}, "
            f"BLEU-1: {avg_bleu.get(1, 0.0):.4f}, BLEU-2: {avg_bleu.get(2, 0.0):.4f}, "
            f"BLEU-3: {avg_bleu.get(3, 0.0):.4f}, BLEU-4: {avg_bleu.get(4, 0.0):.4f}",
            main_process_only=True
        )

        # 返回结果
        return {
            'loss': avg_loss,
            'bleu_1': avg_bleu.get(1, 0.0),
            'bleu_2': avg_bleu.get(2, 0.0),
            'bleu_3': avg_bleu.get(3, 0.0),
            'bleu_4': avg_bleu.get(4, 0.0)
        }

    def tokenize_text(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128, add_special_tokens=True)
        return encoded['input_ids'].squeeze().to(self.device)

    def encode_text(self, text: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            batch_texts = [self.tokenize_text(t) for t in text]
        except Exception as e:
            self.logger.error(f"Error tokenizing text: {e}")
            raise

        max_len = max(t.size(0) for t in batch_texts)
        padded_texts = torch.stack([
            torch.cat([t, torch.tensor([self.tokenizer.pad_token_id] * (max_len - t.size(0)), device=self.device)])
            if t.size(0) < max_len else t for t in batch_texts
        ])

        targets = padded_texts[:, 1:]
        decoder_input_ids = padded_texts[:, :-1]
        return decoder_input_ids, targets

    def compute_translation_loss(self, targets: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 3 or logits.shape[-1] != self.tokenizer.vocab_size:
            raise ValueError(f"Unexpected logits shape {logits.shape}. Expected [batch_size, sequence_length, {self.tokenizer.vocab_size}]")

        if logits.size(1) != targets.size(1):
            min_len = min(logits.size(1), targets.size(1))
            logits = logits[:, :min_len, :]
            targets = targets[:, :min_len]

        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)
        loss = criterion(logits_flat, targets_flat)
        return loss

    def run_training_only(self):
        train_loss_history = []
        val_loss_history = []
        loss_curve_filepath = f'{self.log_dir}/training_loss_curve.jpg'

        self.logger.info("Start training...", main_process_only=True)
        for epoch in range(self.start_epoch, self.max_epochs):
            self.logger.info('\n' + '-' * 50, main_process_only=True)
            # Set epoch for DistributedSampler if applicable
            if isinstance(self.train_sampler, DistributedSampler):
                self.train_sampler.set_epoch(epoch)
                self.logger.info(f"Setting epoch for DistributedSampler: {epoch}, rank: {self.accelerator.process_index}")
            else:
                self.logger.warning(f"Train loader sampler is not DistributedSampler, skipping epoch setting, rank: {self.accelerator.process_index}")

            train_loss = self.training(self.train_loader, epoch, 'train')
            train_loss_history.append(train_loss)

            if train_loss < self.best_loss and self.accelerator.is_main_process:
                self.best_loss = train_loss
                ckpt_path = f"{self.ckpt_dir}/{self.weight_name_prefix}_best.pth"
                self.best_ckpt_path = ckpt_path
                self.logger.info(f"Best epoch {epoch}: {self.best_loss:.4f}", main_process_only=True)
            else:
                ckpt_path = f"{self.ckpt_dir}/{self.weight_name_prefix}.pth"
                if self.best_ckpt_path is None:
                    self.best_ckpt_path = ckpt_path

            if self.accelerator.is_main_process:
                self.save_ckpt(epoch, ckpt_path)

                if os.path.exists(loss_curve_filepath):
                    os.remove(loss_curve_filepath)

                num_subplots = 2 if len(val_loss_history) > 0 else 1
                fig, axes = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 4))
                if num_subplots == 1:
                    axes = [axes]

                epochs = range(self.start_epoch, self.start_epoch + len(train_loss_history))
                axes[0].plot(epochs, train_loss_history, label='Train Loss', marker='o', linestyle='-')
                axes[0].set_xlabel("Epochs")
                axes[0].set_ylabel("Loss")
                axes[0].set_title("Train Loss")
                axes[0].legend()
                axes[0].grid(True)

                if num_subplots == 2:
                    val_epochs = range(self.start_epoch, self.start_epoch + len(val_loss_history))
                    axes[1].plot(val_epochs, val_loss_history, label='Validation Loss', marker='o', linestyle='-')
                    axes[1].set_xlabel("Epochs")
                    axes[1].set_ylabel("Loss")
                    axes[1].set_title("Validation Loss")
                    axes[1].legend()
                    axes[1].grid(True)

                plt.tight_layout()
                plt.savefig(loss_curve_filepath)
                plt.close()
                self.logger.info(f"Saving loss curve to: {loss_curve_filepath}\n", main_process_only=True)

            if self.debug and epoch > 2:
                break

        self.logger.info("Done!", main_process_only=True)
        self.cleanup()

    def run_all(self):
        train_loss_history = []
        val_loss_history = []
        bleu_history = {1: [], 2: [], 3: [], 4: []}

        loss_curve_filepath = f'{self.log_dir}/loss_curve.jpg'
        bleu_curve_filepath = f'{self.log_dir}/bleu_scores.jpg'

        self.logger.info("Start training...", main_process_only=True)

        for epoch in range(self.start_epoch, self.max_epochs):
            self.logger.info('\n' + '-' * 50, main_process_only=True)
            
            if isinstance(self.train_sampler, DistributedSampler):
                self.train_sampler.set_epoch(epoch)
                self.logger.info(f"Setting epoch {epoch} for train DistSampler, rank: {self.accelerator.process_index}/{self.world_size}", main_process_only=True)
            else:
                self.logger.warning(f"Train loader sampler is not DistSampler, skipping epoch setting, rank: {self.accelerator.process_index}")

            
            train_loss = self.training(self.train_loader, epoch, 'train')
            train_loss_history.append(train_loss)

            if not self.val_loader is None:
                if isinstance(self.val_sampler, DistributedSampler):
                    self.val_sampler.set_epoch(epoch)
                    self.logger.info(f"Setting epoch {epoch} for Val DistSampler, rank: {self.accelerator.process_index}/{self.world_size}", main_process_only=True)
                else:
                    self.logger.warning(f"Val loader sampler is not DistSampler, skipping epoch setting, rank: {self.accelerator.process_index}", main_process_only=True)
                
                val_metrics = self.evaluate(self.val_loader, epoch, 'val')
                val_loss_history.append(val_metrics['loss'])
                for n in range(1, 5):
                    bleu_history[n].append(val_metrics[f'bleu_{n}'])

            if train_loss < self.best_loss and self.accelerator.is_main_process:
                self.best_loss = train_loss
                ckpt_path = f"{self.ckpt_dir}/{self.weight_name_prefix}_best.pth"
                self.best_ckpt_path = ckpt_path
                self.logger.info(f"Best epoch {epoch}: {self.best_loss:.4f}", main_process_only=True)
            else:
                ckpt_path = f"{self.ckpt_dir}/{self.weight_name_prefix}.pth"
            if self.accelerator.is_main_process:
                self.save_ckpt(epoch, ckpt_path)

            if self.accelerator.is_main_process:
                if os.path.exists(loss_curve_filepath):
                    os.remove(loss_curve_filepath)

                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                epochs = range(self.start_epoch, self.start_epoch + len(train_loss_history))
                ax.plot(epochs, train_loss_history, label='Train Loss', marker='o', linestyle='-')
                if val_loss_history:
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

                if val_loss_history:
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

            if self.debug and epoch > 1:
                break

        if not self.test_loader is None:
            if self.accelerator.is_main_process:
                self.logger.info("Loading best model for testing...", main_process_only=True)
                self.load_ckpt(self.best_ckpt_path)

                

            if isinstance(self.test_sampler, DistributedSampler):
                self.test_sampler.set_epoch(0)
                self.logger.info(f"Setting epoch for Test DistSampler, rank: {self.accelerator.process_index}/{self.world_size}", main_process_only=True)
            else:
                self.logger.warning(f"Test loader sampler is not DistSampler, skipping epoch setting, rank: {self.accelerator.process_index}")
            self.logger.info("Evaluating on test set...", main_process_only=True)

            test_metrics = self.evaluate(self.test_loader, epoch=0, mode='test')

            if self.accelerator.is_main_process:
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
        self.cleanup()

    def save_ckpt(self, epoch, ckpt_path):
        if not self.accelerator.is_main_process:
            return
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        model_dict = {
            'UniSignModel': self.UniSignModel.module.state_dict() if isinstance(self.UniSignModel, DDP) else self.UniSignModel.state_dict(),
            'encoder': self.feature_encoder.state_dict() if self.use_feature_encoder else None,
            'optimizer_uniSign': self.optimizer_UniSignModel.state_dict(),
            'optimizer_encoder': self.optimizer_encoder.state_dict() if self.use_feature_encoder else None,
            'epoch': epoch,
            'best_loss': self.best_loss,
        }
        torch.save(model_dict, ckpt_path)
        self.logger.info(f'Saving model to: {ckpt_path}', main_process_only=True)

    def load_ckpt_after_acceleratorPrepare(self, ckpt_path):
        # --- 1. 主进程加载 checkpoint dict ---
        ckpt_dict = None
        if self.accelerator.is_main_process:
            ckpt_dict = torch.load(ckpt_path, map_location='cpu')
        self.accelerator.wait_for_everyone()

        # --- 2. 所有进程同步 ckpt_dict ---
        ckpt_list = [ckpt_dict] if self.accelerator.is_main_process else [None]
        if dist.is_initialized():
            dist.broadcast_object_list(ckpt_list, src=0)
        ckpt_dict = ckpt_list[0]

        # --- 3. 加载 UniSignModel（DDP 包装时只需主进程加载，但无害于所有进程执行） ---
        if isinstance(self.UniSignModel, DDP):
            missing_keys, unexpected_keys = self.UniSignModel.module.load_state_dict(
                ckpt_dict['UniSignModel'], strict=False)
        else:
            missing_keys, unexpected_keys = self.UniSignModel.load_state_dict(
                ckpt_dict['UniSignModel'], strict=False)

        if missing_keys or unexpected_keys:
            self.logger.warning(f"UniSignModel - Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}", main_process_only=True)

        # --- 4. 所有进程都加载 feature_encoder（非DDP，需要每个进程同步加载） ---
        if self.use_feature_encoder:
            missing_keys, unexpected_keys = self.feature_encoder.load_state_dict(
                ckpt_dict['encoder'], strict=False)
            if missing_keys or unexpected_keys:
                self.logger.warning(f"Feature encoder - Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}", main_process_only=True)

        # --- 5. 构建并广播 optimizer 状态（主进程构建） ---
        if self.accelerator.is_main_process and not self.finetune:
            optimizer_states = {
                'optimizer_uniSign': ckpt_dict['optimizer_uniSign'],
                'optimizer_encoder': ckpt_dict['optimizer_encoder'] if self.use_feature_encoder else None,
                'start_epoch': ckpt_dict['epoch'],
                'best_loss': ckpt_dict.get('best_loss', float('inf'))
            }
        else:
            optimizer_states = None

        opt_list = [optimizer_states] if self.accelerator.is_main_process else [None]
        if dist.is_initialized():
            dist.broadcast_object_list(opt_list, src=0)
        optimizer_states = opt_list[0]


        # --- 6. 所有进程加载 optimizer 状态和 epoch 等元数据 ---
        if not self.finetune:
            self.optimizer_UniSignModel.load_state_dict(optimizer_states['optimizer_uniSign'])
            if self.use_feature_encoder and optimizer_states['optimizer_encoder']:
                self.optimizer_encoder.load_state_dict(optimizer_states['optimizer_encoder'])

        self.start_epoch = optimizer_states['start_epoch']
        self.best_loss = optimizer_states['best_loss']

        self.logger.info(f"Start epoch: {self.start_epoch}", main_process_only=True)
        self.logger.info(f"Best loss: {self.best_loss}", main_process_only=True)

        self.accelerator.wait_for_everyone()


    def cleanup(self):
        """Clean up distributed environment."""
        if dist.is_initialized():
            dist.destroy_process_group()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=arg_settings["debug"], action="store_true", help="Debug mode")
    parser.add_argument("--model_name", type=str, default='uniSign', help="Model name")
    parser.add_argument("--resume", type=str, default=arg_settings["resume"], help="Resume training from a checkpoint")
    parser.add_argument("--dataset_name", type=str, default=arg_settings["dataset_name"], help="Dataset name")
    parser.add_argument("--feature_encoder", type=str, default=arg_settings["feature_encoder"], help="Feature encoder name")
    parser.add_argument("--modality", type=str, default=arg_settings["modality"], help="Modality, e.g., rgb, pose, rgb_pose")
    parser.add_argument("--finetune", default=arg_settings["finetune"], action="store_true", help="Fine-tune the model")
    parser.add_argument("--n_frames", type=int, default=arg_settings["n_frames"], help="Number of frames")
    parser.add_argument("--batch_size", type=int, default=arg_settings["batch_size"], help="Batch size")
    parser.add_argument("--use_feature_encoder", default=arg_settings["use_feature_encoder"], action="store_true", help="Use feature encoder")
    return parser.parse_args()

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    args = get_args()
    uniSignTrans = UniSignTrans(args)
    uniSignTrans.run_all()
    # uniSignTrans.run_training_only()