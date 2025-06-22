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

from transformers import MBartTokenizer
from transformers import AutoTokenizer



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

class EvalUniSignTrans:
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
        self.eval_batch_size = args.eval_batch_size

        time_stamp = kwargs.get('time_stamp', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        slurm_job_id = kwargs.get('slurm_job_id', os.environ.get('SLURM_JOB_ID', '0'))

        
        assert self.modality in ['pose', 'rgb', 'pose_rgb'], f"Unsupported modality: {self.modality}"
        
        if self.modality == 'pose':
            self.use_feature_encoder = False
        else:
            self.use_feature_encoder = True 

        # Initialize distributed environment
        self.init_distributed()

        last_second_dir = self.resume_path.split('/')[-2] if self.resume_path else ''
        last_third_dir = self.resume_path.split('/')[-3] if self.resume_path else ''
        
        parent_eval_log_dir = os.path.join('zlog', last_third_dir, last_second_dir)
        if os.path.exists(parent_eval_log_dir):
            log_dir = os.path.join(parent_eval_log_dir, f'Eval_{time_stamp}_JID-{slurm_job_id}')
        else:
            log_dir = os.path.join('zlog', f'Eval_{time_stamp}_JID-{slurm_job_id}')
        self.log_dir = log_dir

        if self.accelerator.is_main_process:
            os.makedirs(log_dir, exist_ok=True)
            shutil.copy(__file__, os.path.join(log_dir, f"{__file__.split('/')[-1].split('.')[0]}_{time_stamp}.py"))
            config_filename = os.path.join(log_dir, f"config_{time_stamp}.py")
            shutil.copy("config/config.py", config_filename)

        # Synchronize all processes to ensure directory exists
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

        self.logger.info(f"\n{os.path.basename(__file__)}\n", main_process_only=self.accelerator.is_main_process)
        self.logger.info(f'logging_dir: {self.log_dir}', main_process_only=self.accelerator.is_main_process)

        self.best_loss = float('inf')
        self.logger.info(f"Model name: {self.model_name}", main_process_only=self.accelerator.is_main_process)
        self.logger.info(f"Resume path: {self.resume_path}", main_process_only=self.accelerator.is_main_process)
        self.logger.info(f"Finetune: {self.finetune}", main_process_only=self.accelerator.is_main_process)
        self.logger.info(f"Modality: {self.modality}", main_process_only=self.accelerator.is_main_process)
        self.logger.info(f"Number of frames: {self.n_frames}", main_process_only=self.accelerator.is_main_process)
        self.logger.info(f"Batch size: {self.eval_batch_size}", main_process_only=self.accelerator.is_main_process)
        
        self.logger.info(f"feature encoder: {self.feature_encoder_name}", main_process_only=self.accelerator.is_main_process)

        self.logger.info(f"Use use_feature_encoder: {self.use_feature_encoder}", main_process_only=self.accelerator.is_main_process)

        self.device = self.accelerator.device
        self.logger.info(f"Using device: {self.device}", main_process_only=self.accelerator.is_main_process)

        self.logger.info(f"Dataset name: {self.dataset_name}", main_process_only=self.accelerator.is_main_process)

        if self.debug:
            self.eval_batch_size = 5
        self.logger.info(f"Eval batch size: {self.eval_batch_size}", main_process_only=self.accelerator.is_main_process)

        self.logger.info("Dataloader configuration.", main_process_only=self.accelerator.is_main_process)

        distributed = True if self.accelerator.num_processes  > 1 else False
        self.distributed = distributed
        self.logger.info(f"Distributed: {distributed}", main_process_only=self.accelerator.is_main_process)
        val_loader, val_dataset, self.val_sampler = None, None, None
        # val_loader, val_dataset, self.val_sampler = get_dataloader(
        #     dataset_name=self.dataset_name,
        #     logger=self.logger,
        #     debug=self.debug,
        #     batch_size=self.eval_batch_size,
        #     modality=self.modality,
        #     n_frames=self.n_frames,
        #     distributed=distributed,
        #     world_size=self.accelerator.num_processes,
        #     rank=self.accelerator.process_index,
        #     split = 'val',
        # )
        print('self.dataset_name', self.dataset_name)
        test_loader, test_dataset, self.test_sampler = get_dataloader(
            dataset_name=self.dataset_name,
            logger=self.logger,
            debug=self.debug,
            batch_size=self.eval_batch_size,
            modality=self.modality,
            n_frames=self.n_frames,
            distributed=distributed,
            world_size=self.accelerator.num_processes,
            rank=self.accelerator.process_index,
            split = 'test',
        )
        

        if not val_loader is None and isinstance(val_loader.sampler, DistributedSampler):
            self.logger.info("Val loader correctly uses DistributedSampler", main_process_only=self.accelerator.is_main_process)
        else:
            self.logger.warning("Val loader does NOT use DistributedSampler", main_process_only=self.accelerator.is_main_process)
        
        if not test_loader is None and isinstance(test_loader.sampler, DistributedSampler):
            self.logger.info("Test loader correctly uses DistributedSampler", main_process_only=self.accelerator.is_main_process)
        else:
            self.logger.warning("Test loader does NOT use DistributedSampler", main_process_only=self.accelerator.is_main_process)

        self.val_loader = self.accelerator.prepare(val_loader) if val_loader else None
        self.test_loader = self.accelerator.prepare(test_loader) if test_loader else None
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.logger.info(f"Val dataset size: {len(self.val_dataset) if self.val_dataset else 0}", main_process_only=self.accelerator.is_main_process)
        self.logger.info(f"Test dataset size: {len(self.test_dataset) if self.test_dataset else 0}", main_process_only=self.accelerator.is_main_process)
        

        
        if self.use_feature_encoder:
            self.feature_encoder, encoder_output_size = get_encoder(self.feature_encoder_name, self.device)
            
        else:
            encoder_output_size = 0
            self.feature_encoder = None
        # Log sampler type after accelerator.prepare for debugging


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

        self.tokenizer = AutoTokenizer.from_pretrained(
                "facebook/mbart-large-50",
                src_lang="en_XX",
                tgt_lang="en_XX"
            )

        self.logger.info("Tokenizer loaded.", main_process_only=self.accelerator.is_main_process)

        self.UniSignModel = UniSignNetwork(hidden_dim=256, LLM_name="facebook/mbart-large-50", device=self.device, tokenizer=self.tokenizer)
        self.UniSignModel.float().to(self.device)
        # Wrap model with DDP
        self.UniSignModel = self.accelerator.prepare(self.UniSignModel)

        # self.decoder_start_token_id = self.tokenizer.lang_code_to_id["en_XX"]


        if self.use_feature_encoder:
            self.feature_encoder.to(self.device)
            self.accelerator.prepare(self.feature_encoder)

        self.load_ckpt_after_acceleratorPrepare(self.resume_path)
        self.logger.info(f"Resuming from checkpoint: {self.resume_path}", main_process_only=self.accelerator.is_main_process)
    
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.logger.info(f"[AFTER PREPARE] Distributed training initialized: {dist.is_initialized()}", main_process_only=self.accelerator.is_main_process)
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
        print(f"World size: {self.accelerator.num_processes}, Rank: {self.accelerator.process_index}")



    def evaluate(self, dataloader, mode='val'):
        
        if mode not in ['val', 'test']:
            raise ValueError("Mode must be 'val' or 'test'")

        if mode == 'val':
            sampler = self.val_sampler
        elif mode == 'test':
            sampler = self.test_sampler

        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(0)
            self.logger.info(
                f"Setting epoch for {mode} DistributedSampler: {0}, rank: {self.accelerator.process_index}",
                main_process_only=self.accelerator.is_main_process
            )
        else:
            self.logger.warning(
                f"{mode.capitalize()} loader sampler is not DistributedSampler, skipping epoch setting.",
                main_process_only=self.accelerator.is_main_process
            )



        self.UniSignModel.eval()
        if self.use_feature_encoder:
            self.feature_encoder.eval()

        bleu_scores = {1: [], 2: [], 3: [], 4: []}
        smoothing = SmoothingFunction().method1

        total_steps = len(dataloader)
        update_steps = 20
        step_interval = max(total_steps // update_steps, 1)

        progress_bar = tqdm(
            iterable=dataloader,
            total=total_steps,
            disable=not self.accelerator.is_main_process
        )


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
                # print(f"translated_text: {translated_text}")

                generated_ids = self.tokenizer(
                    translated_text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=128
                )['input_ids'].to(self.device)

                # print(f"generated_ids: {generated_ids}")
                # print(f"decoded: {self.tokenizer.batch_decode(generated_ids)}")


                encoder_outputs = (encoder_hidden,)
                if self.distributed:
                    outputs = self.UniSignModel.module.llm_trans.model(
                        encoder_outputs=encoder_outputs,
                        decoder_input_ids=generated_ids,
                        return_dict=True
                    )
                else:
                    outputs = self.UniSignModel.llm_trans.model(
                        encoder_outputs=encoder_outputs,
                        decoder_input_ids=generated_ids,
                        return_dict=True
                    )
                    
                logits = outputs.logits


                for i, pred_text in enumerate(translated_text):
                    ref_text = self.tokenizer.decode(targets[i], skip_special_tokens=True)
                    if step == 0 and i <= 3 and self.accelerator.is_main_process:
                        self.logger.info(
                            f"Sample {i} - Predicted: {pred_text[:100]}... | Reference: {ref_text[:100]}...",
                            main_process_only=self.accelerator.is_main_process
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
                        f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB at batch size: {self.eval_batch_size}",
                        main_process_only=self.accelerator.is_main_process
                    )
                    self.logger.info(
                        f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB",
                        main_process_only=self.accelerator.is_main_process
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


        # 记录日志
        self.logger.info(
            f"BLEU-1: {avg_bleu.get(1, 0.0):.4f}, BLEU-2: {avg_bleu.get(2, 0.0):.4f}, "
            f"BLEU-3: {avg_bleu.get(3, 0.0):.4f}, BLEU-4: {avg_bleu.get(4, 0.0):.4f}",
            main_process_only=self.accelerator.is_main_process
        )

        # 返回结果
        return {
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


    def run_eval(self, split = 'test'):
        self.logger.info(f"Start {split}...", main_process_only=self.accelerator.is_main_process)

        if isinstance(self.test_sampler, DistributedSampler):
            self.test_sampler.set_epoch(0)
            self.logger.info(f"Setting epoch for Test DistSampler, rank: {self.accelerator.process_index}/{self.world_size}", main_process_only=self.accelerator.is_main_process)
        else:
            self.logger.warning(f"Test loader sampler is not DistSampler, skipping epoch setting, rank: {self.accelerator.process_index}")
        self.logger.info("Evaluating on test set...", main_process_only=self.accelerator.is_main_process)

        test_metrics = self.evaluate(self.test_loader, mode='test')

        if self.accelerator.is_main_process:
            test_results_filepath = f'{self.log_dir}/test_results.txt'
            with open(test_results_filepath, 'w') as f:
                f.write(f"Test Results:\n")
                f.write(f"BLEU-1: {test_metrics['bleu_1']:.4f}\n")
                f.write(f"BLEU-2: {test_metrics['bleu_2']:.4f}\n")
                f.write(f"BLEU-3: {test_metrics['bleu_3']:.4f}\n")
                f.write(f"BLEU-4: {test_metrics['bleu_4']:.4f}\n")
        self.logger.info(f"Test results saved to: {test_results_filepath}", main_process_only=self.accelerator.is_main_process)
        self.logger.info(
            f"BLEU-1: {test_metrics['bleu_1']:.4f}, "
            f"BLEU-2: {test_metrics['bleu_2']:.4f}, "
            f"BLEU-3: {test_metrics['bleu_3']:.4f}, "
            f"BLEU-4: {test_metrics['bleu_4']:.4f}",
            main_process_only=self.accelerator.is_main_process
        )

        self.logger.info("Done!", main_process_only=self.accelerator.is_main_process)
        self.cleanup()

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
            self.logger.warning(f"UniSignModel - Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}", main_process_only=self.accelerator.is_main_process)

        # --- 4. 所有进程都加载 feature_encoder（非DDP，需要每个进程同步加载） ---
        if self.use_feature_encoder:
            missing_keys, unexpected_keys = self.feature_encoder.load_state_dict(
                ckpt_dict['encoder'], strict=False)
            if missing_keys or unexpected_keys:
                self.logger.warning(f"Feature encoder - Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}", main_process_only=self.accelerator.is_main_process)

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
    parser.add_argument("--eval_batch_size", type=int, default=arg_settings["eval_batch_size"], help="Batch size")
    parser.add_argument("--eval_log_dir", type=str, default=arg_settings["eval_log_dir"], help="Directory to save logs")
    return parser.parse_args()

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    args = get_args()
    uniSignTrans = EvalUniSignTrans(args)
    uniSignTrans.run_eval(split='test')
    # uniSignTrans.run_training_only()