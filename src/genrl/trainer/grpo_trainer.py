import gc
import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from genrl.data import DataManager
from genrl.logging_utils.ml_logger import LoggerMixin
from genrl.rewards import RewardManager
from genrl.state import GameState
from genrl.trainer import TrainerModule
from genrl.trainer.trainer_utils import DTYPE_MAP

from genrl.logging_utils.global_defs import get_logger


def create_reference_model(
    model: torch.nn.Module
) -> torch.nn.Module:
    ref_model = deepcopy(model)
    for param in ref_model.parameters():
        param.requires_grad = False
    return ref_model.eval()

@dataclass
class GRPOTrainerConfig:
    epsilon: float = 0.2
    epsilon_high: float = 0.28
    beta: float = 0.0
    temperature: float = 1.0
    dtype: str = "float32"
    enable_gradient_checkpointing: bool = True
    max_new_tokens: int = 256
    num_generations: int = 2
    learning_rate: float = 1e-6
    top_p: float = 1.0
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float = 1.0
    ppo_epochs: int = 1
    minibatch_size: int = 2


class GRPOLanguageTrainerModule(TrainerModule, LoggerMixin):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method.
    Modified to support BitsAndBytes 8-bit quantization.
    """

    def __init__(self, models: List[Any], config: GRPOTrainerConfig, **kwargs):
        if not models or len(models) < 1:
            raise ValueError("At least one model must be provided")

        self.model = models[0]
        self.args = config

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate
        )

        # Tokenizers
        self.processing_class = kwargs.get("processing_class", None)

        # Additional parameters
        self.callbacks = kwargs.get("callbacks", [])
        self.save_dir = kwargs.get("log_dir", "./outputs")
        self.global_step = 0
        assert (
            self.args.num_generations > 1
        ), f"For GRPO training, number of generations must be > 1, got {self.args.num_generations}"
        
        # Detect quantization TRƯỚC khi set dtype
        self.is_quantized = self._detect_quantization()
        
        # Chỉ set dtype nếu KHÔNG phải quantized model
        if self.is_quantized:
            self.dtype = None  # Không cast dtype cho quantized models
            self.compute_dtype = torch.float32  # Dùng float32 cho computation
            print("🔧 Detected 8-bit quantized model - using float32 for computations")
        else:
            self.dtype = DTYPE_MAP.get(self.args.dtype, None)
            self.compute_dtype = self.dtype if self.dtype is not None else torch.float32
        
        self.enable_gradient_checkpointing = self.args.enable_gradient_checkpointing

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Initialize core components
        self._initialize_model(self.enable_gradient_checkpointing)
        self._initialize_tokenizers()
        self._initialize_metrics()
        self._initialize_generation_config()
        self.init_tracker(self.save_dir, log_with=kwargs.get("log_with", None))

    def _detect_quantization(self):
        """Detect if model is quantized"""
        model_type_str = str(type(self.model)).lower()
        is_quant = (
            hasattr(self.model, "is_quantized")
            or hasattr(self.model, "is_loaded_in_4bit")
            or hasattr(self.model, "is_loaded_in_8bit")
            or "bitsandbytes" in model_type_str
            or "bnb" in model_type_str
        )
        
        # Extra check: xem model có quantization_config không
        if hasattr(self.model, "config") and hasattr(self.model.config, "quantization_config"):
            is_quant = True
            
        return is_quant

    def _initialize_model(self, enable_gradient_checkpointing):
        """Initialize the model and reference model."""
        if self.is_quantized:
            # Quantized model - chỉ move device, KHÔNG cast dtype
            print(f"✓ Model is quantized, keeping original precision")
            try:
                self.model = self.model.to(device=self.device)
            except Exception as e:
                # device_map có thể đã được set
                print(f"  Note: Model already on device ({e})")
                pass
            
            # Disable use_cache cho quantized models (QUAN TRỌNG!)
            if hasattr(self.model, "config"):
                try:
                    self.model.config.use_cache = False
                    print("✓ Disabled use_cache for stable training")
                except:
                    pass
            
            # Gradient checkpointing
            if enable_gradient_checkpointing:
                if hasattr(self.model, "gradient_checkpointing_enable"):
                    try:
                        self.model.gradient_checkpointing_enable()
                        print("✓ Enabled gradient checkpointing")
                    except Exception as e:
                        print(f"⚠ Could not enable gradient checkpointing: {e}")
        else:
            # Non-quantized model - cast dtype bình thường
            print(f"✓ Non-quantized model, using dtype: {self.dtype}")
            if self.dtype is not None:
                self.model = self.model.to(device=self.device, dtype=self.dtype)
            else:
                self.model = self.model.to(device=self.device)
            
            if enable_gradient_checkpointing:
                try:
                    self.model.gradient_checkpointing_enable()
                except Exception:
                    pass

        # Reference model setup
        if self.args.beta == 0.0:
            self.ref_model = None
            print("✓ No reference model (beta=0)")
        else:
            try:
                print("✓ Creating reference model...")
                self.ref_model = create_reference_model(self.model).to(device=self.device)
            except Exception as e:
                print(f"⚠ Could not create reference model: {e}")
                self.ref_model = None

    def _initialize_tokenizers(self):
        """Initialize tokenizers for the model and reward models."""
        if self.processing_class is None:
            try:
                model_name = getattr(self.model.config, "_name_or_path", None)
                if model_name:
                    self.processing_class = AutoTokenizer.from_pretrained(
                        model_name, padding_side="left"
                    )
                    print(f"✓ Loaded tokenizer from {model_name}")
            except Exception as e:
                print(f"⚠ Could not load tokenizer: {e}")
                self.processing_class = None

    def _initialize_metrics(self):
        """Initialize metrics tracking for training and evaluation."""
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

    def _initialize_generation_config(self):
        """Set generation config - safer defaults for quantized models"""
        # Cho quantized models, dùng greedy decoding để tránh numerical instability
        if self.is_quantized:
            base_do_sample = False
            temp = 1.0
            top_p = 1.0
            top_k = 1
            print("✓ Using greedy decoding for quantized model")
        else:
            base_do_sample = bool(self.args.temperature and self.args.temperature > 0.0)
            temp = self.args.temperature
            top_p = self.args.top_p
            top_k = self.args.top_k if self.args.top_k is not None else 50
        
        try:
            pad_id = getattr(self.processing_class, "pad_token_id", None)
            bos_id = getattr(self.processing_class, "bos_token_id", None)
            eos_id = getattr(self.processing_class, "eos_token_id", None)
        except Exception:
            pad_id = bos_id = eos_id = None

        self.generation_config = GenerationConfig(
            max_new_tokens=self.args.max_new_tokens,
            do_sample=base_do_sample,
            pad_token_id=pad_id,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            min_p=self.args.min_p,
            repetition_penalty=self.args.repetition_penalty,
        )

    def _process_inputs(self, inputs, with_template=True, for_training=False):
        if hasattr(inputs, "to_dict"):
            inputs = [dict(inputs[i]) for i in range(len(inputs))]
        elif isinstance(inputs, dict):
            inputs = [inputs]

        if with_template:
            if for_training:
                templated_prompts = []
                for item in inputs:
                    for _ in range(self.args.num_generations):
                        templated_prompts.append(
                            self.processing_class.apply_chat_template(
                                item["prompt"], tokenize=False, add_generation_prompt=True
                            )
                        )
            else:
                templated_prompts = [
                    self.processing_class.apply_chat_template(
                        item["prompt"], tokenize=False, add_generation_prompt=True
                    )
                    for item in inputs
                ]
        else:
            if for_training:
                templated_prompts = []
                for generations in inputs:
                    for output in generations:
                        templated_prompts.append(output)
            else:
                templated_prompts = [item[0] for item in inputs]

        input_tokens = self.processing_class(
            text=templated_prompts, return_tensors="pt", padding=True, truncation=True
        )
        return input_tokens

    def generate(
        self, inputs: Any, return_completion_ids: bool = False, stage=0
    ) -> Any:
        """Generate outputs from the model for the given inputs."""
        input_tokens = self._process_inputs(inputs)
        rollout, rollout_ids = [], []
        
        # Prepare generation kwargs - safer for quantized models
        if self.is_quantized:
            gen_kwargs = {
                "do_sample": False,
                "num_beams": 1,
                "use_cache": False,
                "top_k": 1,
                "top_p": 1.0,
                "temperature": 1.0,
                "max_new_tokens": self.args.max_new_tokens,
            }
        else:
            gen_kwargs = {
                "generation_config": self.generation_config,
            }
        
        # Add pad/eos tokens
        pad_id = getattr(self.processing_class, "pad_token_id", None)
        eos_id = getattr(self.processing_class, "eos_token_id", None)
        if pad_id is not None:
            gen_kwargs["pad_token_id"] = pad_id
        if eos_id is not None:
            gen_kwargs["eos_token_id"] = eos_id
        
        input_ids = input_tokens.input_ids.to(self.model.device)
        # CRITICAL: attention_mask MUST always be torch.long
        attention_mask = input_tokens.attention_mask.to(self.model.device, dtype=torch.long)
        prompt_length = input_ids.size(1)
        
        for gen_idx in range(self.args.num_generations):
            with torch.no_grad():
                # Cho quantized models với batch > 1, generate per-sample để stability
                if self.is_quantized and input_ids.size(0) > 1:
                    outputs_list = []
                    for i in range(input_ids.size(0)):
                        ids = input_ids[i:i+1]
                        mask = attention_mask[i:i+1]
                        try:
                            out = self.model.generate(ids, attention_mask=mask, **gen_kwargs)
                            outputs_list.append(out)
                        except Exception as e:
                            print(f"⚠ Generation error for sample {i}: {e}")
                            # Fallback: return input + padding
                            out = torch.cat([ids, torch.full((1, 10), pad_id or 0, device=ids.device)], dim=1)
                            outputs_list.append(out)
                    outputs = torch.cat(outputs_list, dim=0)
                else:
                    try:
                        outputs = self.model.generate(
                            input_ids, attention_mask=attention_mask, **gen_kwargs
                        )
                    except Exception as e:
                        print(f"⚠ Generation error: {e}")
                        # Fallback
                        outputs = torch.cat([input_ids, torch.full((input_ids.size(0), 10), pad_id or 0, device=input_ids.device)], dim=1)

            # Extract completions
            completion_ids = outputs[:, prompt_length:]
            completions = self.processing_class.batch_decode(
                completion_ids, skip_special_tokens=True
            )

            if len(rollout) == 0:
                rollout = [[comp] for comp in completions]
                if return_completion_ids:
                    rollout_ids = [[comp] for comp in completion_ids]
            else:
                for idx, comp in enumerate(completions):
                    rollout[idx].append(comp)
                    if return_completion_ids:
                        rollout_ids[idx].append(completion_ids[idx])
        
        if return_completion_ids:
            return rollout, rollout_ids
        else:
            return rollout

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        """
        Get the per-token log probabilities for the input tokens.
        CRITICAL FIX: Use float32 for loss_mask to avoid dtype issues with quantized models.
        """
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep + 1,
        ).logits
        logits = logits[:, :-1, :]

        # CRITICAL FIX: loss_mask MUST be float32 for quantized models
        # Không dùng logits.dtype vì có thể mixed precision
        loss_mask = (
            attention_mask[:, -logits_to_keep:]
            .to(device=logits.device, dtype=torch.float32)  # ← LUÔN dùng float32
            .contiguous()
        )
        labels = input_ids[:, -logits_to_keep:].contiguous()
        logits = logits[:, -logits_to_keep:].contiguous()

        # Divide logits by temperature (with safety check)
        temp = max(1e-8, float(self.args.temperature))
        logits = logits / temp
        
        logits_shape = logits.shape
        
        # Compute cross entropy in float32 for numerical stability
        logits_for_ce = logits.float() if logits.dtype != torch.float32 else logits
        
        token_log_probs = -torch.nn.functional.cross_entropy(
            logits_for_ce.view(-1, logits_shape[-1]),
            labels.view(-1),
            reduction="none",
        ).view(logits_shape[0], logits_shape[1])
        
        # Apply mask (both in float32)
        token_log_probs = token_log_probs.float()
        token_log_probs = (
            token_log_probs * loss_mask
            + (1.0 - loss_mask) * torch.finfo(torch.float32).min
        )
        
        return token_log_probs

    def compute_loss(
        self, model, inputs, mode="train", return_metrics=False
    ):
        """Compute the GRPO loss."""
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(self.model.device)
        # CRITICAL: attention_mask MUST always be torch.long
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1).to(
            self.model.device, dtype=torch.long
        )
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep
        )

        # Compute KL divergence if beta > 0
        if self.args.beta != 0.0:
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, input_ids, attention_mask, logits_to_keep
                )
            else:
                ref_per_token_logps = per_token_logps.clone()

            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps", None)
        if old_per_token_logps is None:
            old_per_token_logps = per_token_logps.detach()

        # Calculate ratios and loss terms (all in float32)
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(
            coef_1,
            1 - self.args.epsilon,
            1 + (self.args.epsilon_high if self.args.epsilon_high is not None else self.args.epsilon),
        )
        advantages = advantages.unsqueeze(dim=-1)
        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        per_token_loss = torch.clamp(per_token_loss, -10.0, 10.0)

        # Add KL penalty if beta > 0
        if self.args.beta != 0.0:
            per_token_loss = per_token_loss + self.args.beta * per_token_kl

        # Convert completion_mask to float32 for computation
        completion_mask_float = completion_mask.float()
        loss = (per_token_loss * completion_mask_float).sum() / (completion_mask_float.sum() + 1e-8)

        # Check for invalid loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠ WARNING: Invalid loss detected (NaN or Inf), setting to 0")
            loss = torch.tensor(0.0, device=loss.device, requires_grad=True)

        if self.args.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask_float).sum() / (completion_mask_float.sum() + 1e-8)
            self._metrics[mode]["kl"].append(mean_kl.item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask_float).sum() / (completion_mask_float.sum() + 1e-8)
        self._metrics[mode]["clip_ratio"].append(clip_ratio.item())
        self._metrics[mode]["loss"].append(loss.item())

        metrics = {
            "loss": loss.item(),
            "kl": mean_kl.item() if self.args.beta != 0.0 else None,
            "clip_ratio": clip_ratio.item(),
        }

        if return_metrics:
            return loss, metrics
        else:
            return loss

    def train(
        self, state: GameState, data_manager: DataManager, reward_manager: RewardManager
    ) -> None:
        """Train the model using the given game state and reward manager."""
        self.model.train()
        global_step = self.global_step
        for stage in range(state.stage):
            global_step = self.step(
                stage, state, data_manager, reward_manager, global_step
            )
        self.global_step = global_step
        self.model.eval()

    def step(
        self,
        stage: int,
        state: GameState,
        data_manager: DataManager,
        reward_manager: RewardManager,
        global_step: int,
    ) -> int:
        global_step += 1

        # Prepare stage's inputs
        stage_inputs = state.get_stage_state(stage)
        stage_inputs, index_mapping = data_manager.prepare_input(stage_inputs, stage)
        assert stage_inputs is not None, f"No inputs found for stage {stage}"
        
        # Unflatten stage's outputs
        stage_actions = state.get_stage_actions(stage)
        stage_outputs = [
            stage_actions[index_mapping[idx][0]][index_mapping[idx][1]][
                index_mapping[idx][2]
            ]
            for idx, _ in enumerate(index_mapping)
        ]
        assert stage_outputs is not None, f"No outputs found for stage {stage}"

        metrics = {}

        model_inputs = {}
        processed_inputs = self._process_inputs(stage_inputs, for_training=True)
        model_inputs["prompt_ids"] = processed_inputs.input_ids.to(self.model.device)
        # CRITICAL: attention_mask MUST always be torch.long
        model_inputs["prompt_mask"] = processed_inputs.attention_mask.to(
            self.model.device, dtype=torch.long
        )
        
        processed_outputs = self._process_inputs(
            stage_outputs, with_template=False, for_training=True
        )
        model_inputs["completion_ids"] = processed_outputs.input_ids.to(self.model.device)
        # CRITICAL: attention_mask MUST always be torch.long
        model_inputs["completion_mask"] = processed_outputs.attention_mask.to(
            self.model.device, dtype=torch.long
        )

        rewards = reward_manager[stage]
        rewards = [
            rewards[index_mapping[idx][0]][index_mapping[idx][1]][index_mapping[idx][2]]
            for idx, _ in enumerate(index_mapping)
        ]

        assert rewards is not None, f"No rewards found for stage {stage}"
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        # Validate rewards
        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            print(f"⚠ WARNING: Invalid rewards at stage {stage}, cleaning...")
            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
        
        # DEBUG: Print reward stats
        print(f"📊 Stage {stage} | Rewards - mean: {rewards.mean():.4f}, std: {rewards.std():.4f}, min: {rewards.min():.4f}, max: {rewards.max():.4f}")
        
        do_training = (rewards != 0).any()
        
        if do_training:
            with torch.no_grad():
                advantages = rewards - rewards.mean(dim=1, keepdim=True)
                if rewards.shape[1] > 1:
                    std = rewards.std(dim=1, keepdim=True)
                    advantages /= (std + 1e-8)
                
                # DEBUG: Print advantage stats
                print(f"📊 Advantages - mean: {advantages.mean():.4f}, std: {advantages.std():.4f}")

                prompt_ids, prompt_mask = model_inputs["prompt_ids"], model_inputs["prompt_mask"]
                completion_ids, completion_mask = model_inputs["completion_ids"], model_inputs["completion_mask"]
                input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(self.model.device)
                # CRITICAL: attention_mask MUST always be torch.long
                attention_mask = torch.cat([prompt_mask, completion_mask], dim=1).to(
                    self.model.device, dtype=torch.long
                )
                logits_to_keep = completion_ids.size(1)
                old_per_token_logps_full = self._get_per_token_logps(
                    self.model, input_ids, attention_mask, logits_to_keep
                ).detach()

            # CRITICAL: advantages MUST be float32
            advantages = torch.flatten(advantages).to(self.model.device, dtype=torch.float32)
            model_inputs["advantages"] = advantages.squeeze(dim=-1)
            
            num_samples = model_inputs["completion_ids"].size(0)
            updates_per_rollout = self.args.ppo_epochs
            minibatch_size = min(self.args.minibatch_size, num_samples)
            
            loss_vals = []
            for epoch in range(updates_per_rollout):
                perm = torch.randperm(num_samples, device=self.model.device)
                for start in range(0, num_samples, minibatch_size):
                    idx = perm[start : start + minibatch_size]
                    mb_inputs = self._return_minibatch(model_inputs, idx, old_per_token_logps_full)
                    self.model.zero_grad()
                    loss = self.compute_loss(self.model, mb_inputs)
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"⚠ Skipping backward due to invalid loss")
                        continue
                        
                    loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    loss_vals.append(loss.detach().float())

            mean_loss = torch.stack(loss_vals).mean() if loss_vals else torch.tensor(0.0)
            print(f"✓ Training completed | Loss: {mean_loss:.4f}")
        else:
            mean_loss = torch.tensor(0.0)
            print(f"⊘ No training (all rewards are zero)")

        metrics.update({"train/loss": mean_loss.cpu().item()})
        metrics.update({"train/rewards": rewards.cpu().mean().item()})
        self.log(metrics, global_step)

        self.cleanup_step()

        return global_step

    def _return_minibatch(self, model_inputs, idx, old_per_token_logps_full):
        mb_inputs = {
            "prompt_ids": model_inputs["prompt_ids"][idx],
            "prompt_mask": model_inputs["prompt_mask"][idx],
            "completion_ids": model_inputs["completion_ids"][idx],
            "completion_mask": model_inputs["completion_mask"][idx],
            "advantages": model_inputs["advantages"][idx],
            "old_per_token_logps": old_per_token_logps_full[idx],
        }
        return mb_inputs

    @torch.no_grad()
    def evaluate(self, state: GameState, data_manager: DataManager, reward_manager: RewardManager):
        pass

    def save(self, save_dir: str) -> None:
        """Save the model and trainer state to the given directory."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model using HuggingFace method
        try:
            self.model.save_pretrained(save_dir)
            print(f"✓ Model saved to {save_dir}")
        except Exception as e:
            print(f"⚠ Could not use save_pretrained, using torch.save: {e}")
            torch.save(self.model.state_dict(), os.path.join(save_dir, "model_state.pt"))

        # Save additional state
        torch.save(
            {
                "metrics": self._metrics,
                "total_train_tokens": self._total_train_tokens,
                "generation_config": self.generation_config,
            },
            os.path.join(save_dir, "trainer_state.pt"),
        )

    @classmethod
    def load(cls, load_dir: str) -> "GRPOLanguageTrainerModule":
        """Load a trainer module from the given directory."""
        model = AutoModelForCausalLM.from_pretrained(load_dir)
        trainer = cls([model])
        
        trainer_state = torch.load(os.path.join(load_dir, "trainer_state.pt"))
        trainer._metrics = trainer_state["metrics"]
        trainer._total_train_tokens = trainer_state["total_train_tokens"]
        trainer.generation_config = trainer_state["generation_config"]

        return trainer

    def cleanup_step(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    def cleanup(self):
        self.cleanup_trackers()
