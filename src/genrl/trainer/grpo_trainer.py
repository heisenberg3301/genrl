# grpo_trainer_lora.py
# Trainer module adapted from genrl GRPO trainer to support QLoRA (4-bit) via PEFT.
# Drop-in replacement: paste this file into your project and point your Hydra `_target_` to
# `path.to.grpo_trainer_lora.GRPOQLoRATrainerModule` in config.

import gc
import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# PEFT / BitsAndBytes helpers
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

from genrl.data import DataManager
from genrl.logging_utils.ml_logger import LoggerMixin
from genrl.rewards import RewardManager
from genrl.state import GameState
from genrl.trainer import TrainerModule
from genrl.trainer.trainer_utils import DTYPE_MAP

from genrl.logging_utils.global_defs import get_logger


def create_reference_model(model: torch.nn.Module) -> torch.nn.Module:
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
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: float = 1.0
    ppo_epochs: int = 1
    minibatch_size: int = 2

    # LoRA/QLoRA specific
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None

    # k-bit options
    load_in_4bit: bool = False
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"


class GRPOLanguageTrainerModule(TrainerModule, LoggerMixin):
    """
    GRPO trainer that supports training LoRA adapters (QLoRA) on 4-bit quantized models.

    - If config.load_in_4bit is True and PEFT is available, this class will prepare
      the model for k-bit training and wrap it with a LoRA adapter.
    - It keeps masks as torch.long and computes token logprobs in float32 for stability.
    - The training loop is the same GRPO logic but optimizer only updates adapter params.

    NOTE: Do NOT attempt to train on 8-bit int8 (bitsandbytes) without adapters. 8-bit
    models are inference-only for full-weight training.
    """

    def __init__(self, models: List[Any], config: GRPOTrainerConfig, **kwargs):
        if not models or len(models) < 1:
            raise ValueError("At least one model must be provided")

        self.model = models[0]
        self.args = config

        # tokenizers / processing class
        self.processing_class = kwargs.get("processing_class", None)

        # metrics / bookkeeping
        self.callbacks = kwargs.get("callbacks", [])
        self.save_dir = kwargs.get("log_dir", "./outputs")
        self.global_step = 0

        assert (
            self.args.num_generations > 1
        ), f"For GRPO training, number of generations must be > 1, got {self.args.num_generations}"

        # dtype mapping for floats (advantages, etc.)
        self.dtype = DTYPE_MAP.get(self.args.dtype, None)

        # k-bit / LoRA decisions
        self.is_kbit = bool(self.args.load_in_4bit)
        self.use_lora = bool(self.args.use_lora)

        if self.is_kbit and not PEFT_AVAILABLE and self.use_lora:
            raise RuntimeError("PEFT not available in environment; install `peft` to train QLoRA LoRA adapters.")

        # device selection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # initialize model/tokenizer/metrics/generation config
        self._initialize_model()
        self._initialize_tokenizers()
        self._initialize_metrics()
        self._initialize_generation_config()
        self.init_tracker(self.save_dir, log_with=kwargs.get("log_with", None))

    # ---------------------------
    # Model init + LoRA prep
    # ---------------------------
    def _initialize_model(self):
        # If user requests 4-bit QLoRA, prepare model for k-bit training and apply LoRA
        if self.is_kbit:
            print("✓ Request: load_in_4bit -> QLoRA path")
            # model should already be created with from_pretrained(load_in_4bit=True, device_map=..,
            # bnb_4bit_* kwargs). We only prepare and wrap here.
            try:
                # prepare kbit training (changes module dtype for certain ops)
                prepare_model_for_kbit_training(self.model)
            except Exception as e:
                print(f"⚠ prepare_model_for_kbit_training failed: {e}")

            if self.use_lora:
                lora_config = LoraConfig(
                    r=self.args.lora_r,
                    lora_alpha=self.args.lora_alpha,
                    target_modules=self.args.lora_target_modules,
                    lora_dropout=self.args.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                print("✓ Wrapping model with LoRA adapters (PEFT)")
                self.model = get_peft_model(self.model, lora_config)

                # Only adapter params should require grad
                for n, p in self.model.named_parameters():
                    p.requires_grad = any(x in n for x in ["lora"])

                # optimizer on adapter params
                adapter_params = [p for p in self.model.parameters() if p.requires_grad]
                self.optimizer = torch.optim.AdamW(adapter_params, lr=self.args.learning_rate)
            else:
                # Not using LoRA: can't train full weights on k-bit model
                raise RuntimeError("Requested load_in_4bit but use_lora=False. Training full weights on 4-bit models is unsupported.")

        else:
            # Non k-bit path: cast dtype if requested
            if self.dtype is not None:
                try:
                    self.model = self.model.to(device=self.device, dtype=self.dtype)
                except Exception:
                    self.model = self.model.to(device=self.device)
            else:
                self.model = self.model.to(device=self.device)

            # full-weight optimizer
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

        # gradient checkpointing if supported
        if self.args.enable_gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable()
            except Exception:
                pass

        # reference model if beta != 0
        if self.args.beta == 0.0:
            self.ref_model = None
        else:
            try:
                self.ref_model = create_reference_model(self.model).to(device=self.device)
            except Exception:
                self.ref_model = None

    # ---------------------------
    # Tokenizer / metrics / generation
    # ---------------------------
    def _initialize_tokenizers(self):
        if self.processing_class is None:
            try:
                model_name = getattr(self.model.config, "_name_or_path", None)
                if model_name:
                    self.processing_class = AutoTokenizer.from_pretrained(model_name, padding_side="left")
            except Exception:
                self.processing_class = None

    def _initialize_metrics(self):
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

    def _initialize_generation_config(self):
        base_do_sample = bool(self.args.temperature and self.args.temperature > 0.0)
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
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k if self.args.top_k is not None else 50,
            min_p=self.args.min_p,
            repetition_penalty=self.args.repetition_penalty,
        )

    # ---------------------------
    # Input / generate helpers
    # ---------------------------
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
                            self.processing_class.apply_chat_template(item["prompt"], tokenize=False, add_generation_prompt=True)
                        )
            else:
                templated_prompts = [
                    self.processing_class.apply_chat_template(item["prompt"], tokenize=False, add_generation_prompt=True)
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

        input_tokens = self.processing_class(text=templated_prompts, return_tensors="pt", padding=True, truncation=True)
        return input_tokens

    def generate(self, inputs: Any, return_completion_ids: bool = False, stage=0) -> Any:
        input_tokens = self._process_inputs(inputs)
        rollout, rollout_ids = [], []

        # For k-bit models, prefer greedy per-sample generation for stability
        if self.is_kbit:
            gen_kwargs = {"do_sample": False, "num_beams": 1, "use_cache": False, "top_k": 1, "top_p": 1.0, "temperature": 1.0, "max_new_tokens": self.args.max_new_tokens}
        else:
            gen_kwargs = {"generation_config": self.generation_config}

        pad_id = getattr(self.processing_class, "pad_token_id", None)
        eos_id = getattr(self.processing_class, "eos_token_id", None)
        if pad_id is not None:
            gen_kwargs["pad_token_id"] = pad_id
        if eos_id is not None:
            gen_kwargs["eos_token_id"] = eos_id

        input_ids = input_tokens.input_ids.to(self.model.device)
        attention_mask = input_tokens.attention_mask.to(self.model.device, dtype=torch.long)
        prompt_len = input_ids.size(1)

        for _ in range(self.args.num_generations):
            with torch.no_grad():
                if self.is_kbit and input_ids.size(0) > 1:
                    outs = []
                    for i in range(input_ids.size(0)):
                        ids = input_ids[i : i + 1]
                        mask = attention_mask[i : i + 1]
                        try:
                            out = self.model.generate(ids, attention_mask=mask, **gen_kwargs)
                        except Exception as e:
                            print(f"⚠ Gen error sample {i}: {e}")
                            out = torch.cat([ids, torch.full((1, 10), pad_id or 0, device=ids.device)], dim=1)
                        outs.append(out)
                    outputs = torch.cat(outs, dim=0)
                else:
                    try:
                        outputs = self.model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
                    except Exception as e:
                        print(f"⚠ Generation error: {e}")
                        outputs = torch.cat([input_ids, torch.full((input_ids.size(0), 10), pad_id or 0, device=input_ids.device)], dim=1)

            completion_ids = outputs[:, prompt_len:]
            completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

            if len(rollout) == 0:
                rollout = [[c] for c in completions]
                if return_completion_ids:
                    rollout_ids = [[comp] for comp in completion_ids]
            else:
                for idx, c in enumerate(completions):
                    rollout[idx].append(c)
                    if return_completion_ids:
                        rollout_ids[idx].append(completion_ids[idx])

        return (rollout, rollout_ids) if return_completion_ids else rollout

    # ---------------------------
    # per-token logprobs + loss (keep stable numerics)
    # ---------------------------
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]

        loss_mask = attention_mask[:, -logits_to_keep:].to(device=logits.device, dtype=torch.float32).contiguous()
        labels = input_ids[:, -logits_to_keep:].contiguous()
        logits = logits[:, -logits_to_keep:].contiguous()

        temp = max(1e-8, float(self.args.temperature))
        logits = logits / temp

        logits_for_ce = logits.float()
        token_log_probs = -torch.nn.functional.cross_entropy(logits_for_ce.view(-1, logits_for_ce.shape[-1]), labels.view(-1), reduction="none").view(logits_for_ce.shape[0], logits_for_ce.shape[1])

        token_log_probs = token_log_probs.float()
        token_log_probs = token_log_probs * loss_mask + (1.0 - loss_mask) * torch.finfo(torch.float32).min
        return token_log_probs

    def compute_loss(self, model, inputs, mode="train", return_metrics=False):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(self.model.device)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1).to(self.model.device, dtype=torch.long)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        if self.args.beta != 0.0:
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, input_ids, attention_mask, logits_to_keep)
            else:
                ref_per_token_logps = per_token_logps.clone()
            per_token_kl = (torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1)

        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps", None)
        if old_per_token_logps is None:
            old_per_token_logps = per_token_logps.detach()

        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.args.epsilon, 1 + (self.args.epsilon_high if self.args.epsilon_high is not None else self.args.epsilon))
        advantages = advantages.unsqueeze(dim=-1)
        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        per_token_loss = torch.clamp(per_token_loss, -10.0, 10.0)

        if self.args.beta != 0.0:
            per_token_loss = per_token_loss + self.args.beta * per_token_kl

        completion_mask_float = completion_mask.float()
        loss = (per_token_loss * completion_mask_float).sum() / (completion_mask_float.sum() + 1e-8)

        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠ WARNING: Invalid loss detected (NaN or Inf), setting to 0")
            loss = torch.tensor(0.0, device=loss.device, requires_grad=True)

        if self.args.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask_float).sum() / (completion_mask_float.sum() + 1e-8)
            self._metrics[mode]["kl"].append(mean_kl.item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask_float).sum() / (completion_mask_float.sum() + 1e-8)
        self._metrics[mode]["clip_ratio"].append(clip_ratio.item())
        self._metrics[mode]["loss"].append(loss.item())

        metrics = {"loss": loss.item(), "kl": mean_kl.item() if self.args.beta != 0.0 else None, "clip_ratio": clip_ratio.item()}

        if return_metrics:
            return loss, metrics
        else:
            return loss

    # ---------------------------
    # Training loop
    # ---------------------------
    def train(self, state: GameState, data_manager: DataManager, reward_manager: RewardManager) -> None:
        self.model.train()
        global_step = self.global_step
        for stage in range(state.stage):
            global_step = self.step(stage, state, data_manager, reward_manager, global_step)
        self.global_step = global_step
        self.model.eval()

    def step(self, stage: int, state: GameState, data_manager: DataManager, reward_manager: RewardManager, global_step: int) -> int:
        global_step += 1

        stage_inputs = state.get_stage_state(stage)
        stage_inputs, index_mapping = data_manager.prepare_input(stage_inputs, stage)
        assert stage_inputs is not None, f"No inputs found for stage {stage}"

        stage_actions = state.get_stage_actions(stage)
        stage_outputs = [stage_actions[index_mapping[idx][0]][index_mapping[idx][1]][index_mapping[idx][2]] for idx, _ in enumerate(index_mapping)]
        assert stage_outputs is not None, f"No outputs found for stage {stage}"

        metrics = {}

        model_inputs = {}
        processed_inputs = self._process_inputs(stage_inputs, for_training=True)
        model_inputs["prompt_ids"] = processed_inputs.input_ids.to(self.model.device)
        model_inputs["prompt_mask"] = processed_inputs.attention_mask.to(self.model.device, dtype=torch.long)

        processed_outputs = self._process_inputs(stage_outputs, with_template=False, for_training=True)
        model_inputs["completion_ids"] = processed_outputs.input_ids.to(self.model.device)
        model_inputs["completion_mask"] = processed_outputs.attention_mask.to(self.model.device, dtype=torch.long)

        rewards = reward_manager[stage]
        rewards = [rewards[index_mapping[idx][0]][index_mapping[idx][1]][index_mapping[idx][2]] for idx, _ in enumerate(index_mapping)]
        assert rewards is not None, f"No rewards found for stage {stage}"
        rewards = torch.tensor(rewards, dtype=torch.float32)

        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"📊 Stage {stage} | Rewards - mean: {rewards.mean():.4f}, std: {rewards.std():.4f}, min: {rewards.min():.4f}, max: {rewards.max():.4f}")

        do_training = (rewards != 0).any()

        if do_training:
            with torch.no_grad():
                # compute advantages in float32
                advantages = rewards - rewards.mean(dim=1, keepdim=True)
                if rewards.shape[1] > 1:
                    advantages /= (rewards.std(dim=1, keepdim=True) + 1e-8)

                print(f"📊 Advantages - mean: {advantages.mean():.4f}, std: {advantages.std():.4f}")

                prompt_ids, prompt_mask = model_inputs["prompt_ids"], model_inputs["prompt_mask"]
                completion_ids, completion_mask = model_inputs["completion_ids"], model_inputs["completion_mask"]
                input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(self.model.device)
                attention_mask = torch.cat([prompt_mask, completion_mask], dim=1).to(self.model.device, dtype=torch.long)
                logits_to_keep = completion_ids.size(1)
                old_per_token_logps_full = self._get_per_token_logps(self.model, input_ids, attention_mask, logits_to_keep).detach()

            advantages = torch.flatten(advantages).to(self.model.device, dtype=torch.float32)
            model_inputs["advantages"] = advantages.squeeze(dim=-1)

            num_samples = model_inputs["completion_ids"].size(0)
            updates_per_rollout = self.args.ppo_epochs
            minibatch_size = min(self.args.minibatch_size, num_samples)

            loss_vals = []
            for _ in range(updates_per_rollout):
                perm = torch.randperm(num_samples, device=self.model.device)
                for start in range(0, num_samples, minibatch_size):
                    idx = perm[start : start + minibatch_size]
                    mb_inputs = self._return_minibatch(model_inputs, idx, old_per_token_logps_full)
                    self.model.zero_grad()
                    loss = self.compute_loss(self.model, mb_inputs)

                    if torch.isnan(loss) or torch.isinf(loss):
                        print("⚠ Skipping backward due to invalid loss")
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    loss_vals.append(loss.detach().float())

            mean_loss = torch.stack(loss_vals).mean() if loss_vals else torch.tensor(0.0)
            print(f"✓ Training completed | Loss: {mean_loss:.4f}")
        else:
            mean_loss = torch.tensor(0.0)
            print("⊘ No training (all rewards are zero)")

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
        # This function must be implemented in your subclass (application layer).
        # It should request question, generate answer, submit to judge, get score and call reward_manager.add_reward(...)
        return None

    def save(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        try:
            self.model.save_pretrained(save_dir)
            print(f"✓ Model saved to {save_dir}")
        except Exception as e:
            print(f"⚠ Save failed: {e}")
            torch.save(self.model.state_dict(), os.path.join(save_dir, "model_state.pt"))

        torch.save({"metrics": self._metrics, "total_train_tokens": self._total_train_tokens, "generation_config": self.generation_config}, os.path.join(save_dir, "trainer_state.pt"))

    @classmethod
    def load(cls, load_dir: str) -> "GRPOLanguageTrainerModule":
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
