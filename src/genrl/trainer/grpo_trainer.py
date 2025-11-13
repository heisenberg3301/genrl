import gc
import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

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


class GRPOLanguageTrainerModule(TrainerModule, LoggerMixin):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method.
    Modified to safely support BitsAndBytes quantized models (4-bit).
    """

    def __init__(self, models: List[Any], config: GRPOTrainerConfig, **kwargs):
        if not models or len(models) < 1:
            raise ValueError("At least one model must be provided")

        self.model = models[0]
        self.args = config

        # safe dtype lookup: do not KeyError if user passed 'auto' or unknown
        self.dtype = DTYPE_MAP.get(self.args.dtype, None)  # None means "don't cast"

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate
        )

        # Tokenizers / processing class
        self.processing_class = kwargs.get("processing_class", None)

        # Additional parameters
        self.callbacks = kwargs.get("callbacks", [])
        self.save_dir = kwargs.get("log_dir", "./outputs")
        self.global_step = 0
        assert (
            self.args.num_generations > 1
        ), f"For GRPO training, number of generations must be > 1, got {self.args.num_generations}"

        self.enable_gradient_checkpointing = self.args.enable_gradient_checkpointing

        # device selection
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

    # ---------------------------
    # Model initialization
    # ---------------------------
    def _initialize_model(self, enable_gradient_checkpointing):
        """
        Initialize the model and reference model.
        Important: if model is bitsandbytes-quantized, DO NOT cast dtype.
        """

        # detect bitsandbytes / quantized model heuristically
        model_type_str = str(type(self.model)).lower()
        is_quantized = (
            hasattr(self.model, "is_quantized")
            or hasattr(self.model, "is_loaded_in_4bit")
            or "bitsandbytes" in model_type_str
            or "bnb" in model_type_str
        )

        if is_quantized:
            # Move to device but DO NOT cast dtype
            try:
                self.model = self.model.to(device=self.device)
            except Exception:
                # device_map handling may already be applied; ignore
                pass

            # disable use_cache for quantized models (can cause issues)
            try:
                if hasattr(self.model, "config"):
                    self.model.config.use_cache = False
            except Exception:
                pass

            if enable_gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
                try:
                    self.model.gradient_checkpointing_enable()
                except Exception:
                    pass

            # reference model only if beta != 0 and desired
            if self.args.beta == 0.0:
                self.ref_model = None
            else:
                try:
                    self.ref_model = create_reference_model(self.model).to(device=self.device)
                except Exception:
                    self.ref_model = None

            return

        # Non-quantized path: perform dtype cast if dtype was set
        if self.dtype is not None:
            try:
                self.model = self.model.to(device=self.device, dtype=self.dtype)
            except Exception:
                # fallback to device only
                self.model = self.model.to(device=self.device)
        else:
            self.model = self.model.to(device=self.device)

        if enable_gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable()
            except Exception:
                pass

        # reference model
        if self.args.beta == 0.0:
            self.ref_model = None
        else:
            try:
                self.ref_model = create_reference_model(self.model).to(device=self.device, dtype=self.dtype if self.dtype is not None else None)
            except Exception:
                self.ref_model = None

    # ---------------------------
    # Tokenizer init
    # ---------------------------
    def _initialize_tokenizers(self):
        if self.processing_class is None:
            # Try to load AutoTokenizer from model config name if available
            try:
                model_name = getattr(self.model.config, "_name_or_path", None)
                if model_name:
                    self.processing_class = AutoTokenizer.from_pretrained(model_name, padding_side="left")
                else:
                    self.processing_class = None
            except Exception:
                self.processing_class = None

    # ---------------------------
    # Metrics init
    # ---------------------------
    def _initialize_metrics(self):
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

    # ---------------------------
    # Generation config init
    # ---------------------------
    def _initialize_generation_config(self):
        """
        Create a base GenerationConfig.
        For quantized models we will override to GREEDY/safe settings when generating.
        """
        # choose default do_sample based on temperature: but we'll override for quantized models
        base_do_sample = bool(self.args.temperature and self.args.temperature > 0.0)

        # Graceful defaults; note: GenerationConfig stores the values but model.generate
        # can be passed kwargs to override per-call.
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
    # Input processing helper
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

        # tokenization: return tensors with padding
        input_tokens = self.processing_class(
            text=templated_prompts, return_tensors="pt", padding=True, truncation=True
        )
        return input_tokens

    # ---------------------------
    # Generate (used by game manager)
    # ---------------------------
    def generate(self, inputs: Any, return_completion_ids: bool = False, stage=0) -> Any:
        """
        Generate outputs from the model for the given inputs.
        This method keeps backward compatibility with previous genrl API.
        It will call model.generate with safe kwargs when model is quantized.
        """

        input_tokens = self._process_inputs(inputs)
        rollout, rollout_ids = [], []

        # Detect if model is quantized (same heuristic as _initialize_model)
        model_type_str = str(type(self.model)).lower()
        is_quantized = (
            hasattr(self.model, "is_quantized")
            or hasattr(self.model, "is_loaded_in_4bit")
            or "bitsandbytes" in model_type_str
            or "bnb" in model_type_str
        )

        # Prepare per-call generation kwargs. For quantized models, enforce greedy safe config.
        if is_quantized:
            gen_kwargs = {
                "do_sample": False,
                "num_beams": 1,
                "use_cache": False,
                "top_k": 1,
                "top_p": 1.0,
                "temperature": 1.0,
            }
        else:
            # non-quantized: use generation_config defaults but allow overrides
            gen_kwargs = {
                "do_sample": self.generation_config.do_sample,
                "temperature": self.generation_config.temperature,
                "top_k": self.generation_config.top_k,
                "top_p": self.generation_config.top_p,
                "repetition_penalty": self.generation_config.repetition_penalty,
                "use_cache": True,
            }

        # pass pad/eos
        pad_id = getattr(self.processing_class, "pad_token_id", None)
        eos_id = getattr(self.processing_class, "eos_token_id", None)
        if pad_id is not None:
            gen_kwargs["pad_token_id"] = pad_id
        if eos_id is not None:
            gen_kwargs["eos_token_id"] = eos_id

        # run multiple generations
        input_ids = input_tokens.input_ids.to(self.model.device)
        attention_mask = input_tokens.attention_mask.to(self.model.device, dtype=torch.long)

        prompt_len = input_ids.size(1)

        for _ in range(self.args.num_generations):
            with torch.no_grad():
                # For quantized models do per-sample generation for stability
                if is_quantized and input_ids.size(0) > 1:
                    outputs_list = []
                    for i in range(input_ids.size(0)):
                        ids = input_ids[i : i + 1]
                        mask = attention_mask[i : i + 1]
                        out = self.model.generate(ids, attention_mask=mask, **gen_kwargs)
                        outputs_list.append(out)
                    outputs = torch.cat(outputs_list, dim=0)
                else:
                    outputs = self.model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)

            # Extract completions (after prompt length)
            completion_ids = outputs[:, prompt_len:]
            completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

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

    # ---------------------------
    # per-token logprobs helper
    # ---------------------------
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep + 1,
        ).logits
        logits = logits[:, :-1, :]

        loss_mask = attention_mask[:, -logits_to_keep:].to(device=logits.device, dtype=logits.dtype).contiguous()
        labels = input_ids[:, -logits_to_keep:].contiguous()
        logits = logits[:, -logits_to_keep:].contiguous()

        # Divide logits by temperature
        logits = logits / max(1e-8, float(self.args.temperature))

        logits_shape = logits.shape
        token_log_probs = -torch.nn.functional.cross_entropy(
            logits.view(-1, logits_shape[-1]),
            labels.view(-1),
            reduction="none",
        ).view(logits_shape[0], logits_shape[1])
        token_log_probs = token_log_probs * loss_mask + (1.0 - loss_mask) * torch.finfo(logits.dtype).min
        return token_log_probs

    # ---------------------------
    # compute_loss (unchanged logic but using safe dtypes)
    # ---------------------------
    def compute_loss(self, model, inputs, mode="train", return_metrics=False):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(self.model.device)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1).to(self.model.device, dtype=self.dtype if self.dtype is not None else torch.long)
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

        if self.args.beta != 0.0:
            per_token_loss = per_token_loss + self.args.beta * per_token_kl

        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        if self.args.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(clip_ratio.item())
        self._metrics[mode]["loss"].append(loss.item())

        metrics = {"loss": loss.item(), "kl": mean_kl.item() if self.args.beta != 0.0 else None, "clip_ratio": clip_ratio.item()}
        if return_metrics:
            return loss, metrics
        else:
            return loss

    # ---------------------------
    # Training loop + step (unchanged semantics but safe dtypes)
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
        model_inputs["prompt_ids"], model_inputs["prompt_mask"] = processed_inputs.input_ids.to(self.model.device), processed_inputs.attention_mask.to(self.model.device, dtype=self.dtype if self.dtype is not None else torch.long)
        processed_outputs = self._process_inputs(stage_outputs, with_template=False, for_training=True)
        model_inputs["completion_ids"], model_inputs["completion_mask"] = processed_outputs.input_ids.to(self.model.device), processed_outputs.attention_mask.to(self.model.device, dtype=self.dtype if self.dtype is not None else torch.long)

        rewards = reward_manager[stage]
        rewards = [rewards[index_mapping[idx][0]][index_mapping[idx][1]][index_mapping[idx][2]] for idx, _ in enumerate(index_mapping)]
        assert rewards is not None, f"No rewards found for stage {stage}"
        rewards = torch.tensor(rewards)
        do_training = (rewards != 0).any()
        if do_training:
            with torch.no_grad():
                advantages = rewards - rewards.mean(dim=1, keepdim=True)
                if rewards.shape[1] > 1:
                    advantages /= rewards.std(dim=1, keepdim=True) + 1e-8

                prompt_ids, prompt_mask = model_inputs["prompt_ids"], model_inputs["prompt_mask"]
                completion_ids, completion_mask = model_inputs["completion_ids"], model_inputs["completion_mask"]
                input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(self.model.device)
                attention_mask = torch.cat([prompt_mask, completion_mask], dim=1).to(self.model.device, dtype=self.dtype if self.dtype is not None else torch.long)
                logits_to_keep = completion_ids.size(1)
                old_per_token_logps_full = self._get_per_token_logps(self.model, input_ids, attention_mask, logits_to_keep).detach()

            advantages = torch.flatten(advantages).to(self.model.device, dtype=self.dtype if self.dtype is not None else torch.float32)
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
                    loss.backward()
                    self.optimizer.step()
                    loss_vals.append(loss.detach().float())

            mean_loss = torch.stack(loss_vals).mean() if loss_vals else torch.tensor(0.0)
        else:
            mean_loss = torch.tensor(0.0)

        metrics.update({"train/loss": mean_loss.cpu().item()})
        metrics.update({"train/rewards": rewards.cpu().mean().item()})
        self.log(metrics, global_step)

        self.cleanup_step()

        return global_step

    # ---------------------------
    # Helpers
    # ---------------------------
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
        # Keep evaluate minimal here; the calling wrapper (your GRPOTrainerModule subclass) can override
        return None

    # Save/load/cleanup (unchanged)
    def save(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        # saving model using huggingface method
        try:
            self.model.save_pretrained(save_dir)
        except Exception:
            torch.save(self.model.state_dict(), os.path.join(save_dir, "model_state.pt"))

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
        gc.collect()

    def cleanup(self):
        self.cleanup_trackers()
