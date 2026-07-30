from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from deepsearch.data import filter_prompt, selector_state_prompt, selector_system_prompt
from deepsearch.selectors import generate_keywords, parse_selector_output
from deepsearch.types import Action, Paper


class TransformersGenerator:
    """Small lazy wrapper around a Hugging Face causal language model."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        adapter_path: str | None = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
    ):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Install model dependencies with `pip install -e '.[train]'`."
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError("Loading a LoRA adapter requires `peft`.") from exc
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 0.0,
    ) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            kwargs["temperature"] = temperature
        output = self.model.generate(**inputs, **kwargs)
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def next_token_probabilities(
        self,
        messages: list[dict[str, str]],
        candidates: Sequence[str],
    ) -> dict[str, float]:
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1]
        token_ids = []
        for candidate in candidates:
            ids = self.tokenizer.encode(candidate, add_special_tokens=False)
            if not ids:
                raise ValueError(f"Candidate {candidate!r} has no token ID.")
            token_ids.append(ids[0])
        selected = torch.softmax(logits[token_ids].float(), dim=0).tolist()
        return dict(zip(candidates, selected))


class TransformersPaperFilter:
    """Paper filter backed by an SFT or SFT+RL checkpoint."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        adapter_path: str | None = None,
        threshold: float = 0.08,
    ):
        self.generator = TransformersGenerator(
            model_name_or_path,
            adapter_path=adapter_path,
        )
        self.threshold = threshold

    def score(self, topic: str, paper: Paper) -> float:
        metadata = paper.to_dict()
        metadata.pop("references", None)
        metadata.pop("cited_by", None)
        probabilities = self.generator.next_token_probabilities(
            [{"role": "user", "content": filter_prompt(topic, metadata)}],
            ["yes", "no"],
        )
        return probabilities["yes"]


class TransformersActionSelector:
    """Retrieval selector backed by an SFT or SFT+RL checkpoint."""

    def __init__(self, model_name_or_path: str, *, adapter_path: str | None = None):
        self.generator = TransformersGenerator(
            model_name_or_path,
            adapter_path=adapter_path,
        )

    def select(
        self,
        *,
        topic: str,
        collection: Sequence[Paper],
        action_history: Sequence[Action],
        used_keywords: Sequence[str],
        step: int,
        max_steps: int,
        unused_seed_ids: Sequence[str],
    ) -> tuple[Action, list[str]]:
        state = {
            "current_step": step,
            "max_steps": max_steps,
            "previous_actions": [action.value for action in action_history],
            "used_keywords": list(used_keywords),
            "collection_ids": [paper.id for paper in collection],
            "papers": [
                {
                    "id": paper.id,
                    "title": paper.title,
                    "abstract": paper.abstract[:600],
                    "year": paper.year,
                    "citation_count": paper.citation_count,
                }
                for paper in collection[:20]
            ],
        }
        response = self.generator.generate(
            [
                {"role": "system", "content": selector_system_prompt()},
                {"role": "user", "content": selector_state_prompt(topic, state)},
            ],
            max_new_tokens=192,
        )
        action, keywords = parse_selector_output(response)
        if action is None or (action == Action.SEARCH and not keywords):
            return Action.SEARCH, generate_keywords(topic, used_keywords=used_keywords)
        return action, keywords


def paper_metadata_json(paper: Paper) -> str:
    return json.dumps(paper.to_dict(), ensure_ascii=False)
