import os
import torch
import math
from abc import ABC
import torch.nn as nn
from typing import List
from transformers import AutoTokenizer
from transformers import SiglipTextModel
from transformers import T5EncoderModel, T5Tokenizer
from transformers import CLIPTextModel, AutoModel, AutoModelForCausalLM
from transformers import Qwen3VLForConditionalGeneration


class TextEmbedder(ABC, nn.Module):
    """
    Abstract base class for text embedders.
    Subclasses must set: self.model (nn.Module), self.tokenizer, self.emb_dim (int).
    This class provides a shared forward() that returns hidden states with shape (b, n, d).
    """

    emb_dim: int  # required
    max_length: int  # required
    tokenizer: object  # tokenizer
    model: nn.Module  # HF model (encoder or LM; must support output_hidden_states=True)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def forward(self, txt: List[str]):
        tok_out = self.tokenizer(
            txt, return_tensors="pt", padding="max_length", max_length=self.max_length, truncation=True
        )
        tok_out = tok_out.to(self.device)
        txt_emb = self.model(**tok_out, output_hidden_states=True)
        if hasattr(txt_emb, "last_hidden_state"):
            return txt_emb.last_hidden_state
        return txt_emb.hidden_states[-1]


# ===================================================================================================


class ClipTextEmbedder(TextEmbedder):
    def __init__(self, max_length: int = 77, compile: bool = False, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.emb_dim = 768
        self.max_length = max_length
        self.path = "openai/clip-vit-large-patch14"

        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = CLIPTextModel.from_pretrained(self.path, torch_dtype=dtype)
        self.model.requires_grad_(False)
        self.model.eval()
        if compile:
            torch.compile(self.model)

        print(f"[ClipTextEmbedder] {sum([p.numel() for p in self.parameters()]):,}")


class SigLipTextEmbedder(TextEmbedder):
    def __init__(self, max_length: int = 64, compile: bool = False, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.emb_dim = 1152
        self.max_length = max_length
        self.path = "google/siglip-so400m-patch14-384"

        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = SiglipTextModel.from_pretrained(self.path, torch_dtype=dtype, attn_implementation="sdpa")
        self.model.requires_grad_(False)
        self.model.eval()
        if compile:
            torch.compile(self.model)

        print(f"[SigLipTextEmbedder] {sum([p.numel() for p in self.parameters()]):,}")


class T5XXL(TextEmbedder):
    def __init__(self, max_length: int = 512, compile: bool = False, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.emb_dim = 4096
        self.max_length = max_length
        self.path = "google/t5-xxl-lm-adapt"

        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(self.path, max_length=max_length)
        self.model = T5EncoderModel.from_pretrained(self.path, torch_dtype=dtype)
        self.model.requires_grad_(False)
        self.model.eval()
        if compile:
            torch.compile(self.model)

        print(f"[T5XXL] {sum([p.numel() for p in self.parameters()]):,}")


class InternVL3(TextEmbedder):
    def __init__(self, max_length: int = 160, compile: bool = False, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.emb_dim = 896
        self.max_length = max_length
        self.path = "OpenGVLab/InternVL3-1B"

        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True, use_fast=True)
        model = AutoModel.from_pretrained(self.path, trust_remote_code=True, torch_dtype=dtype)
        text_tower = getattr(model, "language_model", None) or getattr(model, "text_model", None)
        assert text_tower is not None, "Could not find text tower (language_model/text_model)."
        self.model = text_tower
        self.model.requires_grad_(False)
        self.model.eval()
        if compile:
            torch.compile(self.model)

        print(f"[InternVL3] {sum([p.numel() for p in self.parameters()]):,}")


class Gemma2B(TextEmbedder):
    def __init__(self, max_length: int = 160, compile: bool = False, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.emb_dim = 2048
        self.max_length = max_length
        self.path = "google/gemma-2b"

        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=dtype)
        self.model.requires_grad_(False)
        self.model.eval()
        if compile:
            torch.compile(self.model)

        print(f"[Gemma2B] {sum([p.numel() for p in self.parameters()]):,}")


class Qwen3VLEmbedder2B(TextEmbedder):
    def __init__(
        self,
        repo: str = "Qwen/Qwen3-VL-Embedding-2B",
        max_length: int = 256,
        compile: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.path = repo
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        full_model = Qwen3VLForConditionalGeneration.from_pretrained(self.path, dtype=dtype)
        text_tower = full_model.model.language_model
        del full_model

        self.model = text_tower
        self.emb_dim = int(self.model.config.hidden_size)
        self.model.requires_grad_(False)
        self.model.eval()
        if compile:
            self.model = torch.compile(self.model)

        print(f"[Qwen3VLEmbedder2B] {sum([p.numel() for p in self.parameters()]):,}")


# ===================================================================================================


if __name__ == "__main__":
    DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_text = ["a red cube on a wooden table, studio lighting, 35mm", "image of a dog"]

    def check(model_cls):
        model = model_cls().to(DEV).eval()
        with torch.no_grad():
            out = model(batch_text)
        print(f"[{model.__class__.__name__}] output shape: {tuple(out.shape)}")
        assert out.shape[-1] == model.emb_dim, f"Mismatch emb_dim: {model.emb_dim} != {out.shape[-1]}"
        assert out.shape[1] == model.max_length, f"Mismatch max_length: {model.max_length} != {out.shape[1]}"

    check(ClipTextEmbedder)
    check(SigLipTextEmbedder)
    check(T5XXL)
    check(Gemma2B)
    check(InternVL3)
    check(Qwen3VLEmbedder2B)
