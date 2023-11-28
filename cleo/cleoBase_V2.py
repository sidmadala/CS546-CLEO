from transformers import AutoTokenizer, AutoModelForCausalLM
from imagebind import data
import torch
import typing
from typing import List
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import logging
import torch.nn as nn

class CLEO(torch.nn.Module):
    def __init__(
        self,
        llm_model_path: str,
        audio_features: int, # 1024 if ImageBind
        host_llm_on_cuda: bool = False,
        max_seq_len: int = 512,
        freeze_llm: bool = True,
        audio_instruction_token: str = "<wav>",
        audio_gpu: str = "cpu"
    ):
        super().__init__()
        self.llm_tokenizer, self.llm_model = self.__load_llm__(
            llm_model_path, freeze_llm, device="auto" if host_llm_on_cuda else "cpu"
        )

        ## projection layer
        self.proj = nn.Linear(audio_features, self.llm_model.config.hidden_size)
        if audio_gpu != "cpu":
            self.proj = self.proj.to(audio_gpu)
        self.max_seq_len = max_seq_len
        self.audio_instruction_token = audio_instruction_token
        self.host_llm_on_cuda = host_llm_on_cuda

    def __load_llm__(self, llm_model, freeze_llm, pad_token_id=None, device="cpu"):
        ## Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(llm_model)
        if pad_token_id is not None:
            tokenizer.pad_token_id = pad_token_id
        else:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(llm_model, device_map=device)
        logging.info("Loaded LLAMA model")
        if freeze_llm:
            for param in model.parameters():
                param.requires_grad = False
            logging.info("Model parameters frozen")
        return tokenizer, model

    def __embed_text_tokens__(self, tokens: torch.Tensor):
        return self.llm_model.base_model.embed_tokens(tokens)

    def __prepare_instruction__(self, instruction, wav_embs):
        ## Split a single instruction segment into different pieces
        instruction_segments = instruction.split(self.audio_instruction_token)

        ## Make sure we have enough
        assert len(instruction_segments) == len(wav_embs) + 1

        embs = []
        for idx, segment in enumerate(instruction_segments):
            ## Tokenize the segment
            tokens = self.llm_tokenizer.encode(segment, padding=False, truncation=False, return_tensors="pt")[0]
            if self.host_llm_on_cuda:
                tokens = tokens.to("cuda:0")

            ## Get the instruction embeddings
            emb = self.__embed_text_tokens__(tokens)
            if idx < len(wav_embs):
                wav_at_position = wav_embs[idx].unsqueeze(0)
                if self.host_llm_on_cuda:
                    wav_at_position = wav_at_position.to("cuda:0")
                emb = torch.cat([emb, wav_at_position], dim=0)
            embs.append(emb)

        embs = torch.cat(embs, dim=0)
        attn = torch.zeros(self.max_seq_len, dtype=torch.long)
        attn[:embs.shape[0]] = 1

        if embs.shape[0] < self.max_seq_len:
            pad_tokens = torch.tensor([self.llm_tokenizer.pad_token_id]).repeat(self.max_seq_len - embs.shape[0])
            if self.host_llm_on_cuda:
                pad_tokens = pad_tokens.to("cuda:0")
            ## Need to create the pad array
            pad_embs = self.__embed_text_tokens__(pad_tokens)
            ## concatenate the embs with pad_embs
            embs = torch.cat([embs, pad_embs], dim=0)
        else:
            embs = embs[:self.max_seq_len]

        if self.host_llm_on_cuda:
            embs = embs.to("cuda:0")
            attn = attn.to("cuda:0")
        return embs, attn

    
