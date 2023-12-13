from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import typing
from typing import List
import logging
import torch.nn as nn
from datasets import load_from_disk, load_dataset
import tqdm
import numpy as np
import os
from transformers import ClapProcessor, ClapModel
from torch.utils.data import Dataset, DataLoader
import os
from cleo.QFormer import BertConfig, BertLMHeadModel
from torch.nn import functional as F
from info_nce import InfoNCE, info_nce


class CLEOQFormer(torch.nn.Module):
    def __init__(
        self,
        llm_model_path: str,
        clapPath: str,
        audio_width: int = 512, # 512 for CLAP, 1024 for ImageBind
        num_query_tokens: int = 32,
        audio_gpu: str = "cpu",
        host_llm_on_cuda: bool = False,
        max_seq_len: int = 512,
        freeze_llm: bool = True,
        freeze_qformer: bool = False,
        audio_instruction_token: str = "<wav>",
        training_objective: str = "ATC",
        pretraining_Qformer: bool = True
    ):

        super().__init__()
        assert training_objective in ["ATC", "ATG", "ATM"]
        self.audio_gpu = audio_gpu
        self.host_llm_on_cuda = host_llm_on_cuda
        self.training_objective = training_objective
        self.pretraining_Qformer = pretraining_Qformer

        ## Load the audio models
        self.clapModelProcessor = ClapProcessor.from_pretrained(clapPath)
        self.clapModel = ClapModel.from_pretrained(clapPath)
        if audio_gpu != "cpu":
            self.clapModel = self.clapModel.to(audio_gpu)
        ## Freeze audio model
        for param in self.clapModel.parameters():
            param.requires_grad = False

        ## Load LLM
        llm_device = "auto" if host_llm_on_cuda else "cpu"
        self.tokenizer, self.llm = self.__load_llm__(llm_model_path, True, device=llm_device)

        ## Load QFormer
        Qformer, query_tokens = self.init_Qformer(num_query_tokens, audio_width, freeze_qformer)
        if audio_gpu != "cpu":
            self.Qformer = Qformer.to(audio_gpu)
            self.query_tokens = query_tokens.to(audio_gpu)
        else:
            self.Qformer = Qformer
            self.query_tokens = query_tokens
        self.Qtokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        ## Create projection layer
        self.proj = nn.Linear(self.Qformer.config.hidden_size, self.llm.config.hidden_size)
        if audio_gpu != "cpu":
            self.proj = self.proj.to(audio_gpu)


    ## Initialize QFormer
    def init_Qformer(self, num_query_token, audio_width, freeze):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = audio_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        if not self.pretraining_Qformer:
            Qformer.cls = None
            Qformer.bert.embeddings.word_embeddings = None
            Qformer.bert.embeddings.position_embeddings = None
            for layer in Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None

        if freeze:
            for name, param in Qformer.named_parameters():
                param.requires_grad = False
            Qformer = Qformer.eval()
            query_tokens.requires_grad = False
            logging.info("freeze Qformer")
        return Qformer, query_tokens

    ## Initialize LLM
    def __load_llm__(self,llm_model, freeze_llm, pad_token_id=None, device="cpu"):
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

    ## Def get audio embeddings
    def get_audio_embeddings(self, audios):
        inputs = self.clapModelProcessor(audios=audios, sampling_rate=48000, return_tensors="pt")
        if self.audio_gpu != "cpu":
            inputs = inputs.to(self.audio_gpu)
        with torch.no_grad():
            embeddings = self.clapModel.get_audio_features(**inputs, return_dict=True)
        return embeddings

    ## Use CLAP to encode the audio and pass it through the QFormer
    def encode_audio(self, audios):
        ## Get the embeddings first
        wav_embs = self.get_audio_embeddings(audios)
        wav_embs = wav_embs.unsqueeze(1)
        if self.audio_gpu != "cpu":
            wav_embs = wav_embs.to(self.audio_gpu)

        ## Create the attention mask for the wav
        wav_attn = torch.ones(wav_embs.size()[:-1], dtype=torch.long)
        if self.audio_gpu != "cpu":
            wav_attn = wav_attn.to(self.audio_gpu)

        ## Expand the query tokens
        wav_query_tokens = self.query_tokens.expand(wav_embs.shape[0], -1, -1)
        if self.audio_gpu != "cpu":
            wav_query_tokens = wav_query_tokens.to(self.audio_gpu)

        ## Create Qformer output
        query_output = self.Qformer.bert(
            query_embeds = wav_query_tokens,
            encoder_hidden_states = wav_embs,
            encoder_attention_mask = wav_attn,
            return_dict = True
        )

        return query_output

    ## Project the QFormer queries through the projection layer
    def project_query(self, query_output):
        wav_input = self.proj(query_output["last_hidden_state"])
        wav_attn = torch.ones(wav_input.size()[:-1], dtype=torch.long)
        if self.audio_gpu != "cpu":
            wav_attn = wav_attn.to(self.audio_gpu)
        return wav_input, wav_attn

    ## Get the embeddings from the text
    def encode_text(self, text):
        tokenized = self.tokenizer.encode(text, return_tensors="pt")
        if self.host_llm_on_cuda:
            tokenized = tokenized.to(self.audio_gpu)

        output_dict = self.llm(tokenized, return_dict=True, output_hidden_states=True)
        return output_dict

    ## Audio-Text Contrastive Loss
    def obj_func_ATC(self, audios, labels):
        ## Get the wav_input and wav_attn
        wav_output = self.encode_audio(audios)

        ## Pass the labels through QFormer
        stuff = self.Qtokenizer(labels, return_tensors="pt", padding=True, truncation=True, max_length=256)
        if self.audio_gpu != "cpu":
            stuff = stuff.to(self.audio_gpu)
            
        text_output = self.Qformer.bert(
            input_ids = stuff["input_ids"],
            attention_mask = stuff["attention_mask"],
            return_dict = True
        )
        best_match = torch.argmax(F.cosine_similarity(wav_output.last_hidden_state[:,:,:], text_output.last_hidden_state[:,0,:].unsqueeze(1), dim=2), dim=1)
        wav_rep = torch.gather(wav_output.last_hidden_state, 1, best_match.view(-1, 1, 1).expand(-1, 1, 768)).squeeze(1)
        cls_rep = text_output.last_hidden_state[:,0,:]
        loss = InfoNCE()
        output = loss(wav_rep, cls_rep)
        return output

    def forward(self, batch):
        if self.training_objective == "ATC":
            return self.obj_func_ATC(batch["audios"], batch["labels"])
        if self.training_objective == "ATM":
            pass
        if self.training_objective == "ATG":
            pass


