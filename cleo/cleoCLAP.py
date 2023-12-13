from transformers import AutoTokenizer, AutoModelForCausalLM
from imagebind import data
import torch
import typing
from typing import List
from transformers import ClapProcessor, ClapModel
import logging
import torch.nn as nn
from cleo.cleoBase_V2 import CLEO
from datasets import load_dataset, Audio

## When we plan to use a different audio processor to get different features, we need the following updates:
# 1. Update the __init__ function and remove the imageBind_model parameter if it is not needed
# 2. Update the encode_audio __get_audio_embeddings__ function to use the new audio processor. The outputs needs to be a NxD tensor, where N is the number of audio segments and D is the dimension of the audio features

class CLEOClap(CLEO):
    def __init__(
        self,
        llm_model_path: str,
        audio_features: int, # 1024 if ImageBind, 512 if CLAP
        clapModelVr: str = "laion/clap-htsat-unfused",
        audio_gpu: str = "cpu",
        host_llm_on_cuda: bool = False,
        max_seq_len: int = 512,
        freeze_llm: bool = True,
        audio_instruction_token: str = "<wav>"
    ):
        super().__init__(llm_model_path, audio_features, host_llm_on_cuda, max_seq_len, freeze_llm, audio_instruction_token, audio_gpu)
        self.clapModel = ClapModel.from_pretrained(clapModelVr)
        self.clapModelProcessor = ClapProcessor.from_pretrained(clapModelVr)
        if audio_gpu != "cpu":
            self.clapModel = self.clapModel.to(audio_gpu)
        self.audio_gpu = audio_gpu
        self.host_llm_on_cuda = host_llm_on_cuda

    def __get_audio_embeddings__(self, audio_array):
        inputs = self.clapModelProcessor(audios=audio_array, sampling_rate=48000, return_tensors="pt")
        if self.audio_gpu != "cpu":
            inputs = inputs.to(self.audio_gpu)
        with torch.no_grad():
            embeddings = self.clapModel.get_audio_features(**inputs)
        return embeddings

    def encode_audio(self, audio_array):
        ## First grab the embeddings from imagebind
        embeddings = self.__get_audio_embeddings__(audio_array)
        ## Pass through the projection layer
        wav_embs = self.proj(embeddings)
        ## Create the attention mask for the layer
        wav_attn = torch.ones(wav_embs.size(), dtype=torch.long)
        return wav_embs, wav_attn

    def __prepare_batch__(self, batch):
        assert "instructions" in batch.keys()
        assert "audio_array" in batch.keys()
        assert "labels" in batch.keys()
        assert len(batch["instructions"]) == len(batch["audio_array"]) == len(batch["labels"])

        ## Get batch size
        batch_size = len(batch["instructions"])
        
        ## Get the processed embeddings and attentions
        processed_embs = []
        processed_attns = []
        for idx in range(batch_size):
            instruction = batch["instructions"][idx]
            audio_array = batch["audio_array"][idx]
            wav_embs, _ = self.encode_audio(audio_array)

            proc_embs, proc_attn = self.__prepare_instruction__(instruction, wav_embs)
            processed_embs.append(proc_embs.unsqueeze(0))
            processed_attns.append(proc_attn.unsqueeze(0))

        input_embs = torch.cat(processed_embs, dim=0)
        input_attn = torch.cat(processed_attns, dim=0)
        if self.host_llm_on_cuda:
            input_embs = input_embs.to("cuda:0")
            input_attn = input_attn.to("cuda:0")

        ## Prepare labels
        labels = self.llm_tokenizer(batch["labels"], padding="max_length", truncation=True, max_length=self.max_seq_len, return_tensors="pt").input_ids
        labels[labels == self.llm_tokenizer.pad_token_id] = -100
        if self.host_llm_on_cuda:
            labels = labels.to("cuda:0")

        return input_embs, input_attn, labels

    def forward(self, batch):
        input_embs, input_attn, labels = self.__prepare_batch__(batch)
        outputs = self.llm_model(
            inputs_embeds = input_embs,
            attention_mask = input_attn,
            return_dict = True,
            labels = labels
        )
        return outputs

    def generate(self, instruction, audio_array, labels, max_new_tokens=15, top_p=.5, top_k=50, temperature=1.5, repetition_penalty=1.5):
        ## Create batch
        batch = {
            "instructions": [instruction],
            "audio_array": [audio_array],
            "labels": [labels]
        }
        
        input_embs, input_attn, labels = self.__prepare_batch__(batch)

        output = self.llm_model.generate(
            inputs_embeds = input_embs,
            attention_mask = input_attn,
            max_new_tokens = max_new_tokens,
            repetition_penalty = repetition_penalty,
            temperature = temperature,
            top_k = top_k,
            top_p = top_p,
            do_sample = True,
        )
        return output

## create the main function
# if __name__ == "__main__":
#     dataset = load_dataset("patrickvonplaten/librispeech_asr_self_contained", split="train.clean.100[0:100]")
#     dataset = dataset.cast_column("audio", Audio(sampling_rate=48000))

#     ## Define the prompt:
#     instruction_prompts = [
#         """Repeat back the information that you see below. Here is an example:
# <wav>
# Information:
# Hello! Is it me you're looking for

# Now it is your turn:
# <wav>

# Information:
# """,
#         """Convert the following information to a graph of triplets:
# <wav>

# Triples:
# """
#     ]

#     audios = [
#         [dataset[0]["audio"]["array"], dataset[1]["audio"]["array"]],
#         [dataset[2]["audio"]["array"]]
#     ]
                        
#     labels = [
#         "Hello | one | three",
#         "help | two | five"
#     ]

#     batch = {
#         "instructions": instruction_prompts,
#         "audio_array": audios,
#         "labels": labels
#     }

    # cleo_model = CLEOClap(
    #     llm_model_path = "/home/models/Llama-2-7b-hf",
    #     audio_features = 512, # 1024 if ImageBind,
    #     host_llm_on_cuda = True
    # )
#     outputs = cleo_model(batch)


