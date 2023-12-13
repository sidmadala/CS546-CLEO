from transformers import AutoTokenizer, AutoModelForCausalLM
from imagebind import data
import torch
import typing
from typing import List
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import logging
import torch.nn as nn
from cleo.cleoBase_V2 import CLEO

## When we plan to use a different audio processor to get different features, we need the following updates:
# 1. Update the __init__ function and remove the imageBind_model parameter if it is not needed
# 2. Update the encode_audio __get_audio_embeddings__ function to use the new audio processor. The outputs needs to be a NxD tensor, where N is the number of audio segments and D is the dimension of the audio features

class CLEOImageBind(CLEO):
    def __init__(
        self,
        llm_model_path: str,
        audio_features: int, # 1024 if ImageBind,
        imageBind_model,
        audio_gpu: str = "cpu",
        host_llm_on_cuda: bool = False,
        max_seq_len: int = 512,
        freeze_llm: bool = True,
        audio_instruction_token: str = "<wav>"
    ):
        super().__init__(llm_model_path, audio_features, host_llm_on_cuda, max_seq_len, freeze_llm, audio_instruction_token, audio_gpu)
        self.imageBind_model = imageBind_model
        self.imageBind_model = self.imageBind_model.to(audio_gpu)
        self.audio_gpu = audio_gpu
        self.host_llm_on_cuda = host_llm_on_cuda

    def __get_audio_embeddings__(self, audio_paths: List[str]):
        inputs = {
            ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, self.audio_gpu, sample_rate=24000),
        }

        with torch.no_grad():
            embeddings = self.imageBind_model(inputs)
        return embeddings["audio"] # number of audio files x dimension of audio features as a tensor

    def encode_audio(self, audio_paths):
        ## First grab the embeddings from imagebind
        embeddings = self.__get_audio_embeddings__(audio_paths)
        ## Pass through the projection layer
        wav_embs = self.proj(embeddings)
        ## Create the attention mask for the layer
        wav_attn = torch.ones(wav_embs.size(), dtype=torch.long)
        return wav_embs, wav_attn

    def __prepare_batch__(self, batch):
        assert "instructions" in batch.keys()
        assert "audio_paths" in batch.keys()
        assert "labels" in batch.keys()
        assert len(batch["instructions"]) == len(batch["audio_paths"]) == len(batch["labels"])

        ## Get batch size
        batch_size = len(batch["instructions"])
        
        ## Get the processed embeddings and attentions
        processed_embs = []
        processed_attns = []
        for idx in range(batch_size):
            instruction = batch["instructions"][idx]
            audio_paths = batch["audio_paths"][idx]
            wav_embs, _ = self.encode_audio(audio_paths)

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

    def generate(self, instruction, audioPath, label, max_new_tokens=15, top_p=.5, top_k=50, temperature=1.5, repetition_penalty=1.5):
        ## Create the batch
        batch = {
            "instructions": [instruction],
            "audio_paths": [[audioPath]],
            "labels": [label]
        }

        ## Get the embeddings
        input_embs, input_attn, labels = self.__prepare_batch__(batch)

        output = self.llm_model.generate(
            inputs_embeds = input_embs,
            attention_mask = input_attn,
            max_new_tokens = max_new_tokens,
            top_p = top_p,
            top_k = top_k,
            temperature = temperature,
            repetition_penalty = repetition_penalty,
            do_sample = True
        )
        return output


## create the main function
# if __name__ == "__main__":
#         ## Define the prompt:
#     instruction_prompts = [
#         """Convert the following information to a graph of triplets. 
#         Here is an example for you:
#         <wav>
#         Triples:
#         Hello | toYou | thing

#         Now it is your turn:
#         <wav>
#         Triples:
#         """,
#         """Convert the following information to a graph of triplets:
#         <wav>

#         Triples:
#         """
#     ]
                        
#     audio_file_path = [
#         ["/home/CS546-CLEO/wav_samples/0a304f91-fdee-479e-978c-62bc1483c92d.wav", "/home/CS546-CLEO/wav_samples/0a304f91-fdee-479e-978c-62bc1483c92d.wav"],
#         ["/home/CS546-CLEO/wav_samples/0b4c9803-3df3-42e2-bedb-dce38215a950.wav"]
#     ]

#     labels = [
#         "Hello | one | three",
#         "help | two | five"
#     ]

#     batch = {
#         "instructions": instruction_prompts,
#         "audio_paths": audio_file_path,
#         "labels": labels
#     }

#     ib_model = imagebind_model.imagebind_huge(pretrained=True)
#     cleo_model = CLEOImageBind(
#         llm_model_path = "/home/models/Llama-2-7b-hf",
#         audio_features = 1024, # 1024 if ImageBind,
#         imageBind_model = ib_model,
#         host_llm_on_cuda = True
#     )
#     outputs = cleo_model(batch)


