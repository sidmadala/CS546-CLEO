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
        llama_model:str = "",
        imagebind_model:str = "",
        use_cuda:bool = True,
        special_tokens:List[str] = None,
        image_bind_feature_size:int = 78336,
        max_seq_len:int = 2048
    ):
        super().__init__()
        ## Determine device allocation
        if use_cuda:
            self.wav_device = "cuda:0"
            self.llm_device = "cuda:1"
        else:
            self.wav_device = "cpu"
            self.llm_device = "cpu"

        ## Load the frozen pre-trained models
        self.llama_model, self.llama_tokenizer = self.__load_llama__(llama_model)
        self.imagebind_model = self.__load_imagebind__(imagebind_model)

        ## Create the projection layer
        self.proj = nn.Linear(image_bind_feature_size, self.llama_model.config.hidden_size)
        self.max_seq_len = max_seq_len

    ## Need to load a frozen llama 2 model & tokenizer
    def __load_llama__(self, llama_model):
        logging.info("Loading LLAMA model")
        pass

    ## Need to load a frozen imagebind model
    def __load_imagebind__(self, imagebind_model):
        logging.info("Loading Imagebind model")
        pass

    ## Needs to process the audio inputs and return a flattened array representing the embedding space
    def __imagebind_helper__(self, audio_paths):
        inputs = {
            ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, self.wav_device, sample_rate=24000),
        }
        return torch.flatten(inputs[ModalityType.AUDIO], start_dim=1)

    ## Encode the audio
    # Inputs: audios: List[str] - A list of audio paths
    # Outputs: 
    #  - inputs_llama: torch.Tensor - The encoded audio embeddings
    #  - audio_atts: torch.Tensor - The attention mask for the audio embeddings
    def encode_audio(self, audios):
        ## Get the flattened audio embeddings from the imagebind model
        audio_embeds = self.__imagebind_helper__(audios)
        ## Pass through the projection layer
        inputs_llama = self.proj(audio_embeds)
        ## Create the attention mask for the layer
        audio_atts = torch.ones(inputs_llama.size(), dtype=torch.long)
        return inputs_llama, audio_atts

    # Ouptut shape is (batch_size, sequence_length, hidden_size)
    # hidden_size is 4096 for LLama-2
    def embed_text_tokens(self, tokens):
        assert type(tokens) is torch.Tensor, "Tokens must be a tensor"
        return self.llama_model.base_model.embed_tokens(tokens)

    def get_input_embeddings(self, instruction, wav_embeds):
        ## 1.  Need to split the instruction everytime we see a special token
        instruction_segments = instruction.split("<wav>")
        assert len(instruction_segments) - 1 == wav_embeds.shape[0], "The number of audio inputs in the instruction does not correspond to the number of audio samples"

        ## 2. Need to create the input embeddings
        embed_segments = []
        for idx, segment in enumerate(instruction_segments):
            emb = self.embed_text_tokens(self.llama_tokenizer.encode(segment, padding=False, truncation=False, return_tensors="pt")[0])
            if idx == len(instruction_segments) - 1:
                emb = torch.cat((emb, wav_embeds[idx]), dim=0)
            embed_segments.append(emb)

        ## 3. Concat the embed segments
        input_embs = torch.cat(embed_segments, dim=0)

        ## 4. Create the attn mask
        if len(input_embs) <= self.max_seq_len:
            ## Padding attn mask
            attn_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
            attn_mask[:len(input_embs)] = 1
            ## Create the padding array
            padding_array = self.embed_text_tokens(torch.tensor([self.llama_tokenizer.pad_token_id]).repeat(self.max_seq_len - len(input_embs),))
            ## Concat the padding array
            input_embs = torch.cat((input_embs, padding_array), dim=0)
        else:
            ## Truncation
            attn_mask = torch.ones(self.max_seq_len, dtype=torch.long)
            input_embs = input_embs[:self.max_seq_len,:]
        
        return input_embs, attn_mask


    ## Expect the inputs to be a dictionary
    def create_embeddings(self, inputs):
        input_keys = inputs.keys()
        assert "audio_path" in input_keys, "Audio paths must be provided"
        assert "instruction" in input_keys, "Instruction must be provided"
        assert "output_graph_tripples" in input_keys, "Output graph tripples must be provided"

        ## 1. Need to get the audio embeddings and attentions
        # Shape - (batch_size, projection_size)
        wav_embeds, wav_attn = self.encode_audio(inputs["audio_path"])

        ## 2. Need to somehow wrap the audio embeddings in the llama tokenizer along with the instruction
        input_embs, input_attn = self.get_input_embeddings(inputs["instruction"], wav_embeds)

        ## 3. Need to process the output tokens
        labels = self.llama_tokenizer.encode(
            inputs["output_graph_tripples"],
            padding=False,
            truncation=False,
            return_tensors="pt"
        )[0]

        return input_embs, input_attn, labels
    
    def forward(self, input):
        input_embs, input_attn, labels = self.create_embeddings(input)

        outputs = self.llama_model(
            input_embeds = input_embs,
            attention_mask = input_attn,
            return_dict = True,
            labels = labels
        )

        loss = outputs.loss
        return {
            "loss": loss
        }

    