from regex import E
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import typing
from typing import List
import logging
import torch.nn as nn
from cleo.cleoQFormer import CLEOQFormer
from datasets import load_from_disk, load_dataset
import tqdm
import numpy as np
import os
from transformers import ClapProcessor, ClapModel
from torch.utils.data import Dataset, DataLoader
import os
from cleo.QFormer import BertConfig, BertLMHeadModel
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

experiment_name = "QFormer_ATC_PreTrain"
BATCH_SIZE = 16
EPOCHS = 100

try:  
    os.mkdir(f"/home/CS546-CLEO/models/{experiment_name}")  
except OSError as error:  
    print(error)   

writer = SummaryWriter(log_dir=f"/home/CS546-CLEO/runs/{experiment_name}")


clapModelVr = "laion/clap-htsat-unfused"
dataset = load_dataset("patrickvonplaten/librispeech_asr_self_contained", split="train.clean.100")
audio_gpu = "cuda:1"
clapModelProcessor = ClapProcessor.from_pretrained(clapModelVr)

class CLEODataset(Dataset):
    def __init__(self, dataset, instruction, processor, sampling_rate = 48000):
        self.dataset = dataset
        self.instruction = instruction
        self.processor = processor
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ## Create the label
        label = self.dataset[idx]["text"].lower()
        
        ## Save the audio
        audio_array = self.dataset[idx]["audio"]["array"]
        return self.instruction, audio_array, label

def custom_collate_fn(original_batch):
    instructions = [each[0] for each in original_batch]
    audios = [each[1] for each in original_batch]
    labels = [each[2] for each in original_batch]
    return instructions, audios, labels

instruction = """Repeat back the information that you see below:
<wav>

Information:
"""
cleoDataset = CLEODataset(dataset, instruction, clapModelProcessor)
train_dataloader = DataLoader(cleoDataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
print("Dataset Loaded...")

cleo_model = CLEOQFormer(
        llm_model_path = "/home/models/Llama-2-7b-hf",
        clapPath = clapModelVr,
        audio_width = 512, # 1024 if ImageBind,
        num_query_tokens = 32,
        host_llm_on_cuda = True,
        audio_gpu = "cuda:1",
        freeze_llm = True,
        freeze_qformer = False,
        training_objective = "ATC"
)
print("Model Loaded...")

optimizer = torch.optim.Adam(cleo_model.parameters(), lr=0.0001)
numIter = 0
for epoch in range(1,EPOCHS):
    loss_avg = []
    with tqdm.tqdm(train_dataloader, unit="batch", total=len(train_dataloader)) as tepoch:
        for batch_idx, preBatch in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch} / Batch {batch_idx}")
            ## Create batch
            batch = {
                "instructions": preBatch[0],
                "audios": preBatch[1],
                "labels": preBatch[2]
            }
            loss = cleo_model(batch)
            loss_val = loss.item()
            writer.add_scalar("Loss/train", loss_val, numIter)
            loss_avg.append(loss_val)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tepoch.set_postfix(loss = loss_val)
            if numIter % 50 == 0:
                torch.save(cleo_model.state_dict(), f"/home/CS546-CLEO/models/{experiment_name}/model.pt")
            numIter += 1
            
    print(f"Average Epoch Loss: {np.mean(loss_avg)}")


