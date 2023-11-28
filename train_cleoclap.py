from transformers import AutoTokenizer, AutoModelForCausalLM
from imagebind import data
import torch
import typing
from typing import List
import logging
import torch.nn as nn
from cleo.cleoCLAP import CLEOClap
from datasets import load_from_disk, load_dataset
import tqdm
import numpy as np
import os
from transformers import ClapProcessor, ClapModel
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import os

EPOCHS = 100
BATCH_SIZE = 8
experiment_name = "CLAP_model"

try:  
    os.mkdir(f"/home/CS546-CLEO/models/{experiment_name}")  
except OSError as error:  
    print(error)   

writer = SummaryWriter(log_dir=f"/home/CS546-CLEO/runs/{experiment_name}")
clapModelVr = "laion/clap-htsat-unfused"
dataset = load_dataset("patrickvonplaten/librispeech_asr_self_contained")
dataset = dataset["train.clean.100"]
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

print("Dataset Loaded...")

cleo_model = CLEOClap(
        llm_model_path = "/home/models/Llama-2-7b-hf",
        audio_features = 512, # 1024 if ImageBind,
        host_llm_on_cuda = True,
        audio_gpu = "cuda:1",
        clapModelVr = clapModelVr
)

print("Models Loaded...")

instruction = """Repeat back the information that you see below:
<wav>

Information:
"""
cleoDataset = CLEODataset(dataset, instruction, clapModelProcessor)
train_dataloader = DataLoader(cleoDataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)

optimizer = torch.optim.Adam(cleo_model.parameters(), lr=0.0001)
print("Begin Training...")

numIter = 0
for epoch in range(1,EPOCHS):
    loss_avg = []
    with tqdm.tqdm(train_dataloader, unit="batch", total=len(train_dataloader)) as tepoch:
        for batch_idx, preBatch in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch} / Batch {batch_idx}")
            ## Create batch
            batch = {
                "instructions": preBatch[0],
                "audio_array": preBatch[1],
                "labels": preBatch[2]
            }
            output = cleo_model(batch)
            loss = output.loss
            loss_val = loss.item()
            writer.add_scalar("Loss/train", loss_val, numIter)
            loss_avg.append(loss_val)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tepoch.set_postfix(loss = loss_val)
            numIter += 1
            if numIter % 50 == 0:
                torch.save(model, f"/home/CS546-CLEO/models/{experiment_name}/model.pt")
            
    print(f"Average Epoch Loss: {np.mean(loss_avg)}")
