from transformers import AutoTokenizer, AutoModelForCausalLM
from imagebind import data
import torch
import typing
from typing import List
import logging
import torch.nn as nn
from cleo.cleoImageBind import CLEOImageBind
from datasets import load_from_disk
import tqdm
import numpy as np
import laion_clap

EPOCHS = 100
BATCH_SIZE = 2

dataset = load_from_disk("/home/CS546-CLEO/data/processed_dataset_with_uuid")
clap_model = laion_clap.CLAP_Module(enable_fusion=False)
clap_model.load_ckpt()
cleo_model = CLEOImageBind(
    llm_model_path = "/home/models/Llama-2-7b-hf",
    audio_features = 1024, # 1024 if ImageBind, 512 if CLAP? 
        # See both https://github.com/LAION-AI/CLAP/blob/817041c079af560fa2c610287c68c7c97ace50b6/src/laion_clap/clap_module/model.py#L28 
        # and https://github.com/LAION-AI/CLAP/blob/817041c079af560fa2c610287c68c7c97ace50b6/src/laion_clap/clap_module/model.py#L532    clap_model = clap_model,
    host_llm_on_cuda = True
)

from torch.utils.data import Dataset, DataLoader
class CLEODataset(Dataset):
    def __init__(self, dataset, instruction):
        self.dataset = dataset
        self.instruction = instruction

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ## Create the label
        label = ""
        for node in self.dataset[idx]["graph"]["elements"]["nodes"]:
            label += node["data"]["value"].replace('"', "") + " | "
        label = label[:-3]

        ## Get the audio
        audio_path = f"/home/CS546-CLEO/wav_samples/{self.dataset[idx]['id']}.wav"
        return self.instruction, audio_path, label

instruction = """Convert the following information to a graph of triplets:
<wav>

Triples:
"""
cleoDataset = CLEODataset(dataset, instruction)
train_dataloader = DataLoader(cleoDataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = torch.optim.Adam(cleo_model.parameters(), lr=0.01)

for epoch in range(1,EPOCHS):
    loss_avg = []
    with tqdm.tqdm(train_dataloader, unit="batch", total=len(train_dataloader)) as tepoch:
        for batch_idx, preBatch in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch} / Batch {batch_idx}")
            ## Create batch
            batch = {
                "instructions": list(preBatch[0]),
                "audio_paths": [[each] for each in list(preBatch[1])],
                "labels": list(preBatch[2])
            }
            output = cleo_model(batch)
            loss = output.loss
            loss_val = loss.item()
            loss_avg.append(loss_val)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            tepoch.set_postfix(loss = loss_val)
    print(f"Average Epoch Loss: {np.mean(loss_avg)}")
