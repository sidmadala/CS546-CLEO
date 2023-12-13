from transformers import AutoTokenizer, AutoModelForCausalLM
from imagebind import data
import torch
import typing
from typing import List
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import logging
import torch.nn as nn
from cleo.cleoImageBind import CLEOImageBind
from datasets import load_from_disk, load_dataset
import tqdm
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

EPOCHS = 100
BATCH_SIZE = 8
experiment_name = "ImageBind_model_short_sentences"
try:  
    os.mkdir(f"/home/CS546-CLEO/models/{experiment_name}")  
except OSError as error:  
    print(error)   

writer = SummaryWriter(log_dir=f"/home/CS546-CLEO/runs/{experiment_name}")

#dataset = load_from_disk("/home/CS546-CLEO/data/processed_dataset_with_uuid")
dataset = load_dataset("patrickvonplaten/librispeech_asr_self_contained", split="train.clean.100")
print("Dataset Loaded...")

ib_model = imagebind_model.imagebind_huge(pretrained=True)
cleo_model = CLEOImageBind(
    llm_model_path = "/home/models/Llama-2-7b-hf",
    audio_features = 1024, # 1024 if ImageBind,
    imageBind_model = ib_model,
    host_llm_on_cuda = True
)
print("Models Loaded...")

sentences = dataset["text"]
sentence_length = []
for sentence in tqdm.tqdm(sentences):
    sentence_length.append(len(cleo_model.llm_tokenizer.encode(sentence, add_special_tokens=False)))
sentence_length = np.array(sentence_length)
dataset = dataset.add_column("sentence_length", sentence_length)
dataset = dataset.select(np.where(sentence_length < 20)[0])
print("Dataset modified...")


from torch.utils.data import Dataset, DataLoader
from scipy.io.wavfile import write as write_wav
import uuid
class CLEODataset(Dataset):
    def __init__(self, dataset, instruction):
        self.dataset = dataset
        self.instruction = instruction

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ## Create the label
        label = self.dataset[idx]["text"].lower()
        
        ## Save the audio
        file_name = f"/home/CS546-CLEO/wav_samples/{str(uuid.uuid4())}.wav"
        audio_file = np.array(self.dataset[idx]["audio"]["array"], dtype=np.float32)
        write_wav(file_name, 16000, audio_file)

        return self.instruction, file_name, label

instruction = """Repeat back the information that you see below:
<wav>

Information:
"""
cleoDataset = CLEODataset(dataset, instruction)
train_dataloader = DataLoader(cleoDataset, batch_size=BATCH_SIZE, shuffle=True)

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
                "instructions": list(preBatch[0]),
                "audio_paths": [[each] for each in list(preBatch[1])],
                "labels": list(preBatch[2])
            }
            output = cleo_model(batch)
            loss = output.loss
            loss_val = loss.item()
            writer.add_scalar("Loss/train", loss_val, numIter)
            loss_avg.append(loss_val)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            for each in preBatch[1]:
                os.remove(each)

            tepoch.set_postfix(loss = loss_val)
            if numIter % 50 == 0:
                ## Need to generate an example
                instruction, audioPath, label = cleoDataset.__getitem__(0)
                valOutput = cleo_model.generate(instruction, audioPath, label)
                decodedOutput = cleo_model.llm_tokenizer.decode(valOutput[0], skip_special_tokens=True)
                saveInfo = f"Model Output:\n{decodedOutput}\n\nActual Label:\n{label}"
                writer.add_text("Model Output", saveInfo, numIter)

                torch.save(cleo_model.state_dict(), f"/home/CS546-CLEO/models/{experiment_name}/model.pt")
            numIter += 1
    print(f"Average Epoch Loss: {np.mean(loss_avg)}")
