from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

from datasets import Dataset
from datasets import load_from_disk
from scipy.io.wavfile import write as write_wav
import numpy as np
import uuid
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt



dataset = load_from_disk("/home/john/Desktop/adv-nlp/processed_dataset")
id_list = [str(uuid.uuid4()) for i in range(0, len(dataset))]
dataset = dataset.add_column("id", id_list)

import os
# print(os.getcwd())
# raise Exception

## Save the audio files
for i in tqdm.tqdm(range(0, len(dataset))):
    file_name = "wav_samples/" + dataset[i]["id"] + ".wav"
    audio_file = np.array(dataset[i]["wav"]["audio"], dtype=np.float32)
    sampling_rate = dataset[i]["wav"]["sampling_rate"]
    write_wav(file_name, sampling_rate, audio_file)


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)
print("model loaded")

def get_text_and_audio_lists(index_list):
    text_list = []
    audio_list = []
    for index in index_list:
        text_list.append(dataset[int(index)]["text"])
        audio_list.append("wav_samples/" + dataset[int(index)]["id"] + ".wav")
    return text_list, audio_list

text_list, audio_list = get_text_and_audio_lists(np.random.randint(0, len(dataset), 3))
# inputs = {
#     ModalityType.AUDIO: imagebind.data.load_and_transform_audio_data(audio_list, device, sample_rate=24000),
#     ModalityType.TEXT: imagebind.data.load_and_transform_text_data(text_list, device),
# }

# Load data
inputs = {
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_list, device, sample_rate=24000),
    ModalityType.TEXT: data.load_and_transform_text(text_list, device),
}

with torch.no_grad():
    embeddings = model(inputs)

## Computes the pairwise distance between the audio & text
print(
    "Audio X Text:",
    torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1)
)

axt = torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1).cpu().numpy()
# np.save('axt.npy', axt)

with open('axt.npy', 'wb') as wb:
    np.save(wb, axt)

# axt = axt.numpy()
sns.set()
plt.figure(figsize = (8, 8))
sns.heatmap(axt, annot=True, cmap='viridis', cbar_kws={'label': 'Embedding Similartiries'})
plt.title('Embedding Similarities Heatmap')
plt.show()