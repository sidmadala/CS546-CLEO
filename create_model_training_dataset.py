from datasets import Dataset
from datasets import load_from_disk
from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from scipy.io.wavfile import write as write_wav
import numpy as np
import uuid
import tqdm
import seaborn as sns

dataset = load_from_disk("/home/CS546-CLEO/data/processed_dataset")
id_list = [str(uuid.uuid4()) for i in range(0, len(dataset))]
dataset = dataset.add_column("id", id_list)

## Save the audio files
for i in tqdm.tqdm(range(0, len(dataset))):
    file_name = "wav_samples/" + dataset[i]["id"] + ".wav"
    audio_file = np.array(dataset[i]["wav"]["audio"], dtype=np.float32)
    sampling_rate = dataset[i]["wav"]["sampling_rate"]
    write_wav(file_name, sampling_rate, audio_file)

## Save the dataset
dataset.save_to_disk("/home/CS546-CLEO/data/processed_dataset_with_uuid")