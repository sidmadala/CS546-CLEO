from transformers import AutoTokenizer, AutoModelForCausalLM
from imagebind import data
import torch
import typing
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

class Cleo(torch.nn.Module):
    def __init__(self, model_path:str, device_map:str="auto", audio_device:str="cuda:0"):
        super().__init__()
        ## Need to load the audio embedding model
        self.imagebind = imagebind_model.imagebind_huge(pretrained=True, device=audio_device)

        ## Need to load the LLama pre-trained model
        self.llama = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map
        )

    def __encode_audio__(self, audio_path: list):
        inputs = {
            ModalityType.AUDIO: data.load_and_transform_audio_data(audio_path, device, sample_rate=24000)
        }
        pass

