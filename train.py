import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import numpy as np
from diffusers import DDPMScheduler
from datasets import load_dataset
from tokenizer import TOKENIZER, INV_TOKENIZER, tokenize_string_random
from models import VoiceEncoder, TextEncoder, AudioModel
from diffusers.optimization import get_cosine_schedule_with_warmup


class LibriSpeechDataset(Dataset):
    def __init__(self, split):
        self.dataset = load_dataset("librispeech_asr", split=split)
        self.transcriptions = self.load_transcriptions()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        audio_array = item['audio']['array'].astype(np.float32)

        transcription = self.transcriptions.get(item['id'], '')
        transcription = tokenize_string_random(transcription, TOKENIZER)

        audio_array = torch.tensor(audio_array)

        transcription = torch.tensor(transcription, dtype=torch.long)

        return audio_array, transcription

    def load_transcriptions(self):
        transcriptions = {}
        with open('transcriptions.txt', 'r') as f:
            for line in f:
                key, value = line.strip().split(': ', 1)
                transcriptions[key] = value
        return transcriptions

def pad_collate(batch):
    lengths = [sample[0].size()[0] for sample in batch]
    max_length = max(lengths)
    
    padded_batch = torch.zeros(len(batch), max_length)
    for i, sample in enumerate(batch):
        padded_batch[i, :lengths[i]] = sample[0]
    
    return padded_batch, lengths

if __name__ == '__main__':

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    voice_encoder = VoiceEncoder(d_model=384, nhead=6, num_layers=4)
    text_encoder = TextEncoder(d_model=384, nhead=6, num_layers=4)
    audio_model = AudioModel(voice_encoder, text_encoder, d_model=384, nhead=6, num_layers=4)

    BATCH_SIZE = 128
    PEAK_LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100

    class MyDataset(data.Dataset):
        def __init__(self):
            self.data = np.random.randn(1000, 1600).astype(np.float32)
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
        
    dataset = MyDataset("train.clean.100")

    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(audio_model.parameters() + voice_encoder.parameters() + text_encoder.parameters(), lr=PEAK_LEARNING_RATE)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=1000,
        num_training_steps=(len(train_dataloader) * NUM_EPOCHS),
    )