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
import random
import logging


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
        
        # Select a random snippet of the audio
        audio_length = len(audio_array)
        if audio_length <= 16000:
            audio_snippet = audio_array
        else:
            snippet_length = random.randint(16000, audio_length)
            start_idx = random.randint(0, audio_length - snippet_length)
            end_idx = start_idx + snippet_length
            audio_snippet = audio_array[start_idx:end_idx]
        
        audio_array = torch.tensor(audio_array)
        audio_snippet = torch.tensor(audio_snippet)
        transcription = torch.tensor(transcription, dtype=torch.long)

        return audio_array, transcription, audio_snippet

    def load_transcriptions(self):
        transcriptions = {}
        with open('transcriptions.txt', 'r') as f:
            for line in f:
                key, value = line.strip().split(': ', 1)
                transcriptions[key] = value
        return transcriptions

def pad_collate(batch):
    audio_lengths = [sample[0].size(0) for sample in batch]
    max_audio_length = max(audio_lengths)
    
    audio_padding_size = 1000 - (max_audio_length % 1000)  # subtracting remainder from 1000
    padded_audio_length = max_audio_length + audio_padding_size
    padded_audio = torch.zeros(len(batch), padded_audio_length, dtype=torch.float32)
    for i, (audio, _, _) in enumerate(batch):
        padded_audio[i, :audio_lengths[i]] = audio
    padded_audio = padded_audio.view(len(batch), -1, 1000)

    text_lengths = [sample[1].size(0) for sample in batch]
    max_text_length = max(text_lengths)
    padded_text = torch.zeros(len(batch), max_text_length, dtype=torch.long)
    for i, (_, text, _) in enumerate(batch):
        padded_text[i, :text_lengths[i]] = text

    snippet_lengths = [sample[2].size(0) for sample in batch]
    max_snippet_length = max(snippet_lengths)
    
    snippet_padding_size = 1000 - (max_snippet_length % 1000)  # subtracting remainder from 1000
    padded_snippet_length = max_snippet_length + snippet_padding_size
    padded_snippet = torch.zeros(len(batch), padded_snippet_length, dtype=torch.float32)
    for i, (_, _, snippet) in enumerate(batch):
        padded_snippet[i, :snippet_lengths[i]] = snippet
    padded_snippet = padded_snippet.view(len(batch), -1, 1000)
    
    # calculate lengths after splitting into 1000-sample tokens
    audio_lengths = [(audio_length - 1) // 1000 + 1 for audio_length in audio_lengths]
    snippet_lengths = [(snippet_length - 1) // 1000 + 1 for snippet_length in snippet_lengths]

    return (padded_audio, padded_text, padded_snippet), (audio_lengths, text_lengths, snippet_lengths)

if __name__ == '__main__':
    logging.basicConfig(filename='training.log', level=logging.INFO, format='%(message)s')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    voice_encoder = VoiceEncoder(d_model=384, nhead=6, num_layers=4).to(device)
    text_encoder = TextEncoder(d_model=384, nhead=6, num_layers=4).to(device)
    audio_model = AudioModel(voice_encoder, text_encoder, d_model=384, nhead=6, num_layers=4, num_train_timesteps=1000).to(device)

    ema_audio_model = AudioModel(voice_encoder, text_encoder, d_model=384, nhead=6, num_layers=4, num_train_timesteps=1000).to(device)
    ema_voice_encoder = VoiceEncoder(d_model=384, nhead=6, num_layers=4).to(device)
    ema_text_encoder = TextEncoder(d_model=384, nhead=6, num_layers=4).to(device)
    ema_audio_model.load_state_dict(audio_model.state_dict())
    ema_voice_encoder.load_state_dict(voice_encoder.state_dict())
    ema_text_encoder.load_state_dict(text_encoder.state_dict())

    voice_encoder = nn.DataParallel(voice_encoder)
    text_encoder = nn.DataParallel(text_encoder)
    audio_model = nn.DataParallel(audio_model)

    ema_audio_model = nn.DataParallel(ema_audio_model)
    ema_voice_encoder = nn.DataParallel(ema_voice_encoder)
    ema_text_encoder = nn.DataParallel(ema_text_encoder)

    BATCH_SIZE = 128
    PEAK_LEARNING_RATE = 1e-3
    NUM_EPOCHS = 100
    EMA_DECAY = 0.9998  # EMA decay rate
    
    dataset = LibriSpeechDataset("train.clean.100")

    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)

    optimizer = torch.optim.AdamW(
        list(audio_model.parameters()) + list(voice_encoder.parameters()) + list(text_encoder.parameters()),
        lr=PEAK_LEARNING_RATE
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=1000,
        num_training_steps=(len(train_dataloader) * NUM_EPOCHS),
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    voice_encoder.train()
    text_encoder.train()
    audio_model.train()

    for epoch in range(NUM_EPOCHS):
        for i_step, (batch, batch_lengths) in enumerate(train_dataloader):
            audio_data, text_data, audio_snippets = batch
            audio_lengths, text_lengths, snippet_lengths = batch_lengths

            audio_mask = torch.arange(audio_data.size(1)).expand(len(audio_lengths), audio_data.size(1)) >= torch.tensor(audio_lengths).unsqueeze(1)
            text_mask = torch.arange(text_data.size(1)).expand(len(text_lengths), text_data.size(1)) >= torch.tensor(text_lengths).unsqueeze(1)
            snippet_mask = torch.arange(audio_snippets.size(1)).expand(len(snippet_lengths), audio_snippets.size(1)) >= torch.tensor(snippet_lengths).unsqueeze(1)

            audio_data = audio_data.to(device)
            text_data = text_data.to(device)
            audio_snippets = audio_snippets.to(device)
            audio_mask = audio_mask.to(device)
            text_mask = text_mask.to(device)
            snippet_mask = snippet_mask.to(device)

            timesteps = torch.randint(
                0, 1000, (audio_data.shape[0],), device=device,
                dtype=torch.int64
            )

            noise = torch.randn(audio_data.shape, device='cuda')

            noisy_audio = noise_scheduler.add_noise(audio_data, noise, timesteps)

            noise_predicted = audio_model(audio_snippets, text_data, audio_data, timesteps, snippet_mask, text_mask, audio_mask)

            loss = F.mse_loss(noise_predicted, noise)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Update EMA models
            for param, ema_param in zip(audio_model.module.parameters(), ema_audio_model.module.parameters()):
                ema_param.data.mul_(EMA_DECAY).add_(param.data, alpha=1 - EMA_DECAY)
            for param, ema_param in zip(voice_encoder.module.parameters(), ema_voice_encoder.module.parameters()):
                ema_param.data.mul_(EMA_DECAY).add_(param.data, alpha=1 - EMA_DECAY)
            for param, ema_param in zip(text_encoder.module.parameters(), ema_text_encoder.module.parameters()):
                ema_param.data.mul_(EMA_DECAY).add_(param.data, alpha=1 - EMA_DECAY)

            logging.info(f"Epoch {epoch}, Step {i_step}, Loss: {loss.item()}")

        # Save model
        torch.save(ema_audio_model.module.state_dict(), f"models/ema_audio_model.pt")
        torch.save(ema_voice_encoder.module.state_dict(), f"models/ema_voice_encoder.pt")
        torch.save(ema_text_encoder.module.state_dict(), f"models/ema_text_encoder.pt")
        
        if epoch % 10 == 0:
            torch.save(ema_audio_model.module.state_dict(), f"models/ema_audio_model_epoch_{epoch}.pt")
            torch.save(ema_voice_encoder.module.state_dict(), f"models/ema_voice_encoder_epoch_{epoch}.pt")
            torch.save(ema_text_encoder.state_dict(), f"models/ema_text_encoder_epoch_{epoch}.pt")

