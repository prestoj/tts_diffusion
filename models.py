import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class VoiceEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout=0.1):
        super(VoiceEncoder, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.token_linear_in = nn.Linear(1000, d_model)

        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        # x: (batch_size, seq_length)
        seq_length = x.size(1)
        x = x.unsqueeze(1)  # (batch_size, 1, seq_length)
        
        # Split audio into tokens of 1000 samples
        x = x.unfold(2, 1000, 1000)  # (batch_size, 1, num_tokens, 1000)
        x = x.squeeze(1) # (batch_size, num_tokens, 1000)
        
        # Add class token and positional encoding
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (batch_size, 1, d_model)
        x = self.token_linear_in(x)  # (batch_size, num_tokens, d_model)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_tokens+1, d_model)
        x = self.positional_encoding(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Extract class token as output
        output = x[:, 0, :]  # (batch_size, d_model)
        
        return output

class TextEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout=0.1):
        super(TextEncoder, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout

        self.text_embedding = nn.Embedding(1001, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        # x: (batch_size, seq_length)
        seq_length = x.size(1)

        x = self.text_embedding(x)  # (batch_size, seq_length, d_model)
        
        x = self.positional_encoding(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        return x

class AudioLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(AudioLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, audio_tokens, voice_text_keys, voice_text_values):

        # Self-attention
        audio_tokens2 = self.self_attn(audio_tokens, audio_tokens, audio_tokens)[0]
        audio_tokens = audio_tokens + self.dropout1(audio_tokens2)
        audio_tokens = self.norm1(audio_tokens)

        # Cross-attention
        audio_tokens2 = self.cross_attn(audio_tokens, voice_text_keys, voice_text_values)[0]
        audio_tokens = audio_tokens + self.dropout2(audio_tokens2)
        audio_tokens = self.norm2(audio_tokens)

        # MLP
        audio_tokens2 = self.linear2(self.dropout(F.relu(self.linear1(audio_tokens))))
        audio_tokens = audio_tokens + self.dropout3(audio_tokens2)
        audio_tokens = self.norm3(audio_tokens)

        return audio_tokens

class AudioModel(nn.Module):
    def __init__(self, voice_encoder, text_encoder, d_model, nhead, num_layers, dropout=0.1):
        super(AudioModel, self).__init__()
        self.voice_encoder = voice_encoder
        self.text_encoder = text_encoder
        self.audio_token_linear_in = nn.Linear(1600, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.voice_keys = nn.Linear(d_model, d_model)
        self.voice_values = nn.Linear(d_model, d_model)
        self.text_keys = nn.Linear(d_model, d_model)
        self.text_values = nn.Linear(d_model, d_model)
        
        self.layers = nn.ModuleList([
            AudioLayer(d_model, nhead, dropout) for _ in range(num_layers)
        ])
        
        self.audio_token_linear_out = nn.Linear(d_model, 1600)

    def forward(self, voice_input, text_input, audio_tokens):
        # Voice encoding
        voice_encoding = self.voice_encoder(voice_input)
        voice_encoding = voice_encoding.unsqueeze(1)  # (batch_size, 1, d_model)

        # Text encoding
        text_encoding = self.text_encoder(text_input)  # (batch_size, num_text, d_model)

        # Audio token projection
        audio_token_input = self.audio_token_linear_in(audio_tokens)  # (batch_size, num_audio_tokens, d_model)
        audio_token_input = self.positional_encoding(audio_token_input)

        voice_keys = self.voice_keys(voice_encoding)
        voice_values = self.voice_values(voice_encoding)
        text_keys = self.text_keys(text_encoding)
        text_values = self.text_values(text_encoding)

        voice_text_keys = torch.cat((voice_keys, text_keys), dim=1)
        voice_text_values = torch.cat((voice_values, text_values), dim=1)

        for layer in self.layers:
            audio_token_input = layer(audio_token_input, voice_text_keys, voice_text_values)
    
        # Project audio tokens back to 1600 dimensions
        audio_token_output = self.audio_token_linear_out(audio_token_input)  # (batch_size, num_audio_tokens, 1600)
        audio_token_output = audio_token_output.view(audio_token_output.size(0), -1)  # (batch_size, num_audio_tokens * 1600)

        return audio_token_output

if __name__ == '__main__':
    voice_encoder = VoiceEncoder(d_model=384, nhead=6, num_layers=4)
    text_encoder = TextEncoder(d_model=384, nhead=6, num_layers=4)
    audio_model = AudioModel(voice_encoder, text_encoder, d_model=384, nhead=6, num_layers=4)
    
    voice_input = torch.randn(1, 1600 * 93)
    text_input = torch.randint(0, 1001, (1,29))
    audio_tokens = torch.randn(1, 123, 1600)

    # print out number of parameters in all models
    print(f"Voice encoder has {sum(p.numel() for p in voice_encoder.parameters())} parameters")
    print(f"Text encoder has {sum(p.numel() for p in text_encoder.parameters())} parameters")
    print(f"Audio model has {sum(p.numel() for p in audio_model.parameters())} parameters")
    
    output = audio_model(voice_input, text_input, audio_tokens)
