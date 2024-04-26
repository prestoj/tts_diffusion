from datasets import load_dataset

dataset = load_dataset("librispeech_asr", split="train.clean.100")
print(dataset[0]['audio']['array'].min(), dataset[0]['audio']['array'].max())