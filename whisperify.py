from datasets import load_dataset
import whisper
import numpy as np

# Load the dataset
dataset = load_dataset("librispeech_asr", split="train.clean.100")

# Load the Whisper model
model = whisper.load_model("tiny.en")

text_data = {}

for i in range(len(dataset)):

    # Get the audio array from the dataset
    audio_array = dataset[i]['audio']['array']

    # Convert the audio array to the appropriate data type
    audio_array = audio_array.astype(np.float32)

    # Pad or trim the audio array to the desired length
    desired_length = 30 * 16000  # Assuming a desired length of 30 seconds at 16kHz sampling rate
    padded_array = whisper.pad_or_trim(audio_array, desired_length)

    # Run the Whisper model on the padded/trimmed audio array
    result = model.transcribe(padded_array)

    # result['text']
    text_data[dataset[i]['id']] = result['text']

    if i % 100 == 0:
        print(f"Processed {i} samples")

# save the text data to a file
with open('transcriptions.txt', 'w') as f:
    for key, value in text_data.items():
        f.write(f"{key}: {value}\n")