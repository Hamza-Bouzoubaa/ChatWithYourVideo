import whisperx
from pyannote.audio import Pipeline
from pydub import AudioSegment  
import os
import pandas as pd

from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv('hf_token')


def transcribe_with_diarization(audio_file_path,OutputFile, model_type="large-v2", device="cpu", hf_token=hf_token):
    # Step 1: Load the WhisperX model with float32 compute type
    model = whisperx.load_model(model_type, device, compute_type="float32")

    # Step 2: Load the audio file
    audio = whisperx.load_audio(audio_file_path)

    # Step 3: Transcribe audio
    result = model.transcribe(audio)

    # Step 4: Load an alignment model for diarization
    align_model, align_metadata = whisperx.load_align_model(language_code=result["language"], device=device)

    # Step 5: Align Whisper output
    result = whisperx.align(result["segments"], align_model, align_metadata, audio, device, return_char_alignments=False)

    # Step 6: Perform speaker diarization
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize_model(audio)

    # Step 7: Assign speaker IDs to word-level segments
    result = whisperx.assign_word_speakers(diarize_segments, result)

    transcription = result["segments"]
    
    text = transcription[0]["speaker"]+": \n" 
    for i in range(len(transcription)):
    
    
    
        if (i>0  and (transcription[i-1]["speaker"] != transcription[i]["speaker"])):
            text += "\n\n"+transcription[i]["speaker"]+": \n"
        text  += transcription[i]["text"]
    
      
    # Write the text to the file
    with open(OutputFile, "w", encoding="utf-8") as file:
        file.write(text)
    
    print(f"Text has been saved to {OutputFile}")


    return transcription

transcribe_with_diarization("audio.wav","transcription.txt")