import whisperx
from pyannote.audio import Pipeline
from pydub import AudioSegment  
import os
import pandas as pd

from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv('hf_token')
print(hf_token)



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

    unique_speakers = set(word.get('speaker', 'UNKNOWN') for word in transcription)

    # Count total number of unique speakers
    total_speakers = len(unique_speakers)

    speaker_dict = {speaker: [] for speaker in unique_speakers}



    audio = AudioSegment.from_mp3(audio_file_path)
    for i in range(len(transcription)):
        sentence = transcription[i]

        text = sentence["text"]
        speaker = sentence["speaker"]
        start = sentence["start"] 
        end = sentence["end"] 

        speaker_dict[speaker].append(f"segment_{speaker}_{i}.wav")

        segment =  audio[start  * 1000 :end  * 1000]
        segment_filename = os.path.join("sub", f"segment_{speaker}_{i}.wav")
        segment.export(segment_filename, format="wav")
        
        Row = {"file" : f"sub/segment_{speaker}_{i}.wav","speaker": speaker, "start":start, "end":end,"utterance":text}
        df.loc[len(df)] = Row  # detailed frame
        text = ""
        startList = []
        for k in range(len(sentence["words"])):

            word = sentence["words"][k]

            text += " "+word["word"]
            try:
                speaker = word["speaker"]
                start = word["start"] 
                end = word["end"]
            except:
                
                #end = sentence["words"][k+1]["start"] 
                try:
                    print("couldn't catch timing")
                    print(word,sentence)
                    speaker = sentence["words"][k-1]["speaker"]
                    start = sentence["words"][k-1]["end"] 
                    end = sentence["words"][k+1]["start"] 
                except:
                    end = sentence["words"][k-1]["end"]
                
            try:
                timediff = sentence["words"][k+1]["start"] - end 

            except:
                timediff = 1
                

        
            


    

        # Export the combined audio to a file
        #combined_audio.export(f"CombinedAudio/{speaker}.mp3", format="mp3")
        


    df.to_excel(DetailedFrame,index = False)
    df2.to_excel(SubDetailedFrame,index = False)




#DetailedFrame
#SubDetailedFrame

                       

        


 
# Example usage:
#audio_file_path = "Audio.wav"
#transcription = transcribe_with_diarization(audio_file_path)


