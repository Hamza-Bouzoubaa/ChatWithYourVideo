from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate

from Transcription import transcribe_with_diarization
from EmbeddingGeneration import GenerateEmbeddings, QuerySimilaritySearch

from dotenv import load_dotenv
import os
load_dotenv()

template = """
    You are a world class problem solver.
    I will share with you a part of a video transcription and you will have to answer to the question asked using knowledge inside that that transcription.

    1/ Response should be very clear and very easy to understand.
    2/ Response should be consice
    3/ You should not make things up and only say fact based things

    Here is the Transcription:
    {transcription}

    Here is the question:
    {question}


"""


llm = AzureChatOpenAI(model_name ="gpt-35-turbo-16k" ,temperature=0)
prompt = PromptTemplate(
    input_variables=["transcription","question"],
    template=template

) 

llm_chain = LLMChain(llm=llm, prompt=prompt)


AudioFile = "audio.wav"
OutputFile = "Transcription.txt"

File = transcribe_with_diarization(AudioFile,OutputFile)
db = GenerateEmbeddings(File)

def generate_response(Question,db):
    Transcription = QuerySimilaritySearch(Question,db)
    response = llm_chain.run(question = Question, transcription = Transcription)
    return response

    