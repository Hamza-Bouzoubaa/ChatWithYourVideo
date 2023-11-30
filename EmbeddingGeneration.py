from langchain.vectorstores import FAISS
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
import os

load_dotenv()


def SplitResumeText(Text,chunk_size=5000,chunk_overlap = 00,separators = ["\n\n"]):

    rec_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size= chunk_size,
    chunk_overlap = chunk_overlap,
    length_function=len,
    separators = separators

    )

    chunks = rec_text_splitter.split_text(Text)

    return chunks

def ReadFile(FilePath):
    # Get the file extension
    file_extension = FilePath.split('.')[-1].lower()

    
    if file_extension == 'txt':
        with open(FilePath, 'r', encoding='utf-8') as file:
            raw_text = file.read()
            
    

    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    return raw_text




def GenerateEmbeddings(FilePath):  #Works with PDF DOCX and TXT

    Transcription = ReadFile(FilePath)  #Extracting raw text from Resume
    
    
    TranscriptionSections = SplitResumeText(Transcription)  # splitting the raw text into sections

    embeddings =  AzureOpenAIEmbeddings(model = "text-embedding-ada-002")  # Setting up OpenAI embedding

    
    dbTranscription  = FAISS.from_texts(TranscriptionSections,embeddings)      # Generating and storing Resume embeddings 
    

    return dbTranscription


def QuerySimilaritySearch(Query,db,k=1):
    similar = db.similarity_search(Query,k)
    page_content_array = [doc.page_content for doc in similar]
    return page_content_array

