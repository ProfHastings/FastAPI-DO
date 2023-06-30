from langchain import PromptTemplate
from langchain.retrievers import PineconeHybridSearchRetriever
import torch
import pinecone
from pinecone_text.sparse import SpladeEncoder
import os
import tiktoken
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
#    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import asyncio
from langchain.embeddings import OpenAIEmbeddings
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from pydantic import BaseModel, ValidationError
from langchain.callbacks.base import AsyncCallbackHandler
import uvicorn
from langchain.text_splitter import CharacterTextSplitter
from typing import List

#init of global gpt-4 model, gpt-3.5-turbo model and OpenAI tokenizer
gpt4_maxtokens = 8192
response_maxtokens = 2048
callback_handler = [StreamingStdOutCallbackHandler()]
gpt4 = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=2048, streaming=True, callbacks=callback_handler)
gpt35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=5, max_retries=20)
gptdataquery = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=256)
tokenizer = tiktoken.encoding_for_model("gpt-4")
text_splitter = CharacterTextSplitter(chunk_size=1000, separator="\n")

#template for final analysis prompt
analysis_system_message = SystemMessage(content="Du bist ein im österreichischen Recht erfahrener Anwalt.")
analysis_template_string = """
Deine Aufgabe ist es die folgende Frage zu beantworten: 
"{question}"
Um die Frage zu beantworten hast du die folgenden Entscheidungen des Österreichischen Obersten Gerichtshofes zur Verfügung:
"{sources}"
Du bist Rechtsanwältsanwärter in einer Anwaltskanzlei. Schreibe einen sehr ausführlichen und detaillierten ersten Entwurf für ein Rechtsgutachten. Zuerst klärst du welche Rechtsfrage sich stellt.
Dann erörterst Du die Rechtsfrage abstrakt und nimmst dabei jeweils im Zuge der Erörterung einzelner Fragen auch auf Fälle Bezug, die vom Obersten Gerichtshof bereits entschieden wurden und gib dazu die Fallzahl an.
Vermeide aber eine bloße Auflistung der Fälle.
Danach wendest Du die so beschriebene Rechtslage auf den Fall an.
Schließlich gib an, wie die Frage deines Erachtens zu lösen ist. Gib immer auch an, wenn du dich in deinen Ausführungen unsicher fühlst.Falls die Lösung nicht eindeutig ist gib an, wie die wahrscheinlichere Lösung lautet. Gib auch an, welche zusätzlichen Sachverhaltselemente hilfreich wären.
Zum Schluss liste bis zu fünf der wichtigsten Entscheidungen und bis zu fünf der wichtigsten Literaturzitate auf, die du in den Entscheidungen findest.
"""
analysis_template = PromptTemplate.from_template(analysis_template_string)

#template for ranking prompt
ranking_system_message = SystemMessage(content="Du bist ein im österreichischen Recht erfahrener Anwalt. Deine Antwort besteht immer nur aus einer Zahl von 1 bis 10")
ranking_template_string = """
Deine Aufgabe ist es zu bewerten wie relevant ein Abschnitt einer Gerichtsentscheidung ist um die folgende Frage zu beantworten: 
"{question}"
Der Abschnitt lautet:
"{case}"
Skaliere die Relevanz auf einer Skala von 1 bis 10 und antworte mit dieser Zahl
"""
ranking_template = PromptTemplate.from_template(ranking_template_string)

#template for database query prompt
dataquery_system_message = SystemMessage(content="You are a lawyer and only write the answers to questions and only answer in german language")
dataquery_template_string = """
Generate a paragraph of a court decision from the U.S. Supreme Court Database in german language about the topic of:
"{question}"
Do not mention the US supreme court and write as if this is a real case
"""

dataquery_template = PromptTemplate.from_template(dataquery_template_string)

#takes Vector Database results, returns highest number of results with sources as string that fit in max_tokens OpenAI
def fill_tokens(results, max_tokens):
    sources = ""
    nr_sources = 0
    token_count = 0
    
    for result in results:
        new_text = f"Inhalt: {result.page_content}\nQuelle: {result.short_source}\n"
        new_tokens = list(tokenizer.encode(new_text))
        new_token_count = len(new_tokens)
        if token_count + new_token_count > max_tokens:
            break
        sources += new_text
        token_count += new_token_count
        nr_sources += 1
    print(f"Using {nr_sources} chunks for analysis")
    return sources

#initializes Pinecone index to make database requests
def get_index():
    env = "eu-west4-gcp"
    api_key = "953b2be8-0621-42a1-99db-8480079a9e23"

    pinecone.init(environment=env, api_key=api_key)
    return pinecone.Index("justiz-openai-full")


#loads dense and sparse encoder models and returns retriever to send requests to the database
def get_retriever():
    index = get_index()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dense_encoder = OpenAIEmbeddings(model="text-embedding-ada-002")
    sparse_encoder = SpladeEncoder(device=device)
    retriever = PineconeHybridSearchRetriever(embeddings=dense_encoder, sparse_encoder=sparse_encoder, index=index, top_k=75, alpha=0.99899) #lower alpha - more sparse
    return retriever

def hybrid_scale(dense, sparse, alpha: float):
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hsparse = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    hdense = [v * alpha for v in dense]
    return hdense, hsparse

def hybrid_query(question, top_k, alpha):
    index = get_index()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sparse_encoder = SpladeEncoder(device=device)
    dense_encoder = OpenAIEmbeddings(model="text-embedding-ada-002")
    sparse_vec = sparse_encoder.encode_documents([question])[0]
    dense_vec = dense_encoder.embed_documents([question])[0]
    dense_vec, sparse_vec = hybrid_scale(dense_vec, sparse_vec, alpha)

    result = index.query(
      vector=dense_vec,
      sparse_vector=sparse_vec,
      top_k=top_k,
    )
    return result

results = hybrid_query("Alfred ist Mechaniker in einer Autowerkstatt. Er hat die letzten Tage immer wieder viele Überstunden gearbeitet. Heute kam er zu spät zur Arbeit weil er verschlafen hat. Kann er gekündigt oder entlassen werden?", 10, 0.99899)
id_list = [match['id'] for match in results['matches']]

print(id_list)