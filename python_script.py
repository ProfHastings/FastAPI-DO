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
import gc
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import asyncio
from pydantic import BaseModel, ValidationError
from langchain.callbacks.base import AsyncCallbackHandler

#init of global gpt-4 model, gpt-3.5-turbo model and OpenAI tokenizer
gpt4_maxtokens = 8192
response_maxtokens = 2048
callback_handler = [StreamingStdOutCallbackHandler()]
gpt4 = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=2048, streaming=True, callbacks=callback_handler)
gpt35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=5, max_retries=20)
gptdataquery = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=256)
tokenizer = tiktoken.encoding_for_model("gpt-4")


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

#template for pruning prompt
pruning_system_message = SystemMessage(content="Du bist ein im österreichischen Recht erfahrener Anwalt. Deine Antwort besteht immer nur aus einem Wort")
pruning_template_string = """
Deine Aufgabe ist es zu evaluieren ob ein Abschnitt einer Gerichtsentscheidung relevant sein könnte um die folgende Frage zu beantworten: 
"{question}"
Der Abschnitt lautet:
"{case}"
Falls du dir sicher bist, dass der Abschnitt die Rechtsfrage nicht betrifft, antworte mit Nein. Ansonsten antworte mit Ja.
"""
pruning_template = PromptTemplate.from_template(pruning_template_string)

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
#dataquery_system_message = SystemMessage(content="Du bist ein im österreichischen Recht erfahrener Anwalt. Du antwortest nur genau mit dem was von dir gefragt ist und ausführlich.")
#dataquery_template_string = """
#Ein Klient kommt zu dir mit der folgenden Frage.
#"{question}"
#Schreibe eine Liste mit den wichtigsten rechtlichen Fragen die sich zu dieser Situation stellen. Verwende die genaue juristische Terminologie.
#"""
#dataquery_system_message = SystemMessage(content="Du bist ein im österreichischen Recht erfahrener Anwalt. Du schreibst nur die Antwort auf Fragen.")
#dataquery_template_string = """
#Du hast die folgende informell formulierte Frage:"
#"{question}" 
#Formuliere die Frage ausführlich um, so wie sie in einer gerichtlichen Entscheidung stehen würde.
#"""
#dataquery_system_message = SystemMessage(content="Du bist ein im österreichischen Recht erfahrener Anwalt. Du antwortest nur genau mit dem was von dir gefragt ist und ausführlich.")
#dataquery_template_string = """
#Ein Klient kommt zu dir mit der folgenden Frage.
#"{question}"
#Identifiziere die rechtlichen Parteien in diesem Fall und nutze dies um eine Liste mit den drei wichtigsten Fragen die sich zu dieser Situation zu schreiben. Nenne die Parteien innerhalt der Liste nur mit ihren rechtlichen Rollen ohne persönliche Namen und schreibe nur die Elemente der Liste ohne Erläuterung. Jedes Element der Liste sollte formuliert sein um die Parteien und Rechtsinteraktion zu referenzieren. Jeder der Punkte soll verständlich sein ohne die Originalfrage gelesen zu haben und die Rechtslage enthalten. Verwende die genaue juristische Terminologie.
#"""

dataquery_template = PromptTemplate.from_template(dataquery_template_string)

#takes Vector Database results, returns highest number of results with sources as string that fit in max_tokens OpenAI
def fill_tokens(results, max_tokens):
    sources = ""
    nr_sources = 0
    token_count = 0
    
    for result in results:
        new_text = f"Inhalt: {result.page_content}\nQuelle: {result.metadata['short_source']}\n"
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
    api_key = "953b2be8-0621-42a1-99db-8480079a9e23"
    env = "eu-west4-gcp"
    pinecone.init(api_key=api_key, environment=env)
    return pinecone.Index("justiz-openai")

#loads dense and sparse encoder models and returns retriever to send requests to the database
def get_retriever():
    index = get_index()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dense_encoder = OpenAIEmbeddings(model="text-embedding-ada-002")
    sparse_encoder = SpladeEncoder(device=device)
    retriever = PineconeHybridSearchRetriever(embeddings=dense_encoder, sparse_encoder=sparse_encoder, index=index, top_k=75, alpha=0.99899) #lower alpha - more sparse
    return retriever

#(next three functions) uses async api calls to prune cases based on relevance
async def async_prune(case, question):
    pruning_userprompt = pruning_template.format(case=case.page_content, question=question)
    pruning_user_message = HumanMessage(content=pruning_userprompt)
    relevance = await gpt35._agenerate([pruning_system_message, pruning_user_message])
    print(case.page_content, "\n", relevance.generations[0].text)
    return (case, relevance.generations[0].text)

async def prune_concurrently(cases, question):
    tasks = [async_prune(case, question) for case in cases]
    results = await asyncio.gather(*tasks)
    #print(results)
    return [case for case, relevance in results if relevance == 'Ja.']

def prune_cases(results, question):
    pruned_results = asyncio.run(prune_concurrently(results, question))
    return pruned_results

#(next three functions) uses async api calls to rank chunks from 1-10 based on relevance
async def async_rank(case, question, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            ranking_userprompt = ranking_template.format(case=case.page_content, question=question)
            ranking_user_message = HumanMessage(content=ranking_userprompt)
            relevance = await gpt35._agenerate([ranking_system_message, ranking_user_message])
            relevance_score = float(relevance.generations[0].text)
            print(case.page_content, "\n", relevance_score, "\n")
            return (case, relevance_score)
        except ValueError:
            print(f"Attempt {attempt + 1} failed, did not return ranking number")
    print(f"All {max_attempts} attempts failed. Returning default relevance score of 1.")
    return (case, 1)

async def rank_concurrently(cases, question):
    tasks = [async_rank(case, question) for case in cases]
    results = await asyncio.gather(*tasks)
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    sum_of_relevance = sum(relevance for _, relevance in results)
    return [case for case, _ in sorted_results], sum_of_relevance

def rank_cases(results, question):
    ranked_results, sum_of_relevance = asyncio.run(rank_concurrently(results, question))
    print(f"Average relevance score: {sum_of_relevance/len(ranked_results)}")
    return ranked_results

#rephrases query as optimized prompt for searching vectorstorage
def get_dataquery(question):
    dataquery_userprompt = dataquery_template.format(question=question)
    dataquery_user_message = HumanMessage(content=dataquery_userprompt)
    dataquery = gptdataquery([dataquery_system_message, dataquery_user_message])
    print(f"Looking in database for: {dataquery.content}")
    return dataquery.content

#attempt to force garbace collection. seems unsuccessful
def smart_retriever(question):
    index = get_index()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dense_encoder = OpenAIEmbeddings(model="text-embedding-ada-002")
    sparse_encoder = SpladeEncoder(device=device)
    retriever = PineconeHybridSearchRetriever(embeddings=dense_encoder, sparse_encoder=sparse_encoder, index=index, top_k=75, alpha=0.99899) #lower alpha - more sparse
    
    dataquery = get_dataquery(question)
    print(f"Looking in database for: {dataquery}")  

    results = retriever.get_relevant_documents(dataquery)
    del (dense_encoder, sparse_encoder, index, device, retriever, dataquery)
    torch.cuda.empty_cache()
    gc.collect()
    return results

async def main(question, streamhandler, queue):
    print("main has started execution")
    if not isinstance(question, str):
        print("Invalid input. Please provide a string.")
        return
    retriever = get_retriever()
    dataquery = get_dataquery(question)
    results = retriever.get_relevant_documents(dataquery)
    print(f"Found this many cases: {len(results)}")
    #results = smart_retriever(question)
    #gc.collect()
    #for result in results:
    #    print (result.page_content, "\n", "\n")
    #return
    #print(f"{len(results)} chunks found in database")
    results, sum_of_relevance = await rank_concurrently(results, question)  # Replaced rank_cases with rank_concurrently

    print(f"Average relevance score: {sum_of_relevance/len(results)}")
    print(f"Rated this many cases: {len(results)}")

    max_tokens = ((gpt4_maxtokens - response_maxtokens) - 30) - (len(list(tokenizer.encode(analysis_template_string))) + len(list(tokenizer.encode(question))))
    sources = fill_tokens(results=results, max_tokens=max_tokens)
    analysis_userprompt = analysis_template.format(question=question, sources=sources)
    print(analysis_userprompt)
    user_message = HumanMessage(content=analysis_userprompt)
    gpt4analysis = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=2048, streaming=True, callbacks=[streamhandler])
    #await queue.put("test2")
    try:
        response = await gpt4analysis.agenerate([[analysis_system_message, user_message]])
    except Exception as e:
        print(f"Exception during gpt4analysis: {e}")
    await queue.put("TABALUGA_IST_ANGEKOMMEN")
    await queue.put("TABALUGA_WARTET")
    text_output = response.generations[0][0].text
    print(text_output)
    #await queue.put("test3")
    return text_output

class Item(BaseModel):
    input: str

app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://www.zeilertech.com",
    "https://www.profhastings.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    input: str


class MyCustomAsyncHandler(AsyncCallbackHandler):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"Async handler being called: token: {token}")
        if self.queue is not None:
            await self.queue.put(token)
        else:
            print("Error: queue is not set")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        print("Waiting for client data...")
        queue = asyncio.Queue()
        data = await websocket.receive_text()
        print(f"Received data: {data}")
        try:
            item = Item(**json.loads(data))
            print(f"Received item: {item.input}")
        except ValidationError as e:
            print(f"Error: {e}")
            continue
        handler = MyCustomAsyncHandler(queue)
        asyncio.create_task(main(item.input, handler, queue))
        #await queue.put("test1")
        print("Started task")
        while True:
            token = await queue.get()
            print(token)
            if token == "TABALUGA_WARTET":
                print("Done sending response")
                break
            await websocket.send_text(token)

if __name__ == "__main__":
    queue = asyncio.Queue()
    asyncio.run(main("Alfred arbeitet in einer Fabrik und schläft wo während er am Fließband arbeitet. Es entsteht ein erheblicher Schaden. Kann er zu Schadenersatz verurteilt werden?", MyCustomAsyncHandler(queue), queue))