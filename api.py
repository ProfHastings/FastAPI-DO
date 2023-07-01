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
from fastapi import FastAPI, WebSocket, Depends, HTTPException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from pydantic import BaseModel, ValidationError
from langchain.callbacks.base import AsyncCallbackHandler
import uvicorn
from langchain.text_splitter import CharacterTextSplitter

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
    pinecone.init(environment=env)
    return pinecone.Index("justiz-openai-full")

#(next two functions) uses async api calls to rank chunks from 1-10 based on relevance
async def async_rank(case, question, max_attempts=5, timeout_seconds=5):
    for attempt in range(max_attempts):
        try:
            ranking_userprompt = ranking_template.format(case=case.page_content, question=question)
            ranking_user_message = HumanMessage(content=ranking_userprompt)
            
            relevance = await asyncio.wait_for(
                gpt35._agenerate([ranking_system_message, ranking_user_message]),
                timeout=timeout_seconds
            )
            
            relevance_score = float(relevance.generations[0].text)
            return (case, relevance_score)
        except ValueError:
            print(f"Attempt {attempt + 1} failed, did not return ranking number")
        except asyncio.TimeoutError:
            print(f"Attempt {attempt + 1} timed out, retrying...")
    print(f"All {max_attempts} attempts failed. Returning default relevance score of 1.")
    return (case, 1)

async def rank_concurrently(cases, question):
    tasks = [async_rank(case, question) for case in cases]
    results = await asyncio.gather(*tasks)
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    sum_of_relevance = sum(relevance for _, relevance in results)    
    for index, (case, relevance_score) in enumerate(sorted_results, start=1):
        print(f"Rank {index}:\n{case.page_content}\nRelevance Score: {relevance_score}\n")    
    return [case for case, _ in sorted_results], sum_of_relevance

#rephrases query as optimized prompt for searching vectorstorage
def get_dataquery(question):
    dataquery_userprompt = dataquery_template.format(question=question)
    dataquery_user_message = HumanMessage(content=dataquery_userprompt)
    dataquery = gptdataquery([dataquery_system_message, dataquery_user_message])
    print(f"Looking in database for: {dataquery.content}")
    return dataquery.content

class Result:
    def __init__(self, page_content, short_source, long_source):
        self.page_content = page_content
        self.short_source = short_source
        self.long_source = long_source

def get_short_source(text):
    lines = text.split("\n")
    for j, line in enumerate(lines):
        if "Geschäftszahl" in line and j < len(lines) - 1:
            return lines[j + 1]
    return None

def get_data(id):
    parts = id.rsplit('_', 1)
    long_source = parts[0]
    chunknr = int(parts[1])
    file_path = os.path.join('/database', f'{long_source}.txt')
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            short_source = get_short_source(text)
            splits = text_splitter.split_text(text)
            cleaned_splits = [split.replace('\n', ' ').replace('\t', ' ') for split in splits]
            page_content = cleaned_splits[chunknr] if chunknr < len(cleaned_splits) else None
            return Result(page_content, short_source, long_source)
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
        return None
    except IndexError:
        print(f"Index {chunknr} out of range for file {file_path}.")
        return None

def getfullresults(ids):
    results = []
    for id in ids:
        result = get_data(id)
        if result is not None:
            results.append(result)
    return results

def hybrid_scale(dense, sparse, alpha: float):
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hsparse = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    hdense = [v * alpha for v in dense]
    return hdense, hsparse

def hybrid_query(dataquery, top_k, alpha):
    index = get_index()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sparse_encoder = SpladeEncoder(device=device)
    dense_encoder = OpenAIEmbeddings(model="text-embedding-ada-002")
    sparse_vec = sparse_encoder.encode_documents([dataquery])[0]
    dense_vec = dense_encoder.embed_documents([dataquery])[0]
    dense_vec, sparse_vec = hybrid_scale(dense_vec, sparse_vec, alpha)

    result = index.query(
      vector=dense_vec,
      sparse_vector=sparse_vec,
      top_k=top_k,
    )
    id_list = [match['id'] for match in result['matches']]

    return id_list

async def main(question, streamhandler, queue):
    print("main has started execution")
    if not isinstance(question, str):
        print("Invalid input. Please provide a string.")
        return
    dataquery = get_dataquery(question)
    id_list = hybrid_query(dataquery=dataquery, top_k=75, alpha=0.99899)
    print(f"id_list: {id_list}")
    results = getfullresults(id_list)
    print(f"Unranked results: {results}")
    print(f"Found this many cases: {len(results)}")

    results, sum_of_relevance = await rank_concurrently(results, question)  # Replaced rank_cases with rank_concurrently

    print(f"Average relevance score: {sum_of_relevance/len(results)}")
    print(f"Rated this many cases: {len(results)}")

    max_tokens = ((gpt4_maxtokens - response_maxtokens) - 30) - (len(list(tokenizer.encode(analysis_template_string))) + len(list(tokenizer.encode(question))))
    sources = fill_tokens(results=results, max_tokens=max_tokens)
    analysis_userprompt = analysis_template.format(question=question, sources=sources)
    print(analysis_userprompt)
    user_message = HumanMessage(content=analysis_userprompt)
    gpt4analysis = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=2048, streaming=True, callbacks=[streamhandler])
    try:
        response = await gpt4analysis.agenerate([[analysis_system_message, user_message]])
    except Exception as e:
        print(f"Exception during gpt4analysis: {e}")
    await queue.put("TABALUGA_IST_ANGEKOMMEN")
    await queue.put("TABALUGA_WARTET")
    text_output = response.generations[0][0].text
    print(text_output)
    return text_output

class Item(BaseModel):
    input: str

app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://www.zeilertech.com",
    "https://www.profhastings.com",
    "https://138.197.186.94:8000",
    "https://138.197.186.94",
    "http://138.197.186.94:8000",
    "http://138.197.186.94"
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
        if self.queue is not None:
            await self.queue.put(token)
        else:
            print("Error: queue is not set")

SECRET_API_KEY = os.environ.get("SECRET_API_KEY")

def get_api_key(websocket: WebSocket):
    api_key = websocket.query_params.get("api_key")
    print(f"Received API Key: {api_key}")  # Debugging line
    if api_key != SECRET_API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")
    return api_key

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, api_key: str = Depends(get_api_key)):
    await websocket.accept()
    try:
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
            print("Started task")
            while True:
                token = await queue.get()
                print(token)
                if token == "TABALUGA_WARTET":
                    print("Done sending response")
                    break
                await websocket.send_text(token)
    except WebSocketDisconnect:
        print("WebSocket connection was closed unexpectedly.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)