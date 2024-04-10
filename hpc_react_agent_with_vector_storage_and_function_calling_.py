"""## Load the model with GGUF quantization using Llama.cpp

### Login to Huggingface
"""
"""### Mistral-7B Code-Instruct (as part of the coding engine and the vector query engine)"""

# from llama_index.llms.llama_cpp import LlamaCPP
# from llama_index.llms.llama_cpp.llama_utils import (
#     messages_to_prompt,
#     completion_to_prompt,
# )

# model_url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-code-ft-GGUF/resolve/main/mistral-7b-instruct-v0.2-code-ft.Q5_K_M.gguf"
#model_url = "https://huggingface.co/TheBloke/Magicoder-S-DS-6.7B-GGUF/resolve/main/magicoder-s-ds-6.7b.Q5_K_M.gguf"
#model_url = "https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct.Q5_K_M.gguf"

# mistral_local_llm = LlamaCPP(
#     # You can pass in the URL to a GGML model to download it automatically
#     model_url=model_url,
#     # optionally, you can set the path to a pre-downloaded model instead of model_url
#     model_path=None,
#     temperature=0.1,
#     max_new_tokens=8000,
#     # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
#     context_window=4096,
#     # kwargs to pass to __call__()
#     generate_kwargs={},
#     # kwargs to pass to __init__()
#     # set to at least 1 to use GPU
#     model_kwargs={"n_gpu_layers": 15000},
#     # transform inputs into Llama2 format
#     messages_to_prompt=messages_to_prompt,
#     completion_to_prompt=completion_to_prompt,
#     verbose=True,
# )

from llama_index.llms.ollama import Ollama

deepseek_coder_model = Ollama(model="deepseek-coder:6.7b", request_timeout=60.0)
# import os

# os.environ["HF_HOME"] = "model/"

# from llama_index.llms.vllm import Vllm

# # specific functions to format for mistral instruct
# def messages_to_prompt(messages):
#     prompt = "\n".join([str(x) for x in messages])
#     return f"<s>[INST] {prompt} [/INST] </s>\n"

# def completion_to_prompt(completion):
#     return f"<s>[INST] {completion} [/INST] </s>\n"

# mistral_code_model = Vllm(
#     model="TheBloke/Mistral-7B-Instruct-v0.2-code-ft-AWQ",
#     temperature=0.1,
#     tensor_parallel_size=4,
#     max_new_tokens=8000,
#     vllm_kwargs={
#         "swap_space": 1, 
#         "gpu_memory_utilization": 0.5,
#         "quantization": "AWQ",
#     },
# )

"""## Load other needed models through APIs"""

import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.callbacks import CallbackManager
from llama_index.llms.huggingface import HuggingFaceInferenceAPI, HuggingFaceLLM
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
from google.ai.generativelanguage import (
    GenerateAnswerRequest,
    HarmCategory,
    SafetySetting,
)
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
#from llama_index.llms.mistralai import MistralAI
#from llama_index.embeddings.mistralai import MistralAIEmbedding

callback_manager = CallbackManager([])

# Using OpenAI API for embeddings/llms

os.environ["MISTRAL_API_KEY"] = "eia4L9DdjWXm982FALAX1foUjfSXa60B"

# Define embedding models
#embed_model = GooglePaLMEmbedding(model_name="models/embedding-gecko-001", api_key=palm_api_key)
#bge_embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", max_length=1024)
gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001",api_key='AIzaSyAXL2kgiPDo7o63rBrogERv0hJUVGKagj4')
cohere_embed_model = CohereEmbedding(
    cohere_api_key='ydbWRav6garWLCLoXg5fUrXUCdBWjHGHkrOVDn1N',
    model_name="embed-multilingual-v3.0",
    input_type="search_document",
    embedding_type="float",
)
# Define Huggingface models
#instruct_local_mistral = HuggingFaceLLM(model=instruct_model, tokenizer=instruct_tokenizer, max_new_tokens=2048)
#local_mistral = HuggingFaceLLM(model=model, tokenizer=tokenizer, max_new_tokens=2048)
#local_neural_chat = HuggingFaceLLM(model=neural_model, tokenizer=neural_tokenizer, max_new_tokens=2048)
hf_remote_mistral = HuggingFaceInferenceAPI(model_name='mistralai/Mistral-7B-v0.1')
hf_remote_mistral_instruct = HuggingFaceInferenceAPI(model_name='mistralai/Mistral-7B-Instruct-v0.2', max_new_tokens=2048)
hf_remote_falcon = HuggingFaceInferenceAPI(model_name='tiiuae/falcon-7b-instruct')
hf_remote_zephyr = HuggingFaceInferenceAPI(model_name='HuggingFaceH4/zephyr-7b-beta')
hf_remote_mixtral = HuggingFaceInferenceAPI(model_name='mistralai/Mixtral-8x7B-Instruct-v0.1', context_window=4096)
hf_remote_mpt = HuggingFaceInferenceAPI(model_name='mosaicml/mpt-7b')
hf_remote_llama2_7b = HuggingFaceInferenceAPI(model_name='meta-llama/Llama-2-7b-hf')
hf_remote_command_r = HuggingFaceInferenceAPI(model_name='CohereForAI/c4ai-command-r-v01')
hf_remote_nous_mixtral = HuggingFaceInferenceAPI(model_name='NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO')

#OpenAI models
#embed_model=OpenAIEmbedding(model="text-embedding-3-small")
#gpt_llm = OpenAI(model="gpt-4-0125-preview")

#Mistral AI models
#mistral_llm = MistralAI(model="mistral-medium", temperature=0.1)
#mistral_embed_model = MistralAIEmbedding(model_name="mistral-embed")

# Define Gemini
gemini_llm = Gemini(
    api_key='AIzaSyAXL2kgiPDo7o63rBrogERv0hJUVGKagj4',
    safety_setting=[
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_NONE,
        ),
    ]
)
#Settings.llm = mistral_llm
Settings.embed_model = gemini_embed_model
Settings.llm = gemini_llm
#Settings.embed_model = embed_model

from llama_index.llms.groq import Groq

mixtral_groq = Groq(model="mixtral-8x7b-32768", api_key='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')

"""## Connecting to Neo4j Vector Database (Option 1)"""

# os.environ["NEO4J_URI"] = "neo4j+s://df3ca389.databases.neo4j.io"
# os.environ["NEO4J_USERNAME"] = "neo4j"
# os.environ["NEO4J_PASSWORD"] = "taRN5hso89An7ZU1iZ7rZQHtGH_oSw5OQThcBh3L8Bw"

from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import StorageContext
from llama_index.core import (
    KnowledgeGraphIndex,
    ServiceContext,
    VectorStoreIndex,
    Document
)
# os.environ["NEO4J_URI"] = os.getenv('NEO4J_URI')
# os.environ["NEO4J_USERNAME"] = os.getenv('NEO4J_USERNAME')
# os.environ["NEO4J_PASSWORD"] = os.getenv('NEO4J_PASSWORD')

# edge_types, rel_prop_names = ["relationship"], [
#     "relationship"
# ]  # default, could be omit if create from an empty kg
# tags = ["entity"]

"""## Initializing to Qdrant Vector Database (Option 2)"""

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.core import Settings

# creates a persistant index to disk
client = QdrantClient(path="./qdrant_hpc_data_final")

# create our vector store with hybrid indexing enabled
# batch_size controls how many nodes are encoded with sparse vectors at once
qdrant_vector_store = QdrantVectorStore(
    "hpc_paper_final", client=client, enable_hybrid=True, batch_size=20
)

qdrant_storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)
Settings.chunk_size = 512

"""### Load the nodes containing information about HPC Lab through the report"""

import pickle

with open('base_nodes_parsed_json_full.pkl', 'rb') as f:
   base_nodes_full = pickle.load(f)
with open('objects_parsed_json_full.pkl', 'rb') as f:
   objects_full = pickle.load(f)

all_nodes = base_nodes_full + objects_full

import uuid

for node in all_nodes:
    try:
      _uuid = uuid.UUID(node.id_)
    except ValueError:
      node.id_ = str(uuid.uuid4())

"""## Creating Indexes for querying

### Qdrant Vector Store Index

Only need to do this once, the folder containing the storage and the index to this storage will stay on our device forever and we'll retrieve the index directly from this storage
"""

qdrant_index = VectorStoreIndex(
    all_nodes[0:10],
    storage_context=qdrant_storage_context,
    llm=mixtral_groq,
    embed_model=cohere_embed_model,
    show_progress=True,
    verbose=True,
)

"""## Vector Query Engine

### Set up the response synthesizer to stream our response
"""

from llama_index.core import get_response_synthesizer

synth = get_response_synthesizer(
    streaming=True,
    response_mode="refine",
    llm=mixtral_groq,
)

"""### Set up our Qdrant Query Engine"""
from llama_index.core.postprocessor import LongContextReorder

reorder = LongContextReorder()

from llama_index.core.postprocessor import SimilarityPostprocessor

similarity = SimilarityPostprocessor(similarity_cutoff=0.5)

qdrant_query_engine = qdrant_index.as_query_engine(
    similarity_top_k=3,
    sparse_top_k=10,
    vector_store_query_mode="hybrid",
    node_postprocessors=[reorder, similarity],
    response_synthesizer=synth,
    llm=hf_remote_command_r,
    embed_model=cohere_embed_model,
)

from llama_index.core import PromptTemplate
from langchain import hub

langchain_prompt = hub.pull("rlm/rag-prompt")

from llama_index.core.prompts import LangchainPromptTemplate

lc_prompt_tmpl = LangchainPromptTemplate(
    template=langchain_prompt,
    template_var_mappings={"query_str": "question", "context_str": "context"},
)

qdrant_query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": lc_prompt_tmpl}
)

from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool, BaseTool

query_engine_tool = QueryEngineTool(
    query_engine=qdrant_query_engine,
    metadata=ToolMetadata(
        name="hpc_query_engine",
        description=(
            "useful for when you want to answer queries that require searching"
            " for information about hpc lab in the vector store"
        ),
    ),
)

# define pydantic model for auto-retrieval function
from pydantic import BaseModel, Field
class CodeRetrieveModel(BaseModel):
    query: str = Field(..., description="natural language query string")

def code(query: str):
    """Use code instruct model to answer code-related questions"""
    return deepseek_coder_model.complete(query)

code_assistant_tool = FunctionTool(
    fn=code,
    metadata=ToolMetadata(
        name="coding_query_engine",
        description=(
            "useful for when you want to answer queries that have"
            " codes in them"
        ),
        fn_schema=CodeRetrieveModel,
    ),  
)

from llama_index.core.agent import ReActAgent

context = """\
You are a hpc helpful assistant who is an expert on Ho Chi Minh University of Technology's HPC Lab's information.\
    You will answer questions about HPC Lab as in the persona of a friendly assistant \
    and a top hpc expert. \
    If there is a code block or multiple code blocks in the question, extract the code block and pass to the 'code_assistant_tool'
"""
agent = ReActAgent.from_tools(
    tools=[query_engine_tool, code_assistant_tool],
    llm=mixtral_groq,
    verbose=True,
    context=context
)

### Feature for Version 1 end here!!!!!

"""## Fast API"""
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import AsyncGenerator
import asyncio
import time
import uvicorn

app = FastAPI()
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Explicitly enable parallelism

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def hello_world():
    return {"message": "Hello, World!"}

@app.post("/chat")
async def chat(request: Request)->StreamingResponse:
    json_data = await request.json()
    user_message = json_data.get('content')

    async def send_stream_data():
        start_time = time.time()
        chatbot_response = agent.stream_chat(user_message)
        for response in chatbot_response.response_gen:
            yield response
        print(time.time() - start_time, user_message)

    return StreamingResponse(send_stream_data(), media_type="text/event-stream")

@app.post("/getChatTitle")
async def get_chat_title(request: Request):
    json_data = await request.json()
    user_prompt = json_data.get('content')

    async def call_model():
        summarization_prompt = f"""Summarize this query and write a suitable title from the following: "{user_prompt}"
Only return the title in your response."""
        response_iter = mixtral_groq.stream_complete(summarization_prompt)
        for response in response_iter:
            yield response.delta

    return StreamingResponse(call_model())

@app.post("/reset")
def reset():
    agent.reset()
    return {"message": "Chat Context Resetted!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3001)