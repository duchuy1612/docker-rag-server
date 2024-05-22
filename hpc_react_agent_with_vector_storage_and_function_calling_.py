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
deepseek_coder_model = Ollama(base_url="http://ollama:11434", model="deepseek-coder:6.7b", request_timeout=60.0)
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
    cohere_api_key='Dseuzx9tizgMjbbJBpOigdXnhOCfhaI73KPyGQla',
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
hf_remote_command_r = HuggingFaceInferenceAPI(model_name='CohereForAI/c4ai-command-r-plus', api_key='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')
hf_remote_nous_mixtral = HuggingFaceInferenceAPI(model_name='NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO', api_key='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')
hf_remote_zephyr_orpho = HuggingFaceInferenceAPI(model_name='HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1', api_key='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')

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
llama3_70b_groq = Groq(model="llama3-70b-8192", api_key='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')
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

"""## Creating Indexes for querying

### Qdrant Vector Store Index

Only need to do this once, the folder containing the storage and the index to this storage will stay on our device forever and we'll retrieve the index directly from this storage
"""
from index import load_index

qdrant_index = load_index()

"""## Vector Query Engine

### Set up the response synthesizer to stream our response
"""

from llama_index.core import get_response_synthesizer

synth = get_response_synthesizer(
    streaming=True,
    response_mode="refine",
    llm=llama3_70b_groq,
)

"""### Set up our Qdrant Query Engine"""
from llama_index.core.postprocessor import LongContextReorder

reorder = LongContextReorder()

from llama_index.core.postprocessor import SimilarityPostprocessor

similarity = SimilarityPostprocessor(similarity_cutoff=0.5)

from llama_index.core.postprocessor import MetadataReplacementPostProcessor
# from llama_index.core.postprocessor import LLMRerank

# llm_rerank = LLMRerank(
#     choice_batch_size=10,
#     top_n=5,
#     llm=mixtral_groq,
# )
metadata_replacement = MetadataReplacementPostProcessor(target_metadata_key="window")

from llama_index.postprocessor.cohere_rerank import CohereRerank

cohere_rerank = CohereRerank(api_key="eCI1CWbErf56zI0hPZOVeey1wrUORKLKvRAJ2KuC", top_n=2)

qdrant_query_engine = qdrant_index.as_query_engine(
    similarity_top_k=3,
    # sparse_top_k=12,
    # vector_store_query_mode="hybrid",
    node_postprocessors=[cohere_rerank, metadata_replacement],
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
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine

hyde = HyDEQueryTransform(include_original=True, llm=mixtral_groq)
hyde_query_engine = TransformQueryEngine(qdrant_query_engine, hyde)

query_engine_tool = QueryEngineTool(
    query_engine=hyde_query_engine,
    metadata=ToolMetadata(
        name="hpc_query_engine",
        description=(
            "useful for when you want to answer queries about"
            " HPC Lab at HCMUT"
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

code_assistant_tool = FunctionTool.from_defaults(
    fn=code,
    tool_metadata=ToolMetadata(
        name="code_assistant_tool",
        description=(
            "useful for when you want to answer queries that have"
            " codes in them or are related to code problems"
        ),
        fn_schema=CodeRetrieveModel,
    ),
    return_direct=True,
)

from llama_index.core.agent import ReActAgent

IRRELEVANT_QUESTION = {
    "default_answer_id": "irrelevant_question",
    "default_answer": "This question is not relevant to HPC Lab, Ho Chi Minh University of Technology or coding problems, please ask a question related to HPC Lab, HPC, Ho Chi Minh University of Technology or code-related questions.",
}

def get_default_answer_id():
    return IRRELEVANT_QUESTION["default_answer_id"]

def get_default_answer():
    return IRRELEVANT_QUESTION["default_answer"]

CONTEXT = """\
You are a hpc helpful assistant who is an expert on Ho Chi Minh University of Technology's HPC Lab's information.\
    You will answer questions about HPC Lab as in the persona of a friendly assistant \
    and a top hpc expert. \
    If there is a code block or multiple code blocks in the question, extract the code block and pass to the 'code_assistant_tool'
"""
PROMPT_TEMPLATE_FOR_QUERY_ENGINE = (
    "Assume you are the administrator of HPC Lab system glad to answer questions from HPC Lab's users, "
    "if the question has anything to do with HPC Lab, or the Supernode-XP system, or HPC knowledge, or Ho Chi Minh University of Technology, or coding problems "
    "give detailed, simple, accurate, precise answer to the question. "
    "Always use the information available in the vector store as the source of truth. Don't try to guess or make up information. "
    "Limited to 1000 words maximum. If the question has nothing to do with HPC Lab, HPC, Ho Chi Minh University of Technology or coding problems at all, please answer "
    f"'{get_default_answer_id()}'.\n"
    "The question is: {query_str}\n"
)
SYSTEM_PROMPT_TEMPLATE_FOR_CHAT_ENGINE = (
    "You are an expert assistant that can find relevant information using the tools at your disposal, and you have "
    "great knowledge about HPC Lab, HPC, Ho Chi Minh University of Technology and coding.\n"
    "Please give detailed, simple, accurate, precise answer to the question, limited to 1000 words maximum.\n"
    "Remove the prefixes 'Thought: ', 'Observation: ' or 'Answer: ' if they exist in your answer. \n"
    "You may need to combine the chat history to fully understand the query of the user.\n"
    "You must always think of a tool in the toolset for every single question.\n"
    "If there are no suitable tools to answer the questions, answer it using your own knowledge.\n"
    "Make sure that you do not make up any information.\n"
    "If the questions are closely related to information about HPC Lab, please use the 'hpc\_query\_engine' tool.\n"
    "If there are code blocks in your response from the 'code\_assistant\_tool' tool, return it with the name of the languae too.\n"
    "If the questions are not related to HPC Lab, HPC, Ho Chi Minh University of Technology and coding at all, still answer it but then append " 
    f"'{get_default_answer_id()}'.\n "
    "to your response." 
)
vistral_llm = Ollama(base_url="http://ollama:11434", model="ontocord/vistral", request_timeout=60.0)

from llama_index.llms.cohere import Cohere
cohere_llm = Cohere(model='command-r-plus', api_key='eCI1CWbErf56zI0hPZOVeey1wrUORKLKvRAJ2KuC')

agent = ReActAgent.from_tools(
    tools=[query_engine_tool, code_assistant_tool],
    llm=llama3_70b_groq,
    verbose=True,
    context=CONTEXT,
    system_prompt=SYSTEM_PROMPT_TEMPLATE_FOR_CHAT_ENGINE,
)

### Feature for Version 1 end here!!!!!

"""## Fast API"""
from fastapi import FastAPI, Request, APIRouter
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import AsyncGenerator
import asyncio
import time
import uvicorn

router = APIRouter(prefix="/api/v1")

os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Explicitly enable parallelism

@router.get("/")
async def hello_world():
    return {"message": "Hello, World!"}

@router.post("/chat")
async def chat(request: Request):
    json_data = await request.json()
    user_message = json_data.get('content')
    start_time = time.time()
    chatbot_response = agent.chat(user_message)
    end_time = time.time()
    print("time", end_time - start_time)
    return {
        "response": chatbot_response.response,
        "time": end_time - start_time
    }

@router.post("/chat-streaming")
async def chat_streaming(request: Request)->StreamingResponse:
    json_data = await request.json()
    user_message = json_data.get('content')

    async def send_stream_data():
        start_time = time.time()
        chatbot_response = await agent.astream_chat(user_message)
        async for token in chatbot_response.async_response_gen():
            yield token
        print(time.time() - start_time, user_message)

    return StreamingResponse(send_stream_data(), media_type="text/event-stream")

@router.post("/getChatTitle")
async def get_chat_title(request: Request):
    json_data = await request.json()
    user_prompt = json_data.get('content')
    summarization_prompt = f"""Summarize this query and write a suitable title from the following: "{user_prompt}"
Only return the title in your response."""
    response = mixtral_groq.complete(summarization_prompt)

    return {
        "message": "Title Generated Successfully!",
        "title": response
    }

@router.post("/reset")
def reset():
    agent.reset()
    return {"message": "Chat Context Resetted!"}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3001)