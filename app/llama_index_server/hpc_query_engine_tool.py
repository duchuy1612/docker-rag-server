from typing import Any
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import get_response_synthesizer
from pydantic import BaseModel, Field
from llama_index.llms.huggingface import HuggingFaceInferenceAPI, HuggingFaceLLM
from llama_index.core.postprocessor import LongContextReorder
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.llms.gemini import Gemini
from llama_index.core.postprocessor import LLMRerank
from llama_index.llms.groq import Groq
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.core import PromptTemplate
from langchain import hub
from llama_index.core.prompts import LangchainPromptTemplate
from google.ai.generativelanguage import (
    GenerateAnswerRequest,
    HarmCategory,
    SafetySetting,
)
from app.llama_index_server.index_storage import IndexStorage, Models, ApiKeys

DEFAULT_NAME = "hpc_query_engine"
DEFAULT_DESCRIPTION = (
    "useful for when you want to answer queries that require searching"
    " for information about hpc lab in the vector store"
)

def build_query_engine_tool()-> QueryEngineTool:
    qdrant_index = IndexStorage().index()

    reorder = LongContextReorder()
    similarity = SimilarityPostprocessor(similarity_cutoff=0.5)
    llm_rerank = LLMRerank(
        choice_batch_size=5,
        top_n=2,
        llm=Groq(model=Models.GROQ_MIXTRAL, api_key=ApiKeys().get_api_key(org_name="groq")),
    ),
    metadata_replacement = MetadataReplacementPostProcessor(target_metadata_key="window")

    synth = get_response_synthesizer(
        streaming=True,
        response_mode="refine",
        llm=Groq(model=Models.GROQ_MIXTRAL, api_key=ApiKeys().get_api_key(org_name="groq")),
    )
    
    qdrant_query_engine = qdrant_index.as_query_engine(
        similarity_top_k=3,
        sparse_top_k=10,
        vector_store_query_mode="hybrid",
        node_postprocessors=[reorder, similarity, llm_rerank, metadata_replacement],
        response_synthesizer=synth,
        llm=HuggingFaceInferenceAPI(model_name='CohereForAI/c4ai-command-r-v01'),
        embed_model=CohereEmbedding(
            cohere_api_key=ApiKeys().get_api_key(org_name="cohere"),
            model_name="embed-multilingual-v3.0",
            input_type="search_document",
            embedding_type="float",
        ),
    )
    
    langchain_prompt = hub.pull("rlm/rag-prompt")
    lc_prompt_tmpl = LangchainPromptTemplate(
        template=langchain_prompt,
        template_var_mappings={"query_str": "question", "context_str": "context"},
    )

    qdrant_query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": lc_prompt_tmpl}
    )
    query_engine_tool = QueryEngineTool(
        query_engine=qdrant_query_engine,
        metadata=ToolMetadata(
            name=DEFAULT_NAME,
            description=DEFAULT_DESCRIPTION,
        ),
    )
    return query_engine_tool