import os
from contextlib import contextmanager
from multiprocessing import Lock
from typing import Tuple
from enum import Enum
from llama_index.llms.huggingface import HuggingFaceInferenceAPI, HuggingFaceLLM
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.callbacks import CallbackManager
from llama_index.llms.groq import Groq
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.core import (
    Settings,
    load_index_from_storage,
    StorageContext,
    VectorStoreIndex,
)
from app.utils.log_util import logger
from app.utils import data_util

CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
LLAMA_INDEX_HOME = os.path.join(PARENT_DIR, "llama_index_server")
os.environ["LLAMA_INDEX_CACHE_DIR"] = f"{LLAMA_INDEX_HOME}/llama_index_cache"
INDEX_PATH = f"{LLAMA_INDEX_HOME}/saved_index"
NODES_PATH = os.path.join(PARENT_DIR, f"{LLAMA_INDEX_HOME}/llama_parse_nodes/base_nodes_parsed_json_full.pkl")
PERSIST_INTERVAL = 3600

class ApiKeys:
    def __init__(self):
        self._api_key = ''

    def __set_api_key(self, org_name: str) -> str:
        if org_name == "cohere":
            self._api_key = 'ydbWRav6garWLCLoXg5fUrXUCdBWjHGHkrOVDn1N'
        if org_name == "huggingface":
            self._api_key = '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'
        if org_name == "groq":
            self._api_key = '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'
        if org_name == "google":
            self._api_key = 'AIzaSyAXL2kgiPDo7o63rBrogERv0hJUVGKagj4'

    def get_api_key(self, org_name: str) -> str:
        self.__set_api_key(org_name)
        return self._api_key

class Models(str, Enum):
    COHERE_COMMAND_R = "CohereForAI/c4ai-command-r-v01"
    GROQ_MIXTRAL = "mixtral-8x7b-32768"
    NOUS_MIXTRAL = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
    GEMINI = "models/gemini-1.0-pro"
    GEMINI_EMBEDDING = "models/embedding-001"

class IndexStorage:
    def __init__(self):
        self._current_model = Models.COHERE_COMMAND_R
        logger.info("initializing qdrant index ...")
        self._index = self.initialize_index()
        logger.info("initializing done")
        self._lock = Lock()
        self._last_persist_time = 0
        self._chat_engine_record = {}

    @property
    def chat_engine_record(self):
        return self._chat_engine_record

    @property
    def current_model(self):
        return self._current_model

    def index(self):
        return self._index

    @contextmanager
    def lock(self):
        # for the write operations on self._index
        with self._lock:
            yield

    def initialize_index(self):
        hf_remote_command_r = HuggingFaceInferenceAPI(model_name=self._current_model)
        mixtral_groq = Groq(model=Models.GROQ_MIXTRAL, api_key=ApiKeys().get_api_key(org_name="groq"))
        cohere_embed_model = CohereEmbedding(
            cohere_api_key=ApiKeys().get_api_key("cohere"),
            model_name="embed-multilingual-v3.0",
            input_type="search_document",
            embedding_type="float",
        )
        Settings.llm = hf_remote_command_r
        Settings.embed_model = cohere_embed_model
        if os.path.exists(INDEX_PATH):
            logger.info(f"Loading index from dir: {INDEX_PATH}")
            qdrant_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=INDEX_PATH),
            )
        else:
            data_util.assert_true(os.path.exists(NODES_PATH), f"parsed nodes not found: {NODES_PATH}")
            all_nodes = data_util.get_nodes()
            valid_nodes = data_util.convert_to_valid_uuid(nodes=all_nodes)

            transformations = [
                SentenceWindowNodeParser.from_defaults(
                    window_size=3,
                    window_metadata_key="window",
                    original_text_metadata_key="original_text",
                )
            ]
            pipeline = IngestionPipeline(transformations=transformations)
            query_nodes = pipeline.run(nodes=valid_nodes, in_place=True, show_progress=True)
            # creates a persistant index to disk
            client = QdrantClient(path="./qdrant_hpc_data_final")

            # create our vector store with hybrid indexing enabled
            # batch_size controls how many nodes are encoded with sparse vectors at once
            qdrant_vector_store = QdrantVectorStore(
                "hpc_paper_final", client=client, enable_hybrid=True, batch_size=20
            )

            qdrant_storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)
            Settings.chunk_size = 512

            qdrant_index = VectorStoreIndex(
                query_nodes,
                storage_context=qdrant_storage_context,
                llm=hf_remote_command_r,
                embed_model=cohere_embed_model,
                show_progress=True,
                verbose=True,
            )            
            qdrant_index.storage_context.persist(persist_dir=INDEX_PATH)
        return qdrant_index