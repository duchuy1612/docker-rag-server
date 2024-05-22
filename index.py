import logging
import os

from constants import STORAGE_DIR
from llama_index.core.indices import load_index_from_storage
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.core.callbacks import CallbackManager
from llama_index.core import Settings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

cohere_embed_model = CohereEmbedding(
    cohere_api_key='Dseuzx9tizgMjbbJBpOigdXnhOCfhaI73KPyGQla',
    model_name="embed-multilingual-v3.0",
    input_type="search_document",
    embedding_type="float",
)

# # creates a persistant index to disk
# client = QdrantClient(path="./qdrant_hpc_data_final")

# # create our vector store with hybrid indexing enabled
# # batch_size controls how many nodes are encoded with sparse vectors at once
# qdrant_vector_store = QdrantVectorStore(
#     "hpc_paper_final", client=client, batch_size=32
# )

# qdrant_storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)
# Settings.chunk_size = 800

import pickle

with open('2nd_base_nodes_parsed_full.pkl', 'rb') as f:
   base_nodes_full = pickle.load(f)
with open('2nd_objects_parsed_full.pkl', 'rb') as f:
   objects_full = pickle.load(f)

all_nodes = base_nodes_full + objects_full

import uuid

for node in all_nodes:
    try:
      _uuid = uuid.UUID(node.id_)
    except ValueError:
      node.id_ = str(uuid.uuid4())

from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.ingestion import IngestionPipeline

transformations = [
    SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    ),
]
pipeline = IngestionPipeline(transformations=transformations)
query_nodes = pipeline.run(nodes=all_nodes, in_place=True, show_progress=True)

logger = logging.getLogger("uvicorn")

# build the existing index
logger.info(f"Building index...")
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore

# dimensions of text-ada-embedding-002
d = 1024
faiss_index = faiss.IndexFlatL2(d)

faiss_vector_store = FaissVectorStore(faiss_index=faiss_index)
faiss_storage_context = StorageContext.from_defaults(vector_store=faiss_vector_store)

faiss_index = VectorStoreIndex(
    query_nodes[0:10],
    storage_context=faiss_storage_context,
    embed_model=cohere_embed_model,
    show_progress=True,
    verbose=True,
)
logger.info(f"Finished building index")
faiss_index.storage_context.persist(STORAGE_DIR)
logger.info(f"Finished creating new index. Stored in {STORAGE_DIR}")

def load_index():
    # check if storage already exists
    if not os.path.exists(STORAGE_DIR):
        raise Exception(
            "StorageContext is empty - call 'python app/engine/generate.py' to generate the storage first"
        )
    
    # load the existing index
    logger.info(f"Loading index from {STORAGE_DIR}...")
    # load index from disk
    faiss_vector_store = FaissVectorStore.from_persist_dir("./storage")
    faiss_storage_context = StorageContext.from_defaults(
        vector_store=faiss_vector_store, persist_dir="./storage"
    )
    faiss_index = load_index_from_storage(storage_context=faiss_storage_context)
    logger.info(f"Finished loading index from {STORAGE_DIR}")
    return faiss_index