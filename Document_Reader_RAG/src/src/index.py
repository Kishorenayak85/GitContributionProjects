import os


from llama_index.core import (
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.llms.groq import Groq
from llama_index.core.embeddings.utils import resolve_embed_model
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Document

from api import *

def build_automerging_index(
    main_doc,
    merging_context,
    index_dir="E:\RAG Llamaindex\merging_index",
    chunk_sizes=None
    ):

    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    
    # Process each document individually and retain metadata
    all_nodes = []
    for doc in main_doc:
        document = Document(text=doc.text, metadata={"file_name": doc.metadata["file_name"]})
        nodes = node_parser.get_nodes_from_documents([document])
        for node in nodes:
            node.metadata["file_name"] = document.metadata["file_name"]
        all_nodes.extend(nodes)
    leaf_nodes = get_leaf_nodes(all_nodes)
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(all_nodes)


    automerging_index = VectorStoreIndex(
        leaf_nodes, storage_context=storage_context, service_context=merging_context
    )
    automerging_index.storage_context.persist(persist_dir=index_dir)

    return automerging_index



def get_automerging_index(merging_context, index_dir="E:\RAG Llamaindex\merging_index"):

    automerging_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=index_dir),
        service_context=merging_context
        )
    
    return automerging_index