import os

from llama_index.core import (
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI
from llama_index.core.embeddings.utils import resolve_embed_model
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine


def get_automerging_query_engine(
    automerging_index,
    similarity_top_k,
    rerank_top_n,
    llm = Groq(model="mixtral-8x7b-32768")
    ):

    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank], llm=llm
    )

    return auto_merging_engine, retriever