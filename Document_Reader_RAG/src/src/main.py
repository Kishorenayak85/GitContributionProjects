import os
import json

from api import *
from data_loader import load_eudr_document_transcripts, load_eudr_document
from index import build_automerging_index, get_automerging_index
from citation import cite_source
from evaluator import create_ragas_dataset, evaluate_ragas_dataset

from llama_index.core import Document
from query_engine import get_automerging_query_engine
from llama_index.core import PromptTemplate
from llama_index.core import ServiceContext
from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI
from llama_index.core.embeddings.utils import resolve_embed_model
from llama_index.embeddings.openai import OpenAIEmbedding
from tqdm import tqdm
import pandas as pd

from datasets import Dataset
eval_dataset = Dataset.from_csv("E:\RAG Llamaindex\eval_data\dataset15.csv")

llm = Groq(model="mixtral-8x7b-32768")
embed_model = resolve_embed_model("local:BAAI/bge-base-en-v1.5")
merging_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )


eudr_pdf_path = "E:\RAG Llamaindex\Data\EUDR_with_metadata"
transcript_files_path = "E:\RAG Llamaindex\Data\Video_Transcripts_with_metadata"
index_dir="E:\RAG Llamaindex\merging_index"
youtube_links_path="E:\RAG Llamaindex\Data\youtube_links.json"

qa_temp = """Context information is below.
---------------------
{context_str}
---------------------
Do not create answer outside of the retrieved context information, No matter what the Query is.
Given the context information and not prior knowledge, answer the query.
Give a clear & concise answer.
When the information is not present in the provided context, just say out of context. Do not give answers to such query.
Query: {query_str}
Answer: """

qa_prompt = PromptTemplate(qa_temp)

if not os.path.exists(index_dir):
    print("Creating indices")
    main_doc = load_eudr_document_transcripts(eudr_pdf_path, transcript_files_path)
    automerging_index = build_automerging_index(main_doc, merging_context, index_dir="E:\RAG Llamaindex\merging_index", chunk_sizes=[1300, 500, 200])
    print("Done")
else:
    print("Calling indices")
    automerging_index = get_automerging_index(merging_context, index_dir="E:\RAG Llamaindex\merging_index")
    print("Done")

auto_merging_engine, retriever = get_automerging_query_engine(automerging_index, similarity_top_k=12, rerank_top_n=6, llm = Groq(model="mixtral-8x7b-32768"))
auto_merging_engine.update_prompts({"response_synthesizer:text_qa_template" : qa_prompt})

def generate_answer(retriever, auto_merging_engine, query_str, qa_prompt):
    retrieved_nodes = retriever.retrieve(query_str)
    context_str = "\n\n".join([n.get_content() for n in retrieved_nodes])

    qa_prompt.format(query_str=query_str, context_str=context_str)

    response = str(auto_merging_engine.query(query_str))

    citations = cite_source(youtube_links_path, retrieved_nodes, response)

    return response, citations

def evaluate_rag_performance(eval_dataset, retriever, auto_merging_engine, qa_prompt):
    ragas_dataset = create_ragas_dataset(eval_dataset, retriever, auto_merging_engine, qa_prompt)
    ragas_performance = evaluate_ragas_dataset(ragas_dataset)

    return ragas_performance, ragas_dataset

if __name__ == "__main__":

    query_str = "what is physics?"

    response, citations = generate_answer(retriever, auto_merging_engine, query_str, qa_prompt)
    print(response)
    print(citations)


