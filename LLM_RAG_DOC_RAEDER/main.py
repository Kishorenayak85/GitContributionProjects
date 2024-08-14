#import required libraries
from langchain.embeddings import HuggingFaceEmbeddings,HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQAWithSourcesChain,RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate,
                                    SystemMessagePromptTemplate,
                                    HumanMessagePromptTemplate)
from langchain.llms import CTransformers
from langchain.llms import LlamaCpp
from langchain.retrievers import BM25Retriever,EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from cohere import Client
#
import chainlit as cl
import PyPDF2
from io import BytesIO
from getpass import getpass
#
import os
from configparser import ConfigParser
env_config = ConfigParser()

os.environ["COHERE_API_KEY"] = "Your_api_key"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)


system_template = """Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

Begin!
 - - - - - - - - 
{summaries}"""
messages = [SystemMessagePromptTemplate.from_template(system_template),HumanMessagePromptTemplate.from_template("{question}"),]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

class TextSplitter:
    def __init__(self, chunk_size=1000):
        self.chunk_size = chunk_size

    def split_text(self, text):
        # Split the text into chunks of the specified size
        return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]

#Decorator to react to the user websocket connection event.

@cl.on_chat_start
async def init():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
        ).send()

        if files:
            file = files[0]
            msg = cl.Message(content=f"Processing `{file.name}`…")
            await msg.send()

            # Read the PDF file
            with open(r"C:\Users\kisho\OneDrive\Desktop\LEARN\Enviseer\Document_Reader_Dev\HR_Policy_Manual_2023.pdf", 'rb') as file:
                pdf_stream = BytesIO(file.read())  # Use the read() method to get the content

            pdf = PyPDF2.PdfReader(pdf_stream)
            pdf_text = ""

            # Extract text from each page
            for page in pdf.pages:
                pdf_text += page.extract_text()

            # Split the text into chunks
            text_splitter = TextSplitter()  # Initialize your text splitter here
            texts = text_splitter.split_text(pdf_text)

            # Create metadata for each chunk
            metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

            # Initialize components
            model_id = "BAAI/bge-small-en-v1.5"
            embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_id,
                model_kwargs={"device": "cpu"}
            )
            bm25_retriever = BM25Retriever.from_texts(texts)
            bm25_retriever.k = 5

            # Store the embeddings in the user session
            cl.user_session.set("embeddings", embeddings)
            docsearch = await cl.make_async(Qdrant.from_texts)(
                texts, embeddings, location=":memory:", metadatas=metadatas
            )

            llm = LlamaCpp(
                streaming=True,
                model_path="zephyr-7b-beta.Q4_K_M.gguf",
                max_tokens=1500,
                temperature=0.75,
                top_p=1,
                gpu_layers=0,
                stream=True,
                verbose=True,
                n_threads=int(os.cpu_count() / 2),
                n_ctx=4096
            )

            # Hybrid Search
            qdrant_retriever = docsearch.as_retriever(search_kwargs={"k": 5})
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, qdrant_retriever],
                weights=[0.5, 0.5]
            )

            # Cohere Reranker
            compressor = CohereRerank(
                client=Client(api_key=os.getenv("COHERE_API_KEY")),
                user_agent='langchain'
            )
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=ensemble_retriever,
            )

            # Create a chain that uses the Chroma vector store
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=compression_retriever,
                return_source_documents=True,
            )

            # Save the metadata and texts in the user session
            cl.user_session.set("metadatas", metadatas)
            cl.user_session.set("texts", texts)

            # Let the user know that the system is ready
            msg.content = f"`{file.name}` processed. You can now ask questions!"
            await msg.update()

            # Store the chain as long as the user session is active
            cl.user_session.set("chain", chain)


import chainlit as cl
from chainlit import Text

@cl.on_message
async def process_response(res: cl.Message):
    # Retrieve the retrieval chain initialized for the current session
    chain = cl.user_session.get("chain")

    if not chain:
        await cl.Message(content="Error: Retrieval chain not found.").send()
        return

    # Chainlit callback handler
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])
    cb.answer_reached = True

    # res.content to extract the content from chainlit.message.Message
    response = await chain.acall(res.content, callbacks=[cb])
    answer = response["result"]
    sources = response["source_documents"]
    source_elements = []

    # Get the metadata and texts from the user session
    metadatas = cl.user_session.get("metadatas")
    texts = cl.user_session.get("texts")

    if not metadatas or not texts:
        await cl.Message(content="Error: Metadata or texts not found in session.").send()
        return

    all_sources = [m["source"] for m in metadatas]

    if sources:
        found_sources = []

        # Add the sources to the message
        for source in sources:
            try:
                source_name = source.metadata["source"]
            except KeyError:
                source_name = "Unknown Source"

            # Get the index of the source
            text = source.page_content
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer, elements=source_elements).send()
