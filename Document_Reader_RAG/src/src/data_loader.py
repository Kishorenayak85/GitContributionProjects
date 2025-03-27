import glob
import os

from llama_index.core import SimpleDirectoryReader


def load_eudr_document_transcripts(eudr_pdf_path, transcript_files_path):
    reader = SimpleDirectoryReader(input_dir=eudr_pdf_path)
    main_doc = reader.load_data()
    transcript_reader = SimpleDirectoryReader(input_dir=transcript_files_path)
    transcript_docs = transcript_reader.load_data()
    len(transcript_docs)
    main_doc.extend(transcript_docs)
    len(main_doc)
    
    return main_doc

def load_eudr_document(eudr_pdf_path):
    reader = SimpleDirectoryReader(input_dir=eudr_pdf_path)
    main_doc = reader.load_data()
    len(main_doc)

    return main_doc