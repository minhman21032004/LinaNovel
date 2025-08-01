from utils import *
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
import argparse

PERSIST_DIRECTORY = 'chroma_db'
EMBEDDING_MODEL = 'text-embedding-3-large'
BACKUP_DIRECTORY = 'backup'

HIERARCHICAL_CONFIG  = {
    'level_2' : 5,
    'level_3' : 3,
    'level_4' : 3,
    'level_5' : 3
}

TEXT_SPLITTER_CONFIG = {
    'chunk_size' : 500,
    'chunk_overlap' : 100
}

def split_into_chunks(data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=TEXT_SPLITTER_CONFIG['chunk_size'], chunk_overlap=TEXT_SPLITTER_CONFIG['chunk_overlap'])
    return splitter.split_documents(data)

def hierarchical_chunking(data):
    if os.path.exists(f"{BACKUP_DIRECTORY}/chunks_level_1.txt"):
        print("Using backup chunks_level_1.txt")
        chunks_level_1 = read_chunks(f"{BACKUP_DIRECTORY}/chunks_level_1.txt")
    else:
        chunks_level_1 = split_into_chunks(data)
        print(f"Completed chunking at level 1 with a total {len(chunks_level_1)} chunks.\n")
        write_chunks(chunks_level_1, f"{BACKUP_DIRECTORY}/chunks_level_1.txt")

    if os.path.exists(f"{BACKUP_DIRECTORY}/chunks_level_2.txt"):
        print("Using backup chunks_level_2.txt")
        chunks_level_2 = read_chunks(f"{BACKUP_DIRECTORY}/chunks_level_2.txt")
    else:
        chunks_level_2 = up_level_chunking(chunks_level_1, n_content_chunks=HIERARCHICAL_CONFIG['level_2'], level=2, log_error=True)
        print(f"Completed chunking at level 2 with a total {len(chunks_level_2)} chunks.\n")
        write_chunks(chunks_level_2, f"{BACKUP_DIRECTORY}/chunks_level_2.txt")

    if os.path.exists(f"{BACKUP_DIRECTORY}/chunks_level_3.txt"):
        print("Using backup chunks_level_3.txt")
        chunks_level_3 = read_chunks(f"{BACKUP_DIRECTORY}/chunks_level_3.txt")
    else:
        chunks_level_3 = up_level_chunking(chunks_level_2, n_content_chunks=HIERARCHICAL_CONFIG['level_3'], level=3, log_error=True)
        print(f"Completed chunking at level 3 with a total {len(chunks_level_3)} chunks.\n")
        write_chunks(chunks_level_3, f"{BACKUP_DIRECTORY}/chunks_level_3.txt")

    if os.path.exists(f"{BACKUP_DIRECTORY}/chunks_level_4.txt"):
        print("Using backup chunks_level_4.txt")
        chunks_level_4 = read_chunks(f"{BACKUP_DIRECTORY}/chunks_level_4.txt")
    else:
        chunks_level_4 = up_level_chunking(chunks_level_3, n_content_chunks=HIERARCHICAL_CONFIG['level_4'], level=4, log_error=True)
        print(f"Completed chunking at level 4 with a total {len(chunks_level_4)} chunks.\n")
        write_chunks(chunks_level_4, f"{BACKUP_DIRECTORY}/chunks_level_4.txt")

    if os.path.exists(f"{BACKUP_DIRECTORY}/chunks_level_5.txt"):
        print("Using backup chunks_level_5.txt")
        chunks_level_5 = read_chunks(f"{BACKUP_DIRECTORY}/chunks_level_5.txt")
    else:
        chunks_level_5 = up_level_chunking(chunks_level_4, n_content_chunks=HIERARCHICAL_CONFIG['level_5'], level=5, log_error=True)
        print(f"Completed chunking at level 5 with a total {len(chunks_level_5)} chunks.\n")
        write_chunks(chunks_level_5, f"{BACKUP_DIRECTORY}/chunks_level_5.txt")

    chunks_map = {
        'level_1': chunks_level_1,
        'level_2': chunks_level_2,
        'level_3': chunks_level_3,
        'level_4': chunks_level_4,
        'level_5': chunks_level_5
    }

    for level, chunks in chunks_map.items():
        retype_metadata(chunks)

    return chunks_map

def embedd_chunks(persist_directory, chunks_map, embedding):
    for level, chunks in chunks_map.items():
        print(f"Embedding chunks {level}")
        directory = f"{persist_directory}/chunk_{level}"
        if os.path.exists(directory):
            print(f"Skipping {level} (already embedded).")
            continue
        Chroma.from_documents(documents=chunks, embedding = embedding, persist_directory=directory)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data/Oregairu Volume 1.pdf")
    return parser.parse_args()

def main():
    args = parse_args()
    file_path = args.file
    print("\n=====Loading Data=====\n")
    if not os.path.exists(file_path):
        print(f"Could not find {file_path}")
        return

    data = load_documents(file_path)
    data = clean_data(data)

    print("\n=====Hirerachical Chunking=====\n")
    chunks_map = hierarchical_chunking(data)

    print("\n=====Embedding=====")
    embedding = AzureOpenAIEmbeddings(
        model = EMBEDDING_MODEL, 
        openai_api_version = OPENAI_API_VERSION, 
        azure_endpoint = OPENAI_API_ENDPOINT, 
        api_key = OPENAI_API_KEY)

    embedd_chunks(PERSIST_DIRECTORY, chunks_map, embedding)

    print("Embedded succesfully")
    

if __name__ == "__main__":
    main()


