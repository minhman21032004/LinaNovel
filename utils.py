import re
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader


load_dotenv()
deployment_name = 'gpt-4.1-nano'
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_API_ENDPOINT = os.environ.get('OPENAI_API_ENDPOINT')
OPENAI_API_VERSION = os.environ.get('OPENAI_API_VERSION')

llm = AzureChatOpenAI(
    deployment_name=deployment_name,
    api_key=OPENAI_API_KEY,
    azure_endpoint=OPENAI_API_ENDPOINT,
    api_version=OPENAI_API_VERSION,
    temperature=0,
)
summarizer = load_summarize_chain(llm, chain_type="stuff")

def clean_text(text):
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def clean_data(data):
    data = [doc for doc in data if len(doc.page_content.strip()) > 20]
    for doc in data:
        doc.page_content = clean_text(doc.page_content)
    return data

def write_chunks(chunks, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in chunks:
            f.write(f"{repr(doc)}\n")

def read_chunks(input_path):
    chunks = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(eval(line))
    return chunks

def load_documents(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File path does not exist: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    elif ext in [".doc", ".docx"]:
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Only .pdf, .txt, and .doc/.docx are supported.")

    data = loader.load()
    print("Loaded succesfully")
    return data
    


def up_level_chunking(chunks, n_content_chunks, level, log_checkpoint = 10, log_error = False):
    high_level_chunks = []
    index = 0
    for i in range(0, len(chunks), n_content_chunks):
        if log_checkpoint > 0 and i % log_checkpoint == 0:
            print(f"Processed chunks {i}/{len(chunks)} so far")
        try:
            group = chunks[i : i + n_content_chunks]
            summary = summarizer.invoke(group)
            doc = Document(
                page_content=summary['output_text'],
                metadata = {"level" : level, "group_index" : [j for j in range(i, i+n_content_chunks)], "chunk_index" : index}
            )
            high_level_chunks.append(doc)
            index += 1
        except Exception as e:
            if log_error:
                print(f"[!] Error at group {i // n_content_chunks}: {e}")
            continue
    return high_level_chunks


