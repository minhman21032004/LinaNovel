import re
import os
from dotenv import load_dotenv

from langchain.chains.summarize import load_summarize_chain
from langchain_openai import AzureChatOpenAI
from langchain.docstore.document import Document
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader, TextLoader

from tqdm import tqdm

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

def retype_metadata(chunks):
    for doc in chunks:
        for key, value in doc.metadata.items():
            if isinstance(value, list):
                doc.metadata[key] = str(value)

def write_chunks(chunks, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
    print(f"Data loaded succesfully with total {len(data)} pages!")
    return data
    
def up_level_chunking(chunks, n_content_chunks, level, log_error=False):
    high_level_chunks = []
    index = 0

    for i in tqdm(range(0, len(chunks), n_content_chunks), desc=f"Level {level} Chunking"):
        try:
            group = chunks[i : i + n_content_chunks]
            summary = summarizer.invoke(group)
            doc = Document(
                page_content=summary['output_text'],
                metadata={
                    "level": level,
                    "group_index": list(range(i, i + n_content_chunks)),
                    "chunk_index": index
                }
            )
            high_level_chunks.append(doc)
            index += 1
        except Exception as e:
            if log_error:
                msg = str(e).lower()
                if "content management policy" in msg or "response was filtered" in msg or "content filter" in msg:
                    print(f"[!] Warning at group {i // n_content_chunks} : {e}")
                else:
                    print(f"[!] Error at group {i // n_content_chunks}: {e}")
            continue

    return high_level_chunks



