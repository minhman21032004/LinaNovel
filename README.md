
# 📚🌸 Lina Novel 

AI-powered reading assistant that helps **understand**, **summarize**, and **explore** novels, or light novels.

## Description
LinaNovel is an AI-powered anime-style reading companion designed to help users understand context of long documents. Built with **RAG (Retrieval-Augmented Generation)** and **Hierarchical Chunking** technique:

- **Question Answering:**  Ask questions about the source content and receive grounded, context-aware answers.

- **Summarization:** Get the general context of events, dialogues.
- **Explores:** Dive into themes, character arcs, relationships, and key events.
- **Citations:** Answers include references to the exact page in the original document.


## RAG Improvement

While traditional **RAG systems** often struggle with long documents and context fragmentation, **LinaNovel** enhances the retrieval and reasoning process through a hierarchical design:

-  **Better Context Preservation:** The hierarchical approach helps maintain the broader context of retrieved information.

-  **Improved Retrieval Efficiency:** Narrows down relevant sections only and to reduce irrelevant output.

-  **Scalability:** This method works for large documents, where flat retrieving cant keep the overall context.

- **Context-Aware Reasoning:** Hierachical structure allow for top-down abstraction and bottom-up detail recovery, enabling nuanced responses that respect both the micro and macro structure of the content.


## Work Flow
**LinaNovel** is built using LangGraph, a stateful agent framework. This structure lets user talk to a structured agent that retrieves and generates answers from deeply indexed books.

It uses a **5-level hierarchical chunking system**, ensuring contextually relevant answers across very long documents.
<img width="4884" height="6128" alt="image" src="https://github.com/user-attachments/assets/f53d1939-51a6-4a7d-aca2-f5077b718ada" />

## Getting Started

### Installation

#### **1. Clone the repository:**

```bash
git clone https://github.com/minhman21032004/LinaNovel.git
cd LinaNovel
```

#### **2. Create virtual environment and install dependencies:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

#### **3. Setup environment variables:**
create .env file in the same directory, in your .env file :
```bash
# Azure OpenAI configuration
OPENAI_API_KEY="your-azure-openai-key"
OPENAI_API_VERSION="2025-08-01"
OPENAI_API_ENDPOINT="https://your-resource-name.openai.azure.com/"
```
- Make sure your Azure OpenAI deployment name and version match the actual deployment you set up on the Azure portal.

#### **4. Embedding your documents:**

```bash
python3 embedding.py --file data/your_document.pdf
```
- To get started, you should place your documents in the data folder.
- Supported file types: .pdf, .txt, .doc
- You can use the sample document provided

### How to run:
Run the streamlit app :
```bash
streamlit run app.py
```

### Demo application:
**Sample Data: My Youth Romantic Comedy Is Wrong, as I Expected Vol.1**
Author : Wataru Watari
Publisher : YEN press
Pages : 294
Document type : Light Novel

**General Question:**
.            |  .
:-------------------------:|:-------------------------:
<img width="940" height="642" alt="demo_2" src="https://github.com/user-attachments/assets/3fcbeb2c-0bbe-4371-9a66-9599412a13e9" />  |   <img width="918" height="783" alt="demo_4" src="https://github.com/user-attachments/assets/1edceea6-670e-46f9-9b52-2a6ae7981ea8" />

**Interpretive Question:**
.            |  . | .
:-------------------------:|:-------------------------: | :-------------------------: 
<img width="947" height="661" alt="demo_5_1" src="https://github.com/user-attachments/assets/7961a272-281f-4007-a6bd-ef96168ab78b" /> | <img width="772" height="652" alt="demo_5_2" src="https://github.com/user-attachments/assets/b634f148-e744-43e2-a9ad-bb7a81874922" /> | <img width="881" height="538" alt="demo_5_3" src="https://github.com/user-attachments/assets/4db6e249-9fc7-40c8-a61d-818c103e89d9" />

**Specific Question:**
.            |  . | .
:-------------------------:|:-------------------------: | :-------------------------: 
<img width="957" height="832" alt="demo_3_1" src="https://github.com/user-attachments/assets/822b6ad3-083f-4082-bbb5-4c7b6b53c815" /> | <img width="972" height="851" alt="demo_3_2" src="https://github.com/user-attachments/assets/ea36912f-55d9-486b-b8bf-764149b1984c" /> | <img width="970" height="546" alt="demo_3_3" src="https://github.com/user-attachments/assets/1a92adb5-5ad8-4023-a83d-234d844ebd42" />

**Context-based Evaluative Question:**
.            |  . | . | .
:-------------------------:|:-------------------------: | :-------------------------: |  :-------------------------: 
<img width="682" height="782" alt="demo_6_1" src="https://github.com/user-attachments/assets/c89c0b1b-84b1-4c4b-b6d6-9dfff1d82413" /> | <img width="982" height="833" alt="demo_6_2" src="https://github.com/user-attachments/assets/a16fe12e-7b8e-48a3-b066-bcf380576d60" /> | <img width="1041" height="832" alt="demo_6_3" src="https://github.com/user-attachments/assets/64698bf1-c3f7-414c-a6a4-7096d9b4cbc5" /> | <img width="950" height="461" alt="demo_6_4" src="https://github.com/user-attachments/assets/10afe3c6-6536-4401-adb3-2443df8ed109" />

Question are answered with source text citation.

### Framework used:
+ LangChain
+ Chroma
+ LangGraph
+ AzureOpenAI
+ Streamlit

## Created By :
Manbell
