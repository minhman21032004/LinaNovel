import ast

from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool 
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage

from typing import Annotated, Sequence, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from utils import OPENAI_API_VERSION, OPENAI_API_KEY, OPENAI_API_ENDPOINT
from embedding import PERSIST_DIRECTORY, EMBEDDING_MODEL
from hierarchical_retriever import SingleRetriever, HierarchicalRetriever


LLM_MODEL = 'gpt-4.1'

levels = [1,2,3,4,5]
TOP_K = [10, 10, 7, 7, 5]

embedding = AzureOpenAIEmbeddings(model = EMBEDDING_MODEL, openai_api_version = OPENAI_API_VERSION, azure_endpoint = OPENAI_API_ENDPOINT, api_key = OPENAI_API_KEY)

llm = AzureChatOpenAI(
    api_version=OPENAI_API_VERSION,
    azure_endpoint=OPENAI_API_ENDPOINT,
    api_key=OPENAI_API_KEY,
    azure_deployment='gpt-4.1',
    temperature = 0
)

vector_stores = [
    Chroma(embedding_function=embedding, persist_directory=f"{PERSIST_DIRECTORY}/chunk_level_{level}")
    for level in levels
]

retrievers = [
    SingleRetriever(store, top_k=top_k, level_name=f"level_{level}")
    for store, top_k, level in zip(vector_stores, TOP_K, levels)
]

hirerachical_retriever = HierarchicalRetriever(retrievers)

@tool
def retrieve_by_level(query: str, level : int) -> str:
    ''' 
    This tool searchs and returns the information from the PDF file using similarity strategy
    Args:
        query (str) : The question user ask
        level (int) : A single number at the current number you want to search
    Return:
        str : The retrieval result
    '''

    level_name = f"level_{level}"
    docs = hirerachical_retriever.retrieve_by_level(query, level_name)

    if not docs:
        return "There is no relevant information in the document"
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)

@tool
def retrieve_across_level(query : str, high_level : int, low_level : int) -> str:
    '''  
    This tool searchs and returns the information from the PDF file using similarity strategy
    Relevant documents indices are searched from high_level (summarization) to low_level (more detail)
    Args:
        query (str) : The question user ask
        start_level : The level where searching start (high level)
        end_level : The level where searching stop (low level)
    Return :
        str : The retrieval result
    '''
    if low_level >= high_level:
        return "High_level should be a higher value"
    
    relevant_indices = []
    for level in range(high_level, low_level, -1):

        current_level = f"level_{level}"
        # print(f"Searching at {current_level} with relevant indices {relevant_indices}")
        
        docs = hirerachical_retriever.retrieve_by_level(query = query, level=current_level, indices=relevant_indices)

        if docs:
            relevant_indices = []
            for doc in docs:
                group_index = doc.metadata['group_index']
                group_index = ast.literal_eval(group_index)
                relevant_indices = relevant_indices + group_index

    final_level = f"level_{low_level}"
    # print(f"Searching at {final_level} with relevant indices {relevant_indices}\n")

    final_results = []
    final_docs = hirerachical_retriever.retrieve_by_level(query, final_level, relevant_indices)
    for i, doc in enumerate(final_docs):
        final_results.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n\n".join(final_results)

@tool
def cite_from_documents(keyword : str, high_level : int) -> list:
    '''  
    This tool searches and return the raw text and page from the PDF file using similarity strategy
    Relevant documents indices are searched from high_level (summarization) straight to level 1
    Args:
        keyword (str) : Some keyword of the result from high_level that you found
        high_level : The level where searching start (high level)
    Return :
        list : List of page : raw_text 
    '''
    low_level = 1
    if low_level >= high_level:
        return "High_level should be a higher value"
    
    relevant_indices = []
    for level in range(high_level, low_level, -1):

        current_level = f"level_{level}"
        # print(f"Searching at {current_level} with relevant indices {relevant_indices}")
        
        docs = hirerachical_retriever.retrieve_by_level(query = keyword, level=current_level, indices=relevant_indices)

        if docs:
            relevant_indices = []
            for doc in docs:
                group_index = doc.metadata['group_index']
                group_index = ast.literal_eval(group_index)
                relevant_indices = relevant_indices + group_index

    final_level = f"level_{low_level}"
    # print(f"Searching at {final_level} with relevant indices {relevant_indices}\n")

    final_docs = hirerachical_retriever.retrieve_by_level(keyword, final_level, relevant_indices)
    final_results = []
    for i, doc in enumerate(final_docs):
        final_results.append(f"Page {doc.metadata['page']}:\n{doc.page_content}")


    return final_docs

tools = [retrieve_by_level, retrieve_across_level, cite_from_documents]
tools_dict = {tool.name : tool for tool in tools}


llm = llm.bind_tools(tools)

system_prompts = ''

try:
    with open('configs/prompts.txt', 'r', encoding='utf-8') as f:
        system_prompt = f.read()
except FileNotFoundError:
    print("Cannot find system prompts file at 'configs/prompts.txt'")
except Exception as e:
    print(f"An error occurred while reading system prompts: {e}")


#Define Agent
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

def call_llm(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=system_prompts)] + list(state['messages'])
    result = llm.invoke(messages)
    return {'messages': [result]}

def tool_call(state: AgentState) -> AgentState:
    tool_calls = state['messages'][-1].tool_calls
    results = []

    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with args: {t['args']}")

        if t['name'] not in tools_dict:
            content = f"Tool '{t['name']}' not found."
        else:
            content = tools_dict[t['name']].invoke(t['args'])
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(content)))

    state['messages'] = state['messages'] + results
    return state

def create_app():
    G = StateGraph(AgentState)
    G.add_node('llm', call_llm)
    G.add_node('retriever_agent', tool_call)

    G.add_conditional_edges(
        source='llm',
        path=should_continue,
        path_map={
            True: 'retriever_agent',
            False: END
        }
    )

    G.add_edge('retriever_agent', 'llm')
    G.set_entry_point('llm')
    app = G.compile()
    return app