"""
Building a Medical Research Assistant with LangGraph and RAG
==========================================================

This tutorial shows how to create a medical research assistant 
that uses Retrieval-Augmented Generation (RAG) to answer questions 
about PubMed papers.

Requirements:
- Check README

Structure:
1. Setup environment and tools
2. Configure the RAG components
3. Create the LangGraph agent
4. Build the interactive chat interface
"""

# 1. SETUP TOOLS AND ENVIRONMENT
# ===============================
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import os
from dotenv import load_dotenv
from rich.console import Console
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.retrievers import WikipediaRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError
from supabase import create_client
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import SupabaseVectorStore

# Load environment variables from .env file
load_dotenv()

# Initialize Rich console for pretty terminal output
console = Console()

# Set recursion limit for the agent
RECURSION_LIMIT = 10  #

# Get API keys and URLs from environment variables
LLM_APIKEY = os.getenv("LLM_APIKEY")
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://llm-api.cyverse.ai/v1")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Verify environment variables are loaded
console.print(f"Environment setup: [green]{'✓' if all([LLM_APIKEY, LLM_API_BASE, SUPABASE_URL, SUPABASE_KEY]) else '✗'}[/green]")


# 2. SETUP RAG COMPONENTS
# =======================

# Create embeddings adapter using SentenceTransformer
class SentenceTransformerEmbeddings:
    """Adapter class for SentenceTransformer to work with LangChain."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()


# Initialize LLM
def create_llm():
    """Create and return the LLM instance."""
    return ChatOpenAI(
        model="Llama-3.2-11B-Vision-Instruct",
        openai_api_key=LLM_APIKEY,
        openai_api_base=LLM_API_BASE,
    )


# Setup Supabase vector store and retriever
def setup_retrievers():
    """Set up and return the retrievers for RAG."""
    # Initialize Supabase client
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Initialize embeddings model
    st_model = SentenceTransformerEmbeddings()
    
    # Create Supabase vector store
    vectorstore = SupabaseVectorStore(
        client=supabase_client,
        embedding=st_model,
        table_name="pubmed_documents",
        query_name="match_documents",
    )
    
    # Create Supabase retriever
    supabase_vs_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Create a custom retriever class
    class PubMedRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str) -> list[Document]:
            return supabase_vs_retriever.get_relevant_documents(query)
    
    # Create retriever tools
    pubmed_retriever_tool = create_retriever_tool(
        retriever=PubMedRetriever(),
        name="pubmed_retriever",
        description="A tool to retrieve documents from my knowledge base created from PubMedCentral Database."
    )
    
    wiki_retriever_tool = create_retriever_tool(
        retriever=WikipediaRetriever(),
        name="wikipedia_retriever",
        description="Wikipedia retriever to search for medical terms and helping for information which is not in the knowledge base."
    )
    
    return [pubmed_retriever_tool, wiki_retriever_tool]


# 3. CREATE LANGGRAPH AGENT
# =========================

def create_agent():
    """Create and return the LangGraph ReAct agent."""
    # Initialize LLM
    llm = create_llm()
    
    # Get retriever tools
    tools = setup_retrievers()
    
    # Define system message for the agent
    system_message = """
    You are a assistant with expertise in analyzing PubMed papers. You have been given a pubmed_retriever tool, which gets the relevant document from the vector database. Your task is to provide answers to user queries according to the documents found in the vector database. You also have access to wikipedia_retriever tool which you can use to get understanding beyond the pubmed_retriever. Dont utilize the full recursion limit, in 5-10 retrievals, summarize your findings.
    """
    
    # Create LangGraph ReAct agent
    langgraph_agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_message
    )
    
    return langgraph_agent


# 4. MAIN CODE: INTERACTIVE CHAT INTERFACE
# ========================================

def process_tool_calls(chunk, tools_dict):
    """Process tool calls from agent and display their outputs."""
    if "agent" in chunk:
        for message in chunk["agent"]["messages"]:
            if "tool_calls" in message.additional_kwargs:
                tool_calls = message.additional_kwargs["tool_calls"]

                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_arguments = eval(tool_call["function"]["arguments"])
                    tool_query = tool_arguments["query"]

                    console.print(
                        f"\nAgent is calling tool [bold cyan]{tool_name}[/bold cyan] "
                        f"with query: [cyan]{tool_query}[/cyan]",
                    )

                    # Find and invoke the correct tool
                    for tool in tools_dict:
                        if tool.name == tool_name:
                            tool_response = tool.invoke({"query": tool_query})
                            break

                    console.print(
                        f"\n[green]Tool Response:[/green]\n{tool_response[:250]} .....\n",
                    )
            else:
                agent_answer = message.content
                console.print(f"\nAgent:\n{agent_answer}", style="bold green")


def main():
    """Main function to run the interactive chat interface."""
    # Create agent
    agent = create_agent()
    
    # Get tools for processing tool calls
    tools = setup_retrievers()
    
    console.print("[bold]Medical Research Assistant[/bold]")
    console.print("Ask questions about medical research papers. Type 'quit' to exit.")
    
    # Loop until the user chooses to quit
    while True:
        # Get user question
        user_question = input("\nYou: ")
        
        # Check if user wants to quit
        if user_question.lower() in ["quit", "exit", "bye"]:
            console.print("\nAgent: Thank you for using the Medical Research Assistant. Goodbye!", 
                         style="bold green")
            break
        
        try:
            # Stream the agent's response
            for chunk in agent.stream(
                {"messages": [HumanMessage(content=user_question)]},
                {"recursion_limit": RECURSION_LIMIT},
            ):
                # Process the chunks from the agent
                process_tool_calls(chunk, tools)
                
        except GraphRecursionError:
            console.print("\nAgent stopped due to reaching maximum recursion limit.", 
                         style="bold red")


if __name__ == "__main__":
    main()


#  What was the main objective of the study comparing lorazepam and pentobarbital? 