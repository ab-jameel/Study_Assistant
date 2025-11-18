# Import necessary libraries
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings  # Corrected import path
from langchain_ollama import OllamaLLM
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
# -------------------------------------------------------
# Create an embeddings model to convert text into vectors
# -------------------------------------------------------
embeddings_model = OllamaEmbeddings(model="nomic-embed-text:latest")
# -------------------------------------------------------
# Create a vector store (Chroma DB) from document chunks
# -------------------------------------------------------
def create_vector_store(chunks):
    # Store document chunks in Chroma DB with embeddings for semantic search
    return Chroma.from_documents(chunks, embeddings_model)
# -------------------------------------------------------
# Generate AI response with source references
# -------------------------------------------------------
def generate_response_with_sources(vector_store, query):
    # Initialize the LLM (Large Language Model) using Ollama's qwen2.5 model
    llm = OllamaLLM(model="qwen2.5:latest")
    # Define a prompt template (replicating the default from the old load_qa_chain)
    prompt = PromptTemplate.from_template(
        "Use the following pieces of context to answer the question at the end. "
        "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n"
        "{context}\n\n"
        "Question: {question}\n"
        "Helpful Answer:"
    )
    # Create the stuff documents chain (replacement for load_qa_chain with chain_type="stuff")
    chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    # Perform semantic search to find matching documents for the query
    matching_docs = vector_store.similarity_search(query)
    # Pass the matching documents and query into the chain
    response = chain.invoke({
        "context": matching_docs,
        "question": query
    })
    # -------------------------------------------------------
    # Build a list of sources for transparency
    # -------------------------------------------------------
    sources = []
    for doc in matching_docs:
        # Get page number if available, otherwise "unknown page"
        src_info = doc.metadata.get("page", "unknown page")
        # Take first 200 characters of text as a snippet (removing newlines)
        snippet = doc.page_content[:200].replace('\n', ' ')
        # Add formatted source info
        sources.append(f"Page: {src_info} | Text snippet: {snippet}")
    return response, sources
