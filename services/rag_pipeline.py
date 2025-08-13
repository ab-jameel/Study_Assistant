# Import necessary libraries
from langchain_chroma import Chroma
from langchain_community import embeddings
from langchain_ollama import OllamaLLM
from langchain.chains.question_answering import load_qa_chain

# -------------------------------------------------------
# Create an embeddings model to convert text into vectors
# -------------------------------------------------------
embeddings_model = embeddings.OllamaEmbeddings(model="nomic-embed-text:latest")

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

    # Create a QA (Question-Answering) chain
    # "stuff" means it feeds all relevant docs into the prompt at once
    chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

    # Perform semantic search to find matching documents for the query
    matching_docs = vector_store.similarity_search(query)

    # Pass the matching documents and query into the QA chain
    response = chain.invoke({
        "input_documents": matching_docs,
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
