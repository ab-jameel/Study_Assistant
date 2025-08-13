# Import necessary libraries
import os
import streamlit as st
from services.pdf_loader import extract_text_from_document, split_text_into_chunks
from services.rag_pipeline import generate_response_with_sources, create_vector_store

# -------------------------------
# Streamlit App Configuration
# -------------------------------
st.set_page_config(page_title="Study Assistant", layout="wide")
st.title("📚 Study Assistant (Streamlit + LangChain + Ollama)")

# Get the current script's directory (used to temporarily save uploaded files)
script_directory = os.path.dirname(os.path.abspath(__file__))

# -------------------------------
# File Upload Section
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload your study material (PDF, PPTX, PPTM, TXT)",
    type=["pdf", "pptx", "pptm", "txt"]  # Allowed file types
)

if uploaded_file is not None:
    st.write(f"📂 Processing {uploaded_file.name}...")

    # Save the uploaded file temporarily to disk
    uploaded_file_path = os.path.join(script_directory, uploaded_file.name)
    with open(uploaded_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract raw text from the uploaded document
    raw_text = extract_text_from_document(uploaded_file_path)

    # Delete the file after processing to save space
    os.remove(uploaded_file_path)
    
    if raw_text:
        # Split extracted text into smaller overlapping chunks for vector search
        chunks = split_text_into_chunks(raw_text)
        
        # Create a vector store from the chunks (for semantic search)
        vector_store = create_vector_store(chunks)
        
        st.success("✅ File processed and indexed!")

        # -------------------------------
        # Query Input Section
        # -------------------------------
        query = st.text_input("Enter your query:")
        
        if query:
            with st.spinner("Thinking..."):
                # Generate AI answer + retrieve sources
                answer, sources = generate_response_with_sources(vector_store, query)

                # Display AI's answer
                st.markdown(f"**Answer:** {answer['output_text']}")

                # Display retrieved sources for transparency
                if sources:
                    st.markdown("### 📑 Sources")
                    for s in sources:
                        st.markdown(f"- {s}")
else:
    st.write("No file uploaded.")  # Show when no file is provided

# -------------------------------
# Footer (Signature)
# -------------------------------
st.markdown(
    """
    <div style='text-align: center; padding-top: 50px; color: gray; font-size: 14px;'>
        Made by <b>Abdullah Gamil</b>
    </div>
    """,
    unsafe_allow_html=True
)
