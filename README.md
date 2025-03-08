# Chat with PDF using Gemini  

This is a **Streamlit application** that allows users to upload PDF files, process their content, and ask questions using **Google's Gemini model**. The application utilizes **LangChain** for text processing and **FAISS** for vector storage and similarity search.  

## Features  
- **Upload PDF Files:** Users can upload one or more PDFs.  
- **Text Extraction:** Extracts text from uploaded PDFs.  
- **Text Chunking:** Splits extracted text into manageable chunks.  
- **Vector Storage:** Converts text chunks into embeddings and stores them in a **FAISS vector database**.  
- **Question Answering:** Users can ask questions about the PDF content and get responses powered by the **Gemini model**.  

## Prerequisites  
Before running the application, ensure you have the following:  

1. **Google API Key:**  
   - Obtain a **Google API key** with access to the **Generative Language API**.  
   - Enable the **Generative Language API** in the **Google Cloud Console**.  

2. **Python 3.9 or higher:**  
   - Ensure **Python** is installed on your system.  

3. **Dependencies:**  
   - Install the required Python packages listed in `requirements.txt`.  

## Dependencies  
The project relies on the following Python packages:  

- `streamlit`: For the web interface.  
- `PyPDF2`: For extracting text from PDFs.  
- `langchain`: For text processing and question-answering.  
- `google-generativeai`: For interacting with Google's **Gemini model**.  
- `faiss-cpu`: For vector storage and similarity search.  
- `python-dotenv`: For loading environment variables.  

For a full list of dependencies, see the `requirements.txt` file. 
