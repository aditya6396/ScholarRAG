# ScholarRAG
## Overview
The RAG (Retrieval-Augmented Generation) Question Generator is a Streamlit-based web application designed to generate questions from uploaded PDF documents using advanced language models. It leverages retrieval-augmented generation techniques, supporting multiple AI models (Gemini, Groq, and xAI's Grok) to create context-aware questions and responses. The application also includes features for refining generated content and translating responses into various languages.

## Features
- **PDF Text Extraction**: Extracts text from uploaded PDF files with optional page range selection.
- **Question Generation**: Generates questions based on the extracted text using configurable AI models (Gemini, Groq, or Grok).
- **Response Refinement**: Allows users to refine generated questions or responses with additional context or adjustments.
- **Translation**: Translates generated or refined responses into multiple languages with domain-specific accuracy (e.g., Technical, Medical, Legal).
- **Model Flexibility**: Supports multiple AI models with customizable API keys for Gemini, Groq, and xAI's Grok.
- **Session Persistence**: Maintains chat history and response states across sessions using Streamlit's session state.

## Prerequisites
- Python 3.9+
- Streamlit
- PyPDF2
- SentenceTransformers
- LangChain
- FAISS
- Google Generative AI
- Groq SDK
- OpenAI SDK
- NumPy
- PyTorch
- Dotenv
- UUID
- Docker (optional for deployment)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd rag-question-generator
   ```

2. **Set Up Environment**:
   Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
   Note: Ensure `requirements.txt` includes all listed prerequisites.

3. **Configure API Keys**:
   - Create a `.env` file in the project root with the following (replace with your keys):
     ```
     GEMINI_API_KEY=your_gemini_api_key
     GROQ_API_KEY=your_groq_api_key
     GROK_API_KEY=your_xai_api_key
     ```
   - Alternatively, input API keys manually via the Streamlit sidebar during runtime.

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
   The application will be accessible at `http://localhost:8501`.

## Project Structure
- **`app.py`**: Main Streamlit application file containing UI configuration, text extraction, question generation, refinement, and translation logic.
- **`.env`**: Environment file for storing API keys (optional, can be managed via sidebar).
- **`requirements.txt`**: List of Python dependencies (to be created based on prerequisites).

## Usage

1. **Upload PDF**:
   - Use the file uploader to select a PDF document.
   - Optionally specify a page range (e.g., "1-5, 7") to extract text from specific pages.

2. **Generate Questions**:
   - Select an AI model (Gemini, Groq, or Grok) from the sidebar.
   - Enter a prompt or use default settings to generate questions based on the PDF content.
   - Review the generated questions and response text.

3. **Refine Responses**:
   - Provide additional context or instructions to refine the generated questions or responses.
   - Submit the refinement task to update the output.

4. **Translate Responses**:
   - Select a target language and domain (e.g., Technical, Medical) from the form.
   - Choose whether to preserve formatting.
   - Submit to translate the response, with the result displayed below.

## Configuration
- **API Keys**: Input via `.env` or sidebar text inputs.
- **Models**: Choose from Gemini, Groq (mixtral-8x7b-32768), or Grok (grok-beta/grok-4).
- **Embedding**: Uses `all-MiniLM-L6-v2` model with FAISS for efficient similarity search.
- **Device**: Automatically detects CUDA if available, otherwise uses CPU.

## Notes
- Ensure API keys are valid and have sufficient quota for the selected models.
- The application caches text extraction and embedding for performance.
- Translated responses are stored in session state for persistence.
- For deployment, consider using Docker with a `Dockerfile` to containerize the app.

## Troubleshooting
- **API Key Errors**: Verify keys in `.env` or sidebar inputs and ensure they are correctly formatted.
- **PDF Processing Issues**: Check file format and page range validity; use PyPDF2-compatible PDFs.
- **Model Errors**: Ensure internet connectivity and sufficient model quotas.
- **Performance**: Increase chunk size or adjust FAISS index parameters if memory issues arise.
