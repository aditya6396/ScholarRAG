import streamlit as st
import os
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import faiss
from dotenv import load_dotenv
import uuid
import torch
from groq import Groq
from openai import OpenAI

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="RAG Question Generator", layout="wide")

# --- Initialize Session State ---
if "response_text" not in st.session_state:
    st.session_state.response_text = None
if "context_text" not in st.session_state:
    st.session_state.context_text = None
if "translated_text" not in st.session_state:
    st.session_state.translated_text = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "task_submitted" not in st.session_state:
    st.session_state.task_submitted = False
if "refine_submitted" not in st.session_state:
    st.session_state.refine_submitted = False

# --- API Key Management ---
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY") or st.sidebar.text_input("🔑 Gemini API Key", type="password", help="Enter your Gemini API key", key="gemini_api_key")
groq_api_key = os.getenv("GROQ_API_KEY") or st.sidebar.text_input("🔑 Groq API Key", type="password", help="Enter your Groq API key", key="groq_api_key")
xai_api_key = os.getenv("GROK_API_KEY") or st.sidebar.text_input("🔑 xAI API Key", type="password", help="Enter your xAI API key for Grok models", key="xai_api_key")
if not gemini_api_key and not groq_api_key and not xai_api_key:
    st.sidebar.warning("Please enter at least one API key (Gemini, Groq, or xAI).")
    st.stop()

# --- Configure APIs ---
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
if xai_api_key:
    xai_client = OpenAI(api_key=xai_api_key, base_url="https://api.x.ai/v1")

# --- Cache Embedder ---
@st.cache_resource
def load_embedder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer("all-MiniLM-L6-v2", device=device)

embedder = load_embedder()

# --- Parse Page Range ---
def parse_page_range(page_range, total_pages):
    selected_pages = set()
    try:
        for part in page_range.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                if start < 1 or end > total_pages or start > end:
                    raise ValueError("Invalid page range.")
                selected_pages.update(range(start - 1, end))
            else:
                page = int(part.strip())
                if page < 1 or page > total_pages:
                    raise ValueError("Invalid page number.")
                selected_pages.add(page - 1)
        return selected_pages
    except ValueError as e:
        st.error(f"Invalid page range format: {str(e)}")
        st.stop()

# --- Extract Text from PDF ---
@st.cache_data
def extract_text_from_pdf(uploaded_file, selected_pages=None, _file_name=None):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        total_pages = len(pdf_reader.pages)
        if selected_pages is None:
            selected_pages = range(total_pages)
        text = ""
        for i in selected_pages:
            if 0 <= i < total_pages:
                page_text = pdf_reader.pages[i].extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return ""

# --- Chunk Text ---
@st.cache_data
def chunk_text(text, _text_hash=None):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)

# --- Create Vector Store ---
@st.cache_data
def create_vector_store(text_chunks, _embedder_hash=None):
    try:
        embeddings = embedder.encode(text_chunks, show_progress_bar=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index, embeddings, text_chunks
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        st.stop()

# --- Sidebar for Task, Model, and Chatbot ---
st.sidebar.title("📋 Task Selection")
task = st.sidebar.selectbox(
    "Choose Your Task",
    ["Generate Questions", "Summarize Context", "Content Writing", "Translation"],
    format_func=lambda x: f"{'📘' if x == 'Generate Questions' else '📝' if x == 'Summarize Context' else '🧾' if x == 'Content Writing' else '🌐'} {x}",
    help="Select a task to perform with your PDF content",
    key="task_selectbox"
)

st.sidebar.title("🤖 Model Selection")
available_models = []
if gemini_api_key:
    available_models.append("Gemini (gemini-2.5-pro)")
if groq_api_key:
    available_models.append("Groq (llama-3.3-70b-versatile)")
if xai_api_key:
    available_models.extend(["Grok (grok-beta)", "Grok (grok-4)"])
if not available_models:
    st.sidebar.error("No models available. Please provide at least one API key.")
    st.stop()
selected_model = st.sidebar.selectbox(
    "Choose Your Model",
    available_models,
    help="Select the AI model to use for content generation and chatbot",
    key="model_selectbox"
)

# --- Chatbot in Sidebar ---
# st.sidebar.title("💬 Chatbot")
# st.sidebar.subheader("Chatbot Settings")
# response_length = st.sidebar.selectbox(
#     "Response Length",
#     ["Short (100 tokens)", "Medium (200 tokens)", "Long (500 tokens)"],
#     help="Control the length of chatbot responses",
#     key="response_length_selectbox"
# )
# max_tokens_map = {"Short (100 tokens)": 100, "Medium (200 tokens)": 200, "Long (500 tokens)": 500}
# chat_max_tokens = max_tokens_map[response_length]

# if st.sidebar.button("Clear Chat History", help="Reset the chatbot conversation", key="clear_chat_button"):
#     st.session_state.chat_history = []
#     st.rerun()

# chat_container = st.sidebar.container()
# with chat_container:
#     with st.expander("View Chat History", expanded=False):
#         for chat in st.session_state.chat_history[-4:]:
#             if chat["role"] == "user":
#                 st.markdown(f"**You**: {chat['content']}")
#             else:
#                 st.markdown(f"**Bot**: {chat['content']}")

# with st.sidebar.form(key="chat_form"):
#     chat_input = st.text_input("Ask a question...", placeholder="Type your message here", key="chat_input")
#     submit_chat = st.form_submit_button("Send")
#     if submit_chat and chat_input.strip():
#         st.session_state.chat_history.append({"role": "user", "content": chat_input})
#         try:
#             with st.spinner("Generating response..."):
#                 if selected_model.startswith("Gemini"):
#                     if not gemini_api_key:
#                         st.sidebar.error("Gemini API key required for chatbot.")
#                         st.stop()
#                     model = genai.GenerativeModel("gemini-2.5-pro")
#                     response = model.generate_content(
#                         chat_input,
#                         generation_config={"max_output_tokens": chat_max_tokens}
#                     )
#                     bot_response = response.text
#                 elif selected_model.startswith("Groq"):
#                     if not groq_api_key:
#                         st.sidebar.error("Groq API key required for chatbot.")
#                         st.stop()
#                     response = groq_client.chat.completions.create(
#                         messages=[
#                             {"role": "system", "content": "You are a helpful AI assistant. Provide concise, clear answers limited to a few sentences unless asked for detail."},
#                             {"role": "user", "content": chat_input}
#                         ],
#                         model="llama-3.3-70b-versatile",
#                         max_tokens=chat_max_tokens
#                     )
#                     bot_response = response.choices[0].message.content
#                 else:  # Grok (xAI)
#                     if not xai_api_key:
#                         st.sidebar.error("xAI API key required for Grok chatbot.")
#                         st.stop()
#                     model_name = "grok-beta" if selected_model == "Grok (grok-beta)" else "grok-4"
#                     response = xai_client.chat.completions.create(
#                         model=model_name,
#                         messages=[
#                             {"role": "system", "content": "You are a helpful AI assistant. Provide concise, clear answers limited to a few sentences unless asked for detail."},
#                             {"role": "user", "content": chat_input}
#                         ],
#                         max_tokens=chat_max_tokens
#                     )
#                     bot_response = response.choices[0].message.content
#                 st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
#                 if len(st.session_state.chat_history) > 10:
#                     st.session_state.chat_history = st.session_state.chat_history[-10:]
#                 st.rerun()
#         except Exception as e:
#             st.sidebar.error(f"Chatbot error: {str(e)}")
########################updated code for chat section################################
# --- Chatbot in Sidebar ---
# --- Chatbot in Sidebar ---
st.sidebar.title("💬 Chatbot")
st.sidebar.subheader("Chatbot Settings")
response_length = st.sidebar.selectbox(
    "Response Length",
    ["Short (600 tokens)", "Medium (1000 tokens)", "Long (2500 tokens)"],
    help="Control the length of chatbot responses (word limits for Gemini, token limits for Groq/Grok)",
    key="response_length_selectbox"
)
max_tokens_map = {"Short (600 tokens)": 600, "Medium (1000 tokens)": 1000, "Long (2500 tokens)": 2500}
chat_max_tokens = max_tokens_map[response_length]

if st.sidebar.button("Clear Chat History", help="Reset the chatbot conversation", key="clear_chat_button"):
    st.session_state.chat_history = []
    st.rerun()

# Initialize session state for chat submission and input
if "chat_submitted" not in st.session_state:
    st.session_state.chat_submitted = False
if "chat_input_key" not in st.session_state:
    st.session_state.chat_input_key = str(uuid.uuid4())

chat_container = st.sidebar.container()
with chat_container:
    with st.expander("View Chat History", expanded=False):
        for chat in st.session_state.chat_history[-4:]:
            if chat["role"] == "user":
                st.markdown(f"**You**: {chat['content']}")
            else:
                st.markdown(f"**Bot**: {chat['content']}")

# Form and response container
with st.sidebar.form(key="chat_form"):
    chat_input = st.text_input(
        "Ask a question...",
        placeholder="Type your message here",
        help="Avoid sensitive topics to prevent responses from being blocked by safety filters.",
        key=st.session_state.chat_input_key
    )
    submit_chat = st.form_submit_button("Send")

# Response container immediately below the form
response_container = st.sidebar.container()
if submit_chat and chat_input.strip() and not st.session_state.chat_submitted:
    st.session_state.chat_submitted = True
    # Check for duplicate user input
    if not st.session_state.chat_history or st.session_state.chat_history[-1]["content"] != chat_input or st.session_state.chat_history[-1]["role"] != "user":
        st.session_state.chat_history.append({"role": "user", "content": chat_input})
    try:
        with st.spinner("Generating response..."):
            if selected_model.startswith("Gemini"):
                if not gemini_api_key:
                    st.sidebar.error("Gemini API key required for chatbot.")
                    st.session_state.chat_submitted = False
                    st.stop()
                # Set system prompt based on response length
                length_instruction = {
                    "Short (600 tokens)": "Keep responses concise, around 2-3 sentences or 30-50 words.",
                    "Medium (1000 tokens)": "Keep responses clear, around 4-6 sentences or 60-100 words.",
                    "Long (2500 tokens)": "Provide detailed responses, around 8-10 sentences or 120-200 words."
                }[response_length]
                system_prompt = f"You are a helpful AI assistant. {length_instruction} Avoid sensitive content to comply with safety policies."
                model = genai.GenerativeModel("gemini-2.5-pro")
                response = model.generate_content(
                    [
                        {"role": "user", "parts": [{"text": system_prompt}]},
                        {"role": "user", "parts": [{"text": chat_input}]}
                    ],
                    safety_settings={
                        genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    }
                )
                # Check if response was blocked or truncated
                if not response.candidates:
                    bot_response = "No response received from Gemini. Please try again."
                elif response.candidates[0].finish_reason == 1:
                    bot_response = "Response was incomplete or truncated. Please try rephrasing your question or select a different model."
                elif response.candidates[0].finish_reason == 2:
                    bot_response = "Sorry, the response was blocked due to safety concerns. Please try rephrasing your question."
                else:
                    try:
                        bot_response = response.text
                    except ValueError as e:
                        bot_response = f"Error accessing response: {str(e)}. Please try a different question."
                        # Log for debugging
                        st.sidebar.warning(f"Debug info: {response.candidates[0].__dict__}")
            elif selected_model.startswith("Groq"):
                if not groq_api_key:
                    st.sidebar.error("Groq API key required for chatbot.")
                    st.session_state.chat_submitted = False
                    st.stop()
                response = groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant. Provide concise, clear answers limited to a few sentences unless asked for detail."},
                        {"role": "user", "content": chat_input}
                    ],
                    model="llama-3.3-70b-versatile",
                    max_tokens=chat_max_tokens
                )
                bot_response = response.choices[0].message.content
            else:  # Grok (xAI)
                if not xai_api_key:
                    st.sidebar.error("xAI API key required for Grok chatbot.")
                    st.session_state.chat_submitted = False
                    st.stop()
                model_name = "grok-beta" if selected_model == "Grok (grok-beta)" else "grok-4"
                response = xai_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant. Provide concise, clear answers limited to a few sentences unless asked for detail."},
                        {"role": "user", "content": chat_input}
                    ],
                    max_tokens=chat_max_tokens
                )
                bot_response = response.choices[0].message.content
            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
            if len(st.session_state.chat_history) > 10:
                st.session_state.chat_history = st.session_state.chat_history[-10:]
            with response_container:
                st.markdown(
                    '<div style="margin-top: 5px;">**Latest Response**</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-top: 5px;">{bot_response}</div>',
                    unsafe_allow_html=True
                )
            # Clear input field by updating the key
            st.session_state.chat_input_key = str(uuid.uuid4())
            st.session_state.chat_submitted = False
            st.rerun()
    except Exception as e:
        with response_container:
            st.error(f"Chatbot error: {str(e)}")
        st.session_state.chat_submitted = False

# --- Main Content ---
st.title("📘 RAG-based Question Generator from PDFs")
st.markdown("Upload your PDFs, select a task and model from the sidebar, and generate questions, summaries, content, or translations. Use the chatbot below for quick questions.")

# --- Upload PDF(s) ---
st.subheader("📄 Upload PDF Files")
pdf_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True, key="pdf_uploader")
if not pdf_files:
    st.info("Please upload at least one PDF file to proceed.")
    st.stop()

# --- Process PDFs ---
st.subheader("📝 Processing PDFs")
all_chunks = []
for file in pdf_files:
    pdf_reader = PyPDF2.PdfReader(file)
    total_pages = len(pdf_reader.pages)
    page_range = st.text_input(
        f"Pages to extract from '{file.name}' (e.g., 1,3-5)",
        key=f"page_range_{file.name}",
        placeholder="Leave empty for all pages",
        help="Specify pages like '1,3-5' or leave blank for all"
    )
    selected_pages = parse_page_range(page_range, total_pages) if page_range else None
    text = extract_text_from_pdf(file, selected_pages, _file_name=file.name)
    if text:
        chunks = chunk_text(text, _text_hash=hash(text))
        all_chunks.extend(chunks)

if not all_chunks:
    st.warning("No valid text chunks extracted from the PDFs.")
    st.stop()

# --- Create Vector Store ---
vector_store, embeddings, documents = create_vector_store(all_chunks, _embedder_hash=hash(str(embedder)))
st.success("✅ Documents processed and vector store created!")

# --- Task-Specific Input ---
st.subheader("🧠 Task Input")
with st.form(key="task_form"):
    query = st.text_input("Input Prompt or Query", placeholder="Enter your query or prompt here", help="Provide a query relevant to your selected task", key="query_input")

    if task == "Generate Questions":
        with st.expander("Question Generation Options", expanded=True):
            difficulty_level = st.slider("Difficulty Level", 1, 10, 5, help="1 = Easy, 10 = Hard", key="difficulty_slider")
            num_questions = st.number_input("Number of Questions", min_value=1, max_value=500, value=3, help="Select how many questions to generate", key="num_questions_input")
            num_options = st.number_input("Number of Options", min_value=0, max_value=6, value=4, help="0 for open-ended questions", key="num_options_input")
            question_length = st.slider("Question Length (words)", min_value=5, max_value=30, value=15, help="Approximate length of each question", key="question_length_slider")
    elif task == "Content Writing":
        with st.expander("Content Writing Options", expanded=True):
            content_type = st.selectbox("Content Type", ["Blog Post", "Product Description", "How-To Guide", "Technical Article"], help="Choose the type of content", key="content_type_selectbox")
            topic = st.text_input("Topic / Title", placeholder="Enter the topic or title", help="Specify the main topic or title", key="topic_input")
            target_audience = st.selectbox("Target Audience", ["General Public", "Beginners", "Tech Professionals"], help="Select the intended audience", key="target_audience_selectbox")
            tone = st.selectbox("Tone", ["Professional", "Conversational", "Persuasive", "Friendly"], help="Choose the tone of the content", key="tone_selectbox")
            seo_keywords = st.text_input("SEO Keywords", placeholder="Enter comma-separated keywords", help="Optional SEO keywords", key="seo_keywords_input")
    elif task == "Translation":
        with st.expander("Translation Options", expanded=True):
            source_language = st.selectbox("Source Language", ["English", "Spanish", "French", "German", "Chinese", "Arabic", "Auto-Detect"], help="Select the source language", key="source_language_selectbox")
            target_language = st.selectbox("Target Language", ["English", "Spanish", "French", "German", "Chinese", "Arabic", "Hindi"], help="Select the target language", key="target_language_selectbox")
            domain = st.selectbox("Domain / Context", ["General", "Technical", "Medical", "Legal", "Marketing"], help="Select the context for translation", key="task_domain_selectbox")
            preserve_formatting = st.checkbox("Preserve Formatting", value=True, help="Keep original formatting if checked", key="task_preserve_formatting")
            context_text = st.text_area("Text to Translate", height=200, placeholder="Enter the text to translate", help="Provide the text you want to translate", key="context_text_area")
    else:  # Summarize Context
        st.info("No additional options required for summarization.")

    submit_button = st.form_submit_button("🚀 Run Task", help="Execute the selected task")

# --- Run Task ---
MODEL_GEMINI = "gemini-2.5-pro"
MODEL_GROQ = "llama-3.3-70b-versatile"
REFINEMENT_MODEL = "gemini-2.5-flash"
if submit_button and not st.session_state.task_submitted:
    st.session_state.task_submitted = True
    if not query.strip():
        st.warning("Please enter a valid query.")
        st.session_state.task_submitted = False
        st.stop()
    if task == "Translation" and not context_text.strip():
        st.warning("Please enter text to translate.")
        st.session_state.task_submitted = False
        st.stop()
    if task == "Content Writing" and not topic.strip():
        st.warning("Please enter a valid topic.")
        st.session_state.task_submitted = False
        st.stop()

    # Create context from documents (except for translation)
    if task != "Translation":
        query_embedding = embedder.encode([query])[0]
        D, I = vector_store.search(np.array([query_embedding]), k=3)
        context = [documents[i] for i in I[0]]
        context_text = "\n".join(context)
    else:
        context_text = context_text  # Use user-provided text for translation

    # Store context for refinement
    st.session_state.context_text = context_text

    # Generate prompt based on task
    if task == "Generate Questions":
        prompt = f"""
You are an expert AI assistant specializing in creating educational content. Based on the user's specifications and the provided context, your task is to generate a set of high-quality questions.

### Parameters:
- **Difficulty Level:** {difficulty_level}
- **Number of Questions:** {num_questions}
- **Number of Options:** {num_options}

### Context:
'''
{context_text}
'''

### Instructions:
1. Generate exactly {num_questions} question(s).
2. Keep each question around {question_length} words.
3. Provide MCQs if options > 1; else open-ended.
4. Include correct answer and short explanation.
5. If context is insufficient, say so.
6. If {num_options} is greater than 1, generate Multiple Choice Questions (MCQs) with {num_options} options each.
   - Ensure only one option is correct based on the context.
   - Make the incorrect options (distractors) plausible but clearly wrong.
   - Maintain variability and uniqueness among the generated questions and try to keep the options in maximum 1 words.
   
7. For each question, provide the correct answer with a brief justification referencing the context.
8. do the numerical indexing of questions generated.
### Example MCQ Output Format (if num_options > 1):
**Question 1:** When was the Eiffel Tower built?
**Options:**
A) 1887
B) 1889
C) 1901
D) 1899
**Answer:** B) 1889. The context states it was "constructed in 1889."

Also output in JSON-style format:
```json
{{
  "query": "{query}",
  "contexts": ["{context[0]}"],
  "ground_truth": "...",
  "response": "...",
  "options": ["A", "B", "C", "D"]
}}
```

**BEGIN TASK**
"""
    elif task == "Summarize Context":
        prompt = f"""
You are a summarization expert. Based on the following context, generate a concise and informative summary.

### Context:
'''
{context_text}
'''

### Instructions:
1. Summarize the context in 20+ relevant sentences.
2. Focus on clarity and accuracy.
3. Avoid hallucination or adding unsupported details.
4. Maintain a neutral tone.
5. Return plain text summary.

**BEGIN TASK**
"""
    elif task == "Content Writing":
        prompt = f"""
You are an expert AI assistant specializing in content writing. Based on the specifications and the provided context, generate a high-quality piece of writing.

### Parameters:
- Content Type: {content_type}
- Topic: {topic}
- Target Audience: {target_audience}
- Tone: {tone}
- SEO Keywords: {seo_keywords}

### Context:
'''
{context_text}
'''

### Instructions:
1. Generate a professional {content_type}.
2. Use the specified tone and audience.
3. Incorporate SEO keywords if relevant.
4. Stay within the provided context.
5. Use headings or paragraphs as needed.
6. Do not add external info.

'''

### Instructions:
1. Maintain the original meaning, tone, and technical accuracy.
2. Adapt terminology appropriately for the {domain.lower()} domain.
3. {'Preserve original formatting, including line breaks and structure.' if preserve_formatting else 'You may reformat for clarity if needed.'}
4. Do not summarize or omit any content.
5. Return only the translated text without explanation or commentary.

**BEGIN TASK**
"""
    try:
        with st.spinner("Generating response..."):
            if selected_model.startswith("Gemini"):
                if not gemini_api_key:
                    st.error("Gemini API key required for Gemini model.")
                    st.session_state.task_submitted = False
                    st.stop()
                model = genai.GenerativeModel(MODEL_GEMINI)
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=2048
                    )
                )
                st.session_state.response_text = response.text
            elif selected_model.startswith("Groq"):
                if not groq_api_key:
                    st.error("Groq API key required for Groq model.")
                    st.session_state.task_submitted = False
                    st.stop()
                response = groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=MODEL_GROQ,
                    max_tokens=1024
                )
                st.session_state.response_text = response.choices[0].message.content
            else:  # Grok (xAI)
                if not xai_api_key:
                    st.error("xAI API key required for Grok model.")
                    st.session_state.task_submitted = False
                    st.stop()
                model_name = "grok-beta" if selected_model == "Grok (grok-beta)" else "grok-4"
                response = xai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024
                )
                st.session_state.response_text = response.choices[0].message.content
            st.session_state.translated_text = None  # Reset translated text
            st.session_state.task_submitted = False
            st.subheader("🧠 Current Output")
            st.markdown(st.session_state.response_text)
    except Exception as e:
        st.error(f"Generation error: {str(e)}")
        st.session_state.task_submitted = False
        st.stop()

# --- Refine Response ---
if st.button("🔄 Refine Response", help="Refine the generated output", key="refine_button") and not st.session_state.refine_submitted:
    st.session_state.refine_submitted = True
    if not st.session_state.response_text or not st.session_state.context_text:
        st.warning("No response available to refine. Please run the task first.")
        st.session_state.refine_submitted = False
        st.stop()
    
    with st.spinner("Refining..."):
        review_prompt = f"""
You are a reviewing assistant. Review the following content based on the context. Improve clarity, fix hallucinations, and correct grammar if needed.

### Context:
{st.session_state.context_text}

### Original Output:
{st.session_state.response_text}

### Instructions:
- Return improved version.
- Do not change structure or meaning.
- No additional content.
- If possible, make it concise.

**Begin Refinement**
"""
        try:
            if selected_model.startswith("Gemini"):
                if not gemini_api_key:
                    st.error("Gemini API key required for refinement.")
                    st.session_state.refine_submitted = False
                    st.stop()
                flash_model = genai.GenerativeModel(REFINEMENT_MODEL)
                refined = flash_model.generate_content(review_prompt)
                st.session_state.response_text = refined.text
            elif selected_model.startswith("Groq"):
                if not groq_api_key:
                    st.error("Groq API key required for refinement.")
                    st.session_state.refine_submitted = False
                    st.stop()
                refined = groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": review_prompt}],
                    model=MODEL_GROQ,
                    max_tokens=1024
                )
                st.session_state.response_text = refined.choices[0].message.content
            else:  # Grok (xAI)
                if not xai_api_key:
                    st.error("xAI API key required for Grok refinement.")
                    st.session_state.refine_submitted = False
                    st.stop()
                model_name = "grok-beta" if selected_model == "Grok (grok-beta)" else "grok-4"
                refined = xai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": review_prompt}],
                    max_tokens=1024
                )
                st.session_state.response_text = refined.choices[0].message.content
            st.session_state.translated_text = None  # Reset translated text
            st.session_state.refine_submitted = False
            st.subheader("🧠 Current Output")
            st.markdown(st.session_state.response_text)
        except Exception as e:
            st.error(f"Refinement error: {str(e)}")
            st.session_state.refine_submitted = False

# --- Translate Generated Response ---
if st.session_state.response_text:
    st.subheader("🌐 Translate Generated Response")
    with st.form(key="translate_form"):
        translate_target_language = st.selectbox(
            "Target Language for Translation",
            ["English", "Spanish", "French", "German", "Chinese", "Arabic", "Hindi"],
            help="Select the language to translate the generated response into",
            key="translate_target_language_selectbox"
        )
        translate_domain = st.selectbox(
            "Domain / Context",
            ["General", "Technical", "Medical", "Legal", "Marketing"],
            help="Select the context for translation",
            key="translate_domain_selectbox"
        )
        translate_preserve_formatting = st.checkbox("Preserve Formatting", value=True, help="Keep original formatting if checked", key="translate_preserve_formatting")
        translate_button = st.form_submit_button("🌍 Translate Response", help="Translate the generated or refined response")
    
    if translate_button:
        if not st.session_state.response_text:
            st.warning("No response available to translate. Please run the task first.")
            st.stop()
        
        translate_prompt = f"""
You are a professional translator with expertise in {translate_domain.lower()} content. Translate the following text to {translate_target_language}.

### Source Text:
'''
{st.session_state.response_text}
'''

### Instructions:
1. Maintain the original meaning, tone, and technical accuracy.
2. Adapt terminology appropriately for the {translate_domain.lower()} domain.
3. {'Preserve original formatting, including line breaks and structure.' if translate_preserve_formatting else 'You may reformat for clarity if needed.'}
4. Do not summarize or omit any content.
5. Return only the translated text without explanation or commentary.

**BEGIN TASK**
"""
        try:
            with st.spinner("Translating..."):
                if selected_model.startswith("Gemini"):
                    if not gemini_api_key:
                        st.error("Gemini API key required for translation.")
                        st.stop()
                    model = genai.GenerativeModel(MODEL_GEMINI)
                    translated_response = model.generate_content(
                        translate_prompt,
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=2048
                        )
                    )
                    st.session_state.translated_text = translated_response.text
                elif selected_model.startswith("Groq"):
                    if not groq_api_key:
                        st.error("Groq API key required for translation.")
                        st.stop()
                    translated_response = groq_client.chat.completions.create(
                        messages=[{"role": "user", "content": translate_prompt}],
                        model=MODEL_GROQ,
                        max_tokens=1024
                    )
                    st.session_state.translated_text = translated_response.choices[0].message.content
                else:  # Grok (xAI)
                    if not xai_api_key:
                        st.error("xAI API key required for Grok translation.")
                        st.stop()
                    model_name = "grok-beta" if selected_model == "Grok (grok-beta)" else "grok-4"
                    translated_response = xai_client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": translate_prompt}],
                        max_tokens=1024
                    )
                    st.session_state.translated_text = translated_response.choices[0].message.content
                st.subheader("🌍 Translated Response")
                st.markdown(st.session_state.translated_text)
        except Exception as e:
            st.error(f"Translation error: {str(e)}")

# --- Persist Translated Response ---
if st.session_state.translated_text:
    st.subheader("🌍 Translated Response")
    st.markdown(st.session_state.translated_text)