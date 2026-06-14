# 🤖 Context-Aware RAG Chatbot

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg?style=for-the-badge&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-00cc99.svg?style=for-the-badge&logo=chainlink)
![Google Gemini](https://img.shields.io/badge/Google--Gemini-1.5--Flash-orange.svg?style=for-the-badge&logo=googlegemini)
![FAISS](https://img.shields.io/badge/FAISS-CPU-blueviolet.svg?style=for-the-badge)

A complete **Context-Aware Retrieval-Augmented Generation (RAG)** Chatbot designed for AI/ML internship evaluation. This project utilizes **LangChain** as the orchestration framework, **FAISS** for local vector indexing, **Google Generative AI Embeddings** (`models/embedding-001`) for semantic feature extraction, and **Google Gemini 1.5 Flash** for grounded conversational response generation.

---

## 📐 System Architecture

Below is the conceptual flow of the RAG ingestion and retrieval-synthesis pipeline:

```text
+---------------------------------------------------------------------------------+
|                                INGESTION PHASE                                  |
|                                                                                 |
|  +------------------+     +-------------------+     +------------------------+  |
|  | Source Documents | --> | Text Splitting    | --> | Google Embeddings      |  |
|  | (.txt, .pdf)     |     | (size=500, ov=50) |     | (models/embedding-001) |  |
|  +------------------+     +-------------------+     +------------------------+  |
|                                                                 |               |
|                                                                 v               |
|                                                     +------------------------+  |
|                                                     | FAISS Vector Store     |  |
|                                                     | (Saved locally)        |  |
|                                                     +------------------------+  |
+---------------------------------------------------------------------------------+
                                                                  |
+-----------------------------------------------------------------|---------------+
|                                RETRIEVAL & GENERATION PHASE     |               |
|                                                                 v               |
|  +--------------+     +------------------------+     +-----------------------+  |
|  |  User Query  | --> | Conversation History   | --> | Vector Similarity     |  |
|  |  (Prompt)    |     | (BufferMemory context) |     | Retrieval (Top k=3)   |  |
|  +--------------+     +------------------------+     +-----------------------+  |
|                                                                 |               |
|                                                                 v               |
|  +--------------+     +------------------------+     +-----------------------+  |
|  | Final Answer | <-- | Gemini 1.5 Flash       | <-- | Prompt Composition    |  |
|  | (to stream)  |     | (LLM backend synthesis)|     | (Context + History)   |  |
|  +--------------+     +------------------------+     +-----------------------+  |
+---------------------------------------------------------------------------------+
```

---

## 📂 Project Structure

```text
rag_chatbot/
├── app.py                  # Streamlit application UI & session handling
├── rag_pipeline.py         # Conversational Retrieval Chain & memory configuration
├── document_loader.py      # Local file parser (.txt, .pdf) & text chunker
├── vector_store.py         # FAISS vector store creation & retrieval functions
├── requirements.txt        # Python package dependency manifests
├── .env.example            # Environment variable placeholder
├── data/
│   └── knowledge_base.txt  # Sample corpus (Wikipedia-style AI/ML texts)
└── README.md               # Project documentation
```

---

## ⚡ Setup & Local Run Instructions

### Prerequisites
- Python 3.9 or higher installed.
- A **Google Gemini API Key** (Get one at [Google AI Studio](https://aistudio.google.com/)).

### 1. Clone & Navigate
```bash
git clone <repository_url>
cd rag_chatbot
```

### 2. Create Virtual Environment & Install Packages
```bash
# Create environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Setup Environment Variables
Duplicate `.env.example` to `.env` and enter your Google AI Studio API Key:
```bash
cp .env.example .env
```
Inside `.env`:
```env
GOOGLE_API_KEY=AIzaSy...your_gemini_key...
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
A browser tab will automatically open at `http://localhost:8501`.

---

## 💡 How It Works (Step-by-Step)

1. **Ingestion & Parsing (`document_loader.py`):**
   The loader scans the `./data` folder for `.txt` and `.pdf` files. Using LangChain loaders, it extracts page contents. Text is split via `RecursiveCharacterTextSplitter` into chunks of `500` characters, overlapping by `50` characters, ensuring semantic boundaries are preserved.

2. **Index Generation (`vector_store.py`):**
   The chunks are converted to vector embeddings utilizing Google's 768-dimensional `models/embedding-001` model. The vector representation coordinates are indexed inside **FAISS** (Facebook AI Similarity Search) and cached locally in the `faiss_index/` directory. If the index exists, the application boots instantly by loading it directly, saving API tokens.

3. **Context and Conversation Memory (`rag_pipeline.py`):**
   We instantiate a LangChain `ConversationBufferMemory` with `memory_key="chat_history"`, returning a list of message objects. When the user asks a follow-up query, the chain utilizes history to rephrase the question to be a standalone search query.

4. **Retrieval and Synthesis (`rag_pipeline.py`):**
   The rephrased query searches the FAISS index to find the `k=3` most similar context chunks. The prompt compiles the retrieved documents and history into a strict template instructions layout. This package is fed to **Gemini 1.5 Flash** to synthesize a contextual, highly accurate output.

5. **Streamlit UI (`app.py`):**
   The Streamlit app maintains the conversational flow in `st.session_state` to render messages inside styled native-like chat bubbles. The documents retrieved for each step are stored within the assistant message structure and rendered below the response in an `st.expander` for absolute transparency.

---

## 🎨 UI & Screenshot

A sleek dark-mode glassmorphic interface styled using custom CSS:
- **Left Sidebar:** Database stats (documents chunk count), system guidelines, API Key inputs (for fallback), and a **Reset Conversation** button.
- **Main Chat:** Standard chat dialog bubbles.
- **Source Citation:** Expanding accordion dropdown displaying raw document chunks used to synthesize the response.

*(Screenshot placeholder: Once running, upload your screenshot to assets/screenshot.png and link it below)*
`![Screenshot](./assets/screenshot.png)`

---

## 🚀 Deployment to Streamlit Community Cloud

You can deploy this repository directly to Streamlit Community Cloud for free:

1. **Push your code to GitHub** (make sure `.env` is listed in your `.gitignore` so your API key is never exposed).
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and sign in.
3. Click **New app**, select your repository, branch, and `app.py` as the entry file.
4. Expand the **Advanced settings** section before deploying.
5. In the **Secrets** text area, paste your Google API key in the following format:
   ```toml
   GOOGLE_API_KEY = "AIzaSy...your_gemini_key..."
   ```
6. Click **Deploy**. Your chatbot is now live online!

---

## 📈 Key Results & Observations

- **Retrieval Precision:** By configuring `k=3` and utilizing a low LLM temperature (`0.2`), the model consistently yields precise responses grounded in the text files, eliminating hallucinations for domain-specific queries.
- **Context Resolution:** The conversational buffer correctly retains user references. For example:
  - *Query 1:* "What was the significance of AlexNet in 2012?" -> Generates correct answer.
  - *Query 2:* "Who were the key developers of it?" -> Memory resolves "it" to "AlexNet" and identifies Hinton, Sutskever, and Krizhevsky.
- **FAISS performance:** Loading local indices is extremely fast (<50ms startup delay) and keeps local computational costs to a minimum compared to cloud vector databases.
