import os
from collections import deque

try:
    from langchain.chains import ConversationalRetrievalChain
except ImportError:
    from langchain_classic.chains import ConversationalRetrievalChain

from langchain_community.vectorstores import FAISS

try:
    from langchain.prompts import (
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
        ChatPromptTemplate,
    )
except ImportError:
    from langchain_core.prompts import (
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
        ChatPromptTemplate,
    )

try:
    from langchain_groq import ChatGroq
except ImportError:
    from langchain_community.llms import ChatGroq


# -----------------------------
# HYBRID MEMORY IMPLEMENTATION
# -----------------------------
class HybridMemory:
    """
    Combines:
    1. Running summary (long-term memory)
    2. Last N messages (short-term memory)
    """

    def __init__(self, window_size=4):
        self.window_size = window_size
        self.chat_history = deque(maxlen=window_size)
        self.summary = ""

    def add_message(self, role, content):
        self.chat_history.append({"role": role, "content": content})

    def clear(self):
        self.chat_history.clear()
        self.summary = ""

    def get_history_text(self):
        return "\n".join(
            [f"{m['role']}: {m['content']}" for m in self.chat_history]
        )

    def update_summary(self, llm):
        """
        Optional: call occasionally to compress history.
        """
        if len(self.chat_history) < self.window_size:
            return

        text = self.get_history_text()

        prompt = f"""
Summarize the following conversation in a short, compact form.
Focus on user intent, business needs, and important details.

Conversation:
{text}

Existing Summary:
{self.summary}

Updated Summary:
"""

        response = llm.invoke(prompt)
        self.summary = response.content if hasattr(response, "content") else response


# -----------------------------
# RAG CHAIN
# -----------------------------
def get_rag_chain(vector_store: FAISS):

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY not found")

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        api_key=groq_key,
        temperature=0.2,
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

    memory = HybridMemory(window_size=4)

    system_template = """You are Cairo, the official AI receptionist of MA Technologia, a web and AI development agency based in Islamabad, Pakistan.

ROLE
Help users understand services and convert them into leads.


PRICING RULE
No fixed pricing. Always suggest free consultation.

BEHAVIOR
- Match user language
- Be concise
- Never hallucinate


Context:
{context}

Long-term Memory Summary:
{summary}

Recent Chat:
{chat_history}
"""

    human_template = "{question}"

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ])

    class CustomRAGChain:
        def __init__(self, llm, retriever, prompt):
            self.llm = llm
            self.retriever = retriever
            self.prompt = prompt
            
        def __call__(self, inputs: dict) -> dict:
            return self.invoke(inputs)
            
        def invoke(self, inputs: dict) -> dict:
            question = inputs.get("question", "")
            chat_history = inputs.get("chat_history", "")
            summary = inputs.get("summary", "")
            
            # Retrieve documents
            docs = self.retriever.invoke(question)
            
            # Format context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Format prompt values
            prompt_value = self.prompt.format_prompt(
                context=context,
                chat_history=chat_history,
                summary=summary,
                question=question
            )
            
            # Invoke LLM
            response = self.llm.invoke(prompt_value.to_messages())
            answer = response.content if hasattr(response, "content") else response
            
            return {
                "answer": answer,
                "source_documents": docs
            }

    chain = CustomRAGChain(llm, retriever, prompt)
    return chain, memory, llm


# -----------------------------
# USAGE PATTERN (IMPORTANT)
# -----------------------------
"""
You must manually handle memory like this in your API layer:

1. memory.add_message("user", question)
2. memory.add_message("ai", answer)

3. Occasionally:
   memory.update_summary(llm)

4. Send to chain:
   {
     question,
     chat_history = memory.get_history_text(),
     summary = memory.summary
   }
"""

if __name__ == "__main__":
    from dotenv import load_dotenv
    from vector_store import get_or_create_vector_store
    load_dotenv()

    # Simple self-test
    try:
        store = get_or_create_vector_store("./data", "./faiss_index")
        chain, memory, llm = get_rag_chain(store)
        print("RAG chain successfully built!")

        # Test query
        test_query = "What services does MA Technologia offer?"
        print(f"Testing Query: '{test_query}'")
        
        memory.add_message("user", test_query)
        chat_history = memory.get_history_text()
        summary = memory.summary
        
        res = chain({
            "question": test_query,
            "chat_history": chat_history,
            "summary": summary
        })

        print("\n--- LLM Response ---")
        print(res.get("answer"))
        print("\n--- Source Documents ---")
        for i, doc in enumerate(res.get("source_documents", [])):
            print(f"[{i+1}] Source: {doc.metadata.get('source')} | Preview: {doc.page_content[:150]}...")

    except Exception as err:
        import traceback
        traceback.print_exc()
        print(f"Pipeline verification failed: {err}")