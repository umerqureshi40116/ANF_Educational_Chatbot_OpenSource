################################################STREAMLIT APP BELOW with GROQ VERSION################################################33


import streamlit as st
import os
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load env variables
load_dotenv()

# Retrieval settings
K_RETRIEVE = 20     # get more candidates first
K_RETURN   = 3      # finally send top 3 to the LLM
THRESHOLD  = 0.4 # filter out weak matches

# LLM (Groq via LangChain)
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="deepseek-r1-distill-llama-70b",
    temperature=0.0
)

# Prompt template
PROMPT_TEMPLATE = """
You are ANF Academy EduBot.
- If the context contains relevant information, answer strictly based on it.
- If the context does NOT have the answer, then act as a friendly assistant and help the user conversationally.
- Always be polite, supportive, and clear.
- For legal questions, cite the statute/section numbers from the documents (not filenames).
- For general questions, give helpful and friendly answers.
- **Do not include any internal reasoning, thinking steps, or <think> tags in your response. Only provide the final answer.**

Context:
{context}

Conversation history:
{history}

User's question: {question}
"""

# Embedding function (FREE HuggingFace)
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
    model_kwargs={"device": "cpu"}
)

# Pinecone setup
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "anf-educational-bot"
db = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding_function
)

# Streamlit UI header
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {padding: 0; margin: 0; background-color: white;}
.chat {margin: 0; padding: 0; width: 100%; font-family: sans-serif;}
.header {width: 100%; background-color: #072e22; color: #fbfbfb; text-align: center; padding: 20px 0; margin: 0;}
.header img {width: 60px; height: 60px; border-radius: 10px; display: block; margin: 0 auto 10px auto;}
.header .title h1 {font-size: 30px; margin: 0; color: #fbfbfb;}
</style>
<div class="chat">
  <div class="header">
    <img src="https://upload.wikimedia.org/wikipedia/en/7/70/Anti-Narcotics_Force_Logo.png"/>
    <div class="title"><h1>ANF Academy Educational Chatbot</h1></div>
  </div>
</div>
""", unsafe_allow_html=True)

# Session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if query := st.chat_input("Ask a legal question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Build history
    history = ""
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    # Step 1: Retrieve many candidates
    raw_results = db.similarity_search_with_relevance_scores(query, k=K_RETRIEVE)

    # Step 2: Filter by threshold
    filtered = [(doc, score) for doc, score in raw_results if score >= THRESHOLD]

    # Step 3: Re-rank by cosine similarity (more accurate)
    if filtered:
        query_emb = embedding_function.embed_query(query)
        doc_embs = [embedding_function.embed_query(doc.page_content) for doc, _ in filtered]
        sims = cosine_similarity([query_emb], doc_embs)[0]
        reranked = sorted(zip(filtered, sims), key=lambda x: x[1], reverse=True)
        top_docs = [doc.page_content for (doc, _), _ in reranked[:K_RETURN]]
        context_text = "\n\n".join(top_docs)
    else:
        context_text = ""

    # Debug logs
    print("🔍 Retrieved Results (after filtering):")
    for (doc, score), sim in reranked[:K_RETURN] if filtered else []:
        print(f"Score: {score:.2f} | Cosine: {sim:.2f} | File: {doc.metadata.get('filename')}")

    # Step 4: Build prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, history=history, question=query)

    # Step 5: Get answer from LLM
    response_text = llm.predict(prompt)

    # Show assistant message
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    with st.chat_message("assistant"):
        st.markdown(response_text)