import os
import PyPDF2
from sentence_transformers import SentenceTransformer, util
import faiss
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import streamlit as st

tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdfs(uploaded_files):
    texts = []
    for file in uploaded_files:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        texts.append(text)
    return texts

def clean_text(text):
    text = text.replace("\n", " ").replace("\t", " ").strip()
    return " ".join(text.split())

def index_texts(texts):
    index = faiss.IndexFlatL2(384)
    documents = []
    for text in texts:
        chunks = [text[i:i+512] for i in range(0, len(text), 512)]
        for chunk in chunks:
            vector = embedding_model.encode(chunk)
            index.add(vector.reshape(1, -1))
            documents.append(chunk)
    return index, documents

def search_index(index, documents, query, top_k=3):
    query_vector = embedding_model.encode(query).reshape(1, -1)
    D, I = index.search(query_vector, k=top_k)

    if len(I[0]) == 0 or I[0][0] == -1:
        return "Não consegui encontrar informações relevantes no índice."

    combined_context = " ".join([documents[i] for i in I[0] if i != -1])
    return combined_context

def answer_question(context, question):
    chunks = split_context(context, max_tokens=512)
    best_answer = ""
    highest_score = float("-inf")

    for chunk in chunks:
        print("Context being passed to the model:", chunk)

        inputs = tokenizer(question, chunk, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)

        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        score = outputs.start_logits[0, answer_start].item() + outputs.end_logits[0, answer_end - 1].item()

        print(f"Start Logits: {outputs.start_logits}")
        print(f"End Logits: {outputs.end_logits}")
        print(f"Answer Start: {answer_start.item()} | Answer End: {answer_end.item()}")
        
        answer = tokenizer.decode(inputs.input_ids[0][answer_start:answer_end])
        print("Decoded Answer:", answer)
        if score > highest_score:
            highest_score = score
            best_answer = answer

    if not best_answer.strip():
        return "Não consegui encontrar uma resposta no contexto fornecido."
    return best_answer

def split_context(context, max_tokens=512):
    words = context.split()
    chunks = []
    chunk = []
    current_length = 0

    for word in words:
        token_length = len(tokenizer.encode(word))
        if current_length + token_length > max_tokens:
            if chunk:
                chunks.append(" ".join(chunk))
            chunk = [word]
            current_length = token_length
        else:
            chunk.append(word)
            current_length += token_length

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks

def validate_query(query):
    if len(query.strip()) < 2:
        return False
    return True

st.title("Assistente Conversacional Baseado em LLM")

uploaded_files = st.file_uploader("Envie arquivos PDF", type="pdf", accept_multiple_files=True)
if uploaded_files:
    st.info("Indexando documentos... Isso pode levar alguns instantes.")
    texts = extract_text_from_pdfs(uploaded_files)
    index, documents = index_texts(texts)
    st.success("Documentos indexados com sucesso!")
    st.session_state["index"] = index
    st.session_state["documents"] = documents

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "index" in st.session_state and "documents" in st.session_state:
    query_input = st.chat_input("Digite sua pergunta:")
    if query_input:
        query = query_input.strip()

        if not validate_query(query):
            answer = "Por favor, faça uma pergunta mais detalhada."
        else:
            context = search_index(st.session_state["index"], st.session_state["documents"], query)
            answer = answer_question(context, query)

        st.session_state["messages"].append({"role": "user", "content": query})
        st.session_state["messages"].append({"role": "assistant", "content": answer})


for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
