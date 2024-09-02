# server.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb import Client
from langchain_community.llms.ollama import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.documents import Document

app = Flask(__name__)

# Cargar el archivo CSV con delimitador ;
file_path = 'Base_conocimiento_pre.csv'
df = pd.read_csv(file_path, delimiter=';')
df['Pregunta'] = df['Pregunta'].fillna('')

# Dividir los datos en fragmentos si es necesario
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = [Document(page_content=row['Pregunta']) for _, row in df.iterrows()]
all_splits = text_splitter.split_documents(documents)

# Crear embeddings para las preguntas
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}

try:
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
except Exception as ex:
    print("Exception: ", ex)
    local_model_path = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings_model = HuggingFaceEmbeddings(model_name=local_model_path, model_kwargs=model_kwargs)

# Configurar Chroma
settings = Settings()
client = Client(settings=settings)
collection = client.create_collection(name="qa_collection")

# Generar embeddings para todos los fragmentos
all_embeddings = [embeddings_model.embed_documents([doc.page_content])[0] for doc in all_splits]
ids = [str(i) for i in range(len(all_splits))]

# Agregar todos los documentos a la colección
collection.add(
    documents=[doc.page_content for doc in all_splits],
    embeddings=all_embeddings,
    ids=ids
)

# Configuración de los modelos Ollama
llm_gemma2 = Ollama(model="gemma2")
llm_llama3 = Ollama(model="llama3")

prompt_template_gemma2 = ChatPromptTemplate.from_messages(
    [
        ("system", """Eres una IA llamada Espesito. Tu tarea principal es responder preguntas simples sobre la Universidad de las Fuerzas Armadas "ESPE", basándote en la información proporcionada en las preguntas frecuentes.
        Cuando no tengas una respuesta exacta, responde de acuerdo a lo que has aprendido y el contexto dado en el historial de chat."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

prompt_template_llama3 = ChatPromptTemplate.from_messages(
    [
        ("system", """Eres una IA llamada Espesito. Tu tarea principal es responder preguntas detalladas y complejas sobre la Universidad de las Fuerzas Armadas "ESPE", basándote en la información proporcionada en las preguntas frecuentes.
        Usa toda la información que tengas disponible para proporcionar una respuesta completa y precisa."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

chain_gemma2 = prompt_template_gemma2 | llm_gemma2
chain_llama3 = prompt_template_llama3 | llm_llama3

def obtener_respuesta(pregunta: str, collection, df, model):
    query_embedding = model.embed_documents([pregunta])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    ids_list = results['ids'][0]
    distances_list = results['distances'][0]

    if not ids_list:
        return None, float('inf'), []

    results_combined = list(zip(ids_list, distances_list))
    sorted_results = sorted(results_combined, key=lambda x: x[1])
    best_result = sorted_results[0]
    doc_id, distance = best_result
    index = int(doc_id)
    respuesta = df.loc[index, 'Respuesta']
    additional_context = [df.loc[int(doc_id), 'Respuesta'] for doc_id, _ in sorted_results[1:3]]
    return respuesta, distance, additional_context

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/about')
# def about():
#     return render_template('about.html')

# @app.route('/chatbot')
# def chatbot():
#     return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    pregunta = data.get('pregunta', '')
    chat_history = data.get('chat_history', [])
    
    if len(pregunta.split()) > 10:
        model = llm_llama3
        chain = chain_llama3
    else:
        model = llm_gemma2
        chain = chain_gemma2

    answer, distance, additional_context = obtener_respuesta(pregunta, collection, df, embeddings_model)
    
    if answer and distance < 0.5:
        context = f"La respuesta a tu pregunta es: {answer}. Información adicional: {' '.join(additional_context)}"
        chat_history.append({"role": "user", "content": pregunta})
        chat_history.append({"role": "assistant", "content": answer})
        
        refined_response = chain.invoke({"input": context, "chat_history": chat_history})
        chat_history.append({"role": "assistant", "content": refined_response})
        return jsonify({"respuesta": refined_response, "chat_history": chat_history})
    else:
        response = chain.invoke({"input": pregunta, "chat_history": chat_history})
        chat_history.append({"role": "user", "content": pregunta})
        chat_history.append({"role": "assistant", "content": response})
        return jsonify({"respuesta": response, "chat_history": chat_history})

if __name__ == '__main__':
    app.run(debug=True)
