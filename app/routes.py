from flask import Blueprint, request, jsonify
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from openai import ChatCompletion

bp = Blueprint('routes', __name__)

@bp.route('/process', methods=['POST'])
def process_files():
    # Retrieve patient and policy files from the request
    patient_data = request.files['patient_file'].read().decode('utf-8')
    policy_data = request.files['policy_file'].read().decode('utf-8')

    # Chunking the data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    patient_chunks = text_splitter.split_text(patient_data)
    policy_chunks = text_splitter.split_text(policy_data)

    # Embedding the chunks using OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    patient_store = FAISS.from_texts(patient_chunks, embeddings)
    policy_store = FAISS.from_texts(policy_chunks, embeddings)

    # Example logic for generating a recommendation
    # For the sake of this demo, let's retrieve a few relevant chunks and pass them to GPT
    relevant_patient = patient_store.similarity_search('specific condition or data', k=3)
    relevant_policy = policy_store.similarity_search('specific policy rule', k=3)

    # Now process the chunks with GPT
    gpt_input = "\n".join(relevant_patient + relevant_policy)
    response = ChatCompletion.create(
        model="gpt-4",
        prompt=gpt_input,
        max_tokens=500
    )
    
    # Return the GPT's recommendation
    return jsonify({'recommendation': response.choices[0].text.strip()})
