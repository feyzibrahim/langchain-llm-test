from openai import OpenAI
import requests
from flask import Blueprint, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter

client = OpenAI()

bp = Blueprint('routes', __name__)

OCR_SERVER_URL = ''

def extract_text_from_ocr_server(file):
    # Send the file to the OCR server
    response = requests.post(OCR_SERVER_URL, files={'file': file})
    response.raise_for_status()  # Raise an error for bad responses
    return response.text

@bp.route('/process', methods=['POST'])
def process_files():
    # Retrieve and process PDF files
    patient_file = request.files['patient_file']
    policy_file = request.files['policy_file']
    
    # Extract text from PDFs using the OCR server
    patient_text = extract_text_from_ocr_server(patient_file)
    policy_text = extract_text_from_ocr_server(policy_file)
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    patient_chunks = text_splitter.split_text(patient_text)
    policy_chunks = text_splitter.split_text(policy_text)

    # Summarize and analyze
    patient_summary = summarize_chunks(patient_chunks)
    policy_summary = summarize_chunks(policy_chunks)

 

    # Use the new OpenAI API interface
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that provides recommendations based on the provided data."},
            {"role": "user", "content": f"Based on the following patient data:\n{patient_summary}\nand policy data:\n{policy_summary}, provide a recommendation for prior authorization."}
        ],
        max_tokens=1000
    )
    
    return jsonify({'recommendation': response.choices[0].message['content'].strip()})

def summarize_chunks(chunks):
    summary = ""
    for chunk in chunks:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant that summarizes text."},
                {"role": "user", "content": f"Summarize the following text:\n\n{chunk}"}
            ],
            max_tokens=500
        )
        summary += response.choices[0].message['content'].strip() + "\n"
    return summary
