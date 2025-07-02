import os
import requests
import json
from flask import Flask
from twilio.twiml.messaging_response import MessagingResponse
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Load environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

# Validate required environment variables
required_env_vars = {
    "OPENROUTER_API_KEY": OPENROUTER_API_KEY,
    "TWILIO_ACCOUNT_SID": TWILIO_ACCOUNT_SID,
    "TWILIO_AUTH_TOKEN": TWILIO_AUTH_TOKEN
}

for var_name, var_value in required_env_vars.items():
    if not var_value:
        raise ValueError(f"Environment variable {var_name} is required but not set")

# Your custom GPT system prompt
CUSTOM_SYSTEM_PROMPT = """
-Respond as complete and concise as possible, make sure the information given is accurate. 
-Do not answer questions outside of the knowledge files. 
-For each response, give source, reference, and page number at the end of each response for each information mentioned (reference to the documents within the documents because each file uploaded may contain multiple documents).
-Crosscheck all of the information in the response with the reference. 
-Give the short conclusion first and follow with the explanation
"""

# ─── DOCUMENT PROCESSING & VECTOR SEARCH SETUP ─────────────────────────────────
# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Paths for storing documents and embeddings
DOCUMENTS_DIR = "documents"
EMBEDDINGS_FILE = "embeddings.pkl"
FAISS_INDEX_FILE = "faiss_index.bin"

# Ensure documents directory exists
Path(DOCUMENTS_DIR).mkdir(exist_ok=True)

# Global variables for vector search
document_chunks = []
document_metadata = []
faiss_index = None

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += f"[Page {page_num + 1}] {page_text}\n"
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text

def extract_text_from_docx(file_path):
    """Extract text from Word document"""
    text = ""
    try:
        doc = docx.Document(file_path)
        for para_num, paragraph in enumerate(doc.paragraphs):
            if paragraph.text.strip():
                text += f"[Paragraph {para_num + 1}] {paragraph.text}\n"
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
    return text

def extract_text_from_txt(file_path):
    """Extract text from text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            lines = content.split('\n')
            text = ""
            for line_num, line in enumerate(lines):
                if line.strip():
                    text += f"[Line {line_num + 1}] {line}\n"
            return text
    except Exception as e:
        print(f"Error reading TXT {file_path}: {e}")
        return ""

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def process_documents():
    """Process all documents in the documents directory and create embeddings"""
    global document_chunks, document_metadata, faiss_index
    
    document_chunks = []
    document_metadata = []
    
    # Process all documents in the documents directory
    for file_path in Path(DOCUMENTS_DIR).iterdir():
        if file_path.is_file():
            print(f"Processing {file_path.name}...")
            
            # Extract text based on file type
            if file_path.suffix.lower() == '.pdf':
                text = extract_text_from_pdf(file_path)
            elif file_path.suffix.lower() == '.docx':
                text = extract_text_from_docx(file_path)
            elif file_path.suffix.lower() == '.txt':
                text = extract_text_from_txt(file_path)
            else:
                print(f"Unsupported file type: {file_path.suffix}")
                continue
            
            if not text.strip():
                continue
                
            # Split into chunks
            chunks = chunk_text(text)
            
            # Store chunks and metadata
            for i, chunk in enumerate(chunks):
                document_chunks.append(chunk)
                document_metadata.append({
                    'file_name': file_path.name,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                })
    
    if document_chunks:
        # Create embeddings
        print("Creating embeddings...")
        embeddings = embedding_model.encode(document_chunks)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        faiss_index.add(embeddings.astype('float32'))
        
        print(f"Processed {len(document_chunks)} chunks from {len(set(meta['file_name'] for meta in document_metadata))} documents")
    else:
        print("No documents found to process")

def search_documents(query, top_k=3):
    """Search for relevant document chunks based on query"""
    global faiss_index, document_chunks, document_metadata
    
    if faiss_index is None or len(document_chunks) == 0:
        return []
    
    # Create query embedding
    query_embedding = embedding_model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    # Search
    scores, indices = faiss_index.search(query_embedding.astype('float32'), top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(document_chunks):  # Valid index
            results.append({
                'content': document_chunks[idx],
                'metadata': document_metadata[idx],
                'score': float(score)
            })
    
    return results

# Initialize document processing on startup
print("Loading documents...")
process_documents()

# ─── COLAB TESTING FUNCTIONS ─────────────────────────────────────────────────────
def test_chat_without_flask(user_message):
    """
    Test the chat functionality without Flask context - perfect for Google Colab
    """
    print(f"User message: {user_message}")
    
    # Search for relevant documents
    relevant_docs = search_documents(user_message, top_k=3)
    
    # Build context from retrieved documents
    context = ""
    if relevant_docs:
        context = "\n\nRelevant Information from Knowledge Base:\n"
        for i, doc in enumerate(relevant_docs, 1):
            context += f"\n--- Source {i}: {doc['metadata']['file_name']} ---\n"
            context += f"{doc['content']}\n"
        print("Found relevant documents!")
    else:
        print("No relevant documents found.")
    
    # Build the conversation payload with context
    system_prompt = CUSTOM_SYSTEM_PROMPT
    if context:
        system_prompt += f"\n\nUse the following context to answer the user's question:{context}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    # Call OpenRouter API
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://colab-test.example.com",
        "X-Title": "Colab Test Bot",
    }
    
    try:
        print("Calling OpenRouter API...")
        resp = requests.post(
            url=OPENROUTER_URL,
            headers=headers,
            data=json.dumps({
                "model": "deepseek/deepseek-r1-0528:free",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 500
            })
        )
        
        resp.raise_for_status()
        data = resp.json()
        
        # Extract the AI's reply
        answer = data["choices"][0]["message"]["content"].strip()
        if not answer:
            answer = "Sorry, I couldn't generate a response. Please try again."
            
        print(f"\nAI Response:\n{answer}")
        return answer
        
    except Exception as e:
        error_msg = f"Error calling API: {e}"
        print(error_msg)
        return error_msg

def debug_documents():
    """
    Debug function to check document processing status
    """
    print("=== DOCUMENT PROCESSING DEBUG ===")
    print(f"Documents directory: {DOCUMENTS_DIR}")
    print(f"Directory exists: {Path(DOCUMENTS_DIR).exists()}")
    
    files = list(Path(DOCUMENTS_DIR).iterdir())
    print(f"Files found: {len(files)}")
    
    for file_path in files:
        if file_path.is_file():
            file_size = os.path.getsize(file_path)
            print(f"- {file_path.name}: {file_size} bytes")
    
    print(f"\nProcessed chunks: {len(document_chunks)}")
    print(f"Document metadata: {len(document_metadata)}")
    print(f"FAISS index exists: {faiss_index is not None}")
    
    if document_metadata:
        print("\nDocument sources:")
        sources = set(meta['file_name'] for meta in document_metadata)
        for source in sources:
            chunk_count = sum(1 for meta in document_metadata if meta['file_name'] == source)
            print(f"- {source}: {chunk_count} chunks")
            
    return len(document_chunks) > 0

# ────────────────────────────────────────────────────────────────────────────────

@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    from flask import request  # Import only when needed
    
    # 1. Read the incoming WhatsApp message
    incoming_msg = request.values.get("Body", "").strip()
    
    # 2. Search for relevant documents
    relevant_docs = search_documents(incoming_msg, top_k=3)
    
    # 3. Build context from retrieved documents
    context = ""
    if relevant_docs:
        context = "\n\nRelevant Information from Knowledge Base:\n"
        for i, doc in enumerate(relevant_docs, 1):
            context += f"\n--- Source {i}: {doc['metadata']['file_name']} ---\n"
            context += f"{doc['content']}\n"
    
    # 4. Build the conversation payload with context
    system_prompt = CUSTOM_SYSTEM_PROMPT
    if context:
        system_prompt += f"\n\nUse the following context to answer the user's question:{context}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": incoming_msg}
    ]

    # 5. Call OpenRouter API
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://whatsapp-bot.example.com",  # Optional. Site URL for rankings on openrouter.ai.
        "X-Title": "WhatsApp Bot",  # Optional. Site title for rankings on openrouter.ai.
    }
    
    # Debug logging
    print(f"Making request to: {OPENROUTER_URL}")
    print(f"Headers: {headers}")
    print(f"Messages: {messages}")
    
    resp = requests.post(
        url=OPENROUTER_URL,
        headers=headers,
        data=json.dumps({
            "model": "deepseek/deepseek-r1-0528:free",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 500
        })
    )
    
    # Debug response
    print(f"Response status: {resp.status_code}")
    print(f"Response headers: {dict(resp.headers)}")
    print(f"Response text: {resp.text}")
    
    resp.raise_for_status()
    
    # Add error handling for JSON parsing
    try:
        data = resp.json()
    except requests.exceptions.JSONDecodeError:
        print(f"Response status: {resp.status_code}")
        print(f"Response headers: {dict(resp.headers)}")
        print(f"Response text: {resp.text}")
        raise Exception(f"API returned non-JSON response: {resp.text}")

    # 6. Extract the AI's reply
    try:
        answer = data["choices"][0]["message"]["content"].strip()
        if not answer:
            answer = "Sorry, I couldn't generate a response. Please try again."
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error extracting answer from API response: {e}")
        print(f"API response data: {data}")
        answer = "Sorry, there was an error processing your request. Please try again."

    # 7. Send it back via Twilio
    twilio_resp = MessagingResponse()
    twilio_resp.message(answer)
    
    response_xml = str(twilio_resp)
    print(f"Twilio Response XML: {response_xml}")
    print(f"Response length: {len(response_xml)}")
    
    return response_xml, 200

@app.route("/test", methods=["GET"])
def test_connection():
    """Simple test endpoint to verify the tunnel is working"""
    return "✅ Tunnel is working! Your WhatsApp bot is accessible.", 200

@app.route("/upload", methods=["GET", "POST"])
def upload_document():
    """Upload documents and reprocess the knowledge base"""
    from flask import request  # Import only when needed
    
    if request.method == "GET":
        # Return a simple upload form
        return '''
        <!doctype html>
        <html>
        <head><title>Document Upload</title></head>
        <body>
            <h2>Upload Documents to Knowledge Base</h2>
            <form method="POST" enctype="multipart/form-data">
                <input type="file" name="file" accept=".pdf,.docx,.txt" required>
                <input type="submit" value="Upload">
            </form>
            <hr>
            <h3>Current Documents:</h3>
            <ul>
        ''' + ''.join([f"<li>{f.name}</li>" for f in Path(DOCUMENTS_DIR).iterdir() if f.is_file()]) + '''
            </ul>
        </body>
        </html>
        '''
    
    # Handle file upload
    if 'file' not in request.files:
        return "No file selected", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    
    # Save file
    if file and file.filename.lower().endswith(('.pdf', '.docx', '.txt')):
        file_path = Path(DOCUMENTS_DIR) / file.filename
        file.save(str(file_path))
        
        # Reprocess all documents
        process_documents()
        
        return f"Document '{file.filename}' uploaded successfully and knowledge base updated!", 200
    
    return "Invalid file type. Please upload PDF, DOCX, or TXT files.", 400

@app.route("/status", methods=["GET"])
def status():
    """Check the status of the knowledge base"""
    num_docs = len(set(meta['file_name'] for meta in document_metadata))
    num_chunks = len(document_chunks)
    
    return {
        "status": "active",
        "documents": num_docs,
        "chunks": num_chunks,
        "files": [f.name for f in Path(DOCUMENTS_DIR).iterdir() if f.is_file()]
          }

if __name__ == "__main__":
    # In production, use a real WSGI server and set PORT via env
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
