# WhatsApp Bot with Document Knowledge Base

This WhatsApp bot uses AI to answer questions based on uploaded documents. It implements RAG (Retrieval Augmented Generation) to provide accurate responses with source citations.

## Features

- **Document Support**: Upload PDF, DOCX, and TXT files
- **Vector Search**: Uses embeddings to find relevant document sections
- **Source Citations**: Provides references to source documents and pages
- **WhatsApp Integration**: Responds to messages via Twilio WhatsApp API
- **Web Interface**: Upload and manage documents through a web interface

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Create Documents Directory**:
   The bot will automatically create a `documents/` folder on first run.

3. **Configure Environment Variables**:
   Create a `.env` file in the project root with your API credentials:
   ```bash
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   TWILIO_ACCOUNT_SID=your_twilio_account_sid_here
   TWILIO_AUTH_TOKEN=your_twilio_auth_token_here
   ```
   
   **Get your credentials from**:
   - OpenRouter API Key: https://openrouter.ai/
   - Twilio credentials: https://www.twilio.com/console

4. **Upload Documents**:
   - Place documents in the `documents/` folder, OR
   - Use the web interface at `http://localhost:5000/upload`

5. **Run the Bot**:
   ```bash
   python interface.py
   ```

## Usage

### Uploading Documents

**Method 1: Web Interface**
1. Go to `http://localhost:5000/upload`
2. Select a PDF, DOCX, or TXT file
3. Click "Upload"
4. The knowledge base will automatically update

**Method 2: Direct File Placement**
1. Copy files to the `documents/` folder
2. Restart the bot to process new files

### WhatsApp Integration

1. Configure your Twilio WhatsApp webhook to point to `http://your-domain.com/webhook`
2. Send messages to your WhatsApp number
3. The bot will search the knowledge base and respond with relevant information

### Checking Status

Visit `http://localhost:5000/status` to see:
- Number of documents loaded
- Number of text chunks processed
- List of all files in the knowledge base

## How It Works

1. **Document Processing**: Documents are split into chunks with page/paragraph references
2. **Embeddings**: Text chunks are converted to vector embeddings using sentence transformers
3. **Vector Search**: User queries are matched against document chunks using FAISS
4. **Response Generation**: Relevant context is sent to the LLM with source attribution
5. **Citation**: Responses include references to source documents and locations

## Supported File Types

- **PDF**: Extracts text with page numbers
- **DOCX**: Extracts text with paragraph numbers  
- **TXT**: Extracts text with line numbers

## System Prompt

The bot is configured to:
- Respond completely and concisely
- Only answer questions based on uploaded documents
- Provide source references and page numbers
- Cross-check information with references
- Give conclusions first, then explanations

## API Endpoints

- `POST /webhook` - WhatsApp webhook endpoint
- `GET/POST /upload` - Document upload interface
- `GET /status` - Knowledge base status 