from flask import Flask, render_template, request, Response, stream_with_context
import json
import asyncio
from datetime import datetime, timezone
import os
from reasoning_quality_checking_async_improved import process_batch, process_guidelines
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

def save_uploaded_file(file, filename):
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        return filepath
    return None

async def process_validation(pdf_path, excel1_path, excel2_path, excel3_path, batch_size, concurrent_calls, top_k):
    # Setup
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = Chroma(persist_directory='./chroma_langchain', embedding_function=embeddings)
    llm = ChatAnthropic(model="claude-3-haiku-20240307", api_key=os.environ['ANTHROPIC_API_KEY'])
    
    # Initialize metrics
    total_records = 0
    current_progress = 0
    start_time = datetime.now(timezone.utc)
    
    try:
        # Process guidelines
        await process_guidelines(pdf_path, excel1_path, excel2_path)
        
        # Process records in chunks
        async for chunk_result in process_batch(excel3_path, vectorstore, llm, batch_size, concurrent_calls, top_k):
            total_records += len(chunk_result['results'])
            current_progress = (total_records / chunk_result['total_expected']) * 100
            
            # Calculate metrics
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            avg_duration = duration / total_records if total_records > 0 else 0
            
            yield json.dumps({
                'progress': round(current_progress, 1),
                'total_records': total_records,
                'llm_calls': chunk_result['llm_calls'],
                'errors': chunk_result['errors'],
                'avg_duration': avg_duration,
                'results': chunk_result['results']
            }) + '\n'
            
    except Exception as e:
        yield json.dumps({
            'error': str(e),
            'progress': current_progress
        }) + '\n'

@app.route('/api/validate', methods=['POST'])
def validate():
    try:
        # Save uploaded files
        pdf_path = save_uploaded_file(request.files['pdf'], 'guidelines.pdf')
        excel1_path = save_uploaded_file(request.files['excel1'], 'score_scale.xlsx')
        excel2_path = save_uploaded_file(request.files['excel2'], 'examples.xlsx')
        excel3_path = save_uploaded_file(request.files['excel3'], 'records.xlsx')
        
        batch_size = int(request.form.get('batchSize', 50))
        concurrent_calls = int(request.form.get('concurrent', 5))
        top_k = int(request.form.get('topK', 5))
        
        return Response(
            stream_with_context(process_validation(
                pdf_path, excel1_path, excel2_path, excel3_path,
                batch_size, concurrent_calls, top_k
            )),
            mimetype='text/event-stream'
        )
        
    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)