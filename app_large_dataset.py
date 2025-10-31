"""
Flask API Server for Large Dataset Clustering
Provides endpoints for file upload, column analysis, and clustering execution
"""

from flask import Flask, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime
import pandas as pd
from enhanced_clustering_analysis import (
    LargeDatasetLoader,
    ColumnAnalyzer,
    EnhancedClusteringEngine,
    ResultsExporter,
    create_progress_tracker
)
import threading
import time

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Global storage for analysis jobs
analysis_jobs = {}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Serve the main HTML page"""
    return app.send_static_file('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload and return initial analysis
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only CSV and Excel files allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and analyze file
        if filename.endswith('.csv'):
            df_sample = LargeDatasetLoader.load_csv(filepath, sample_size=100)
            preview = LargeDatasetLoader.get_preview(filepath, rows=10)
        else:
            # For Excel files
            df_sample = pd.read_excel(filepath, nrows=100)
            preview = {
                "preview_rows": df_sample.head(10).to_dict(orient='records'),
                "columns": df_sample.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df_sample.dtypes.items()}
            }
        
        # Analyze columns
        analysis = ColumnAnalyzer.analyze_dataframe(df_sample)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'analysis': analysis,
            'preview': preview
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_dataset():
    """
    Start clustering analysis on uploaded dataset
    
    Expected JSON payload:
    {
        "filepath": "path/to/file.csv",
        "text_columns": ["column1", "column2"],
        "feature_columns": ["column3"],
        "min_cluster_size": 3,
        "min_samples": 2,
        "batch_size": 32
    }
    """
    try:
        data = request.json
        
        filepath = data.get('filepath')
        text_columns = data.get('text_columns', [])
        feature_columns = data.get('feature_columns', [])
        min_cluster_size = data.get('min_cluster_size', 3)
        min_samples = data.get('min_samples', 2)
        batch_size = data.get('batch_size', 32)
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'Invalid file path'}), 400
        
        if not text_columns:
            return jsonify({'error': 'At least one text column must be selected'}), 400
        
        # Create job ID
        job_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        
        # Initialize progress tracker
        update_progress, get_progress = create_progress_tracker()
        
        # Store job info
        analysis_jobs[job_id] = {
            'status': 'queued',
            'progress': 0,
            'stage': 'initializing',
            'created_at': datetime.now().isoformat(),
            'update_progress': update_progress,
            'get_progress': get_progress,
            'results': None
        }
        
        # Start analysis in background thread
        def run_analysis():
            try:
                # Load data
                if filepath.endswith('.csv'):
                    df = LargeDatasetLoader.load_csv(filepath)
                else:
                    df = pd.read_excel(filepath)
                
                # Initialize clustering engine
                engine = EnhancedClusteringEngine(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples
                )
                
                # Run analysis with progress tracking
                def progress_callback(progress_data):
                    analysis_jobs[job_id]['status'] = 'running'
                    analysis_jobs[job_id]['stage'] = progress_data.get('stage', 'processing')
                    analysis_jobs[job_id]['progress'] = progress_data.get('progress', 0)
                    update_progress(progress_data)
                
                results = engine.analyze_dataset(
                    df,
                    text_columns=text_columns,
                    feature_columns=feature_columns,
                    batch_size=batch_size,
                    progress_callback=progress_callback
                )
                
                # Save results
                results_filename = f"results_{job_id}.json"
                results_filepath = os.path.join(app.config['RESULTS_FOLDER'], results_filename)
                ResultsExporter.export_to_json(results, results_filepath)
                
                # Save CSV with clusters
                if 'data_with_clusters' in results:
                    df_result = pd.DataFrame(results['data_with_clusters'])
                    csv_filename = f"clusters_{job_id}.csv"
                    csv_filepath = os.path.join(app.config['RESULTS_FOLDER'], csv_filename)
                    ResultsExporter.export_to_csv(df_result, csv_filepath)
                    results['csv_export'] = csv_filename
                
                # Generate summary report
                report_filename = f"report_{job_id}.txt"
                report_filepath = os.path.join(app.config['RESULTS_FOLDER'], report_filename)
                ResultsExporter.export_summary_report(results, report_filepath)
                results['report_export'] = report_filename
                
                analysis_jobs[job_id]['status'] = 'completed'
                analysis_jobs[job_id]['progress'] = 100
                analysis_jobs[job_id]['results'] = results
                analysis_jobs[job_id]['results_file'] = results_filename
                
            except Exception as e:
                analysis_jobs[job_id]['status'] = 'error'
                analysis_jobs[job_id]['error'] = str(e)
        
        # Start background thread
        thread = threading.Thread(target=run_analysis)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Analysis started'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """
    Get status of an analysis job
    """
    if job_id not in analysis_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = analysis_jobs[job_id]
    
    response = {
        'job_id': job_id,
        'status': job['status'],
        'progress': job['progress'],
        'stage': job['stage'],
        'created_at': job['created_at']
    }
    
    if job['status'] == 'error':
        response['error'] = job.get('error')
    
    if job['status'] == 'completed' and job.get('results'):
        # Return summary of results
        results = job['results']
        clustering = results.get('clustering', {})
        
        response['results'] = {
            'num_clusters': clustering.get('num_clusters', 0),
            'num_noise': clustering.get('num_noise', 0),
            'cluster_sizes': clustering.get('cluster_sizes', {}),
            'total_rows': results.get('total_rows', 0),
            'results_file': job.get('results_file')
        }
    
    return jsonify(response)


@app.route('/api/results/<job_id>', methods=['GET'])
def get_job_results(job_id):
    """
    Get full results of a completed analysis job
    """
    if job_id not in analysis_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = analysis_jobs[job_id]
    
    if job['status'] != 'completed':
        return jsonify({'error': 'Job not completed yet'}), 400
    
    return jsonify(job['results'])


@app.route('/api/export/<job_id>/<export_type>', methods=['GET'])
def export_results(job_id, export_type):
    """
    Export results in different formats
    export_type: 'json', 'csv', 'report'
    """
    if job_id not in analysis_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = analysis_jobs[job_id]
    
    if job['status'] != 'completed':
        return jsonify({'error': 'Job not completed yet'}), 400
    
    results = job['results']
    
    if export_type == 'json':
        filename = f"results_{job_id}.json"
        filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True, download_name=filename)
    
    elif export_type == 'csv':
        csv_filename = results.get('csv_export')
        if csv_filename:
            filepath = os.path.join(app.config['RESULTS_FOLDER'], csv_filename)
            if os.path.exists(filepath):
                return send_file(filepath, as_attachment=True, download_name=csv_filename)
    
    elif export_type == 'report':
        report_filename = results.get('report_export')
        if report_filename:
            filepath = os.path.join(app.config['RESULTS_FOLDER'], report_filename)
            if os.path.exists(filepath):
                return send_file(filepath, as_attachment=True, download_name=report_filename)
    
    return jsonify({'error': 'Export file not found'}), 404


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_jobs': len([j for j in analysis_jobs.values() if j['status'] == 'running'])
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
