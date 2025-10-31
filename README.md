# Large Dataset Clustering Analysis System

A comprehensive web-based clustering analysis system for large CSV datasets with interactive column selection, real-time progress tracking, and exportable results.

## Features

### 📁 **File Upload & Analysis**
- Drag-and-drop CSV/Excel file upload
- Automatic column type detection (text, categorical, numerical, datetime)
- Memory-efficient processing for large datasets (1000s+ rows)
- Interactive data preview (first 10 rows)
- Column statistics and recommendations

### 🎯 **Interactive Column Selection**
- Visual column selection interface
- Smart recommendations for clustering
- Column type indicators
- Null value statistics
- Support for multiple text columns

### ⚙️ **Configurable Clustering**
- Adjustable clustering parameters:
  - Minimum cluster size
  - Minimum samples
  - Batch size for embeddings
- Real-time parameter recommendations
- Configuration summary before execution

### 📊 **Results Visualization**
- Cluster distribution charts
- Statistical summaries (total clusters, noise points, etc.)
- Sample data from each cluster
- Interactive results dashboard

### 💾 **Export Functionality**
- JSON export (complete results)
- CSV export (data with cluster assignments)
- Text report (summary report)

## Architecture

### Backend Components

#### `enhanced_clustering_analysis.py`
Core clustering engine with the following classes:

- **`ColumnAnalyzer`**: Analyzes CSV columns and detects types
- **`LargeDatasetLoader`**: Memory-efficient CSV loading with chunking support
- **`EnhancedClusteringEngine`**: Main clustering engine using:
  - Sentence Transformers for embeddings
  - HDBSCAN for clustering
  - Batch processing for large datasets
- **`ResultsExporter`**: Export results in multiple formats

#### `app_large_dataset.py`
Flask API server providing:

- `/api/upload` - File upload and initial analysis
- `/api/analyze` - Start clustering analysis
- `/api/status/<job_id>` - Check analysis progress
- `/api/results/<job_id>` - Get full results
- `/api/export/<job_id>/<type>` - Export results
- `/api/health` - Health check endpoint

### Frontend Components

#### `static/index.html`
Modern HTML interface with:
- Multi-step workflow (Upload → Select → Configure → Results)
- Responsive design
- Progress tracking
- Results visualization

#### `static/style.css`
Comprehensive styling with:
- Modern gradient design
- Responsive layouts
- Interactive hover effects
- Progress indicators
- Chart visualizations

#### `static/script.js`
JavaScript logic handling:
- File upload (drag-and-drop)
- API communication
- Progress polling
- Results rendering
- Export functionality

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Clustering_Themes
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the server**
```bash
python app_large_dataset.py
```

The application will start on `http://localhost:5001`

## Usage

### Step 1: Upload Dataset
1. Navigate to `http://localhost:5001`
2. Drag and drop your CSV file or click to browse
3. Wait for the file to be analyzed

### Step 2: Select Columns
1. Review the column analysis
2. Select columns to use for clustering (text columns work best)
3. Review the data preview
4. Click "Next: Configure Parameters"

### Step 3: Configure Parameters
1. Adjust clustering parameters:
   - **Minimum Cluster Size**: Minimum items to form a cluster (recommended: 3-5)
   - **Minimum Samples**: Controls clustering conservativeness (recommended: 2-3)
   - **Batch Size**: Items processed at once (recommended: 32)
2. Review the configuration summary
3. Click "Start Clustering Analysis"

### Step 4: View Results
1. Monitor the progress bar
2. Once complete, view:
   - Cluster statistics
   - Distribution charts
   - Sample data from clusters
3. Export results in your preferred format

## Technical Details

### Clustering Algorithm
- **Embedding Model**: Sentence Transformers (all-MiniLM-L6-v2)
- **Clustering Algorithm**: HDBSCAN (Hierarchical Density-Based Spatial Clustering)
- **Similarity Metric**: Cosine similarity for embeddings

### Memory Management
- Chunk-based CSV loading for large files
- Batch processing for embedding generation
- Progressive results streaming
- Background job processing

### API Design
- RESTful API endpoints
- JSON request/response format
- Background job processing with status polling
- File-based result storage

## File Structure

```
/
├── enhanced_clustering_analysis.py  # Core clustering logic
├── app_large_dataset.py            # Flask API server
├── requirements.txt                # Python dependencies
├── sample_data.csv                 # Example dataset
├── static/
│   ├── index.html                  # Main HTML interface
│   ├── style.css                   # Styling
│   ├── script.js                   # JavaScript logic
│   ├── uploads/                    # Uploaded files (excluded from git)
│   └── results/                    # Analysis results (excluded from git)
└── README.md                       # This file
```

## Example Dataset Format

CSV files should have at least one text column for clustering:

```csv
id,customer_name,feedback_text,category,rating
1,John Smith,The pricing is too high...,Pricing,2
2,Jane Doe,Product quality has declined...,Quality,3
```

## Configuration

### Clustering Parameters

- **min_cluster_size** (default: 3)
  - Minimum number of items to form a cluster
  - Lower values create more, smaller clusters
  - Higher values create fewer, larger clusters

- **min_samples** (default: 2)
  - Controls how conservative clustering is
  - Lower values allow more items to be clustered
  - Higher values increase noise points

- **batch_size** (default: 32)
  - Number of items processed at once for embeddings
  - Adjust based on available memory
  - Higher values are faster but use more memory

## Browser Compatibility

- Chrome/Edge (recommended)
- Firefox
- Safari
- Mobile browsers (responsive design)

## Performance Notes

- **Small datasets (< 1000 rows)**: Real-time processing (< 1 minute)
- **Medium datasets (1000-10000 rows)**: 2-5 minutes
- **Large datasets (> 10000 rows)**: 5-15 minutes

Processing time depends on:
- Number of rows
- Text length
- Number of selected columns
- Hardware capabilities

## Troubleshooting

### Upload fails
- Check file format (CSV or Excel)
- Ensure file size is under 500MB
- Verify file encoding (UTF-8 recommended)

### Clustering takes too long
- Reduce batch size
- Select fewer columns
- Increase min_cluster_size

### No clusters found
- Decrease min_cluster_size
- Decrease min_samples
- Ensure text columns have meaningful content

## Integration with Existing System

This system integrates with the existing LangGraph-based clustering in `thematic_analysis_langgraph_agents.py` by:

1. Providing a user-friendly frontend for data upload and column selection
2. Using the same underlying clustering technologies (HDBSCAN, Sentence Transformers)
3. Supporting similar configuration options
4. Enabling batch processing for large datasets

## Security Considerations

### Production Deployment
- **Debug Mode**: Disabled by default. Set `FLASK_DEBUG=true` environment variable only for development
- **File Upload Security**: 
  - Filename sanitization using `secure_filename()`
  - File size limits (500MB, configurable)
  - File type validation (CSV, Excel only)
  - Path traversal prevention
- **Error Handling**: Generic error messages to prevent information disclosure
- **File Storage**: 
  - Uploaded files stored in `static/uploads/` with timestamped filenames
  - Results stored in `static/results/`
  - Add these directories to `.gitignore` to avoid committing sensitive data
- **Path Validation**: All file paths are normalized and validated to prevent directory traversal
- **Input Validation**: All user inputs are validated before processing

### Recommendations for Production
- Add authentication and authorization
- Implement rate limiting
- Use HTTPS
- Regular cleanup of old files
- Monitor disk usage
- Set up proper logging
- Use a production WSGI server (e.g., Gunicorn, uWSGI)
- Configure CORS appropriately
- Add CSRF protection for API endpoints
- Implement file scanning for malware

## Future Enhancements

- Theme generation using LLM (similar to `thematic_analysis_langgraph_agents.py`)
- Multi-file analysis
- Comparison between analyses
- Custom embedding models
- Real-time collaboration
- Advanced visualizations (t-SNE, UMAP plots)
- API key management for LLM features

## License

[Add appropriate license]

## Contributing

[Add contribution guidelines]

## Contact

[Add contact information]
