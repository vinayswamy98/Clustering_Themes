# Quick Start Guide

## Installation

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Start the Server**
```bash
python app_large_dataset.py
```

The application will start at `http://localhost:5001`

## Using the Application

### Step 1: Upload Your Dataset
1. Visit `http://localhost:5001` in your browser
2. Drag and drop your CSV file, or click "Choose File" to browse
3. Supported formats: CSV, Excel (.xlsx, .xls)
4. Maximum file size: 500MB

### Step 2: Review Column Analysis
After upload, you'll see:
- Total rows and columns
- Memory usage estimate
- Detailed analysis of each column:
  - Column type (text, categorical, numerical, datetime)
  - Non-null count and percentage
  - Statistics (avg length for text, unique values for categorical, etc.)
  - Recommendations for clustering

### Step 3: Select Columns
- Check the columns you want to use for clustering
- Text columns with meaningful content work best
- The system automatically recommends suitable columns
- You can select multiple columns - they'll be combined for analysis

### Step 4: Configure Parameters
Adjust clustering parameters based on your needs:

- **Minimum Cluster Size** (default: 3)
  - Smaller values (2-3): More clusters, some very small
  - Larger values (5-10): Fewer, larger clusters
  - Recommendation: Start with 3-5

- **Minimum Samples** (default: 2)
  - Controls how conservative clustering is
  - Lower values: More points assigned to clusters
  - Higher values: More points marked as noise
  - Recommendation: Use 2-3 for balanced results

- **Batch Size** (default: 32)
  - Number of items processed at once
  - Larger values: Faster but more memory
  - Smaller values: Slower but less memory
  - Recommendation: Use 32 for most systems, reduce to 16 if memory issues

### Step 5: View Results
Once analysis completes, you'll see:

1. **Statistics Summary**
   - Total clusters found
   - Total rows processed
   - Noise points (items that don't fit any cluster)
   - Largest cluster size

2. **Cluster Distribution**
   - Visual bar chart showing cluster sizes
   - Easy identification of major themes

3. **Sample Data**
   - Representative examples from each cluster
   - Helps understand what each cluster represents

### Step 6: Export Results
Choose your preferred export format:

- **JSON**: Complete results including all metadata
- **CSV**: Your original data with cluster assignments added
- **Text Report**: Human-readable summary report

## Example Workflow

Using the included `sample_data.csv`:

1. **Upload** the sample_data.csv file
2. **Select** the "feedback_text" column (automatically recommended)
3. **Configure** parameters:
   - Minimum Cluster Size: 3
   - Minimum Samples: 2
   - Batch Size: 32
4. **Start Analysis** and wait for completion (~30 seconds for 50 rows)
5. **Review Results**:
   - Should find 8-10 clusters representing different feedback themes
   - Examples: Pricing concerns, Quality issues, Support problems, etc.
6. **Export** as CSV to see cluster assignments for each feedback item

## Tips for Best Results

### Column Selection
- ✅ **Good**: Customer feedback, product descriptions, survey responses
- ✅ **Good**: Long text fields with varied content
- ❌ **Avoid**: Names, dates, IDs
- ❌ **Avoid**: Very short text (< 20 characters)

### Parameter Tuning
- **Too many small clusters?** Increase min_cluster_size
- **Too many noise points?** Decrease min_samples or min_cluster_size
- **Clusters too broad?** Decrease min_cluster_size
- **Out of memory?** Reduce batch_size

### Dataset Size Guidelines
- **< 1,000 rows**: Process in < 1 minute
- **1,000 - 10,000 rows**: Process in 2-5 minutes
- **10,000 - 100,000 rows**: Process in 5-30 minutes
- **> 100,000 rows**: Consider pre-filtering or sampling

## Troubleshooting

### "Upload failed"
- Check file format (must be CSV or Excel)
- Verify file isn't corrupted
- Ensure file size is under 500MB

### "No clusters found"
- Reduce min_cluster_size (try 2)
- Reduce min_samples (try 1)
- Check that selected columns have meaningful text
- Verify data isn't too homogeneous

### "Analysis takes too long"
- Reduce batch_size
- Select fewer columns
- Process on a smaller sample first

### "Out of memory"
- Reduce batch_size (try 16 or 8)
- Close other applications
- Process smaller datasets
- Consider cloud deployment with more RAM

## API Usage

You can also use the API directly:

### Upload File
```bash
curl -X POST http://localhost:5001/api/upload \
  -F "file=@sample_data.csv"
```

### Start Analysis
```bash
curl -X POST http://localhost:5001/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "filepath": "static/uploads/20251031_120000_sample_data.csv",
    "text_columns": ["feedback_text"],
    "min_cluster_size": 3,
    "min_samples": 2,
    "batch_size": 32
  }'
```

### Check Status
```bash
curl http://localhost:5001/api/status/{job_id}
```

### Download Results
```bash
curl http://localhost:5001/api/export/{job_id}/csv -o results.csv
```

## Production Deployment

For production use:

1. **Security**:
   ```bash
   # Set environment variable to disable debug mode
   export FLASK_DEBUG=false
   ```

2. **Use Production Server**:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5001 app_large_dataset:app
   ```

3. **Configure Reverse Proxy** (nginx example):
   ```nginx
   location / {
       proxy_pass http://127.0.0.1:5001;
       proxy_set_header Host $host;
       proxy_set_header X-Real-IP $remote_addr;
   }
   ```

4. **Set Up Monitoring** and log rotation

5. **Implement Authentication** if needed

## Support

For issues or questions:
- Check the main README.md
- Review the test_system.py for examples
- Examine the sample_data.csv format

## Next Steps

After getting familiar with the basic workflow:
1. Try with your own datasets
2. Experiment with different parameter values
3. Integrate with your existing workflow
4. Consider adding LLM-based theme naming (see thematic_analysis_langgraph_agents.py)
