// Global state
let appState = {
    uploadedFile: null,
    fileAnalysis: null,
    selectedTextColumns: [],
    selectedFeatureColumns: [],
    currentJobId: null,
    currentSection: 'upload-section'
};

// API Base URL
const API_BASE = '';

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initializeUpload();
    initializeEventListeners();
});

// Initialize file upload functionality
function initializeUpload() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');

    // Click to upload
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileUpload(file);
        }
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        
        const file = e.dataTransfer.files[0];
        if (file) {
            handleFileUpload(file);
        }
    });
}

// Initialize other event listeners
function initializeEventListeners() {
    document.getElementById('next-to-config').addEventListener('click', () => {
        if (appState.selectedTextColumns.length > 0) {
            showSection('config-section');
            updateConfigSummary();
        }
    });

    document.getElementById('start-analysis').addEventListener('click', startAnalysis);
    document.getElementById('start-new').addEventListener('click', resetApp);
    
    // Export buttons
    document.getElementById('export-json').addEventListener('click', () => exportResults('json'));
    document.getElementById('export-csv').addEventListener('click', () => exportResults('csv'));
    document.getElementById('export-report').addEventListener('click', () => exportResults('report'));
}

// Handle file upload
async function handleFileUpload(file) {
    // Validate file type
    const validTypes = ['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(csv|xlsx|xls)$/i)) {
        alert('Please upload a CSV or Excel file');
        return;
    }

    // Show loading state
    const uploadArea = document.getElementById('upload-area');
    uploadArea.innerHTML = '<div class="spinner"></div><p>Uploading and analyzing file...</p>';

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            appState.uploadedFile = result.filename;
            appState.fileAnalysis = result.analysis;
            
            displayFileInfo(result);
            displayColumnAnalysis(result.analysis, result.preview);
            showSection('column-section');
        } else {
            throw new Error(result.error || 'Upload failed');
        }
    } catch (error) {
        alert('Error uploading file: ' + error.message);
        resetUploadArea();
    }
}

// Reset upload area
function resetUploadArea() {
    const uploadArea = document.getElementById('upload-area');
    uploadArea.innerHTML = `
        <div class="upload-icon">📁</div>
        <p class="upload-text">Drag & drop your CSV file here</p>
        <p class="upload-subtext">or click to browse</p>
        <button class="btn btn-primary" onclick="document.getElementById('file-input').click()">
            Choose File
        </button>
    `;
}

// Display file information
function displayFileInfo(result) {
    const fileInfo = document.getElementById('file-info');
    const fileDetails = document.getElementById('file-details');
    
    const analysis = result.analysis;
    
    fileDetails.innerHTML = `
        <p><strong>File:</strong> ${result.filename}</p>
        <p><strong>Total Rows:</strong> ${analysis.total_rows.toLocaleString()}</p>
        <p><strong>Total Columns:</strong> ${analysis.total_columns}</p>
        <p><strong>Memory Usage:</strong> ${analysis.memory_usage_mb.toFixed(2)} MB</p>
    `;
    
    fileInfo.style.display = 'block';
}

// Display column analysis
function displayColumnAnalysis(analysis, preview) {
    const columnAnalysis = document.getElementById('column-analysis');
    
    let html = '<div style="margin-bottom: 20px;">';
    html += '<p><strong>Select columns to use for clustering:</strong></p>';
    html += '</div>';
    
    analysis.columns.forEach(col => {
        const recommendationText = {
            'clustering': '✅ Recommended for clustering',
            'feature': 'ℹ️ Can be used as feature',
            'none': '⚠️ Not recommended'
        }[col.recommendation] || '';
        
        html += `
            <div class="column-item" data-column="${col.name}">
                <div class="column-header">
                    <span class="column-name">${col.name}</span>
                    <span class="column-type ${col.type}">${col.type}</span>
                </div>
                <div class="column-stats">
                    <p>Non-null: ${col.non_null_count.toLocaleString()} 
                       (${(100 - col.null_percentage).toFixed(1)}%)</p>
        `;
        
        if (col.stats) {
            if (col.type === 'text') {
                html += `<p>Avg Length: ${col.stats.avg_length?.toFixed(0)} characters</p>`;
            } else if (col.type === 'categorical') {
                html += `<p>Unique Values: ${col.stats.unique_values}</p>`;
            } else if (col.type === 'numerical') {
                html += `<p>Range: ${col.stats.min?.toFixed(2)} - ${col.stats.max?.toFixed(2)}</p>`;
            }
        }
        
        html += `</div>`;
        
        if (recommendationText) {
            html += `<div class="column-recommendation">${recommendationText}</div>`;
        }
        
        html += `
                <div class="column-checkbox">
                    <input type="checkbox" id="col-${col.name}" 
                           data-column="${col.name}" 
                           data-type="${col.type}"
                           ${col.recommendation === 'clustering' ? 'checked' : ''}>
                    <label for="col-${col.name}">Use for clustering</label>
                </div>
            </div>
        `;
    });
    
    columnAnalysis.innerHTML = html;
    
    // Add event listeners to checkboxes
    const checkboxes = columnAnalysis.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(cb => {
        cb.addEventListener('change', updateColumnSelection);
        
        // Initialize state
        if (cb.checked) {
            appState.selectedTextColumns.push(cb.dataset.column);
            cb.closest('.column-item').classList.add('selected');
        }
    });
    
    // Display preview
    displayDataPreview(preview);
    
    // Update next button state
    updateNextButton();
}

// Update column selection
function updateColumnSelection(e) {
    const checkbox = e.target;
    const columnName = checkbox.dataset.column;
    const columnItem = checkbox.closest('.column-item');
    
    if (checkbox.checked) {
        if (!appState.selectedTextColumns.includes(columnName)) {
            appState.selectedTextColumns.push(columnName);
        }
        columnItem.classList.add('selected');
    } else {
        appState.selectedTextColumns = appState.selectedTextColumns.filter(c => c !== columnName);
        columnItem.classList.remove('selected');
    }
    
    updateNextButton();
}

// Update next button state
function updateNextButton() {
    const nextButton = document.getElementById('next-to-config');
    nextButton.disabled = appState.selectedTextColumns.length === 0;
}

// Display data preview
function displayDataPreview(preview) {
    const dataPreview = document.getElementById('data-preview');
    
    if (!preview.preview_rows || preview.preview_rows.length === 0) {
        dataPreview.innerHTML = '<p>No preview available</p>';
        return;
    }
    
    const columns = preview.columns;
    const rows = preview.preview_rows;
    
    let html = '<table class="preview-table"><thead><tr>';
    columns.forEach(col => {
        html += `<th>${col}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    rows.forEach(row => {
        html += '<tr>';
        columns.forEach(col => {
            const value = row[col] !== null && row[col] !== undefined ? row[col] : '';
            const displayValue = String(value).length > 50 ? String(value).substring(0, 50) + '...' : value;
            html += `<td>${displayValue}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    dataPreview.innerHTML = html;
}

// Update configuration summary
function updateConfigSummary() {
    const summary = document.getElementById('config-summary');
    
    let html = '<div>';
    html += '<p><strong>Selected Columns for Clustering:</strong></p>';
    html += '<ul>';
    appState.selectedTextColumns.forEach(col => {
        html += `<li>${col}</li>`;
    });
    html += '</ul>';
    html += `<p><strong>Total Rows:</strong> ${appState.fileAnalysis.total_rows.toLocaleString()}</p>`;
    html += '</div>';
    
    summary.innerHTML = html;
}

// Start analysis
async function startAnalysis() {
    const minClusterSize = parseInt(document.getElementById('min-cluster-size').value);
    const minSamples = parseInt(document.getElementById('min-samples').value);
    const batchSize = parseInt(document.getElementById('batch-size').value);
    
    // Validate parameters
    if (minClusterSize < 2 || minSamples < 1 || batchSize < 8) {
        alert('Please check your parameter values');
        return;
    }
    
    // Show results section with progress
    showSection('results-section');
    document.getElementById('progress-container').style.display = 'block';
    document.getElementById('results-dashboard').style.display = 'none';
    
    try {
        // Start analysis
        const response = await fetch(`${API_BASE}/api/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filepath: `static/uploads/${appState.uploadedFile}`,
                text_columns: appState.selectedTextColumns,
                feature_columns: appState.selectedFeatureColumns,
                min_cluster_size: minClusterSize,
                min_samples: minSamples,
                batch_size: batchSize
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            appState.currentJobId = result.job_id;
            pollJobStatus(result.job_id);
        } else {
            throw new Error(result.error || 'Failed to start analysis');
        }
    } catch (error) {
        alert('Error starting analysis: ' + error.message);
    }
}

// Poll job status
async function pollJobStatus(jobId) {
    const interval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/api/status/${jobId}`);
            const status = await response.json();
            
            updateProgress(status);
            
            if (status.status === 'completed') {
                clearInterval(interval);
                await loadFullResults(jobId);
            } else if (status.status === 'error') {
                clearInterval(interval);
                alert('Analysis failed: ' + (status.error || 'Unknown error'));
            }
        } catch (error) {
            console.error('Error polling status:', error);
        }
    }, 1000);
}

// Update progress display
function updateProgress(status) {
    const progressBar = document.getElementById('progress-bar');
    const progressStage = document.getElementById('progress-stage');
    const progressPercent = document.getElementById('progress-percent');
    
    progressBar.style.width = `${status.progress}%`;
    progressStage.textContent = status.stage.charAt(0).toUpperCase() + status.stage.slice(1) + '...';
    progressPercent.textContent = `${status.progress}%`;
}

// Load full results
async function loadFullResults(jobId) {
    try {
        const response = await fetch(`${API_BASE}/api/results/${jobId}`);
        const results = await response.json();
        
        displayResults(results);
        
        // Hide progress, show dashboard
        document.getElementById('progress-container').style.display = 'none';
        document.getElementById('results-dashboard').style.display = 'block';
    } catch (error) {
        alert('Error loading results: ' + error.message);
    }
}

// Display results
function displayResults(results) {
    const clustering = results.clustering;
    
    // Update stats
    document.getElementById('stat-clusters').textContent = clustering.num_clusters;
    document.getElementById('stat-rows').textContent = results.total_rows.toLocaleString();
    document.getElementById('stat-noise').textContent = clustering.num_noise.toLocaleString();
    
    // Find largest cluster
    const sizes = Object.values(clustering.cluster_sizes).filter((s, i) => Object.keys(clustering.cluster_sizes)[i] !== '-1');
    const largest = sizes.length > 0 ? Math.max(...sizes) : 0;
    document.getElementById('stat-largest').textContent = largest;
    
    // Display cluster distribution
    displayClusterChart(clustering.cluster_sizes);
    
    // Display cluster samples
    displayClusterSamples(results.cluster_samples);
}

// Display cluster chart
function displayClusterChart(clusterSizes) {
    const chartContainer = document.getElementById('cluster-chart');
    
    // Filter out noise (-1) and sort by size
    const clusters = Object.entries(clusterSizes)
        .filter(([id, _]) => id !== '-1')
        .sort((a, b) => b[1] - a[1]);
    
    if (clusters.length === 0) {
        chartContainer.innerHTML = '<p>No clusters found</p>';
        return;
    }
    
    const maxSize = Math.max(...clusters.map(c => c[1]));
    
    let html = '';
    clusters.forEach(([clusterId, size]) => {
        const percentage = (size / maxSize) * 100;
        html += `
            <div class="cluster-bar">
                <div class="cluster-bar-label">
                    <span>Cluster ${clusterId}</span>
                    <span>${size} items</span>
                </div>
                <div class="cluster-bar-visual" style="width: ${percentage}%"></div>
            </div>
        `;
    });
    
    chartContainer.innerHTML = html;
}

// Display cluster samples
function displayClusterSamples(clusterSamples) {
    const container = document.getElementById('cluster-samples-container');
    
    if (!clusterSamples || Object.keys(clusterSamples).length === 0) {
        container.innerHTML = '<p>No sample data available</p>';
        return;
    }
    
    let html = '';
    
    Object.entries(clusterSamples).forEach(([clusterId, samples]) => {
        html += `
            <div class="cluster-sample-group">
                <div class="cluster-sample-header">Cluster ${clusterId} - Sample Items</div>
        `;
        
        samples.forEach(sample => {
            const text = Object.values(sample).join(' | ');
            const displayText = text.length > 200 ? text.substring(0, 200) + '...' : text;
            html += `<div class="sample-item">${displayText}</div>`;
        });
        
        html += '</div>';
    });
    
    container.innerHTML = html;
}

// Export results
async function exportResults(type) {
    if (!appState.currentJobId) {
        alert('No results to export');
        return;
    }
    
    try {
        window.location.href = `${API_BASE}/api/export/${appState.currentJobId}/${type}`;
    } catch (error) {
        alert('Error exporting results: ' + error.message);
    }
}

// Show section
function showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(section => {
        section.style.display = 'none';
    });
    
    // Show target section
    document.getElementById(sectionId).style.display = 'block';
    appState.currentSection = sectionId;
}

// Reset app
function resetApp() {
    appState = {
        uploadedFile: null,
        fileAnalysis: null,
        selectedTextColumns: [],
        selectedFeatureColumns: [],
        currentJobId: null,
        currentSection: 'upload-section'
    };
    
    resetUploadArea();
    showSection('upload-section');
    document.getElementById('file-input').value = '';
    document.getElementById('file-info').style.display = 'none';
}
