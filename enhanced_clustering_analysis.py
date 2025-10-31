"""
Enhanced Clustering Analysis for Large Datasets
Supports memory-efficient processing, column selection, and configurable clustering
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Iterator
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime
import os


class ColumnAnalyzer:
    """Analyze CSV columns and detect their types"""
    
    @staticmethod
    def analyze_column(series: pd.Series) -> Dict:
        """Analyze a single column and return its characteristics"""
        non_null_count = series.notna().sum()
        null_count = series.isna().sum()
        total_count = len(series)
        
        # Detect column type
        if pd.api.types.is_numeric_dtype(series):
            col_type = "numerical"
            stats = {
                "min": float(series.min()) if non_null_count > 0 else None,
                "max": float(series.max()) if non_null_count > 0 else None,
                "mean": float(series.mean()) if non_null_count > 0 else None,
                "median": float(series.median()) if non_null_count > 0 else None
            }
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_type = "datetime"
            stats = {
                "min": str(series.min()) if non_null_count > 0 else None,
                "max": str(series.max()) if non_null_count > 0 else None
            }
        else:
            # String/categorical
            unique_count = series.nunique()
            unique_ratio = unique_count / total_count if total_count > 0 else 0
            
            # Determine if categorical or text
            if unique_ratio < 0.5 and unique_count < 100:
                col_type = "categorical"
                stats = {
                    "unique_values": unique_count,
                    "top_values": series.value_counts().head(10).to_dict()
                }
            else:
                col_type = "text"
                avg_length = series.astype(str).str.len().mean() if non_null_count > 0 else 0
                stats = {
                    "avg_length": float(avg_length),
                    "unique_values": unique_count
                }
        
        # Determine recommendation
        recommendation = "none"
        if col_type == "text" and non_null_count > 0:
            avg_len = stats.get("avg_length", 0)
            if avg_len > 20:  # Likely meaningful text
                recommendation = "clustering"
        elif col_type == "categorical" and non_null_count > 0:
            recommendation = "feature"
        elif col_type == "numerical" and non_null_count > 0:
            recommendation = "feature"
        
        return {
            "name": series.name,
            "type": col_type,
            "non_null_count": int(non_null_count),
            "null_count": int(null_count),
            "null_percentage": float(null_count / total_count * 100) if total_count > 0 else 0,
            "stats": stats,
            "recommendation": recommendation
        }
    
    @staticmethod
    def analyze_dataframe(df: pd.DataFrame) -> Dict:
        """Analyze all columns in a dataframe"""
        columns_info = []
        for col in df.columns:
            try:
                col_info = ColumnAnalyzer.analyze_column(df[col])
                columns_info.append(col_info)
            except Exception as e:
                columns_info.append({
                    "name": col,
                    "type": "error",
                    "error": str(e),
                    "recommendation": "none"
                })
        
        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": columns_info,
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024)
        }


class LargeDatasetLoader:
    """Load CSV files efficiently, even for large datasets"""
    
    @staticmethod
    def load_csv(filepath: str, 
                 chunk_size: Optional[int] = None,
                 sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load CSV file with optional chunking for large files
        
        Args:
            filepath: Path to CSV file
            chunk_size: If provided, read in chunks
            sample_size: If provided, return only a sample
        """
        if sample_size:
            # For preview, just load a sample
            return pd.read_csv(filepath, nrows=sample_size)
        
        if chunk_size:
            # For very large files, read in chunks
            chunks = []
            for chunk in pd.read_csv(filepath, chunksize=chunk_size):
                chunks.append(chunk)
            return pd.concat(chunks, ignore_index=True)
        
        # Normal load
        return pd.read_csv(filepath)
    
    @staticmethod
    def get_preview(filepath: str, rows: int = 10) -> Dict:
        """Get a preview of the CSV file"""
        df_preview = pd.read_csv(filepath, nrows=rows)
        
        return {
            "preview_rows": df_preview.to_dict(orient='records'),
            "columns": df_preview.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df_preview.dtypes.items()}
        }


class EnhancedClusteringEngine:
    """Enhanced clustering engine with column selection and batch processing"""
    
    def __init__(self, 
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 min_cluster_size: int = 3,
                 min_samples: int = 2):
        """
        Initialize clustering engine
        
        Args:
            embedding_model: Sentence transformer model name
            min_cluster_size: Minimum cluster size for HDBSCAN
            min_samples: Minimum samples for HDBSCAN
        """
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        
    def _load_embedding_model(self):
        """Lazy load embedding model"""
        if self.embedding_model is None:
            print(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
    
    def generate_embeddings(self, 
                          texts: List[str], 
                          batch_size: int = 32,
                          show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings with batch processing
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
        """
        self._load_embedding_model()
        
        embeddings = self.embedding_model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def perform_clustering(self, embeddings: np.ndarray) -> Dict:
        """
        Perform HDBSCAN clustering on embeddings
        
        Args:
            embeddings: Numpy array of embeddings
            
        Returns:
            Dictionary with cluster labels and statistics
        """
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean'
        )
        
        labels = clusterer.fit_predict(embeddings)
        
        # Calculate cluster statistics
        unique_labels = np.unique(labels)
        cluster_sizes = {int(label): int(np.sum(labels == label)) 
                        for label in unique_labels}
        
        # Group indices by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)
        
        # Calculate cluster centroids
        centroids = {}
        for label, indices in clusters.items():
            if label != -1:  # Skip noise
                cluster_embeddings = embeddings[indices]
                centroid = np.mean(cluster_embeddings, axis=0)
                centroids[label] = centroid
        
        return {
            "labels": labels.tolist(),
            "num_clusters": len([l for l in unique_labels if l != -1]),
            "num_noise": int(cluster_sizes.get(-1, 0)),
            "cluster_sizes": cluster_sizes,
            "clusters": clusters,
            "centroids": {k: v.tolist() for k, v in centroids.items()}
        }
    
    def analyze_dataset(self,
                       df: pd.DataFrame,
                       text_columns: List[str],
                       feature_columns: Optional[List[str]] = None,
                       batch_size: int = 32,
                       progress_callback: Optional[callable] = None) -> Dict:
        """
        Perform complete clustering analysis on a dataset
        
        Args:
            df: Input dataframe
            text_columns: Columns to use for text clustering
            feature_columns: Additional feature columns (not implemented yet)
            batch_size: Batch size for embedding generation
            progress_callback: Optional callback for progress updates
            
        Returns:
            Complete analysis results
        """
        results = {
            "status": "processing",
            "timestamp": datetime.now().isoformat(),
            "total_rows": len(df),
            "text_columns": text_columns,
            "feature_columns": feature_columns or []
        }
        
        try:
            # Combine text columns
            if progress_callback:
                progress_callback({"stage": "preparing", "progress": 0})
            
            combined_texts = []
            for idx, row in df.iterrows():
                text_parts = [str(row[col]) for col in text_columns if pd.notna(row[col])]
                combined_text = " | ".join(text_parts)
                combined_texts.append(combined_text)
            
            # Generate embeddings
            if progress_callback:
                progress_callback({"stage": "embedding", "progress": 20})
            
            embeddings = self.generate_embeddings(
                combined_texts, 
                batch_size=batch_size,
                show_progress=False
            )
            
            # Perform clustering
            if progress_callback:
                progress_callback({"stage": "clustering", "progress": 70})
            
            clustering_results = self.perform_clustering(embeddings)
            
            # Prepare final results
            if progress_callback:
                progress_callback({"stage": "finalizing", "progress": 90})
            
            # Add cluster assignments to dataframe
            df_result = df.copy()
            df_result['cluster_id'] = clustering_results['labels']
            
            # Extract representative samples from each cluster
            cluster_samples = {}
            for cluster_id, indices in clustering_results['clusters'].items():
                if cluster_id != -1:
                    # Get up to 5 representative samples
                    sample_indices = indices[:min(5, len(indices))]
                    samples = df.iloc[sample_indices][text_columns].to_dict(orient='records')
                    cluster_samples[cluster_id] = samples
            
            results.update({
                "status": "completed",
                "clustering": clustering_results,
                "cluster_samples": cluster_samples,
                "embeddings_shape": embeddings.shape,
                "data_with_clusters": df_result.to_dict(orient='records')
            })
            
            if progress_callback:
                progress_callback({"stage": "completed", "progress": 100})
            
        except Exception as e:
            results.update({
                "status": "error",
                "error": str(e)
            })
        
        return results


class ResultsExporter:
    """Export clustering results in various formats"""
    
    @staticmethod
    def export_to_json(results: Dict, filepath: str):
        """Export results to JSON"""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    @staticmethod
    def export_to_csv(df: pd.DataFrame, filepath: str):
        """Export dataframe with cluster assignments to CSV"""
        df.to_csv(filepath, index=False)
    
    @staticmethod
    def export_summary_report(results: Dict, filepath: str):
        """Export a human-readable summary report"""
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CLUSTERING ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Timestamp: {results.get('timestamp', 'N/A')}\n")
            f.write(f"Total Rows: {results.get('total_rows', 0)}\n")
            f.write(f"Text Columns: {', '.join(results.get('text_columns', []))}\n\n")
            
            clustering = results.get('clustering', {})
            f.write(f"Number of Clusters: {clustering.get('num_clusters', 0)}\n")
            f.write(f"Noise Points: {clustering.get('num_noise', 0)}\n\n")
            
            f.write("Cluster Sizes:\n")
            cluster_sizes = clustering.get('cluster_sizes', {})
            for cluster_id, size in sorted(cluster_sizes.items()):
                if cluster_id != -1:
                    f.write(f"  Cluster {cluster_id}: {size} items\n")
            
            f.write("\n" + "=" * 80 + "\n")


# Utility functions for progress tracking
def create_progress_tracker():
    """Create a simple progress tracking closure"""
    progress_data = {"current": 0, "stage": "initializing"}
    
    def update(data: Dict):
        progress_data.update(data)
        return progress_data.copy()
    
    def get():
        return progress_data.copy()
    
    return update, get
