"""
Test script for the clustering analysis system
Tests core functionality without requiring model downloads
"""

import pandas as pd
import json
from enhanced_clustering_analysis import (
    ColumnAnalyzer,
    LargeDatasetLoader,
    ResultsExporter
)

def test_column_analyzer():
    """Test column analysis functionality"""
    print("=" * 80)
    print("TEST 1: Column Analyzer")
    print("=" * 80)
    
    # Create sample dataframe
    df = pd.read_csv('sample_data.csv')
    
    # Analyze columns
    analysis = ColumnAnalyzer.analyze_dataframe(df)
    
    print(f"\n✓ Analyzed {analysis['total_columns']} columns in {analysis['total_rows']} rows")
    print(f"  Memory usage: {analysis['memory_usage_mb']:.2f} MB")
    
    # Check recommendations
    text_cols = [c for c in analysis['columns'] if c['recommendation'] == 'clustering']
    feature_cols = [c for c in analysis['columns'] if c['recommendation'] == 'feature']
    
    print(f"\n✓ Found {len(text_cols)} text columns recommended for clustering:")
    for col in text_cols:
        print(f"    - {col['name']} (avg length: {col['stats'].get('avg_length', 0):.0f} chars)")
    
    print(f"\n✓ Found {len(feature_cols)} feature columns:")
    for col in feature_cols:
        print(f"    - {col['name']} ({col['type']})")
    
    return True

def test_data_loader():
    """Test data loading functionality"""
    print("\n" + "=" * 80)
    print("TEST 2: Data Loader")
    print("=" * 80)
    
    # Test preview
    preview = LargeDatasetLoader.get_preview('sample_data.csv', rows=5)
    
    print(f"\n✓ Loaded preview with {len(preview['preview_rows'])} rows")
    print(f"  Columns: {', '.join(preview['columns'])}")
    
    # Test full load
    df = LargeDatasetLoader.load_csv('sample_data.csv')
    print(f"\n✓ Loaded full dataset: {len(df)} rows × {len(df.columns)} columns")
    
    # Test sample load
    df_sample = LargeDatasetLoader.load_csv('sample_data.csv', sample_size=10)
    print(f"✓ Loaded sample dataset: {len(df_sample)} rows")
    
    return True

def test_results_exporter():
    """Test results export functionality"""
    print("\n" + "=" * 80)
    print("TEST 3: Results Exporter")
    print("=" * 80)
    
    # Create mock results
    mock_results = {
        'timestamp': '2025-10-31T00:00:00',
        'total_rows': 50,
        'text_columns': ['feedback_text'],
        'clustering': {
            'num_clusters': 5,
            'num_noise': 3,
            'cluster_sizes': {0: 10, 1: 8, 2: 12, 3: 9, 4: 8, -1: 3}
        }
    }
    
    # Test JSON export
    json_path = '/tmp/test_results.json'
    ResultsExporter.export_to_json(mock_results, json_path)
    print(f"\n✓ Exported JSON results to {json_path}")
    
    # Verify
    with open(json_path, 'r') as f:
        loaded = json.load(f)
    print(f"✓ Verified JSON export: {loaded['clustering']['num_clusters']} clusters")
    
    # Test summary report
    report_path = '/tmp/test_report.txt'
    ResultsExporter.export_summary_report(mock_results, report_path)
    print(f"\n✓ Exported summary report to {report_path}")
    
    with open(report_path, 'r') as f:
        report = f.read()
    print(f"✓ Report contains {len(report)} characters")
    
    return True

def test_api_imports():
    """Test API server imports"""
    print("\n" + "=" * 80)
    print("TEST 4: API Server Imports")
    print("=" * 80)
    
    try:
        import app_large_dataset
        print("\n✓ Flask app imports successfully")
        print(f"✓ Upload folder: {app_large_dataset.UPLOAD_FOLDER}")
        print(f"✓ Results folder: {app_large_dataset.RESULTS_FOLDER}")
        return True
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error importing Flask app: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("CLUSTERING ANALYSIS SYSTEM - TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Column Analyzer", test_column_analyzer),
        ("Data Loader", test_data_loader),
        ("Results Exporter", test_results_exporter),
        ("API Imports", test_api_imports)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == '__main__':
    exit(main())
