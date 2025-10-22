"""
Thematic Analysis System using LangGraph
Requirements: langgraph, langchain, langchain-ollama, sentence-transformers, chromadb, sklearn, pandas, numpy
"""

import json
import asyncio
from typing import TypedDict, List, Dict, Annotated
import operator
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from langgraph.graph import StateGraph, END
from openai import OpenAI
import os

api_key = os.getenv("openeouter_api_key")


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

SUMMARIZATION_SYSTEM_PROMPT = """You are a summarization specialist extracting key information from text data.

TASK: For each row provided, generate:
1. A concise summary (2-3 sentences) capturing the main point
2. 3-5 relevant keywords that represent key concepts
3. The primary topic this row discusses (not a consolidated theme, just what THIS row is about)

GUIDELINES:
- Focus on factual content, not interpretation
- Summaries should be self-contained (understandable without original text)
- Keywords should be specific nouns, phrases, or concepts (not generic words)
- Primary topic should be a short label (2-5 words) describing the subject matter
- Process all rows in the batch consistently

OUTPUT FORMAT (JSON array):
[
  {
    "row_id": 1,
    "summary": "...",
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "primary_topic": "Topic Label"
  }
]

CONSTRAINTS:
- Summary max length: 200 words
- Keywords: 3-5 per row
- Primary topic: 2-5 words
- Maintain consistent style across all rows in batch"""

THEME_GENERATION_SYSTEM_PROMPT = """You are a theme identification specialist who analyzes clusters of similar content to identify underlying themes.

TASK: Given a cluster of related summaries, determine:
1. An intuitive theme name that captures what unifies this cluster
2. A clear description explaining what this theme represents

GUIDELINES:
- Theme names should be descriptive and professional (3-6 words)
- Themes represent PATTERNS across multiple rows, not individual mentions
- Descriptions should explain what ties these items together (2-3 sentences)
- Look for the commonality: shared topic, concern, goal, or issue
- Avoid generic themes like "General Discussion" unless truly miscellaneous

EXAMPLES OF GOOD THEMES:
✓ "Budget Constraints and Resource Allocation"
✓ "Employee Satisfaction and Retention Challenges"
✓ "Technical Infrastructure Modernization"
✓ "Customer Feedback on Product Quality"

EXAMPLES OF POOR THEMES:
✗ "Various Topics" (too vague)
✗ "Row mentions budget" (describes single row, not pattern)
✗ "Important Issues" (not specific)

OUTPUT FORMAT (JSON):
{
  "theme_name": "...",
  "theme_description": "..."
}

QUALITY CHECKS:
- Does this theme name clearly communicate the shared topic?
- Would someone unfamiliar with the data understand what this theme covers?
- Are the sampled summaries genuinely related to this theme?"""

# ============================================================================
# STATE DEFINITION
# ============================================================================

class ThematicAnalysisState(TypedDict):
    """State for the thematic analysis workflow"""
    # Input data
    raw_data: List[Dict]
    batch_size: int
    model_name: str
    
    # Summarization outputs
    summaries: Annotated[List[Dict], operator.add]
    
    # Clustering outputs
    embeddings: List[np.ndarray]
    cluster_labels: List[int]
    clusters: Dict[int, List[int]]  # cluster_id -> list of row_ids
    
    # Theme generation outputs
    themes: List[Dict]
    
    # Final assignments
    assignments: List[Dict]
    
    # Metadata
    current_batch: int
    total_batches: int


# ============================================================================
# MOCK DATA GENERATION
# ============================================================================

def generate_mock_data(n_rows: int = 50) -> List[Dict]:
    """Generate mock customer feedback data"""
    
    feedback_templates = [
        # Budget/Pricing theme
        "The pricing is too high for small businesses. We need more affordable plans.",
        "Budget constraints make it difficult to justify the cost. Please offer discounts.",
        "Cost is a major concern. The value doesn't match the price point.",
        "Expensive compared to competitors. Need better pricing tiers.",
        "Price increase was unexpected. Hard to budget for this.",
        
        # Product Quality theme
        "Product quality has declined recently. Many bugs in the latest release.",
        "The software crashes frequently. Quality control issues are evident.",
        "Features don't work as advertised. Poor quality overall.",
        "Reliability issues causing productivity loss. Quality needs improvement.",
        "Too many defects in the product. QA process seems insufficient.",
        
        # Customer Support theme
        "Support team is unresponsive. Takes days to get a reply.",
        "Customer service needs improvement. Long wait times for tickets.",
        "Support staff is helpful but response time is too slow.",
        "Need better support documentation and faster resolution times.",
        "Support experience has been frustrating. Hard to reach anyone.",
        
        # Features/Functionality theme
        "Missing critical features that competitors have. Need more functionality.",
        "Would love to see integration with other tools. Feature requests ignored.",
        "The feature set is limited. Need more customization options.",
        "Lacking advanced features for enterprise use. Please add more capabilities.",
        "Feature requests take too long to implement. More innovation needed.",
        
        # User Experience theme
        "Interface is confusing and not intuitive. UX needs redesign.",
        "Navigation is difficult. User experience could be much better.",
        "Too many clicks to complete simple tasks. Poor usability.",
        "Design is outdated. Modern UI/UX would greatly improve the experience.",
        "Learning curve is steep. Need better onboarding and UX.",
        
        # Performance theme
        "System is slow and laggy. Performance issues affecting work.",
        "Load times are unacceptable. Need significant performance improvements.",
        "Application freezes frequently. Performance optimization required.",
        "Speed has degraded over time. Performance monitoring needed.",
        "Response time is poor during peak hours. Scalability concerns.",
        
        # Security/Privacy theme
        "Security concerns with data handling. Need better encryption.",
        "Privacy policy is unclear. Want more transparency about data usage.",
        "Worried about data breaches. Security measures seem inadequate.",
        "Compliance requirements not met. Security certifications needed.",
        "Data protection features are insufficient. Privacy is a major concern.",
        
        # Integration theme
        "Doesn't integrate well with our existing tools. API limitations.",
        "Need better integration options with third-party services.",
        "API documentation is poor. Integration is challenging.",
        "Lack of webhook support makes automation difficult.",
        "Integration capabilities are limited compared to alternatives.",
        
        # Training/Documentation theme
        "Documentation is outdated and incomplete. Hard to learn the system.",
        "Need more training resources and tutorials. Onboarding is difficult.",
        "Knowledge base articles are not helpful. Better documentation needed.",
        "Training materials are insufficient for new users.",
        "Would benefit from video tutorials and better guides.",
        
        # Positive Feedback theme
        "Great product overall! Team loves using it daily.",
        "Excellent value for money. Very satisfied with the features.",
        "Support team is fantastic. Always helpful and responsive.",
        "Product has improved our workflow significantly. Highly recommend.",
        "Easy to use and reliable. Best tool we've implemented this year.",
    ]
    
    mock_data = []
    for i in range(n_rows):
        mock_data.append({
            'row_id': i + 1,
            'text': feedback_templates[i % len(feedback_templates)],
            'date': f"2025-10-{(i % 28) + 1:02d}",
            'customer_id': f"CUST_{(i % 20) + 1:03d}"
        })
    
    return mock_data


# ============================================================================
# TOOLS
# ============================================================================

class ContextManager:
    """Manages context and batching"""
    
    def __init__(self, model_context_limit: int = 4096, safety_margin: float = 0.8):
        self.context_limit = int(model_context_limit * safety_margin)
        
    def calculate_batch_size(self, avg_tokens_per_row: int) -> int:
        """Calculate optimal batch size"""
        # Reserve tokens for output and prompt
        available_tokens = self.context_limit - 500
        batch_size = max(1, available_tokens // avg_tokens_per_row)
        return min(batch_size, 20)  # Cap at 20 for manageable outputs


class EmbeddingGenerator:
    """Generate embeddings using sentence transformers"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
    
    def generate(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        return self.model.encode(texts, show_progress_bar=True)


class ClusteringTool:
    """Perform clustering on embeddings"""
    
    def __init__(self, min_cluster_size: int = 3, min_samples: int = 2):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
    
    def cluster(self, embeddings: np.ndarray) -> tuple:
        """Cluster embeddings and return labels and cluster mapping"""
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean'
        )
        labels = clusterer.fit_predict(embeddings)
        
        # Group row indices by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)
        
        return labels, clusters


class SmartAssignmentTool:
    """Assign rows to themes with confidence scoring"""
    
    def __init__(self, 
                 primary_threshold: float = 0.75,
                 secondary_threshold: float = 0.60,
                 ambiguity_gap: float = 0.10):
        self.primary_threshold = primary_threshold
        self.secondary_threshold = secondary_threshold
        self.ambiguity_gap = ambiguity_gap
    
    def assign(self, 
               row_embedding: np.ndarray, 
               theme_centroids: Dict[int, np.ndarray],
               themes: List[Dict]) -> Dict:
        """Assign a row to primary and secondary themes"""
        
        similarities = {}
        for cluster_id, centroid in theme_centroids.items():
            sim = cosine_similarity([row_embedding], [centroid])[0][0]
            similarities[cluster_id] = sim
        
        # Sort by similarity
        sorted_clusters = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_clusters:
            return {
                'primary_theme': 'Miscellaneous',
                'secondary_themes': [],
                'confidence': 'low',
                'method': 'no_match'
            }
        
        highest_score = sorted_clusters[0][1]
        second_highest = sorted_clusters[1][1] if len(sorted_clusters) > 1 else 0
        gap = highest_score - second_highest
        
        # Get theme names
        theme_map = {t['cluster_id']: t['theme_name'] for t in themes}
        
        # Deterministic assignment
        if highest_score >= self.primary_threshold and gap >= self.ambiguity_gap:
            primary = theme_map.get(sorted_clusters[0][0], 'Miscellaneous')
            secondary = [
                theme_map.get(c[0], '') 
                for c in sorted_clusters[1:4] 
                if c[1] >= self.secondary_threshold and theme_map.get(c[0])
            ]
            
            return {
                'primary_theme': primary,
                'secondary_themes': secondary,
                'confidence': 'high',
                'method': 'deterministic'
            }
        
        elif highest_score < self.secondary_threshold:
            return {
                'primary_theme': 'Miscellaneous',
                'secondary_themes': [],
                'confidence': 'low',
                'method': 'no_match'
            }
        
        else:
            # Would trigger LLM validation in production
            primary = theme_map.get(sorted_clusters[0][0], 'Miscellaneous')
            secondary = [
                theme_map.get(c[0], '') 
                for c in sorted_clusters[1:4] 
                if c[1] >= self.secondary_threshold and theme_map.get(c[0])
            ]
            
            return {
                'primary_theme': primary,
                'secondary_themes': secondary,
                'confidence': 'medium',
                'method': 'needs_validation'
            }


# ============================================================================
# LANGGRAPH NODES
# ============================================================================

def initialize_node(state: ThematicAnalysisState) -> ThematicAnalysisState:
    """Initialize the workflow"""
    print("\n" + "="*80)
    print("INITIALIZING THEMATIC ANALYSIS SYSTEM")
    print("="*80)
    
    # Generate mock data
    raw_data = generate_mock_data(50)
    print(f"✓ Generated {len(raw_data)} mock data rows")
    
    # Calculate batching
    context_mgr = ContextManager(model_context_limit=4096)
    batch_size = context_mgr.calculate_batch_size(avg_tokens_per_row=150)
    total_batches = (len(raw_data) + batch_size - 1) // batch_size
    
    print(f"✓ Batch size: {batch_size} rows per batch")
    print(f"✓ Total batches: {total_batches}")
    
    return {
        **state,
        'raw_data': raw_data,
        'batch_size': batch_size,
        'total_batches': total_batches,
        'current_batch': 0,
        'summaries': []
    }

from langchain_openai import ChatOpenAI
from langchain_core.runnables.retry import RunnableRetry
def summarization_node(state: ThematicAnalysisState) -> ThematicAnalysisState:
    """Summarize rows using LLM with async parallel processing"""
    print("\n" + "="*80)
    print("SUMMARIZATION AGENT (PARALLEL)")
    print("="*80)
    
    async def process_batch_async(batch: List[Dict], batch_num: int, total_batches: int, model_name: str) -> List[Dict]:
        """Process a single batch asynchronously"""
        print(f"\n📝 Processing batch {batch_num}/{total_batches} ({len(batch)} rows)")
        
        # Create prompt
        batch_text = "\n\n".join([
            f"Row {row['row_id']}: {row['text']}" 
            for row in batch
        ])
        
        
        llm = ChatOpenAI(model="qwen/qwen3-4b:free",
                     temperature=0.1,
                     base_url="https://openrouter.ai/api/v1",
                     api_key = api_key)
        
        # Wrap LLM with retry logic
        llm_with_retry = RunnableRetry(
            bound=llm,
            max_attempt_number=2,
            wait_exponential_jitter=True
        )
        
        messages = [
            SystemMessage(content=SUMMARIZATION_SYSTEM_PROMPT),
            HumanMessage(content=f"Analyze these rows:\n\n{batch_text}")
        ]
        
        try:
            response = await llm_with_retry.ainvoke(messages)
            result = json.loads(response.content)
            print(f"   ✓ Batch {batch_num} completed: {len(result)} rows summarized")
            return result
        except Exception as e:
            print(f"   ✗ Error in batch {batch_num}: {e}")
            # Fallback summaries
            fallback = []
            for row in batch:
                fallback.append({
                    'row_id': row['row_id'],
                    'summary': row['text'][:100] + "...",
                    'keywords': ['feedback', 'customer'],
                    'primary_topic': 'General Feedback'
                })
            return fallback
    
    async def process_all_batches():
        """Process all batches in parallel"""
        batch_size = state['batch_size']
        raw_data = state['raw_data']
        
        # Create batch tasks
        tasks = []
        for i in range(0, len(raw_data), batch_size):
            batch = raw_data[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            task = process_batch_async(batch, batch_num, state['total_batches'], state['model_name'])
            tasks.append(task)
        
        # Run all batches in parallel
        print(f"\n🚀 Launching {len(tasks)} parallel batch processes...")
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_summaries = []
        for batch_result in results:
            all_summaries.extend(batch_result)
        
        return all_summaries
    
    # Run async processing
    summaries = asyncio.run(process_all_batches())
    
    print(f"\n✓ Total summaries generated: {len(summaries)}")
    
    return {
        **state,
        'summaries': summaries
    }


def embedding_node(state: ThematicAnalysisState) -> ThematicAnalysisState:
    """Generate embeddings for all summaries"""
    print("\n" + "="*80)
    print("EMBEDDING GENERATION")
    print("="*80)
    
    embedding_gen = EmbeddingGenerator()
    
    summaries_text = [s['summary'] for s in state['summaries']]
    embeddings = embedding_gen.generate(summaries_text)
    
    print(f"✓ Generated embeddings: {embeddings.shape}")
    
    return {
        **state,
        'embeddings': embeddings
    }


def clustering_node(state: ThematicAnalysisState) -> ThematicAnalysisState:
    """Cluster embeddings to find themes"""
    print("\n" + "="*80)
    print("CLUSTERING")
    print("="*80)
    
    clustering_tool = ClusteringTool(min_cluster_size=3, min_samples=2)
    
    labels, clusters = clustering_tool.cluster(state['embeddings'])
    
    print(f"✓ Found {len(clusters)} clusters")
    for cluster_id, indices in clusters.items():
        if cluster_id == -1:
            print(f"   • Cluster {cluster_id} (Noise): {len(indices)} rows")
        else:
            print(f"   • Cluster {cluster_id}: {len(indices)} rows")
    
    return {
        **state,
        'cluster_labels': labels.tolist(),
        'clusters': clusters
    }


def theme_generation_node(state: ThematicAnalysisState) -> ThematicAnalysisState:
    """Generate theme names and descriptions using LLM with async parallel processing"""
    print("\n" + "="*80)
    print("THEME GENERATION (PARALLEL)")
    print("="*80)
    
    async def generate_theme_async(cluster_id: int, indices: List[int], summaries: List[Dict], model_name: str) -> Dict:
        """Generate theme for a single cluster asynchronously"""
        
        if cluster_id == -1:
            # Noise cluster becomes Miscellaneous
            return {
                'cluster_id': cluster_id,
                'theme_name': 'Miscellaneous',
                'theme_description': 'Items that do not fit clearly into identified themes',
                'row_count': len(indices)
            }
        
        llm = ChatOllama(model=model_name, temperature=0.3)
        
        # Sample summaries from cluster (max 10)
        sample_indices = indices[:min(10, len(indices))]
        sample_summaries = [summaries[i]['summary'] for i in sample_indices]
        
        # Collect keywords
        all_keywords = []
        for i in sample_indices:
            all_keywords.extend(summaries[i]['keywords'])
        
        # Count keyword frequency
        from collections import Counter
        keyword_counts = Counter(all_keywords)
        top_keywords = [k for k, _ in keyword_counts.most_common(5)]
        
        print(f"\n🎯 Generating theme for Cluster {cluster_id} ({len(indices)} rows)")
        print(f"   Keywords: {', '.join(top_keywords)}")
        
        # Create prompt
        summaries_text = "\n".join([f"- {s}" for s in sample_summaries])
        
        user_prompt = f"""Cluster size: {len(indices)} items
Top keywords: {', '.join(top_keywords)}

Sample summaries:
{summaries_text}

What theme connects these items?"""
        
        try:
            response = await llm.ainvoke([
                SystemMessage(content=THEME_GENERATION_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ])
            
            result = json.loads(response.content)
            print(f"   ✓ Cluster {cluster_id}: {result['theme_name']}")
            
            return {
                'cluster_id': cluster_id,
                'theme_name': result['theme_name'],
                'theme_description': result['theme_description'],
                'row_count': len(indices)
            }
            
        except Exception as e:
            print(f"   ✗ Error in Cluster {cluster_id}: {e}")
            return {
                'cluster_id': cluster_id,
                'theme_name': f'Theme {cluster_id}',
                'theme_description': 'Auto-generated theme',
                'row_count': len(indices)
            }
    
    async def generate_all_themes():
        """Generate all themes in parallel"""
        tasks = []
        for cluster_id, indices in state['clusters'].items():
            task = generate_theme_async(cluster_id, indices, state['summaries'], state['model_name'])
            tasks.append(task)
        
        print(f"\n🚀 Launching {len(tasks)} parallel theme generation tasks...")
        themes = await asyncio.gather(*tasks)
        return themes
    
    # Run async processing
    themes = asyncio.run(generate_all_themes())
    
    print(f"\n✓ Generated {len(themes)} themes")
    
    return {
        **state,
        'themes': themes
    }


def assignment_node(state: ThematicAnalysisState) -> ThematicAnalysisState:
    """Assign rows to themes"""
    print("\n" + "="*80)
    print("SMART ASSIGNMENT")
    print("="*80)
    
    # Calculate theme centroids
    theme_centroids = {}
    for theme in state['themes']:
        cluster_id = theme['cluster_id']
        if cluster_id in state['clusters']:
            indices = state['clusters'][cluster_id]
            cluster_embeddings = [state['embeddings'][i] for i in indices]
            centroid = np.mean(cluster_embeddings, axis=0)
            theme_centroids[cluster_id] = centroid
    
    # Assign each row
    assignment_tool = SmartAssignmentTool()
    assignments = []
    
    method_counts = {'deterministic': 0, 'needs_validation': 0, 'no_match': 0}
    
    for i, embedding in enumerate(state['embeddings']):
        assignment = assignment_tool.assign(embedding, theme_centroids, state['themes'])
        
        assignments.append({
            'row_id': state['summaries'][i]['row_id'],
            'primary_theme': assignment['primary_theme'],
            'secondary_themes': assignment['secondary_themes'],
            'confidence': assignment['confidence'],
            'method': assignment['method']
        })
        
        method_counts[assignment['method']] += 1
    
    print(f"\n✓ Assigned {len(assignments)} rows")
    print(f"   • Deterministic: {method_counts['deterministic']}")
    print(f"   • Needs validation: {method_counts['needs_validation']}")
    print(f"   • No match (Misc): {method_counts['no_match']}")
    
    return {
        **state,
        'assignments': assignments
    }


def output_node(state: ThematicAnalysisState) -> ThematicAnalysisState:
    """Format and display results"""
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    # Display themes
    print("\n📊 DISCOVERED THEMES:")
    print("-" * 80)
    for theme in state['themes']:
        print(f"\n{theme['theme_name']} ({theme['row_count']} rows)")
        print(f"   {theme['theme_description']}")
    
    # Display assignment summary
    print("\n\n📋 ASSIGNMENT SUMMARY:")
    print("-" * 80)
    
    theme_distribution = {}
    for assignment in state['assignments']:
        theme = assignment['primary_theme']
        theme_distribution[theme] = theme_distribution.get(theme, 0) + 1
    
    for theme, count in sorted(theme_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"   {theme}: {count} rows")
    
    # Sample assignments
    print("\n\n📝 SAMPLE ASSIGNMENTS (First 10 rows):")
    print("-" * 80)
    for assignment in state['assignments'][:10]:
        row_id = assignment['row_id']
        summary = state['summaries'][row_id - 1]['summary'][:60] + "..."
        primary = assignment['primary_theme']
        secondary = ", ".join(assignment['secondary_themes']) if assignment['secondary_themes'] else "None"
        
        print(f"\nRow {row_id}: {summary}")
        print(f"   Primary: {primary}")
        print(f"   Secondary: {secondary}")
        print(f"   Confidence: {assignment['confidence']}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80 + "\n")
    
    return state


# ============================================================================
# LANGGRAPH WORKFLOW
# ============================================================================

def create_workflow() -> StateGraph:
    """Create the LangGraph workflow"""
    
    workflow = StateGraph(ThematicAnalysisState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("summarize", summarization_node)
    workflow.add_node("embed", embedding_node)
    workflow.add_node("cluster", clustering_node)
    workflow.add_node("generate_themes", theme_generation_node)
    workflow.add_node("assign", assignment_node)
    workflow.add_node("output", output_node)
    
    # Add edges
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "summarize")
    workflow.add_edge("summarize", "embed")
    workflow.add_edge("embed", "cluster")
    workflow.add_edge("cluster", "generate_themes")
    workflow.add_edge("generate_themes", "assign")
    workflow.add_edge("assign", "output")
    workflow.add_edge("output", END)
    
    return workflow.compile()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

# def main():
#     """Run the thematic analysis system"""
    
#     print("\n🚀 Starting Thematic Analysis System")
#     print("=" * 80)
    
#     # Create workflow
#     app = create_workflow()
    
#     # Initial state
#     initial_state = {
#         'model_name': 'qwen3:0.6b',
#         'raw_data': [],
#         'batch_size': 10,
#         'summaries': [],
#         'embeddings': [],
#         'cluster_labels': [],
#         'clusters': {},
#         'themes': [],
#         'assignments': [],
#         'current_batch': 0,
#         'total_batches': 0
#     }
    
#     # Run workflow
#     final_state = app.invoke(initial_state)
    
#     return final_state


# if __name__ == "__main__":
#     main()
