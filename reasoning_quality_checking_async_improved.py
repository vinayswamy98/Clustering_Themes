import os
import asyncio
import pandas as pd
from datetime import datetime, timezone
from PyPDF2 import PdfReader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_core.output_parsers import PydanticOutputParser
from tenacity import retry, stop_after_attempt, wait_exponential
import argparse
import logging
import json
from typing import List, Dict, Any, Optional
from prometheus_client import Counter, Histogram
import time
import aiofiles
from aiohttp import ClientSession, TCPConnector

# Constants
CURRENT_UTC = "2025-10-30 05:17:44"
CURRENT_USER = "vinayswamy98"
MAX_CONCURRENT_CALLS = 5
CHUNK_SIZE = 50

# Set up logging with UTC time
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s UTC - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Metrics setup
class Metrics:
    validation_duration = Histogram('validation_duration_seconds', 'Time spent validating records')
    llm_calls = Counter('llm_calls_total', 'Total number of LLM API calls')
    errors = Counter('validation_errors_total', 'Total number of validation errors')

# Rate limiter for API calls
class RateLimiter:
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.tokens = asyncio.Queue(maxsize=rate_limit)
        for _ in range(rate_limit):
            self.tokens.put_nowait(None)
    
    async def acquire(self):
        await self.tokens.get()
    
    async def release(self):
        await self.tokens.put(None)

# Custom exceptions
class ValidationError(Exception):
    def __init__(self, message: str, recoverable: bool = True):
        self.recoverable = recoverable
        super().__init__(message)

# Pydantic models for validation
class InputRecord(BaseModel):
    title: str
    description: str
    parameters: List[str]
    scores: Dict[str, float]
    
    @validator('scores')
    def validate_scores(cls, v):
        if not all(0 <= score <= 5 for score in v.values()):
            raise ValueError("Scores must be between 0 and 5")
        return v

class ReasonFeedback(BaseModel):
    reason_aligned_with_guidelines: bool = Field(..., description="True if reason follows guideline structure")
    missing_guideline_elements: List[str] = Field(default_factory=list)
    description_relevant_points_used: bool = Field(..., description="True if key description points are used")
    reason_quality: str = Field(..., description="STRONG, ADEQUATE, or WEAK")
    improved_reason: str = Field(..., description="Full improved reason text")
    confidence: float = Field(..., ge=0.0, le=1.0)
    validated_by: str = Field(default=CURRENT_USER)
    validated_at: str = Field(default=CURRENT_UTC)

    @validator('reason_quality')
    def valid_quality(cls, v):
        if v not in ["STRONG", "ADEQUATE", "WEAK"]:
            raise ValueError("Must be STRONG, ADEQUATE, or WEAK")
        return v

class ValidationResult(BaseModel):
    __root__: Dict[str, ReasonFeedback] = Field(..., description="Mapping of parameter to feedback")

async def safe_file_read(file_path: str) -> str:
    try:
        async with aiofiles.open(file_path, 'rb') as f:
            content = await f.read()
        reader = PdfReader(content)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise ValidationError(f"Failed to read file: {e}", recoverable=False)

async def process_guidelines(pdf_path: str, excel1_path: str, excel2_path: str) -> List[Dict[str, Any]]:
    docs = []
    
    # PDF processing
    pdf_text = await safe_file_read(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(pdf_text)
    docs.extend([{"page_content": c, "metadata": {"type": "pdf", "parameter": "general"}} for c in chunks])

    # Excel processing
    try:
        df1 = pd.read_excel(excel1_path)
        df2 = pd.read_excel(excel2_path)
        
        # Process score scale
        for _, row in df1.iterrows():
            param = row['Parameter']
            for r in range(1, 6):
                content = str(row.get(f'score {r}', '')).strip()
                if content:
                    docs.append({
                        "page_content": content,
                        "metadata": {"type": "score_scale", "parameter": param, "score": r}
                    })

        # Process examples
        for _, row in df2.iterrows():
            param = row['Parameter']
            for ex_type, content in [
                ("good_reason", row.get('Good reason Example', '')),
                ("weak_reason", row.get('Weak reason Example', ''))
            ]:
                if content and str(content).strip():
                    docs.append({
                        "page_content": str(content).strip(),
                        "metadata": {"type": ex_type, "parameter": param}
                    })
    except Exception as e:
        logger.error(f"Error processing Excel files: {e}")
        raise ValidationError(f"Failed to process Excel files: {e}", recoverable=False)

    return docs

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def robust_llm_call(prompt: str, llm: ChatAnthropic) -> Any:
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, llm.invoke, prompt)
        Metrics.llm_calls.inc()
        return result
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        Metrics.errors.inc()
        raise

async def validate_record(
    record: InputRecord,
    vectorstore: Chroma,
    llm: ChatAnthropic,
    rate_limiter: RateLimiter,
    top_k: int
) -> Dict[str, ReasonFeedback]:
    start_time = time.time()
    
    try:
        await rate_limiter.acquire()
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k * 3})
        retrieved = await asyncio.get_event_loop().run_in_executor(
            None, retriever.invoke, "score scale definitions and good/weak reason examples"
        )
        
        context = "\n".join([
            f"[{d.metadata.get('parameter','general')}] {d.page_content}"
            for d in retrieved
        ])

        param_inputs = []
        for p in record.parameters:
            param_inputs.append(f"""
            PARAMETER: {p}
            score: {record.scores.get(p, '')}
            reason: {record.reasons.get(p, '')}
            """
        )

        parser = PydanticOutputParser(pydantic_object=ValidationResult)
        prompt = f"""
        You are validating **reasons against guidelines** using description for context.
        Current validator: {CURRENT_USER}
        Validation time (UTC): {CURRENT_UTC}

        Title: {record.title}
        Description (context only): {record.description}

        Guidelines:
        {context}

        For each parameter:
        - Check alignment with good reason patterns
        - Identify missing guideline elements
        - Confirm use of key description points
        - Suggest improved reason

        Parameters:
        {''.join(param_inputs)}

        {parser.get_format_instructions()}
        """

        result = await robust_llm_call(prompt, llm)
        return result.__root__
    except Exception as e:
        logger.error(f"Validation error for record {record.title}: {e}")
        raise
    finally:
        await rate_limiter.release()
        Metrics.validation_duration.observe(time.time() - start_time)

async def process_batch(
    batch: pd.DataFrame,
    vectorstore: Chroma,
    llm: ChatAnthropic,
    rate_limiter: RateLimiter,
    top_k: int
) -> List[Dict[str, Any]]:
    results = []
    for _, row in batch.iterrows():
        try:
            record = InputRecord(
                title=row['Title'],
                description=row['Description'],
                parameters=[col.replace('_score', '') for col in row.index if col.endswith('_score')],
                scores={col.replace('_score', ''): row[col] for col in row.index if col.endswith('_score')}
            )
            result = await validate_record(record, vectorstore, llm, rate_limiter, top_k)
            results.append({
                'Title': record.title,
                'Description': record.description,
                **{f"{p}_score": record.scores.get(p, '') for p in record.parameters},
                **{f"{p}_feedback": json.dumps(result.get(p, {}).dict()) for p in record.parameters}
            })
        except Exception as e:
            logger.error(f"Error processing record: {e}")
            results.append({
                'Title': row['Title'],
                'Description': row['Description'],
                'error': str(e)
            })
    return results

async def main(args):
    # Setup
    embeddings = SentenceTransformerEmbeddings(model_name=args.embedding_model)
    vectorstore = Chroma(persist_directory=args.chroma_db_path, embedding_function=embeddings)
    llm = ChatAnthropic(model="claude-3-haiku-20240307", api_key=os.environ['ANTHROPIC_API_KEY'])
    rate_limiter = RateLimiter(MAX_CONCURRENT_CALLS)

    # Process guidelines
    try:
        await process_guidelines(args.pdf, args.excel1, args.excel2)
    except ValidationError as e:
        if not e.recoverable:
            logger.error("Fatal error in guidelines processing")
            return

    # Process records in chunks
    all_results = []
    for chunk in pd.read_excel(args.excel3, chunksize=CHUNK_SIZE):
        results = await process_batch(chunk, vectorstore, llm, rate_limiter, args.top_k)
        all_results.extend(results)
        
        # Periodic save
        temp_df = pd.DataFrame(all_results)
        temp_df.to_excel(f"{args.output}.temp", index=False)
        
    # Final save
    final_df = pd.DataFrame(all_results)
    final_df.to_excel(args.output, index=False)
    
    # Save metrics
    with open(f"{args.output}.metrics.json", 'w') as f:
        json.dump({
            'total_records': len(all_results),
            'total_llm_calls': Metrics.llm_calls._value.get(),
            'total_errors': Metrics.errors._value.get(),
            'validation_time': datetime.now(timezone.utc).isoformat(),
            'validator': CURRENT_USER
        }, f, indent=2)

    logger.info(f"Validation complete. Report saved to: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Async Reason Validation System")
    parser.add_argument('--pdf', required=True, help='Path to guidelines PDF')
    parser.add_argument('--excel1', required=True, help='Path to score scale Excel')
    parser.add_argument('--excel2', required=True, help='Path to reason examples Excel')
    parser.add_argument('--excel3', required=True, help='Path to records Excel')
    parser.add_argument('--output', default='reason_validation_report.xlsx', help='Output Excel path')
    parser.add_argument('--embedding_model', default='all-MiniLM-L6-v2', help='Embedding model')
    parser.add_argument('--chroma_db_path', default='./chroma_langchain', help='Chroma vectorstore path')
    parser.add_argument('--top_k', type=int, default=5, help='Top K for retrieval')
    args = parser.parse_args()

    asyncio.run(main(args))
