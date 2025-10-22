import json
import pandas as pd
from typing import List, Iterator
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import jsonlines

# Setup Logging
logging.basicConfig(level=logging.INFO, filename='theme_consolidation.log')
logger = logging.getLogger(__name__)

# Pydantic Models (unchanged)
class Category(BaseModel):
    category_name: str
    description: str
    keywords: List[str]
    example_themes: List[str]
    estimated_theme_count: int

class Proposal(BaseModel):
    proposed_categories: List[Category]
    batch_id: int

class ConsolidatedTheme(BaseModel):
    consolidated_name: str
    description: str
    keywords: List[str]
    aliases: List[str]
    merged_from: List[str]

class MergeSummary(BaseModel):
    original_proposals: int
    final_categories: int
    merge_ratio: str

class ConsolidatedDefinitions(BaseModel):
    final_consolidated_themes: List[ConsolidatedTheme]
    merge_summary: MergeSummary

class ThemeMapping(BaseModel):
    original: str
    consolidated: str

class MappingOutput(BaseModel):
    mappings: List[ThemeMapping]

# LangChain Setup
llm = OpenAI()
executor = ThreadPoolExecutor(max_workers=4)

# SQLite Setup for Chunked Storage
def init_mapping_db(db_path: str = 'theme_mapping.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS mappings
                     (original TEXT PRIMARY KEY, consolidated TEXT)''')
    conn.commit()
    return conn

def save_mapping_chunk(conn: sqlite3.Connection, mappings: dict):
    cursor = conn.cursor()
    cursor.executemany('INSERT OR REPLACE INTO mappings VALUES (?, ?)',
                      [(k, v) for k, v in mappings.items()])
    conn.commit()

def load_mappings(db_path: str = 'theme_mapping.db') -> dict:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT original, consolidated FROM mappings')
    mappings = dict(cursor.fetchall())
    conn.close()
    return mappings

# Stream Unique Themes
def stream_unique_themes(df: pd.DataFrame) -> Iterator[str]:
    primary = df['Primary_Theme'].dropna().unique()
    secondary = df['Secondary_Theme'].dropna().unique()
    unique_themes = set(primary).union(secondary)
    for theme in sorted(unique_themes):
        yield theme

# Stage 1: Analyze Themes
async def stage1_analyze_all_batches(theme_iterator: Iterator[str], batch_size: int = 100) -> List[Proposal]:
    parser = PydanticOutputParser(pydantic_object=Proposal)
    prompt = PromptTemplate(
        input_variables=["themes", "batch_num", "batch_size"],
        template="""Analyze these themes and propose consolidated categories.

THEMES ({batch_size} themes):
{themes}

TASK:
Identify patterns and propose 3-8 consolidated categories.

OUTPUT FORMAT (JSON):
{parser.get_format_instructions()}
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)
    proposals = []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def call_llm_async(chain: LLMChain, **kwargs) -> str:
        try:
            return await chain.arun(**kwargs)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    async def process_batch(batch: List[str], batch_num: int):
        logger.info(f"Batch {batch_num}: Analyzing {len(batch)} themes...")
        themes_str = "\n".join([f"{j+1}. {theme}" for j, theme in enumerate(batch)])
        try:
            proposal = await call_llm_async(chain, themes=themes_str, batch_num=batch_num, batch_size=len(batch))
            logger.info(f"✓ Batch {batch_num}: {len(proposal.proposed_categories)} categories proposed")
            return proposal
        except Exception as e:
            logger.error(f"Batch {batch_num} failed: {e}")
            return None

    batch = []
    batch_num = 0
    for theme in theme_iterator:
        batch.append(theme)
        if len(batch) >= batch_size:
            batch_num += 1
            result = await process_batch(batch, batch_num)
            if result:
                proposals.append(result)
            batch = []

    if batch:  # Handle remaining themes
        batch_num += 1
        result = await process_batch(batch, batch_num)
        if result:
            proposals.append(result)

    return proposals

# Stage 2: Merge Proposals (unchanged)
async def stage2_merge_proposals(proposals: List[Proposal]) -> ConsolidatedDefinitions:
    parser = PydanticOutputParser(pydantic_object=ConsolidatedDefinitions)
    prompt = PromptTemplate(
        input_variables=["categories"],
        template="""Merge these proposed categories into a final taxonomy.

PROPOSED CATEGORIES:
{categories}

TASK:
1. Identify duplicates/overlaps
2. Merge into 10-30 consolidated themes

OUTPUT FORMAT (JSON):
{parser.get_format_instructions()}
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)
    
    all_categories = []
    for p in proposals:
        for cat in p.proposed_categories:
            all_categories.append(f"- **{cat.category_name}** (Batch {p.batch_id}): {cat.description}\n  Keywords: {', '.join(cat.keywords)}")
    
    try:
        consolidated = await call_llm_async(chain, categories="\n".join(all_categories))
        logger.info(f"✓ Created {len(consolidated.final_consolidated_themes)} categories")
        return consolidated
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        raise

# Stage 3: Map Themes
async def stage3_map_all_themes(theme_iterator: Iterator[str], definitions: ConsolidatedDefinitions, batch_size: int = 100, db_path: str = 'theme_mapping.db') -> None:
    parser = PydanticOutputParser(pydantic_object=MappingOutput)
    prompt = PromptTemplate(
        input_variables=["context", "themes", "batch_size"],
        template="""Map themes to consolidated categories.

CONSOLIDATED CATEGORIES:
{context}

ORIGINAL THEMES ({batch_size} themes):
{themes}

OUTPUT FORMAT (JSON):
{parser.get_format_instructions()}
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)
    conn = init_mapping_db(db_path)
    context = "\n".join([f"- **{t.consolidated_name}**: {t.description}" for t in definitions.final_consolidated_themes])
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def call_llm_async(chain: LLMChain, **kwargs) -> str:
        try:
            return await chain.arun(**kwargs)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    async def process_batch(batch: List[str], batch_num: int):
        logger.info(f"Batch {batch_num}: Mapping {len(batch)} themes...")
        themes_str = "\n".join([f"{j+1}. {theme}" for j, theme in enumerate(batch)])
        try:
            mappings = await call_llm_async(chain, context=context, themes=themes_str, batch_size=len(batch))
            batch_mapping = {m.original: m.consolidated for m in mappings.mappings}
            save_mapping_chunk(conn, batch_mapping)
            logger.info(f"✓ Batch {batch_num}: {len(batch_mapping)} mapped")
        except Exception as e:
            logger.error(f"Batch {batch_num} mapping failed: {e}")

    batch = []
    batch_num = 0
    for theme in theme_iterator:
        batch.append(theme)
        if len(batch) >= batch_size:
            batch_num += 1
            await process_batch(batch, batch_num)
            batch = []

    if batch:  # Handle remaining themes
        batch_num += 1
        await process_batch(batch, batch_num)

    conn.close()
    unmapped = set(theme_iterator) - set(load_mappings(db_path).keys())
    if unmapped:
        logger.warning(f"Unmapped themes: {len(unmapped)} - {list(unmapped)[:5]}")

# Main Workflow
async def consolidate_large_dataset_no_sampling(df: pd.DataFrame, batch_size: int = 100, db_path: str = 'theme_mapping.db') -> tuple[dict, pd.DataFrame, ConsolidatedDefinitions]:
    logger.info("=" * 50)
    logger.info("THEME CONSOLIDATION STARTED")
    logger.info("=" * 50)
    
    theme_iterator = stream_unique_themes(df)
    theme_count = len(set(df['Primary_Theme'].dropna().to_list() + df['Secondary_Theme'].dropna().to_list()))
    logger.info(f"Total rows: {len(df):,}, Unique themes: {theme_count:,}")

    proposals = await stage1_analyze_all_batches(theme_iterator, batch_size)
    definitions = await stage2_merge_proposals(proposals)
    
    # Reset iterator for mapping
    theme_iterator = stream_unique_themes(df)
    await stage3_map_all_themes(theme_iterator, definitions, batch_size, db_path)

    mapping = load_mappings(db_path)
    df['Consolidated_Primary_Theme'] = df['Primary_Theme'].map(mapping).fillna(df['Primary_Theme'])
    df['Consolidated_Secondary_Theme'] = df['Secondary_Theme'].map(mapping).fillna(df['Secondary_Theme'])

    logger.info(f"Coverage: {len(mapping) / theme_count * 100:.1f}%")
    logger.info("Top 10 Themes:")
    for theme, count in df['Consolidated_Primary_Theme'].value_counts().head(10).items():
        logger.info(f"  {theme}: {count:,} ({count/len(df)*100:.1f}%)")

    return mapping, df, definitions

# Unit Tests (unchanged, included for completeness)
class TestThemeConsolidationChains(unittest.TestCase):
    def setUp(self):
        self.llm = AsyncMock()
        self.themes = ["theme1", "theme2"]
        self.batch_size = 2
        self.proposal = Proposal(
            proposed_categories=[
                Category(
                    category_name="Test Category",
                    description="Test description",
                    keywords=["key1", "key2"],
                    example_themes=["theme1"],
                    estimated_theme_count=2
                )
            ],
            batch_id=1
        )
        self.definitions = ConsolidatedDefinitions(
            final_consolidated_themes=[
                ConsolidatedTheme(
                    consolidated_name="Final Category",
                    description="Final description",
                    keywords=["key1"],
                    aliases=["alias1"],
                    merged_from=["Test Category"]
                )
            ],
            merge_summary=MergeSummary(
                original_proposals=1,
                final_categories=1,
                merge_ratio="1:1"
            )
        )
        self.mapping_output = MappingOutput(
            mappings=[
                ThemeMapping(original="theme1", consolidated="Final Category"),
                ThemeMapping(original="theme2", consolidated="Final Category")
            ]
        )

    @patch("langchain.chains.LLMChain.arun")
    async def test_stage1_chain(self, mock_arun):
        mock_arun.return_value = self.proposal.json()
        parser = PydanticOutputParser(pydantic_object=Proposal)
        prompt = PromptTemplate(
            input_variables=["themes", "batch_num", "batch_size"],
            template="""Analyze themes.\nTHEMES ({batch_size} themes):\n{themes}\n\nOUTPUT FORMAT (JSON):\n{parser.get_format_instructions()}"""
        )
        chain = LLMChain(llm=self.llm, prompt=prompt, output_parser=parser)
        themes_str = "\n".join([f"{i+1}. {t}" for i, t in enumerate(self.themes)])
        result = await call_llm_async(chain, themes=themes_str, batch_num=1, batch_size=2)
        self.assertIsInstance(result, Proposal)
        self.assertEqual(len(result.proposed_categories), 1)
        self.assertEqual(result.batch_id, 1)

    @patch("langchain.chains.LLMChain.arun")
    async def test_stage2_chain(self, mock_arun):
        mock_arun.return_value = self.definitions.json()
        parser = PydanticOutputParser(pydantic_object=ConsolidatedDefinitions)
        prompt = PromptTemplate(
            input_variables=["categories"],
            template="""Merge categories.\nPROPOSED CATEGORIES:\n{categories}\n\nOUTPUT FORMAT (JSON):\n{parser.get_format_instructions()}"""
        )
        chain = LLMChain(llm=self.llm, prompt=prompt, output_parser=parser)
        categories = "- **Test Category** (Batch 1): Test description\n  Keywords: key1, key2"
        result = await call_llm_async(chain, categories=categories)
        self.assertIsInstance(result, ConsolidatedDefinitions)
        self.assertEqual(len(result.final_consolidated_themes), 1)
        self.assertEqual(result.merge_summary.final_categories, 1)

    @patch("langchain.chains.LLMChain.arun")
    async def test_stage3_chain(self, mock_arun):
        mock_arun.return_value = self.mapping_output.json()
        parser = PydanticOutputParser(pydantic_object=MappingOutput)
        prompt = PromptTemplate(
            input_variables=["context", "themes", "batch_size"],
            template="""Map themes.\nCONSOLIDATED CATEGORIES:\n{context}\n\nORIGINAL THEMES ({batch_size} themes):\n{themes}\n\nOUTPUT FORMAT (JSON):\n{parser.get_format_instructions()}"""
        )
        chain = LLMChain(llm=self.llm, prompt=prompt, output_parser=parser)
        context = "- **Final Category**: Final description"
        themes_str = "\n".join([f"{i+1}. {t}" for i, t in enumerate(self.themes)])
        result = await call_llm_async(chain, context=context, themes=themes_str, batch_size=2)
        self.assertIsInstance(result, MappingOutput)
        self.assertEqual(len(result.mappings), 2)
        self.assertEqual(result.mappings[0].original, "theme1")

# Usage
if __name__ == "__main__":
    df = pd.read_csv('your_10k_dataset.csv')
    mapping, df_consolidated, definitions = asyncio.run(consolidate_large_dataset_no_sampling(df, batch_size=100))

    with open('theme_mapping_full.json', 'w') as f:
        json.dump(mapping, f, indent=2)
    with open('consolidated_theme_definitions.json', 'w') as f:
        json.dump(definitions.dict(), f, indent=2)
    df_consolidated.to_csv('dataset_with_consolidated_themes.csv', index=False)
    
    logger.info(f"Saved: mappings ({len(mapping):,}), definitions, dataset")
    
    unittest.main(argv=[''], exit=False)