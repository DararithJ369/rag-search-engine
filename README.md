# RAG Search Engine

A comprehensive Retrieval-Augmented Generation (RAG) search engine for movie information, combining multiple search strategies (keyword, semantic, hybrid) with LLM-powered answer generation and reranking.

## Features

- **Keyword Search**: Fast text-based retrieval using TF-IDF and term frequency
- **Semantic Search**: Dense vector-based search using sentence transformers
- **Hybrid Search**: Combines keyword and semantic search with Reciprocal Rank Fusion (RRF) and weighted scoring
- **Reranking**: LLM-based and cross-encoder reranking for result refinement
- **LLM Integration**: Google Gemini API for query augmentation, answer generation, and citation-based responses
- **Citation Support**: Both numbered [1], [2], etc. and doc_id-based citation formats
- **Evaluation**: Metrics-based evaluation for search quality assessment
- **Multimodal Search**: Image-based search capabilities (description-based matching)

## Project Structure

```
rag-search-engine/
├── cli/                           # Command-line interfaces
│   ├── augumented_generation_cli.py   # RAG orchestration
│   ├── semantic_search_cli.py         # Semantic search interface
│   ├── keyword_search_cli.py          # Keyword search interface
│   ├── hybrid_search_cli.py           # Hybrid search interface
│   ├── multimodal_search_cli.py       # Image search interface
│   ├── evaluation_cli.py              # Evaluation framework
│   └── lib/                       # Core modules
│       ├── llm.py                 # LLM integration (Gemini API)
│       ├── semantic_search.py     # Dense vector search
│       ├── keyword_search.py      # TF-IDF and term frequency search
│       ├── hybrid_search.py       # Hybrid search orchestration
│       ├── multimodal_search.py   # Image/multimodal search
│       ├── rag.py                 # RAG pipeline
│       ├── evaluation.py          # Evaluation metrics
│       ├── search_utils.py        # Utility functions
│       └── prompts/               # LLM prompt templates
│           ├── answer_with_citations.md
│           ├── answer_question_detailed.md
│           ├── answer_question.md
│           ├── summarization.md
│           ├── expand.md
│           ├── rewrite.md
│           ├── spelling.md
│           ├── llm_judge.md
│           ├── rerank.py
│           ├── batch_rerank.md
│           └── individual_rerank.md
├── data/                          # Data files
│   ├── movies.json                # Movie dataset
│   ├── golden_dataset.json        # Evaluation dataset
│   ├── stopwords.txt              # Stopwords for filtering
│   └── paddington.jpeg            # Sample image for multimodal search
├── cache/                         # Cached embeddings and indices
│   ├── chunk_embeddings.npy       # Precomputed embeddings
│   ├── chunk_metadata.json        # Chunk metadata
│   ├── docmap.pkl                 # Document mapping
│   ├── index.pkl                  # Search index
│   ├── doc_lengths.pkl            # Document lengths
│   └── term_frequencies.pkl       # Term frequency cache
├── pyproject.toml                 # Project configuration
├── .env                           # Environment variables (API keys)
└── README.md                      # This file
```

## Installation

### Prerequisites
- Python 3.11+
- Google Gemini API key

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-search-engine
```

2. Create virtual environment (using uv):
```bash
uv venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
uv pip install -e .
```

Or using pip:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your generative api key. i.e., GEMINI_API_KEY
```

## Usage

### Keyword Search
```bash
python cli/keyword_search_cli.py "search query"
```

### Semantic Search
```bash
python cli/semantic_search_cli.py "search query"
```

### Hybrid Search
```bash
python cli/hybrid_search_cli.py "search query" --method rrf
```

### RAG with Citations
```bash
python cli/augumented_generation_cli.py "What is this movie about?" --method citations
```

### Detailed Question Answering
```bash
python cli/augumented_generation_cli.py "Detailed question?" --method detailed
```

### Image-Based Search
```bash
python cli/multimodal_search_cli.py --image path/to/image.jpg
```

### Evaluation
```bash
python cli/evaluation_cli.py --dataset golden_dataset.json
```

## Citation Formats

The system supports two citation formats:

### [1], [2], etc. (Numbered Citations)
Used in `answer_with_citations` for user-friendly output:
```
The movie stars actors from [1] and follows a storyline similar to [2].
```

### (doc_id: <id>) (ID-Based Citations)
Used in `answer_question_detailed` for precise source tracking:
```
The protagonist (doc_id: 54) faces challenges similar to (doc_id: 12).
```

## Architecture

### Search Pipeline
1. **Query Processing**: Spelling correction, query expansion, query rewriting
2. **Retrieval**: Multiple search methods (keyword, semantic, hybrid)
3. **Reranking**: LLM-based or cross-encoder reranking
4. **Answer Generation**: LLM generates answers with citations

### Document Representation
- Chunks: Documents split into manageable segments
- Embeddings: Dense vectors for semantic search (sentence-transformers)
- Metadata: doc_id, title, description for each chunk

## Configuration

Key settings in `cli/lib/search_utils.py`:
- `CHUNK_SIZE`: Document chunk size (default: 500 characters)
- `OVERLAP`: Chunk overlap (default: 100 characters)
- `TOP_K`: Number of results to retrieve (default: 5)
- `RERANK_METHOD`: Reranking strategy (default: "llm")

## Future Work

### 1. **Enhanced Semantic Search**
   - [ ] Add support for multi-lingual embeddings
   - [ ] Implement approximate nearest neighbor (ANN) indexing (FAISS, Annoy)
   - [ ] Support for domain-specific embeddings fine-tuning

### 2. **Advanced Reranking**
   - [ ] Implement learning-to-rank (LTR) models
   - [ ] Add diversity-aware reranking
   - [ ] Support for user feedback-based reranking
   - [ ] Multi-stage reranking pipeline with confidence thresholds

### 3. **Citation & Attribution**
   - [ ] Extract and highlight cited passages in source documents
   - [ ] Support for full-text quotes in citations
   - [ ] Citation confidence scoring
   - [ ] Handling conflicting information across sources

### 4. **Multimodal Search**
   - [ ] Improve image-to-text matching algorithms
   - [ ] Add video frame extraction and search
   - [ ] Cross-modal retrieval (image + text queries)
   - [ ] Vision-language model integration

### 5. **LLM Enhancements**
   - [ ] Support for multiple LLM providers (OpenAI, Anthropic, Llama)
   - [ ] Local LLM inference support
   - [ ] Fine-tuned models for domain-specific tasks
   - [ ] Chain-of-thought reasoning for complex queries

### 6. **Performance Optimization**
   - [ ] Batch processing for large-scale queries
   - [ ] Caching strategies for frequent queries
   - [ ] GPU acceleration for embedding computation
   - [ ] Query result pagination

### 7. **Evaluation & Metrics**
   - [ ] NDCG (Normalized Discounted Cumulative Gain)
   - [ ] MAP (Mean Average Precision)
   - [ ] Citation accuracy metrics
   - [ ] Human evaluation framework
   - [ ] A/B testing infrastructure

### 8. **User Interface**
   - [ ] Web UI (React/Vue.js frontend)
   - [ ] Real-time streaming answers
   - [ ] Chat-based interface
   - [ ] Visual result exploration

### 9. **Knowledge Integration**
   - [ ] Knowledge graph integration
   - [ ] Structured fact extraction
   - [ ] Entity disambiguation
   - [ ] Relationship-based search

### 10. **Production Deployment**
   - [ ] Docker containerization
   - [ ] API server (FastAPI)
   - [ ] Database integration (PostgreSQL)
   - [ ] Monitoring and logging infrastructure
   - [ ] Rate limiting and authentication
   - [ ] Horizontal scaling support

## Dependencies

- `google-genai`: Google Gemini API client
- `sentence-transformers`: Semantic embeddings
- `numpy`: Numerical computing
- `nltk`: Natural language processing
- `pillow`: Image processing
- `python-dotenv`: Environment configuration
- `huggingface-hub`: Model loading

## Environment Variables

```env
GEMINI_API_KEY=your_api_key_here
```

## Notes

- Cache files (`__pycache__`, `*.pyc`) are gitignored
- Environment files (`.env`, `.python-version`) are gitignored
- Data files and cache are pre-generated; ensure they're available before running
- API keys must be set via `.env` file before execution

## License

Add license information here.

## Contributors

- [DararithJ369]

## References

- [Boot dev]

---

**Last Updated**: April 2026
