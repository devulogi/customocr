# Enhanced Document Processor for RAG

## Overview
The Enhanced Document Processor combines OCR text extraction with document structure analysis to create high-quality semantic chunks optimized for RAG (Retrieval-Augmented Generation) applications.

## Processing Pipeline

### 1. PDF to Image Conversion
- Converts PDF pages to OpenCV images using PyMuPDF
- Maintains original resolution and quality

### 2. Image Enhancement
- **Contrast & Sharpness**: Enhanced by 1.3x and 1.2x respectively
- **Denoising**: Applied using fastNlMeansDenoisingColored
- **Adaptive Sizing**: Scales images to 1200-2400px for optimal OCR

### 3. Dual Analysis
- **OCR**: PaddleOCR extracts text with confidence scores and bounding boxes
- **Structure**: PPStructureV3 identifies document elements (titles, headers, text, figures, tables)

### 4. Element Combination
- Matches OCR text with structure elements using bounding box overlap (50% threshold)
- Creates unified elements with both text content and structural classification

### 5. Hierarchical Structure Building
- **Reading Order**: Sorts elements by Y-position, then X-position
- **Hierarchy Levels**: doc_title(1) > title(2) > header(3) > paragraph_title(4) > text/figure/table(5)
- **Parent-Child Relationships**: Builds document tree structure

### 6. Quality Filtering
- **OCR Confidence**: Minimum 0.85 threshold
- **Content Validation**: Filters garbled text, special characters, short content
- **Alpha Ratio**: Minimum 50% alphabetic characters

### 7. Semantic Chunking
- **Token Range**: 150-400 tokens per chunk for optimal embedding
- **Section Breaks**: Respects document structure boundaries
- **Coherent Content**: Combines titles with content for standalone readability

## Output Structure

Each chunk contains complete, contextually-aware passages:

```json
{
  "content": "Title - Subtitle: Main content text",
  "metadata": {
    "page": 0,
    "type": "semantic_chunk",
    "section_titles": ["Chapter 1", "Introduction"],
    "primary_section": "Introduction",
    "context_hierarchy": "Chapter 1 > Introduction", 
    "bbox": [x1, y1, x2, y2],
    "ocr_confidence": 0.942,
    "elements_count": 5,
    "token_count": 287,
    "hierarchy_levels": [2, 3, 5]
  }
}
```

## Key Features

### OCR Text Processing
- **Error Correction**: Fixes common OCR mistakes ("intllience" → "intelligence")
- **Spacing Fixes**: Corrects merged words and excessive whitespace
- **Quality Thresholds**: Filters low-confidence and garbled text

### Structure Recognition
- **Document Elements**: Automatically identifies titles, headers, paragraphs, figures, tables
- **Bounding Box Matching**: Links OCR text with structural elements
- **Hierarchical Classification**: Assigns proper document hierarchy levels

### Semantic Chunking
- **Context Preservation**: Maintains title-content relationships
- **Optimal Sizing**: 150-400 tokens for embedding models
- **Standalone Readability**: Each chunk is self-contained with context

## Metadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `content` | String | "Title - Subtitle: Content" format for standalone readability |
| `page` | Number | Zero-indexed page number (0 = page 1) |
| `type` | String | Always "semantic_chunk" |
| `section_titles` | Array | All titles/headers in chunk |
| `primary_section` | String | Main section title |
| `context_hierarchy` | String | Full path ("Chapter > Section > Subsection") |
| `bbox` | Array[4] | Bounding box [x1, y1, x2, y2] |
| `ocr_confidence` | Number | Average confidence (0.85+ filtered) |
| `elements_count` | Number | OCR elements combined |
| `token_count` | Number | Word count (150-400 range) |
| `hierarchy_levels` | Array | Structure levels present |

## Installation

### Optional: Install ccache for faster compilation

To improve PaddlePaddle compilation performance and eliminate ccache warnings:

```bash
# Ubuntu/Debian
sudo apt-get install ccache

# macOS
brew install ccache

# CentOS/RHEL
sudo yum install ccache
```

**Note**: ccache is optional and only affects initial compilation time, not runtime performance. Without ccache, you may see a warning during first run, but functionality remains unaffected.

## Usage Example

```python
from enhanced_document_processor import process_document, get_pdf_path

# Process document
pdf_path = get_pdf_path("document.pdf")
chunks = process_document(pdf_path)

# Each chunk contains:
for chunk in chunks:
    print(f"Content: {chunk['content'][:100]}...")
    print(f"Page: {chunk['metadata']['page']}")
    print(f"Section: {chunk['metadata']['primary_section']}")
    print(f"Confidence: {chunk['metadata']['ocr_confidence']:.3f}")
    print(f"Tokens: {chunk['metadata']['token_count']}")
    print("---")
```

## Quality Assurance

- **OCR Confidence**: ≥0.85 threshold with adaptive filtering
- **Content Validation**: Removes garbled text and special characters
- **Token Optimization**: 150-400 tokens for embedding models
- **Structure Preservation**: Maintains document hierarchy
- **Context Completeness**: Self-contained chunks with title context

## System Architecture

### Activity Diagram
Shows the complete processing pipeline from PDF input to RAG-ready chunks:

![Activity Diagram](diagrams/Enhanced%20Document%20Processor%20Activity%20Diagram.png)

### Sequence Diagram
Illustrates detailed interaction flow between system components:

![Sequence Diagram](diagrams/Enhanced%20Document%20Processor%20Sequence%20Diagram.png)

### Creating Diagrams

To generate new diagrams or update existing ones:

```bash
# Download PlantUML (one-time setup)
wget -O diagrams/plantuml.jar https://github.com/plantuml/plantuml/releases/download/v1.2024.7/plantuml-1.2024.7.jar

# Create/edit .puml files in diagrams/ folder
# Example: diagrams/my_diagram.puml

# Generate PNG from PlantUML source
cd diagrams
java -jar plantuml.jar -tpng my_diagram.puml

# This creates: my_diagram.png
```

**PlantUML Syntax Examples:**
- Activity: `@startuml` → `start` → `:Activity;` → `stop` → `@enduml`
- Sequence: `@startuml` → `A -> B: message` → `@enduml`
- Class: `@startuml` → `class MyClass { +method() }` → `@enduml`

## Elasticsearch Integration

### Index Setup

```python
# Create index with mapping
def create_index(index_name="document_chunks"):
    mapping = {
        "mappings": {
            "properties": {
                "content": {"type": "text", "analyzer": "standard"},
                "embeddings": {"type": "dense_vector", "dims": 384},
                "metadata": {
                    "properties": {
                        "page": {"type": "integer"},
                        "type": {"type": "keyword"},
                        "section_titles": {"type": "keyword"},
                        "primary_section": {"type": "text"},
                        "context_hierarchy": {"type": "text"},
                        "bbox": {"type": "float"},
                        "ocr_confidence": {"type": "float"},
                        "elements_count": {"type": "integer"},
                        "token_count": {"type": "integer"},
                        "hierarchy_levels": {"type": "integer"}
                    }
                }
            }
        }
    }
    es.indices.create(index=index_name, body=mapping)
```

### Elasticsearch Mapping Fields

| Field | ES Type | Purpose | Usage |
|-------|---------|---------|-------|
| `content` | text | Full-text search on chunk content | Keyword search, text matching |
| `embeddings` | dense_vector | Vector similarity search | Semantic search, cosine similarity |
| `metadata.page` | integer | Filter by page number | Page-specific retrieval |
| `metadata.type` | keyword | Exact match on chunk type | Filter by "semantic_chunk" |
| `metadata.section_titles` | keyword | Exact match on section names | Filter by specific sections |
| `metadata.primary_section` | text | Search within section content | Find chunks in specific topics |
| `metadata.context_hierarchy` | text | Search hierarchical paths | Navigate document structure |
| `metadata.bbox` | float | Spatial coordinates | Position-based filtering |
| `metadata.ocr_confidence` | float | Quality filtering | Filter high-confidence content |
| `metadata.elements_count` | integer | Chunk complexity filtering | Find detailed vs simple chunks |
| `metadata.token_count` | integer | Size-based filtering | Control chunk length for context |
| `metadata.hierarchy_levels` | integer | Structure depth filtering | Find specific document levels |

### CRUD Operations

#### CREATE (POST)
```python
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

es = Elasticsearch(["http://localhost:9200"])
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create chunk without vector (initial upload)
def create_chunk_content(chunk, index_name="document_chunks"):
    doc = {
        "content": chunk["content"],
        "metadata": chunk["metadata"]
    }
    fragment_id = chunk["fragment_id"]
    return es.index(index=index_name, id=fragment_id, body=doc)

# Create chunk with vector embedding
def create_chunk_with_vector(chunk, index_name="document_chunks"):
    embedding = model.encode(chunk["content"]).tolist()
    doc = {
        "content": chunk["content"],
        "embeddings": embedding,
        "metadata": chunk["metadata"]
    }
    fragment_id = chunk["fragment_id"]
    return es.index(index=index_name, id=fragment_id, body=doc)

# Bulk create chunks without vectors
def bulk_create_chunks(chunks, index_name="document_chunks"):
    from elasticsearch.helpers import bulk
    actions = []
    for chunk in chunks:
        actions.append({
            "_index": index_name,
            "_id": chunk["fragment_id"],
            "_source": {
                "content": chunk["content"],
                "metadata": chunk["metadata"]
            }
        })
    return bulk(es, actions)
```

#### READ (GET)
```python
# Get chunk by fragment_id
def get_chunk(fragment_id, index_name="document_chunks"):
    return es.get(index=index_name, id=fragment_id)

# Get chunk content for vector generation
def get_chunk_content(fragment_id, index_name="document_chunks"):
    result = es.get(index=index_name, id=fragment_id, _source=["content"])
    return result["_source"]["content"]

# Get multiple chunks by fragment_ids
def get_chunks(fragment_ids, index_name="document_chunks"):
    return es.mget(index=index_name, body={"ids": fragment_ids})

# Get all chunks without vectors (for processing)
def get_chunks_without_vectors(index_name="document_chunks"):
    return es.search(
        index=index_name,
        body={
            "query": {"bool": {"must_not": {"exists": {"field": "embeddings"}}}}
        },
        size=1000
    )
```

#### UPDATE (PUT)
```python
# Add vector embedding to existing chunk
def add_vector_to_chunk(fragment_id, index_name="document_chunks"):
    # Get content from ES
    content = get_chunk_content(fragment_id, index_name)
    # Generate embedding
    embedding = model.encode(content).tolist()
    # Update chunk with vector
    return es.update(
        index=index_name, 
        id=fragment_id, 
        body={"doc": {"embeddings": embedding}}
    )

# Update chunk content and regenerate vector
def update_chunk_content(fragment_id, new_content, index_name="document_chunks"):
    embedding = model.encode(new_content).tolist()
    return es.update(
        index=index_name,
        id=fragment_id,
        body={"doc": {"content": new_content, "embeddings": embedding}}
    )

# Bulk add vectors to chunks without embeddings
def bulk_add_vectors(index_name="document_chunks"):
    from elasticsearch.helpers import bulk
    # Get chunks without vectors
    chunks = get_chunks_without_vectors(index_name)
    actions = []
    
    for hit in chunks["hits"]["hits"]:
        fragment_id = hit["_id"]
        content = hit["_source"]["content"]
        embedding = model.encode(content).tolist()
        
        actions.append({
            "_op_type": "update",
            "_index": index_name,
            "_id": fragment_id,
            "doc": {"embeddings": embedding}
        })
    
    return bulk(es, actions)
```

#### DELETE
```python
# Delete chunk by fragment_id
def delete_chunk(fragment_id, index_name="document_chunks"):
    return es.delete(index=index_name, id=fragment_id)

# Delete chunks by page
def delete_chunks_by_page(page_num, index_name="document_chunks"):
    return es.delete_by_query(
        index=index_name,
        body={"query": {"term": {"metadata.page": page_num}}}
    )

# Delete chunks without vectors
def delete_chunks_without_vectors(index_name="document_chunks"):
    return es.delete_by_query(
        index=index_name,
        body={"query": {"bool": {"must_not": {"exists": {"field": "embeddings"}}}}}
    )

# Delete entire index
def delete_index(index_name="document_chunks"):
    return es.indices.delete(index=index_name)
```

### Search Operations

#### Semantic Search (Vector)
```python
def semantic_search(query, top_k=5, index_name="document_chunks"):
    query_vector = model.encode(query).tolist()
    return es.search(
        index=index_name,
        body={
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embeddings') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            },
            "size": top_k
        }
    )
```

#### Filtered Search
```python
def filtered_search(query, min_confidence=0.85, page=None, index_name="document_chunks"):
    query_vector = model.encode(query).tolist()
    filters = [{"range": {"metadata.ocr_confidence": {"gte": min_confidence}}}]
    if page is not None:
        filters.append({"term": {"metadata.page": page}})
    
    return es.search(
        index=index_name,
        body={
            "query": {
                "script_score": {
                    "query": {"bool": {"filter": filters}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embeddings') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            },
            "size": 5
        }
    )
```

#### Text Search
```python
def text_search(query, section_name=None, index_name="document_chunks"):
    must_clauses = [{"match": {"content": query}}]
    if section_name:
        must_clauses.append({"match": {"metadata.primary_section": section_name}})
    
    return es.search(
        index=index_name,
        body={
            "query": {"bool": {"must": must_clauses}},
            "highlight": {"fields": {"content": {}}}
        }
    )
```

#### Hybrid Search
```python
def hybrid_search(query, alpha=0.7, index_name="document_chunks"):
    query_vector = model.encode(query).tolist()
    return es.search(
        index=index_name,
        body={
            "query": {
                "bool": {
                    "should": [
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": f"{alpha} * (cosineSimilarity(params.query_vector, 'embeddings') + 1.0)",
                                    "params": {"query_vector": query_vector}
                                }
                            }
                        },
                        {
                            "function_score": {
                                "query": {"match": {"content": query}},
                                "script_score": {
                                    "script": {"source": f"{1-alpha} * _score"}
                                }
                            }
                        }
                    ]
                }
            }
        }
    )
```

### Vector Embedding Workflow

```python
# 1. Upload chunks without vectors (fast initial upload)
bulk_create_chunks(chunks)

# 2. Process chunks to add vector embeddings
bulk_add_vectors()

# 3. Verify all chunks have vectors
chunks_without_vectors = get_chunks_without_vectors()
print(f"Chunks without vectors: {chunks_without_vectors['hits']['total']['value']}")

# 4. Individual chunk vector update
fragment_id = "chunk_0_a1b2c3d4e5f6"
add_vector_to_chunk(fragment_id)
```

### Best Practices

1. **Fragment ID**: Use `fragment_id` for all ES operations instead of auto-generated IDs
2. **Two-Phase Upload**: Upload content first, then add vectors in separate operation
3. **Batch Processing**: Use bulk operations for vector generation to improve performance
4. **Vector Dimensions**: Match embedding model dimensions (384 for all-MiniLM-L6-v2)
5. **Confidence Filtering**: Filter by `ocr_confidence >= 0.85` for reliable results
6. **Source Attribution**: Use `fragment_id`, `page`, `primary_section`, `bbox` for responses

## Integration Notes

- **Vector Embeddings**: Use `content` field for similarity search
- **Metadata Filtering**: Use `metadata` fields for precise retrieval
- **Source Citations**: Combine `page`, `primary_section`, and `bbox` for attribution
- **Quality Assurance**: Monitor `ocr_confidence` for content reliability