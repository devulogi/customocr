"""Lambda Chunk Processor - Lightweight Chunking and Elasticsearch Integration"""

import json
import boto3
import hashlib
from typing import List, Dict, Any
from elasticsearch import Elasticsearch

# Cache hierarchy levels for faster lookup
HIERARCHY_LEVELS = {
    "doc_title": 1,
    "title": 2,
    "header": 3,
    "paragraph_title": 4,
    "text": 5,
    "figure": 5,
    "chart": 5,
    "table": 5,
    "figure_title": 5,
}

TITLE_TYPES = frozenset(["doc_title", "title", "header", "paragraph_title"])

def is_quality_content(content: str, ocr_confidence: float) -> bool:
    """Optimized single-pass content quality check."""
    if not content or ocr_confidence < 0.65:
        return False

    stripped = content.strip()
    length = len(stripped)

    if length < 2:
        return False

    alnum_count = 0
    special_count = 0
    char_counts = {}
    max_char_count = 0

    for char in stripped.lower():
        if char.isalnum():
            alnum_count += 1
        elif not char.isspace():
            special_count += 1

        char_counts[char] = char_counts.get(char, 0) + 1
        max_char_count = max(max_char_count, char_counts[char])

    if alnum_count / length < 0.5:
        return False
    if special_count / length > 0.5:
        return False
    if length >= 4 and max_char_count / length > 0.7:
        return False

    return True

def build_hierarchical_structure(elements: List[Dict[str, Any]], page_num: int) -> List[Dict[str, Any]]:
    """Optimized hierarchical structure building."""
    elements.sort(key=lambda x: (x["page_position"]["y"], x["page_position"]["x"]))

    hierarchy = []
    parent_stack = []

    for elem_index, element in enumerate(elements):
        current_level = HIERARCHY_LEVELS.get(element["structure_type"], 5)

        while parent_stack and parent_stack[-1]["level"] >= current_level:
            parent_stack.pop()

        parent_id = parent_stack[-1]["id"] if parent_stack else None

        hier_element = {
            "id": elem_index,
            "parent_id": parent_id,
            "level": current_level,
            "type": element["structure_type"],
            "content": element["content"],
            "page": page_num,
            "bbox": element["bbox"],
            "ocr_confidence": element["ocr_confidence"],
        }

        hierarchy.append(hier_element)

        if current_level <= 4:
            parent_stack.append(hier_element)

    return hierarchy

def create_semantic_chunks(hierarchy: List[Dict[str, Any]], page_num: int, item_id: str) -> List[Dict[str, Any]]:
    """Optimized semantic chunking with token caching."""
    chunks = []

    quality_elements = []
    for e in hierarchy:
        if is_quality_content(e["content"], e["ocr_confidence"]):
            e["_token_count"] = len(e["content"].split())
            quality_elements.append(e)

    current_chunk = []
    current_tokens = 0
    target_min, target_max = 150, 400
    title_types = frozenset(["doc_title", "title"])

    for element in quality_elements:
        element_tokens = element["_token_count"]

        if (
            current_tokens + element_tokens > target_max
            and current_tokens >= target_min
        ) or (
            element["type"] in title_types
            and current_chunk
            and current_tokens >= target_min
        ):

            if current_chunk:
                chunks.append(create_chunk(current_chunk, page_num, item_id))
            current_chunk = [element]
            current_tokens = element_tokens
        else:
            current_chunk.append(element)
            current_tokens += element_tokens

    if current_chunk:
        chunks.append(create_chunk(current_chunk, page_num, item_id))

    return chunks

def create_chunk(elements: List[Dict[str, Any]], page_num: int, item_id: str) -> Dict[str, Any]:
    """Ultra-optimized chunk creation with minimal allocations."""
    if not elements:
        return {}

    title_contents = []
    content_parts = []
    total_confidence = 0.0
    total_tokens = 0
    hierarchy_levels = set()
    
    for e in elements:
        total_confidence += e["ocr_confidence"]
        hierarchy_levels.add(e["level"])
        total_tokens += e.get("_token_count", len(e["content"].split()))
        
        content = e["content"]
        if e["type"] in TITLE_TYPES:
            title_contents.append(content)
        else:
            content_parts.append(content)

    # Build content
    if title_contents and content_parts:
        content = " - ".join(title_contents) + ": " + " ".join(content_parts)
    elif title_contents:
        content = " - ".join(title_contents)
    elif content_parts:
        content = " ".join(content_parts)
    else:
        content = ""

    # Generate fragment ID
    primary_bbox = elements[0]["bbox"]
    content_sample = elements[0]["content"][:50]
    fragment_id = f"{item_id}_chunk_{page_num}_{hashlib.md5(f'{item_id}_{page_num}_{primary_bbox}_{content_sample}'.encode()).hexdigest()[:12]}"

    return {
        "item_id": item_id,
        "content": content,
        "metadata": {
            "fragment_id": fragment_id,
            "page": page_num,
            "type": "semantic_chunk",
            "section_titles": title_contents,
            "primary_section": title_contents[0] if title_contents else None,
            "context_hierarchy": " > ".join(title_contents) if title_contents else None,
            "bbox": primary_bbox,
            "ocr_confidence": total_confidence / len(elements),
            "elements_count": len(elements),
            "token_count": total_tokens,
            "hierarchy_levels": list(hierarchy_levels),
        },
    }

def save_chunks_to_elasticsearch(chunks: List[Dict[str, Any]], es_index: str = "document-chunks"):
    """Save chunks to Elasticsearch."""
    es = Elasticsearch([{"host": "localhost", "port": 9200}])
    
    for chunk in chunks:
        fragment_id = chunk["metadata"]["fragment_id"]
        
        doc = {
            "content": chunk["content"],
            "metadata": chunk["metadata"]
        }
        
        es.index(index=es_index, id=fragment_id, body=doc)

def lambda_handler(event, context):
    """
    Chunk Lambda Handler - Lightweight Processing
    
    Input SQS Message:
    {
        "Records": [{
            "body": "{\"bucket\": \"ocr-results-bucket\", \"key\": \"doc123/pages_0-5.json\", \"item_id\": \"doc123\"}"
        }]
    }
    """
    try:
        fragment_ids = []
        
        # Parse SQS message
        for record in event['Records']:
            message = json.loads(record['body'])
            bucket = message.get("bucket")
            key = message.get("key")
            item_id = message.get("item_id")
            page_range = message.get("page_range", {})

            if not all([bucket, key, item_id]):
                continue

            # Download OCR results from S3
            s3 = boto3.client('s3')
            response = s3.get_object(Bucket=bucket, Key=key)
            ocr_data = json.loads(response['Body'].read())

            all_chunks = []
            
            # Process each page's OCR results
            for page_result in ocr_data["ocr_results"]:
                page_num = page_result["page_num"]
                elements = page_result["elements"]
                
                # Build hierarchy and create chunks
                hierarchy = build_hierarchical_structure(elements, page_num)
                page_chunks = create_semantic_chunks(hierarchy, page_num, item_id)
                all_chunks.extend(page_chunks)

            # Save chunks to Elasticsearch
            save_chunks_to_elasticsearch(all_chunks)
            
            # Collect fragment IDs
            for chunk in all_chunks:
                fragment_ids.append(chunk["metadata"]["fragment_id"])

            print(f"Chunk processing completed for {item_id}, generated {len(all_chunks)} chunks")

        # Send completion message to vectorization queue (optional)
        if fragment_ids:
            sqs = boto3.client('sqs')
            vectorization_queue_url = "https://sqs.region.amazonaws.com/account/vectorization-queue"
            
            vectorization_message = {
                "fragment_ids": fragment_ids,
                "elasticsearch_index": "document-chunks",
                "item_id": item_id
            }

            sqs.send_message(
                QueueUrl=vectorization_queue_url,
                MessageBody=json.dumps(vectorization_message)
            )

        return {"statusCode": 200}

    except Exception as e:
        print(f"Chunk processing error: {str(e)}")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

# For local testing
if __name__ == "__main__":
    test_event = {
        "Records": [{
            "body": json.dumps({
                "bucket": "ocr-results-bucket",
                "key": "E-Invoice/pages_0-2.json",
                "item_id": "E-Invoice",
                "page_range": {"start": 0, "end": 2}
            })
        }]
    }
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))