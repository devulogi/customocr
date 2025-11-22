"""Lambda Document Processor - OCR and Structure Analysis for RAG"""

import json
import hashlib
import re
import pymupdf as fitz
import cv2
import numpy as np
from paddleocr import PaddleOCR, PPStructureV3
from PIL import Image, ImageEnhance
from typing import List, Dict, Any
import tempfile
from collections import Counter
from dataclasses import dataclass
from typing import Tuple
import os
import time

# Global model instances for Lambda container reuse
_models = None


def get_models():
    global _models
    if _models is None:
        _models = {"ocr": PaddleOCR(), "structure": PPStructureV3()}
    return _models


def should_enhance_image(img: np.ndarray) -> bool:
    """Determine if image needs enhancement based on quality metrics."""
    # Quick quality check - if image is already high quality, skip enhancement
    height, width = img.shape[:2]

    # Skip enhancement for high-resolution images (likely good quality)
    if height >= 1800 and width >= 1800:
        return False

    # Check image sharpness using Laplacian variance
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # If image is already sharp enough, skip enhancement
    return laplacian_var < 100


def enhance_image_for_ocr(img: np.ndarray) -> np.ndarray:
    """Conditionally enhance image for OCR with optimized operations."""
    if not should_enhance_image(img):
        # Just resize if needed, skip expensive enhancements
        height, width = img.shape[:2]
        min_size, max_size = 1200, 2400

        if height < min_size or width < min_size:
            scale = max(min_size / height, min_size / width)
            new_width, new_height = int(width * scale), int(height * scale)
            return cv2.resize(
                img, (new_width, new_height), interpolation=cv2.INTER_CUBIC
            )
        elif height > max_size or width > max_size:
            scale = min(max_size / height, max_size / width)
            new_width, new_height = int(width * scale), int(height * scale)
            return cv2.resize(
                img, (new_width, new_height), interpolation=cv2.INTER_AREA
            )

        return img

    # Fast OpenCV-only enhancement (no PIL conversions)
    enhanced = img.copy()

    # Fast contrast and brightness adjustment using OpenCV
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.3, beta=10)

    # Light denoising with reduced parameters
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 15)

    # Adaptive sizing
    height, width = enhanced.shape[:2]
    min_size, max_size = 1200, 2400

    if height < min_size or width < min_size:
        scale = max(min_size / height, min_size / width)
        new_width, new_height = int(width * scale), int(height * scale)
        enhanced = cv2.resize(
            enhanced, (new_width, new_height), interpolation=cv2.INTER_CUBIC
        )
    elif height > max_size or width > max_size:
        scale = min(max_size / height, max_size / width)
        new_width, new_height = int(width * scale), int(height * scale)
        enhanced = cv2.resize(
            enhanced, (new_width, new_height), interpolation=cv2.INTER_AREA
        )

    return enhanced


def bbox_overlap(box1, box2, threshold=0.5) -> bool:
    """Optimized bounding box overlap with early exit and no type conversions."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Early exit for non-overlapping boxes
    if x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1:
        return False

    # Calculate intersection without type conversions
    x1_i, y1_i = max(x1_1, x1_2), max(y1_1, y1_2)
    x2_i, y2_i = min(x2_1, x2_2), min(y2_1, y2_2)

    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    return intersection > threshold * min(area1, area2)


def is_quality_content(content: str, ocr_confidence: float) -> bool:
    """Optimized single-pass content quality check."""
    if not content or ocr_confidence < 0.65:
        return False

    stripped = content.strip()
    length = len(stripped)

    if length < 2:
        return False

    # Single-pass character analysis
    alnum_count = 0
    special_count = 0
    char_counts = {}
    max_char_count = 0

    for char in stripped.lower():
        if char.isalnum():
            alnum_count += 1
        elif not char.isspace():
            special_count += 1

        # Track character frequency for repetition check
        char_counts[char] = char_counts.get(char, 0) + 1
        max_char_count = max(max_char_count, char_counts[char])

    # Early exit checks
    if alnum_count / length < 0.5:  # Less than 50% alphanumeric
        return False

    if special_count / length > 0.5:  # More than 50% special characters
        return False

    if length >= 4 and max_char_count / length > 0.7:  # Repetitive pattern
        return False

    return True


@dataclass
class StructureElement:
    """Optimized structure element for spatial indexing."""

    bbox: Tuple[float, float, float, float]
    label: str
    confidence: float
    x_center: float
    y_center: float


def create_spatial_index(structure_elements: List[Dict]) -> List[StructureElement]:
    """Create spatial index for faster structure matching."""
    indexed_elements = []
    for elem in structure_elements:
        bbox = elem["bbox"]
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        indexed_elements.append(
            StructureElement(
                bbox=tuple(bbox),
                label=elem["label"],
                confidence=elem["confidence"],
                x_center=x_center,
                y_center=y_center,
            )
        )
    # Sort by y-coordinate for faster spatial queries
    return sorted(indexed_elements, key=lambda x: x.y_center)


def find_matching_structure(
    box: List[float], indexed_structures: List[StructureElement]
) -> str:
    """Fast structure matching using spatial indexing."""
    box_y_center = (box[1] + box[3]) / 2

    # Binary search-like approach for y-coordinate
    for struct_elem in indexed_structures:
        # Early exit if we're too far in y-direction
        if abs(struct_elem.y_center - box_y_center) > 100:  # Reasonable threshold
            continue

        if bbox_overlap(box, struct_elem.bbox):
            return struct_elem.label

    return "text"


def combine_ocr_and_structure(ocr_result, structure_result) -> List[Dict[str, Any]]:
    """Optimized OCR and structure combination with spatial indexing."""
    combined_elements = []

    if not (isinstance(ocr_result, dict) and "rec_texts" in ocr_result):
        return combined_elements

    ocr_texts = ocr_result["rec_texts"]
    ocr_scores = ocr_result["rec_scores"]
    ocr_boxes = ocr_result["rec_boxes"]

    # Extract and index structure data
    structure_elements = []
    if "parsing_res_list" in structure_result:
        for item in structure_result["parsing_res_list"]:
            if hasattr(item, "bbox") and hasattr(item, "label"):
                structure_elements.append(
                    {
                        "bbox": item.bbox,
                        "label": item.label,
                        "confidence": getattr(item, "score", 0.0),
                    }
                )

    # Create spatial index for faster matching
    indexed_structures = create_spatial_index(structure_elements)

    # Match OCR text with structure elements using spatial index
    for text, score, box in zip(ocr_texts, ocr_scores, ocr_boxes):
        bbox_list = box.tolist() if hasattr(box, "tolist") else list(box)

        element = {
            "content": text,
            "ocr_confidence": float(score),
            "bbox": bbox_list,
            "structure_type": find_matching_structure(bbox_list, indexed_structures),
            "page_position": {
                "x": int(box[0]),
                "y": int(box[1]),
                "width": int(box[2] - box[0]),
                "height": int(box[3] - box[1]),
            },
        }

        combined_elements.append(element)

    return combined_elements


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


def build_hierarchical_structure(
    elements: List[Dict[str, Any]], page_num: int
) -> List[Dict[str, Any]]:
    """Optimized hierarchical structure building."""
    # Sort once with optimized key function
    elements.sort(key=lambda x: (x["page_position"]["y"], x["page_position"]["x"]))

    hierarchy = []
    parent_stack = []

    for elem_index, element in enumerate(elements):
        current_level = HIERARCHY_LEVELS.get(element["structure_type"], 5)

        # Optimize parent stack management
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


def create_semantic_chunks(
    hierarchy: List[Dict[str, Any]], page_num: int, item_id: str
) -> List[Dict[str, Any]]:
    """Optimized semantic chunking with token caching."""
    chunks = []

    # Pre-filter quality elements and cache token counts
    quality_elements = []
    for e in hierarchy:
        if is_quality_content(e["content"], e["ocr_confidence"]):
            # Cache token count to avoid repeated splitting
            e["_token_count"] = len(e["content"].split())
            quality_elements.append(e)

    current_chunk = []
    current_tokens = 0
    target_min, target_max = 150, 400
    title_types = frozenset(["doc_title", "title"])  # Use frozenset for faster lookup

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


# Pre-compiled frozen sets for maximum lookup performance
TITLE_TYPES = frozenset(["doc_title", "title", "header", "paragraph_title"])


def create_chunk(
    elements: List[Dict[str, Any]], page_num: int, item_id: str
) -> Dict[str, Any]:
    """Ultra-optimized chunk creation with minimal allocations."""
    if not elements:
        return {}

    # Pre-allocate lists with estimated capacity
    title_contents = []
    content_parts = []
    total_confidence = 0.0
    total_tokens = 0
    hierarchy_levels = set()
    
    # Single-pass processing with minimal operations
    for e in elements:
        total_confidence += e["ocr_confidence"]
        hierarchy_levels.add(e["level"])
        total_tokens += e.get("_token_count", len(e["content"].split()))
        
        content = e["content"]
        if e["type"] in TITLE_TYPES:
            title_contents.append(content)
        else:
            content_parts.append(content)

    # Optimized content building
    if title_contents and content_parts:
        content = " - ".join(title_contents) + ": " + " ".join(content_parts)
    elif title_contents:
        content = " - ".join(title_contents)
    elif content_parts:
        content = " ".join(content_parts)
    else:
        content = ""

    # Optimized fragment ID with minimal string operations
    primary_bbox = elements[0]["bbox"]
    content_sample = elements[0]["content"][:50]
    fragment_id = f"{item_id}_chunk_{page_num}_{hashlib.md5(f'{item_id}_{page_num}_{primary_bbox}_{content_sample}'.encode()).hexdigest()[:12]}"

    return {
        "item_id": item_id,
        "content": content,
        "vector": None,
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


def save_chunks_for_vectorization(chunks: List[Dict[str, Any]], output_path: str):
    """Save chunks in format ready for vector database."""
    from pathlib import Path

    output_data = {
        "chunks": chunks,
        "total_chunks": len(chunks),
        "chunk_types": list({chunk["metadata"]["type"] for chunk in chunks}),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def process_image_in_memory(img: np.ndarray, ocr, structure_pipeline):
    """Process image in memory without temp files."""
    # Try to process directly with numpy array if models support it
    try:
        # Some PaddleOCR versions support numpy arrays directly
        ocr_result = ocr.predict(img)[0]
        structure_result = structure_pipeline.predict(img)[0]
        return ocr_result, structure_result
    except:
        # Fallback to temp file if direct processing fails
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_file:
            cv2.imwrite(temp_file.name, img)
            ocr_result = ocr.predict(temp_file.name)[0]
            structure_result = structure_pipeline.predict(temp_file.name)[0]
            return ocr_result, structure_result


def lambda_handler(event, context):
    """
    Optimized Lambda handler for document processing

    Parameters:
    - item_id: Unique string to identify the document
    - page_range: Dictionary with 'start' and 'end' fields (0-indexed)
    """
    try:
        item_id = event.get("item_id")
        page_range = event.get("page_range", {})

        if not item_id:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "item_id is required"}),
            }

        start_page = page_range.get("start", 0)
        end_page = page_range.get("end")

        # Get PDF from local pdfs directory
        from pathlib import Path

        pdf_path = str(Path.cwd() / "pdfs" / f"{item_id}.pdf")

        if not os.path.exists(pdf_path):
            return {
                "statusCode": 404,
                "body": json.dumps({"error": f"Document {item_id} not found"}),
            }

        # Get models (singleton for container reuse)
        models = get_models()
        ocr = models["ocr"]
        structure_pipeline = models["structure"]

        # Process PDF
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        if end_page is None:
            end_page = total_pages - 1

        # Validate page range
        start_page = max(0, min(start_page, total_pages - 1))
        end_page = max(start_page, min(end_page, total_pages - 1))

        all_chunks = []

        for page_num in range(start_page, end_page + 1):
            # Convert page to image
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Conditionally enhance image
            enhanced_img = enhance_image_for_ocr(img)

            # Process in memory (no temp files)
            ocr_result, structure_result = process_image_in_memory(
                enhanced_img, ocr, structure_pipeline
            )

            # Combine results
            combined_elements = combine_ocr_and_structure(ocr_result, structure_result)

            # Build hierarchy and create chunks
            hierarchy = build_hierarchical_structure(combined_elements, page_num)
            page_chunks = create_semantic_chunks(hierarchy, page_num, item_id)
            all_chunks.extend(page_chunks)

        doc.close()

        # Save chunks for vectorization
        output_path = str(Path.cwd() / "output" / f"{item_id}_chunks.json")
        save_chunks_for_vectorization(all_chunks, output_path)

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "item_id": item_id,
                    "processed_pages": f"{start_page}-{end_page}",
                    "total_chunks": len(all_chunks),
                    "chunks": all_chunks,
                }
            ),
        }

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}


# For local testing
if __name__ == "__main__":
    SYSTEM_DESIGN_PROCESS = "SYSTEM_DESIGN_PROCESS"
    EInvoice = "E-Invoice"
    System_Design = "System_Design"
    test_event = {
        "item_id": EInvoice,
        "page_range": {"start": 0, "end": 2},
    }
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))
