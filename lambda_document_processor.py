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
import os

from regex import E

# Global model instances for Lambda container reuse
_models = None


def get_models():
    global _models
    if _models is None:
        _models = {"ocr": PaddleOCR(), "structure": PPStructureV3()}
    return _models


def enhance_image_for_ocr(img: np.ndarray) -> np.ndarray:
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Enhance contrast and sharpness
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.3)
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(1.2)

    enhanced = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)

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


def bbox_overlap(box1, box2, threshold=0.5):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1_i, y1_i = max(x1_1, x1_2), max(y1_1, y1_2)
    x2_i, y2_i = min(x2_1, x2_2), min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return False

    intersection = float(x2_i - x1_i) * float(y2_i - y1_i)
    area1 = float(x2_1 - x1_1) * float(y2_1 - y1_1)
    area2 = float(x2_2 - x1_2) * float(y2_2 - y1_2)

    if area1 <= 0 or area2 <= 0:
        return False

    return (intersection / min(area1, area2)) > threshold


def post_process_ocr_text(text: str) -> str:
    corrections = {
        "intllience": "intelligence",
        "artifical": "artificial",
        "wasestimated": "was estimated",
        "projectedto": "projected to",
        "andthe": "and the",
        "theend": "the end",
    }

    cleaned_text = text
    for error, correction in corrections.items():
        cleaned_text = cleaned_text.replace(error, correction)

    cleaned_text = re.sub(r"([a-z])([A-Z])", r"\1 \2", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    return cleaned_text.strip()


def is_quality_content(content: str, ocr_confidence: float) -> bool:
    if not content or not content.strip() or ocr_confidence < 0.85:
        return False

    alpha_ratio = sum(c.isalpha() for c in content) / max(len(content), 1)
    if alpha_ratio < 0.5 or len(content.strip()) < 2:
        return False

    special_char_ratio = sum(
        not c.isalnum() and not c.isspace() for c in content
    ) / len(content)
    return special_char_ratio <= 0.5


def combine_ocr_and_structure(ocr_result, structure_result) -> List[Dict[str, Any]]:
    combined_elements = []

    if not (isinstance(ocr_result, dict) and "rec_texts" in ocr_result):
        return combined_elements

    ocr_texts = ocr_result["rec_texts"]
    ocr_scores = ocr_result["rec_scores"]
    ocr_boxes = ocr_result["rec_boxes"]

    # Extract structure data
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

    # Match OCR text with structure elements
    for text, score, box in zip(ocr_texts, ocr_scores, ocr_boxes):
        element = {
            "content": post_process_ocr_text(text),
            "ocr_confidence": float(score),
            "bbox": box.tolist() if hasattr(box, "tolist") else list(box),
            "structure_type": "text",
            "page_position": {
                "x": int(box[0]),
                "y": int(box[1]),
                "width": int(box[2] - box[0]),
                "height": int(box[3] - box[1]),
            },
        }

        # Find matching structure element
        for struct_elem in structure_elements:
            if bbox_overlap(box, struct_elem["bbox"]):
                element["structure_type"] = struct_elem["label"]
                break

        combined_elements.append(element)

    return combined_elements


def build_hierarchical_structure(
    elements: List[Dict[str, Any]], page_num: int
) -> List[Dict[str, Any]]:
    elements.sort(key=lambda x: (x["page_position"]["y"], x["page_position"]["x"]))

    hierarchy_levels = {
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

    hierarchy = []
    parent_stack = []

    for elem_index, element in enumerate(elements):
        current_level = hierarchy_levels.get(element["structure_type"], 5)

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
    chunks = []
    quality_elements = [
        e for e in hierarchy if is_quality_content(e["content"], e["ocr_confidence"])
    ]

    current_chunk = []
    current_tokens = 0
    target_min, target_max = 150, 400

    for element in quality_elements:
        element_tokens = len(element["content"].split())

        if (
            current_tokens + element_tokens > target_max
            and current_tokens >= target_min
        ) or (
            element["type"] in ["doc_title", "title"]
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


def create_chunk(
    elements: List[Dict[str, Any]], page_num: int, item_id: str
) -> Dict[str, Any]:
    # Separate titles from content
    titles = [
        e
        for e in elements
        if e["type"] in ["doc_title", "title", "header", "paragraph_title"]
    ]
    content_elements = [
        e
        for e in elements
        if e["type"] not in ["doc_title", "title", "header", "paragraph_title"]
    ]

    # Build coherent content
    parts = []
    if titles:
        parts.append(" - ".join([t["content"] for t in titles]))
    if content_elements:
        parts.append(" ".join([e["content"] for e in content_elements]))

    content = ": ".join(parts) if len(parts) > 1 else parts[0] if parts else ""

    # Generate fragment ID
    primary = content_elements[0] if content_elements else elements[0]
    content_hash = hashlib.md5(
        f"{item_id}_page_{page_num}_bbox_{primary['bbox']}_content_{elements[0]['content'][:50]}".encode()
    ).hexdigest()[:12]
    fragment_id = f"{item_id}_chunk_{page_num}_{content_hash}"

    # Create metadata
    title_contents = [e["content"] for e in titles]
    avg_confidence = sum(e["ocr_confidence"] for e in elements) / len(elements)

    return {
        "fragment_id": fragment_id,
        "content": content,
        "metadata": {
            "item_id": item_id,
            "page": page_num,
            "type": "semantic_chunk",
            "section_titles": title_contents,
            "primary_section": title_contents[0] if title_contents else None,
            "context_hierarchy": " > ".join(title_contents) if title_contents else None,
            "bbox": primary["bbox"],
            "ocr_confidence": avg_confidence,
            "elements_count": len(elements),
            "token_count": sum(len(e["content"].split()) for e in elements),
            "hierarchy_levels": list({e["level"] for e in elements}),
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


def lambda_handler(event, context):
    """
    Lambda handler for document processing

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

            # Enhance image
            enhanced_img = enhance_image_for_ocr(img)

            # Save to temp file for processing
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                cv2.imwrite(temp_file.name, enhanced_img)
                temp_path = temp_file.name

            try:
                # Run OCR and structure analysis
                ocr_result = ocr.predict(temp_path)[0]
                structure_result = structure_pipeline.predict(temp_path)[0]

                # Combine results
                combined_elements = combine_ocr_and_structure(
                    ocr_result, structure_result
                )

                # Build hierarchy and create chunks
                hierarchy = build_hierarchical_structure(combined_elements, page_num)
                page_chunks = create_semantic_chunks(hierarchy, page_num, item_id)
                all_chunks.extend(page_chunks)

            finally:
                # Cleanup temp file
                os.unlink(temp_path)

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
    test_event = {
        "item_id": EInvoice,
        "page_range": {"start": 0, "end": 2},
    }
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))
