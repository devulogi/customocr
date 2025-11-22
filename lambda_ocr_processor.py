"""Lambda OCR Processor - Heavy ML Processing for Document OCR"""

import json
import boto3
import pymupdf as fitz
import cv2
import numpy as np
from paddleocr import PaddleOCR, PPStructureV3
from typing import List, Dict, Any
import tempfile
from dataclasses import dataclass
from typing import Tuple
import os

# Global model instances for Lambda container reuse
_models = None

def get_models():
    global _models
    if _models is None:
        _models = {"ocr": PaddleOCR(), "structure": PPStructureV3()}
    return _models

def should_enhance_image(img: np.ndarray) -> bool:
    """Determine if image needs enhancement based on quality metrics."""
    height, width = img.shape[:2]
    if height >= 1800 and width >= 1800:
        return False
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < 100

def enhance_image_for_ocr(img: np.ndarray) -> np.ndarray:
    """Conditionally enhance image for OCR with optimized operations."""
    if not should_enhance_image(img):
        height, width = img.shape[:2]
        min_size, max_size = 1200, 2400

        if height < min_size or width < min_size:
            scale = max(min_size / height, min_size / width)
            new_width, new_height = int(width * scale), int(height * scale)
            return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        elif height > max_size or width > max_size:
            scale = min(max_size / height, max_size / width)
            new_width, new_height = int(width * scale), int(height * scale)
            return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return img

    enhanced = img.copy()
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.3, beta=10)
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 15)

    height, width = enhanced.shape[:2]
    min_size, max_size = 1200, 2400

    if height < min_size or width < min_size:
        scale = max(min_size / height, min_size / width)
        new_width, new_height = int(width * scale), int(height * scale)
        enhanced = cv2.resize(enhanced, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    elif height > max_size or width > max_size:
        scale = min(max_size / height, max_size / width)
        new_width, new_height = int(width * scale), int(height * scale)
        enhanced = cv2.resize(enhanced, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return enhanced

def bbox_overlap(box1, box2, threshold=0.5) -> bool:
    """Optimized bounding box overlap with early exit."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    if x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1:
        return False

    x1_i, y1_i = max(x1_1, x1_2), max(y1_1, y1_2)
    x2_i, y2_i = min(x2_1, x2_2), min(y2_1, y2_2)

    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    return intersection > threshold * min(area1, area2)

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
    return sorted(indexed_elements, key=lambda x: x.y_center)

def find_matching_structure(box: List[float], indexed_structures: List[StructureElement]) -> str:
    """Fast structure matching using spatial indexing."""
    box_y_center = (box[1] + box[3]) / 2

    for struct_elem in indexed_structures:
        if abs(struct_elem.y_center - box_y_center) > 100:
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
                structure_elements.append({
                    "bbox": item.bbox,
                    "label": item.label,
                    "confidence": getattr(item, "score", 0.0),
                })

    indexed_structures = create_spatial_index(structure_elements)

    # Match OCR text with structure elements
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

def process_image_in_memory(img: np.ndarray, ocr, structure_pipeline):
    """Process image in memory without temp files."""
    try:
        ocr_result = ocr.predict(img)[0]
        structure_result = structure_pipeline.predict(img)[0]
        return ocr_result, structure_result
    except:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_file:
            cv2.imwrite(temp_file.name, img)
            ocr_result = ocr.predict(temp_file.name)[0]
            structure_result = structure_pipeline.predict(temp_file.name)[0]
            return ocr_result, structure_result

def lambda_handler(event, context):
    """
    OCR Lambda Handler - Heavy ML Processing
    
    Input SQS Message:
    {
        "Records": [{
            "body": "{\"item_id\": \"doc123\", \"page_range\": {\"start\": 0, \"end\": 5}}"
        }]
    }
    """
    try:
        # Parse SQS message
        for record in event['Records']:
            message = json.loads(record['body'])
            item_id = message.get("item_id")
            page_range = message.get("page_range", {})

            if not item_id:
                continue

            start_page = page_range.get("start", 0)
            end_page = page_range.get("end")

            # Get PDF from S3 or local
            from pathlib import Path
            pdf_path = str(Path.cwd() / "pdfs" / f"{item_id}.pdf")

            if not os.path.exists(pdf_path):
                print(f"Document {item_id} not found")
                continue

            # Get models (singleton for container reuse)
            models = get_models()
            ocr = models["ocr"]
            structure_pipeline = models["structure"]

            # Process PDF
            doc = fitz.open(pdf_path)
            total_pages = len(doc)

            if end_page is None:
                end_page = total_pages - 1

            start_page = max(0, min(start_page, total_pages - 1))
            end_page = max(start_page, min(end_page, total_pages - 1))

            # Process pages and collect OCR results
            ocr_results = []
            for page_num in range(start_page, end_page + 1):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                enhanced_img = enhance_image_for_ocr(img)
                ocr_result, structure_result = process_image_in_memory(enhanced_img, ocr, structure_pipeline)
                combined_elements = combine_ocr_and_structure(ocr_result, structure_result)
                
                ocr_results.append({
                    "page_num": page_num,
                    "elements": combined_elements
                })

            doc.close()

            # Save OCR results to S3
            s3 = boto3.client('s3')
            bucket = "ocr-results-bucket"
            key = f"{item_id}/pages_{start_page}-{end_page}.json"
            
            ocr_data = {
                "item_id": item_id,
                "page_range": {"start": start_page, "end": end_page},
                "total_pages": end_page - start_page + 1,
                "ocr_results": ocr_results
            }

            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(ocr_data),
                ContentType='application/json'
            )

            # Send message to chunking queue
            sqs = boto3.client('sqs')
            chunk_queue_url = "https://sqs.region.amazonaws.com/account/chunk-processing-queue"
            
            chunk_message = {
                "bucket": bucket,
                "key": key,
                "item_id": item_id,
                "page_range": {"start": start_page, "end": end_page}
            }

            sqs.send_message(
                QueueUrl=chunk_queue_url,
                MessageBody=json.dumps(chunk_message)
            )

            print(f"OCR processing completed for {item_id}, pages {start_page}-{end_page}")

        return {"statusCode": 200}

    except Exception as e:
        print(f"OCR processing error: {str(e)}")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

# For local testing
if __name__ == "__main__":
    test_event = {
        "Records": [{
            "body": json.dumps({
                "item_id": "E-Invoice",
                "page_range": {"start": 0, "end": 2}
            })
        }]
    }
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))