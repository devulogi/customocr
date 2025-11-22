"""Content Orchestrator Lambda - Downloads PDFs from S3 and orchestrates processing"""

import json
import boto3
import pymupdf as fitz
import tempfile
import os
from typing import List, Dict, Any

# AWS clients
s3_client = boto3.client('s3')
sqs_client = boto3.client('sqs')
lambda_client = boto3.client('lambda')

# Configuration
PAGES_PER_BATCH = 5
PROCESSING_QUEUE_URL = os.environ.get('PROCESSING_QUEUE_URL')
DOCUMENT_PROCESSOR_FUNCTION = os.environ.get('DOCUMENT_PROCESSOR_FUNCTION', 'lambda_document_processor')


def download_pdf_from_s3(bucket: str, key: str, item_id: str) -> str:
    """Download PDF from S3 to temporary file."""
    temp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    try:
        s3_client.download_file(bucket, key, temp_file.name)
        return temp_file.name
    except Exception as e:
        os.unlink(temp_file.name)
        raise Exception(f"Failed to download {key} from {bucket}: {str(e)}")


def get_pdf_page_count(pdf_path: str) -> int:
    """Get total page count from PDF."""
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
        return page_count
    except Exception as e:
        raise Exception(f"Failed to read PDF: {str(e)}")


def create_page_batches(total_pages: int, batch_size: int = PAGES_PER_BATCH) -> List[Dict[str, int]]:
    """Create page range batches for processing."""
    batches = []
    for start in range(0, total_pages, batch_size):
        end = min(start + batch_size - 1, total_pages - 1)
        batches.append({"start": start, "end": end})
    return batches


def send_processing_jobs_to_sqs(item_id: str, batches: List[Dict[str, int]]) -> None:
    """Send processing jobs to SQS queue."""
    if not PROCESSING_QUEUE_URL:
        raise Exception("PROCESSING_QUEUE_URL environment variable not set")
    
    for batch_num, batch in enumerate(batches):
        message = {
            "item_id": item_id,
            "page_range": batch,
            "batch_number": batch_num,
            "total_batches": len(batches)
        }
        
        sqs_client.send_message(
            QueueUrl=PROCESSING_QUEUE_URL,
            MessageBody=json.dumps(message),
            MessageAttributes={
                'item_id': {
                    'StringValue': item_id,
                    'DataType': 'String'
                },
                'batch_number': {
                    'StringValue': str(batch_num),
                    'DataType': 'Number'
                }
            }
        )


def invoke_processor_directly(item_id: str, batches: List[Dict[str, int]]) -> List[Dict]:
    """Alternative: Invoke document processor Lambda directly."""
    results = []
    
    for batch in batches:
        payload = {
            "item_id": item_id,
            "page_range": batch
        }
        
        response = lambda_client.invoke(
            FunctionName=DOCUMENT_PROCESSOR_FUNCTION,
            InvocationType='Event',  # Asynchronous
            Payload=json.dumps(payload)
        )
        
        results.append({
            "batch": batch,
            "status_code": response['StatusCode']
        })
    
    return results


def lambda_handler(event, context):
    """
    Content Orchestrator Lambda Handler
    
    Expected SQS event format:
    {
        "item_id": "document_123",
        "s3_bucket": "my-documents-bucket", 
        "s3_key": "documents/document_123.pdf",
        "processing_mode": "sqs" | "direct"  # Optional, defaults to "sqs"
    }
    """
    try:
        # Handle SQS event format
        if 'Records' in event:
            # Process each SQS message
            results = []
            for record in event['Records']:
                message_body = json.loads(record['body'])
                result = process_document(message_body)
                results.append(result)
            return {"processed_documents": len(results), "results": results}
        else:
            # Direct invocation
            return process_document(event)
            
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }


def process_document(message: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single document orchestration request."""
    item_id = message.get('item_id')
    s3_bucket = message.get('s3_bucket')
    s3_key = message.get('s3_key')
    processing_mode = message.get('processing_mode', 'sqs')
    
    if not all([item_id, s3_bucket, s3_key]):
        raise Exception("Missing required fields: item_id, s3_bucket, s3_key")
    
    pdf_path = None
    try:
        # Download PDF from S3
        pdf_path = download_pdf_from_s3(s3_bucket, s3_key, item_id)
        
        # Get page count
        total_pages = get_pdf_page_count(pdf_path)
        
        # Create page batches
        batches = create_page_batches(total_pages, PAGES_PER_BATCH)
        
        # Send processing jobs
        if processing_mode == 'direct':
            # Invoke document processor directly
            processing_results = invoke_processor_directly(item_id, batches)
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "item_id": item_id,
                    "total_pages": total_pages,
                    "total_batches": len(batches),
                    "pages_per_batch": PAGES_PER_BATCH,
                    "processing_mode": "direct",
                    "batches": batches,
                    "processing_results": processing_results
                })
            }
        else:
            # Send to SQS queue (default)
            send_processing_jobs_to_sqs(item_id, batches)
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "item_id": item_id,
                    "total_pages": total_pages,
                    "total_batches": len(batches),
                    "pages_per_batch": PAGES_PER_BATCH,
                    "processing_mode": "sqs",
                    "batches": batches,
                    "message": f"Sent {len(batches)} processing jobs to SQS"
                })
            }
            
    finally:
        # Cleanup temporary file
        if pdf_path and os.path.exists(pdf_path):
            os.unlink(pdf_path)


# For local testing
if __name__ == "__main__":
    # Test event
    test_event = {
        "item_id": "System_Design",
        "s3_bucket": "my-documents-bucket",
        "s3_key": "documents/System_Design.pdf",
        "processing_mode": "sqs"
    }
    
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))