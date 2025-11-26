# Lambda Prototype - Split Architecture

This prototype demonstrates splitting the monolithic document processor into two event-driven Lambda functions.

## Architecture

```
Lambda 1 (OCR) → [Mock SQS] → Lambda 2 (Chunker)
```

## Files

- `lambda_ocr_processor.py` - Lambda 1: Pure OCR and structure analysis
- `lambda_chunker.py` - Lambda 2: Hierarchy building and chunking
- `test_prototype.py` - Test script to run the prototype
- `README.md` - This file

## Lambda 1: OCR Processor

**Responsibilities:**

- PDF to image conversion
- PaddleOCR text extraction
- PPStructure document analysis
- Combine OCR + structure results
- Send to SQS (mocked)

**Input:**

```json
{
  "item_id": "document_name",
  "page_range": {"start": 0, "end": 5}
}
```

**Output to SQS:**

```json
{
  "item_id": "document_name",
  "page_num": 0,
  "combined_elements": [...]
}
```

## Lambda 2: Chunker

**Responsibilities:**

- Build hierarchical structure
- Create semantic chunks
- Store chunks (mocked)

**Input from SQS:**

```json
{
  "Records": [{
    "body": "{\"item_id\": \"doc\", \"page_num\": 0, \"combined_elements\": [...]}"
  }]
}
```

## Mock Integration

Since this is a prototype without real SQS:

- `mock_send_to_sqs()` directly calls Lambda 2
- `store_chunks()` prints to console
- Integration points are clearly marked for real implementation

## Usage

```bash
cd prototype_lambda
python test_prototype.py
```

## Real Implementation Changes

1. Replace `mock_send_to_sqs()` with:

```python
import boto3
sqs = boto3.client('sqs')
sqs.send_message(QueueUrl=QUEUE_URL, MessageBody=json.dumps(message_data))
```

2. Replace `store_chunks()` with actual storage (Elasticsearch, DynamoDB, etc.)

3. Deploy as separate Lambda functions with SQS trigger on Lambda 2

### One-Off Cold Start Analysis

2800 Pages Performance:
First batch: 25s (20s cold start + 5s processing)

Remaining 279 batches: 5s each (warm container)

Total time: 25s + (279 × 5s) = ~24 minutes

Cost Breakdown:
Cold start penalty: 20s (one-time)

Actual processing: 2800 × 0.5s = 23 minutes

Total: ~24 minutes for 2800 pages

Key Insight:
The 20-second cold start becomes negligible when processing large batches. It's only 1.4% of total time (20s out of 1440s).

Performance Per Page:
With cold start: 25.4s ÷ 10 pages = 2.54s/page

Warm container: 5s ÷ 10 pages = 0.5s/page

Bottleneck Shifts:
Small jobs: Cold start dominates (80% overhead)

Large jobs: Processing dominates (cold start <2% overhead)

Your architecture is actually well-optimized for large document processing! The cold start penalty amortizes across the entire job.

### 100 Pages Performance

Batch Breakdown:
Total batches: 100 ÷ 10 = 10 Lambda invocations

First batch: 25s (20s cold start + 5s processing)

Remaining 9 batches: 5s each (warm container)

Total time: 25s + (9 × 5s) = 70 seconds

Cold Start Impact:
Cold start overhead: 20s out of 70s = 29% penalty

Actual processing: 100 × 0.5s = 50s

Efficiency: 50s processing + 20s overhead = 71% efficient

Performance Comparison:
Document Size Cold Start Impact Total Time Efficiency
10 pages 80% penalty 25s 20% efficient
100 pages 29% penalty 70s 71% efficient
2800 pages 1.4% penalty 24min 98% efficient
100 Pages Sweet Spot:
Reasonable overhead (29% vs 80% for small docs)

Fast completion (70s vs 24min for large docs)

Good cost efficiency (~$2-3 total cost)

100 pages hits the sweet spot where cold start penalty is manageable but processing time remains fast.
