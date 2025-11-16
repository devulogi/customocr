"""sample_ocr_test.py"""

from pathlib import Path
from paddleocr import PaddleOCR
from paddlex.inference.pipelines.ocr.result import OCRResult
import fitz  # PyMuPDF


def get_pdf_path(filename: str) -> str:
    """
    Returns the absolute path to a PDF file located in the 'pdfs' directory
    relative to this script's location.

    Args:
        filename (str): The name of the PDF file.

    Returns:
        Path: The absolute path to the PDF file.
    """
    pdf = Path.cwd() / "pdfs" / filename
    return str(pdf.resolve())


def pdf_to_images(pdf_path: str) -> list:
    """Converts each page of the PDF to an image."""
    doc: fitz.Document = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        images.append(pix)
    return images


def classify_text_element(text: str, height: int, y_pos: int, confidence: float) -> str:
    """Classify text elements into document structure types."""
    text_lower = text.lower().strip()

    # Header/Title detection (larger height, top position, short text)
    if height > 25 and len(text) < 100:
        return "TITLE"
    elif height > 20 and y_pos < 200:
        return "HEADER"

    # List item detection
    if text_lower.startswith(("•", "-", "◦", "▪")) or text_lower.startswith(
        ("1.", "2.", "3.", "4.", "5.")
    ):
        return "LIST_ITEM"

    # Chapter/Section detection
    if any(word in text_lower for word in ["chapter", "section", "figure"]):
        return "SECTION"

    # Short lines might be captions or standalone elements
    if len(text) < 50:
        return "CAPTION"

    # Default to paragraph for longer text
    return "PARAGRAPH"


def sample_ocr_test(book: str) -> None:
    """Test function to convert a sample PDF to images."""
    try:
        # Get PDF path
        pdf_path = get_pdf_path(book)
        # Convert PDF to images
        images = pdf_to_images(pdf_path)
        # Initialize OCR
        ocr: PaddleOCR = PaddleOCR(
            use_doc_orientation_classify=True, use_doc_unwarping=True
        )
        for i, img in enumerate(images):
            # Save image temporarily
            img_path = f"temp_page_{i}.png"
            img.save(img_path)
            # Perform OCR
            result = ocr.predict(img_path)

            # Analyze result structure for document elements
            print(f"\n=== PAGE {i} ANALYSIS ===")
            for res in result:
                # Extract structured data from OCR result
                if "rec_texts" in res and isinstance(res, OCRResult):
                    texts = res["rec_texts"]
                    scores = res["rec_scores"]
                    boxes = res["rec_boxes"]

                    print(f"Found {len(texts)} text elements:")

                    # Analyze each text element for structure
                    for j, (text, score, box) in enumerate(zip(texts, scores, boxes)):
                        # Calculate text properties
                        height = box[3] - box[1]  # y2 - y1
                        width = box[2] - box[0]  # x2 - x1
                        y_pos = box[1]  # top position

                        # Classify text type based on properties
                        text_type = classify_text_element(text, height, y_pos, score)

                        print(
                            f"{j:2d}. [{text_type:10s}] {text[:50]:<50} (h:{height:3d}, y:{y_pos:3d}, conf:{score:.2f})"
                        )

                res.print()
                # save ocr result to outputs folder
                output_dir = Path("outputs")
                png_dir = output_dir / "png"
                json_dir = output_dir / "json"
                # txt_dir = output_dir / "txt"
                png_dir.mkdir(parents=True, exist_ok=True)
                json_dir.mkdir(parents=True, exist_ok=True)
                img_path = png_dir / f"ocr_result_page_{i}.png"
                json_path = json_dir / f"ocr_result_page_{i}.json"
                res.save_to_img(str(img_path))
                res.save_to_json(str(json_path))
            # Clean up temporary image file
            Path(f"temp_page_{i}.png").unlink()
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error processing PDF: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return
    finally:
        print("Sample OCR test completed.")


if __name__ == "__main__":
    BOOK = "Simple_Guide.pdf"
    sample_ocr_test(BOOK)
