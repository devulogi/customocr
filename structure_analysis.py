"""structure_analysis.py - Document structure analysis using PaddleStructureV3"""

from pathlib import Path
import fitz  # PyMuPDF
import cv2
import numpy as np
from paddleocr import PPStructureV3

from paddlex.inference.pipelines.layout_parsing.layout_objects import LayoutBlock
from paddlex.inference.pipelines.layout_parsing.result_v2 import LayoutParsingResultV2


def get_pdf_path(filename: str) -> str:
    """Returns the absolute path to a PDF file in the 'pdfs' directory."""
    pdf = Path.cwd() / "pdfs" / filename
    return str(pdf.resolve())


def pdf_to_images(pdf_path: str) -> list:
    """Converts each page of the PDF to an image."""
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        # Convert to numpy array for OpenCV
        img_data = pix.tobytes("png")
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        images.append(img)
    return images


def draw_structure_boxes(image, result):
    """Draw bounding boxes with structure labels on the image."""
    img_copy = image.copy()

    # Color mapping for different structure types
    colors = {
        "title": (255, 0, 0),  # Red
        "text": (0, 255, 0),  # Green
        "header": (0, 0, 255),  # Blue
        "figure": (255, 255, 0),  # Cyan
        "table": (255, 0, 255),  # Magenta
        "footer": (128, 128, 128),  # Gray
        "reference": (0, 128, 255),  # Orange
    }

    # Handle single result object - check if it's a dict with boxes
    if isinstance(result["parsing_res_list"], list):
        for item in result["parsing_res_list"]:
            # Get bounding box coordinates
            bbox = getattr(item, "bbox", [])
            x1, y1, x2, y2 = map(int, bbox)

            # Get structure type
            struct_type = getattr(item, "label", "unknown")
            color = colors.get(struct_type.lower(), (128, 128, 128))

            # Draw bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

            # Add label
            label_text = f"{struct_type} ({getattr(item, 'score', 0):.2f})"
            cv2.putText(
                img_copy,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    return img_copy


def structure_analysis_test(book: str) -> None:
    """Test function for document structure analysis."""
    try:
        # Get PDF path
        pdf_path = get_pdf_path(book)

        # Convert PDF to images
        images = pdf_to_images(pdf_path)

        # Initialize layout parsing model
        pipeline = PPStructureV3()

        # Create output directories
        output_dir = Path("outputs")
        structure_dir = output_dir / "structure"
        structure_dir.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(images):
            print(f"\n=== ANALYZING PAGE {i} ===")

            # Save original image temporarily
            temp_path = f"temp_page_{i}.png"
            cv2.imwrite(temp_path, img)

            # Perform structure analysis
            result = pipeline.predict(temp_path)

            # Print structure analysis results
            for res in result:
                print("Structure analysis results:")
                print(f"Result type: {type(res)}")
                print(
                    f"Available attributes: {[attr for attr in res if not attr.startswith('_')]}"
                )

                # Check for parsing_res_list
                if "parsing_res_list" in res and isinstance(res, LayoutParsingResultV2):
                    if res["parsing_res_list"]:
                        print(
                            f"Found parsing_res_list with {len(res["parsing_res_list"])} items:"
                        )
                        for j, item in enumerate(res["parsing_res_list"]):
                            print(f"  Item {j}: {type(item)}")
                            if isinstance(item, LayoutBlock):
                                print(f"    Keys: {list(item.__dict__.keys())}")
                                if hasattr(item, "label"):
                                    print(f"    Label: {item.label}")
                                if hasattr(item, "content"):
                                    content = str(item.content)[:100]
                                    print(f"    Content: {content}...")
                                if hasattr(item, "bbox"):
                                    print(f"    BBox: {item.bbox}")
                    else:
                        print("parsing_res_list exists but is empty")
                else:
                    print("No parsing_res_list attribute found")

                # Draw bounding boxes on image
                annotated_img = draw_structure_boxes(img, res)

                # Save annotated image
                output_path = structure_dir / f"structure_page_{i}.png"
                cv2.imwrite(str(output_path), annotated_img)
                print(f"Saved annotated image: {output_path}")

                output_path1 = structure_dir / f"structure_page_{i}_boxes.png"
                res.save_to_img(str(output_path1))
                print(f"Saved image with boxes: {output_path1}")
                # Save results to JSON
                json_path = structure_dir / f"structure_page_{i}.json"
                res.save_to_json(str(json_path))
                print(f"Saved results: {json_path}")

            # Clean up temporary file
            Path(temp_path).unlink()

    except Exception as e:
        print(f"Error during structure analysis: {e}")
        return
    finally:
        print("Structure analysis completed.")


if __name__ == "__main__":
    BOOK = "MLSystemDesign.pdf"
    structure_analysis_test(BOOK)
