# OCR Dev Container Setup

## Quick Start

1. **Open in Container**: `Ctrl+Shift+P` → "Dev Containers: Reopen in Container"
2. **Wait for Build**: Container will install all OCR dependencies
3. **Verify Setup**: Run `python enhanced_document_processor.py`

## What's Included

### System Dependencies
- OpenCV with full image processing support
- PaddleOCR with all language models
- PyMuPDF for PDF processing
- System libraries for computer vision

### VS Code Extensions
- Python language support
- Black formatter
- Pylint linting
- Jupyter notebooks
- Spell checker

### Environment
- Python 3.9 optimized for ML/CV
- Pre-configured paths and variables
- Mounted volumes for pdfs/ and outputs/
- Port 8000 exposed for web interfaces

## Usage

```bash
# Process documents
python enhanced_document_processor.py

# Install additional packages
pip install package-name

# Run tests
python -m pytest
```

## Troubleshooting

- **Build fails**: Check Docker is running
- **Import errors**: Restart container to refresh environment
- **Permission issues**: Container runs as root for ML library compatibility