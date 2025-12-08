# TTB Label Verifier

A web application for verifying alcohol label compliance with TTB regulations using OCR technology. Upload a label image, enter product information, and get instant verification results.

## Features

- **Clean, minimalist UI** - Simple two-panel layout for easy verification
- **Advanced OCR** - Tesseract with multiple preprocessing strategies for accurate text extraction
- **Fuzzy Matching** - Levenshtein distance matching handles OCR errors gracefully
- **Government Warning Validation** - N-Gram token overlap (65%) validates mandatory warning text
- **Real-time Processing** - Progress bar with percentage during OCR processing
- **Manual Audit** - View both OCR text and form entries side-by-side for verification
- **Automatic Form Clearing** - Form resets after successful verification for batch processing

## Tech Stack

**Frontend:** React 19, TypeScript, Vite, Tailwind CSS, React Hook Form, Zod  
**Backend:** FastAPI, Tesseract OCR, OpenCV, Pillow, TheFuzz (Levenshtein distance)

## Development

### Tools Used

- **Cursor** - Primary IDE for programming
- **VS Code** - Secondary editor for development
- **ChatGPT** - Research assistance and image generation for test data
- **qwen3-coder:30b** - Local model on Ollama for support and verification 

### Key Files

- `server/api.py` - Backend with OCR and verification logic
- `client/src/pages/VerificationPage.tsx` - Main frontend component
- `client/src/lib/api.ts` - API client
- `client/src/constants.ts` - Application-wide constants (progress values, UI delays)

### Code Structure

- **OCR Processing:** `perform_ocr_with_retry()` in `server/api.py`
- **Fuzzy Matching:** `fuzzy_match()` using TheFuzz library (85% threshold)
- **Text Normalization:** `normalize_text()` with regex
- **Government Warning:** `check_government_warning()` with N-Gram token overlap (65% threshold)
- **Configuration:** Constants defined at module level for easy tuning
- **Logging:** Structured logging with Python `logging` module

## Project Structure

```
├── client/              # React frontend
│   ├── src/
│   │   ├── components/ # UI components
│   │   ├── lib/        # Utilities and API client
│   │   ├── pages/      # Page components
│   │   └── constants.ts # Application constants
├── server/
│   └── api.py         # FastAPI backend (with constants and logging)
├── test-images/       # Sample test images
├── Dockerfile         # Docker configuration
├── requirements.txt   # Python dependencies
├── package.json       # Node.js dependencies
├── CODE_QUALITY_REVIEW.md      # Comprehensive code quality analysis
└── CODE_QUALITY_IMPROVEMENTS.md # Summary of improvements
```
## How It Works

### 1. Image Processing & OCR

The system uses Tesseract OCR with optimized preprocessing:

- **Processing Order:**
  1. Variant 2 (OpenCV Enhanced Otsu) - Advanced denoising, deskewing, CLAHE
  2. Variant 3 (Standard PIL) - Contrast/sharpness enhancement, Otsu binarization
  3. Variant 1 (OpenCV Enhanced Adaptive) - Only if high-quality result not found

- **PSM Modes:** Tests 3 modes (0, 1, 3) per variant
- **Early Exit:** Stops processing if high-quality result found early
- **Result Selection:** Automatically picks best OCR result based on quality scoring

### 2. Text Matching

Uses **Levenshtein distance** (TheFuzz library) with 85% similarity threshold:

- Normalizes text (lowercase, remove special chars, normalize spaces)
- Tries exact substring match first (fastest)
- Falls back to fuzzy matching with sliding windows
- Handles OCR errors like character substitutions, missing characters

### 3. Government Warning Validation

Validates mandatory warning label using:

- **Header Matching:** Levenshtein distance (85% threshold) for "GOVERNMENT WARNING" and "SURGEON GENERAL"
- **Content Validation:** N-Gram token overlap (65% threshold) for warning text body
- Allows ~35% error rate in OCR text while maintaining accuracy
- Thresholds are configurable via constants in `server/api.py`

### 4. Field Extraction

Smart extraction of:
- **Brand Name:** First 2-5 words from OCR
- **Product Class:** 2-4 words following brand (includes keywords like whiskey, bourbon, wine, etc.)
- **Alcohol Content:** Number followed by % sign
- **Net Contents:** Volume measurements (mL, L, Fl Oz, Pint, etc.)

## OCR Technology Decisions

### Why Tesseract Only?

This application was designed to be lightweight and deployable on free-tier infrastructure (Render). As such:

- **EasyOCR/PaddleOCR:** Not used due to memory constraints (require 2-4GB+ RAM)
- **Cloud Vision APIs (Google/AWS):** Not used to avoid:
  - Increased latency (network round-trip)
  - API costs
  - Dependency on external services
  - Beyond MVP scope

**Note:** These alternatives could provide significantly better OCR results, especially for:
- Complex backgrounds
- Reflections on glass bottles
- Low-contrast text (gold on black)
- Curved surfaces

Tesseract with advanced preprocessing provides good results while keeping the application lightweight and fast.

## Performance

- **Processing Time:** ~1 minute per image on Render (Free tier) and 1-2 seconds on local (M3 Ultra, 96 GB unifed memory)
- **Memory Usage:** ~100-200MB (Tesseract only, no deep learning models)
- **Accuracy:** Good for clear labels, handles common OCR errors with fuzzy matching
- **Code Quality:** Production-ready with structured logging, proper error handling, and maintainable constants

## Code Quality

The codebase follows best practices:

- **Constants:** All magic numbers extracted to named constants for easy configuration
  - Fuzzy matching threshold: 85%
  - Token overlap threshold: 65%
  - OCR quality thresholds configurable
- **Logging:** Structured logging with Python `logging` module (info, warning, error, debug levels)
- **Error Handling:** Specific exception handling with proper logging
- **Type Safety:** TypeScript for frontend, type hints for Python backend
- **Documentation:** Comprehensive code review and improvement documentation

See `CODE_QUALITY_REVIEW.md` and `CODE_QUALITY_IMPROVEMENTS.md` for detailed analysis.

## Testing & Known Issues

### Test Images

The `test-images/` folder contains sample images for testing:

- **Images ending with `_updated`:** OCR-friendly versions with corrected text
- **Other images:** May contain incorrect information by design for testing edge cases

### Installation Issues

**python-Levenshtein build failure (Python 3.12+):**
If you get build errors when installing `python-Levenshtein`, it's optional. The `thefuzz` library works fine without it, just a bit slower. You can install dependencies without it:
```bash
pip install fastapi==0.115.0 "uvicorn[standard]==0.32.0" python-multipart==0.0.12 pytesseract==0.3.10 Pillow==10.4.0 numpy==1.26.4 opencv-python-headless==4.10.0.84 thefuzz==0.19.0
```

### Error Types Found During Testing

Both **Type I errors** (false positives) and **Type II errors** (false negatives) have been observed:

- **Type I (False Positive):** System reports match when text doesn't actually match
  - Can occur with fuzzy matching when similarity threshold is too low
  - May match partial text incorrectly

- **Type II (False Negative):** System reports no match when text actually matches
  - Can occur with poor OCR quality
  - May miss matches due to normalization differences

**Mitigation:** The system uses 85% threshold for fuzzy matching and 65% token overlap for warnings to balance accuracy and error tolerance. These thresholds are configurable via constants in `server/api.py`. Manual review of OCR text is recommended for critical verifications.

## Future Enhancements Ideas

1. **Multiple Image Support**
   - Process front and back labels in single request
   - Combine OCR results from multiple images
   - Show which label each match came from

2. **Enhanced Loading Bar**
   - Task-based progress (OCR preprocessing, OCR execution, text matching, etc.)
   - More accurate time estimates based on image size and complexity

3. **Processing History**
   - View previous verifications
   - Track compliance over time
   - Search and filter history

4. **PDF Report Generation**
   - Download verification reports as PDF
   - Include OCR text, matches, compliance status
   - Customizable report templates

5. **Advanced OCR Options**
   - EasyOCR/PaddleOCR integration (for higher memory infrastructure)
   - Cloud Vision API integration (Google/AWS) for better accuracy
   - Configurable OCR engine selection

6. **Batch Processing**
   - Upload multiple images at once
   - Process queue with progress tracking
   - Export results to CSV/JSON

7. **Confidence Scoring**
   - Show confidence scores for each match
   - Highlight low-confidence matches for manual review
   - Allow manual override of matches


## Quick Start

### Prerequisites

- Node.js 20+
- Python 3.11+
- Tesseract OCR

#### Install Tesseract

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

### Local Development

1. **Clone the repository**
```bash
git clone <repository-url>
cd AI-Alcohol-Label-Verification
```

2. **Install dependencies**
```bash
# Install Node.js dependencies
npm install

# Install Python dependencies
# Option 1: Using pip (if using venv or conda)
pip install -r requirements.txt

# Option 2: If python-Levenshtein fails to build (Python 3.12+), install without it
# Thefuzz will work without it, just slower
pip install fastapi==0.115.0 "uvicorn[standard]==0.32.0" python-multipart==0.0.12 pytesseract==0.3.10 Pillow==10.4.0 numpy==1.26.4 opencv-python-headless==4.10.0.84 thefuzz==0.19.0
```

3. **Run the application**

**Option A: Full stack (recommended)**
```bash
# Terminal 1: Start backend
python -m uvicorn server.api:app --reload --port 8000
# or just: uvicorn server.api:app --reload --port 8000

# Terminal 2: Start frontend dev server
npm run dev
```

Frontend will be available at `http://localhost:5173` and will proxy API requests to `http://localhost:8000`.

**Option B: Production build**
```bash
# Build frontend
npm run build

# Start backend (serves built frontend)
python -m uvicorn server.api:app --reload --port 8000
# or just: uvicorn server.api:app --reload --port 8000
```

Application will be available at `http://localhost:8000`.


## Deployment

### Docker

```bash
# Build image
docker build -t alcohol-label-verifier .

# Run container
docker run -p 8000:8000 -e PORT=8000 alcohol-label-verifier
```

### Render

1. Connect GitHub repository to Render
2. Create new Web Service
3. Set environment to Docker
4. Use provided `render.yaml` for configuration

The application is optimized for Render's free tier with memory-efficient OCR processing.

## API Endpoints

### POST `/api/verify`

Verify label against form data.

**Request (multipart/form-data):**
- `brand_name` (string, required)
- `product_type` (string, required)
- `alcohol_content` (string, required)
- `net_contents` (string, optional)
- `image` (file, required)

**Response:**
```json
{
  "success": boolean,
  "ocr_text_snippet": string,
  "results": {
    "brand_name": {"match": boolean, "value": string},
    "product_type": {"match": boolean, "value": string},
    "alcohol_content": {"match": boolean, "value": string},
    "net_contents": {"match": boolean, "value": string},
    "compliance": {
      "compliant": boolean,
      "details": {
        "has_warning_label": boolean,
        "has_surgeon_general": boolean,
        "has_pregnancy_warning": boolean,
        "has_driving_warning": boolean
      }
    }
  },
  "highlights": Array<{x, y, w, h}>,
  "image_dimensions": {width: number, height: number}
}
```

### GET `/api/health`

Health check endpoint.

## License

MIT
