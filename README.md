# TTB Label Verifier

A web application for verifying alcohol label compliance with TTB regulations using OCR technology. Upload a label image, enter product information, and get instant verification results.

## Features

- **Clean, minimalist UI** - Simple two-panel layout for easy verification
- **Advanced OCR** - Tesseract with multiple preprocessing strategies for accurate text extraction
- **Fuzzy Matching** - Levenshtein distance matching handles OCR errors gracefully
- **Government Warning Validation** - N-Gram token overlap (80%) validates mandatory warning text
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

### Key Files

- `server/api.py` - Backend with OCR and verification logic
- `client/src/pages/VerificationPage.tsx` - Main frontend component
- `client/src/lib/api.ts` - API client

### Code Structure

- **OCR Processing:** `perform_ocr_with_retry()` in `server/api.py`
- **Fuzzy Matching:** `fuzzy_match()` using TheFuzz library
- **Text Normalization:** `normalize_text()` with regex
- **Government Warning:** `check_government_warning()` with N-Gram token overlap

## Project Structure

```
├── client/              # React frontend
│   ├── src/
│   │   ├── components/ # UI components
│   │   ├── lib/        # Utilities and API client
│   │   └── pages/      # Page components
├── server/
│   └── api.py         # FastAPI backend
├── test-images/       # Sample test images
├── Dockerfile         # Docker configuration
├── requirements.txt   # Python dependencies
└── package.json       # Node.js dependencies
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

- **Header Matching:** Levenshtein distance (85%) for "GOVERNMENT WARNING" and "SURGEON GENERAL"
- **Content Validation:** N-Gram token overlap (80%) for warning text body
- Allows ~20% error rate in OCR text while maintaining accuracy

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

- **Processing Time:** ~1 minute per image
- **Memory Usage:** ~100-200MB (Tesseract only, no deep learning models)
- **Accuracy:** Good for clear labels, handles common OCR errors with fuzzy matching

## Testing & Known Issues

### Test Images

The `test-images/` folder contains sample images for testing:

- **Images ending with `_updated`:** OCR-friendly versions with corrected text
- **Other images:** May contain incorrect information by design for testing edge cases

### Error Types Found During Testing

Both **Type I errors** (false positives) and **Type II errors** (false negatives) have been observed:

- **Type I (False Positive):** System reports match when text doesn't actually match
  - Can occur with fuzzy matching when similarity threshold is too low
  - May match partial text incorrectly

- **Type II (False Negative):** System reports no match when text actually matches
  - Can occur with poor OCR quality
  - May miss matches due to normalization differences

**Mitigation:** The system uses 85% threshold for fuzzy matching and 80% token overlap for warnings to balance accuracy and error tolerance. Manual review of OCR text is recommended for critical verifications.

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
pip install -r requirements.txt
```

3. **Run the application**

**Option A: Full stack (recommended)**
```bash
# Terminal 1: Start backend
uvicorn server.api:app --reload --port 8000

# Terminal 2: Start frontend dev server
npm run dev
```

Frontend will be available at `http://localhost:5173` and will proxy API requests to `http://localhost:8000`.

**Option B: Production build**
```bash
# Build frontend
npm run build

# Start backend (serves built frontend)
uvicorn server.api:app --reload --port 8000
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
