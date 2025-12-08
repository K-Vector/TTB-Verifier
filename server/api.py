from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
import re
from typing import Optional, List, Dict, Any
import os
import cv2
import numpy as np
from thefuzz import fuzz
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration constants
FUZZY_MATCH_THRESHOLD = 85
TOKEN_OVERLAP_THRESHOLD = 0.65
FUZZY_MATCH_SHORT_STRING_LENGTH = 20
OCR_MIN_TEXT_LENGTH = 150
OCR_MIN_WORD_COUNT = 15
OCR_QUALITY_SCORE_THRESHOLD = 300
EXTRACTION_CONTEXT_WINDOW = 20
CANNY_EDGE_LOW = 50
CANNY_EDGE_HIGH = 150
HOUGH_LINES_LIMIT = 20
DESKEW_ANGLE_THRESHOLD = 0.5

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files in production
static_dir = "dist"
assets_dir = os.path.join(static_dir, "assets")

# Serve static assets (JS, CSS, images, etc.) from dist/assets
if os.path.exists(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

# Health check endpoint (not at root, so it doesn't interfere with SPA)
@app.get("/api/health")
async def health():
    return {"status": "ok", "message": "TTB Verifier API is running"}

def normalize_text(text: str) -> str:
    """Clean up text for comparison - lowercase, strip special chars, normalize spaces"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text)  # collapse whitespace
    return text.strip()

def fuzzy_match(source: str, target: str, threshold: int = FUZZY_MATCH_THRESHOLD) -> bool:
    """Fuzzy match using Levenshtein distance - handles OCR errors"""
    if not target:
        return True
    
    if not source:
        return False
    
    source_norm = normalize_text(source)
    target_norm = normalize_text(target)
    
    # quick exact match first
    if target_norm in source_norm:
        return True
    
    # try sliding window approach for fuzzy matching
    target_words = target_norm.split()
    if not target_words:
        return False
    
    source_words = source_norm.split()
    window_size = len(target_words)
    
    if len(source_words) >= window_size:
        for i in range(len(source_words) - window_size + 1):
            window = " ".join(source_words[i:i + window_size])
            score = fuzz.token_sort_ratio(target_norm, window)
            if score >= threshold:
                return True
    
    # fallback for short strings
    if len(target_norm) <= FUZZY_MATCH_SHORT_STRING_LENGTH:
        score = fuzz.partial_ratio(target_norm, source_norm)
        if score >= threshold:
            return True
    
    return False

def simple_match(source: str, target: str) -> bool:
    """Basic substring check - used as fallback"""
    source_norm = normalize_text(source)
    target_norm = normalize_text(target)
    
    if not target_norm:
        return True
    if not source_norm:
        return False
    
    return target_norm in source_norm

def token_overlap_ratio(text1: str, text2: str) -> float:
    """Calculate token overlap - used for warning text validation"""
    tokens1 = set(normalize_text(text1).split())
    tokens2 = set(normalize_text(text2).split())
    
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

def check_government_warning(text: str) -> Dict[str, Any]:
    """Validate government warning label compliance"""
    norm_text = normalize_text(text)
    original_text = text.lower()
    
    # check for warning header
    warning_phrases = ["government warning", "government warn ing", "govern ment warning"]
    has_warning_label = False
    for phrase in warning_phrases:
        if fuzzy_match(original_text, phrase, threshold=FUZZY_MATCH_THRESHOLD):
            has_warning_label = True
            break
    
    # regex fallback for common OCR errors
    if not has_warning_label:
        warning_pattern = r"government\s*warn[il1]ng"
        has_warning_label = bool(re.search(warning_pattern, norm_text))
    
    # surgeon general check
    surgeon_phrases = ["surgeon general", "surgeon gen eral"]
    has_surgeon_general = False
    for phrase in surgeon_phrases:
        if fuzzy_match(original_text, phrase, threshold=FUZZY_MATCH_THRESHOLD):
            has_surgeon_general = True
            break
    
    if not has_surgeon_general:
        surgeon_general_pattern = r"surgeon\s*general"
        has_surgeon_general = bool(re.search(surgeon_general_pattern, norm_text))
    
    # pregnancy warning - using token overlap to handle OCR errors
    pregnancy_text = "according to the surgeon general women should not drink alcoholic beverages during pregnancy because of the risk of birth defects"
    pregnancy_overlap = token_overlap_ratio(original_text, pregnancy_text)
    has_pregnancy_warning = pregnancy_overlap >= TOKEN_OVERLAP_THRESHOLD
    
    # driving/machinery warning
    driving_text = "consumption of alcoholic beverages impairs your ability to drive a car or operate machinery and may cause health problems"
    driving_overlap = token_overlap_ratio(original_text, driving_text)
    has_driving_warning = driving_overlap >= TOKEN_OVERLAP_THRESHOLD
    
    is_compliant = has_warning_label and has_surgeon_general and has_pregnancy_warning and has_driving_warning
    
    return {
        "compliant": is_compliant,
        "details": {
            "has_warning_label": has_warning_label,
            "has_surgeon_general": has_surgeon_general,
            "has_pregnancy_warning": has_pregnancy_warning,
            "has_driving_warning": has_driving_warning
        }
    }

def extract_brand_name(ocr_text: str) -> Optional[str]:
    """Grab first few words as brand name"""
    words = ocr_text.split()
    if len(words) >= 2:
        brand_words = words[:min(5, len(words))]
        return " ".join(brand_words)
    return None

def extract_product_class(ocr_text: str, brand_name: Optional[str] = None) -> Optional[str]:
    """Find product type keywords and extract surrounding text"""
    product_keywords = [
        "whiskey", "bourbon", "rye", "rye malt", "corn", "blended", "light", "spirit",
        "scotch", "irish", "canadian", "kentucky", "tennessee", "rum",
        "indian pale ale", "red", "white", "rosé", "port", "sherry", "madeira",
        "champagne", "wine", "cider", "vermouth", "lager", "ale", "stout",
        "porter", "wheat beer", "malt liquor", "hard seltzers", "brandy", "cognac"
    ]
    
    norm_text = normalize_text(ocr_text)
    
    # skip past brand name if we know it
    if brand_name:
        brand_norm = normalize_text(brand_name)
        brand_pos = norm_text.find(brand_norm)
        if brand_pos != -1:
            search_text = norm_text[brand_pos + len(brand_norm):]
        else:
            search_text = norm_text
    else:
        search_text = norm_text
    
    # look for product keywords
    for keyword in product_keywords:
        keyword_norm = normalize_text(keyword)
        pos = search_text.find(keyword_norm)
        if pos != -1:
            words = search_text[max(0, pos-EXTRACTION_CONTEXT_WINDOW):pos+len(keyword_norm)+EXTRACTION_CONTEXT_WINDOW].split()
            keyword_idx = -1
            for i, word in enumerate(words):
                if keyword_norm in normalize_text(word):
                    keyword_idx = i
                    break
            
            if keyword_idx != -1:
                start = max(0, keyword_idx - 1)
                end = min(len(words), keyword_idx + 3)
                return " ".join(words[start:end])
    
    return None

def extract_alcohol_content(ocr_text: str) -> Optional[str]:
    """Find alcohol percentage in text"""
    patterns = [
        r'(\d+\.?\d*)\s*%',
        r'(\d+\.?\d*)\s*percent',
        r'(\d+\.?\d*)\s*alc\s*/\s*vol',
        r'(\d+\.?\d*)\s*abv',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, ocr_text, re.IGNORECASE)
        if match:
            start = max(0, match.start() - EXTRACTION_CONTEXT_WINDOW)
            end = min(len(ocr_text), match.end() + EXTRACTION_CONTEXT_WINDOW)
            context = ocr_text[start:end]
            words = context.split()
            for i, word in enumerate(words):
                if match.group(1) in word:
                    start_idx = max(0, i - 1)
                    end_idx = min(len(words), i + 3)
                    return " ".join(words[start_idx:end_idx])
    
    return None

def extract_net_contents(ocr_text: str) -> Optional[str]:
    """Find volume measurements like 750ml, 1L, etc"""
    volume_patterns = [
        r'(\d+\.?\d*)\s*(ml|milliliter|millilitre)',
        r'(\d+\.?\d*)\s*(l|liter|litre)',
        r'(\d+\.?\d*)\s*(fl\s*oz|fluid\s*ounce)',
        r'(\d+\.?\d*)\s*(pint|pt)',
    ]
    
    for pattern in volume_patterns:
        match = re.search(pattern, ocr_text, re.IGNORECASE)
        if match:
            start = max(0, match.start() - EXTRACTION_CONTEXT_WINDOW)
            end = min(len(ocr_text), match.end() + EXTRACTION_CONTEXT_WINDOW)
            context = ocr_text[start:end]
            words = context.split()
            for i, word in enumerate(words):
                if match.group(1) in word:
                    start_idx = max(0, i - 1)
                    end_idx = min(len(words), i + 3)
                    return " ".join(words[start_idx:end_idx])
    
    return None

def validate_alcohol_content(ocr_text: str, form_value: str, product_type: str) -> bool:
    """
    Validate alcohol content with product-type specific logic.
    """
    # Basic numeric check first
    clean_form_val = re.sub(r'[^0-9.]', '', form_value)
    
    # If it's wine and no ABV is stated, check for "Table Wine"
    if "wine" in product_type.lower() and not clean_form_val:
        return "table wine" in normalize_text(ocr_text)
        
    # Standard check
    if not clean_form_val:
        return True # If not provided in form, skip check (or fail depending on strictness)
        
    # Look for the number in the text
    # Allow for "Alc. X% by Vol" or "X% Alc/Vol" or just "X%"
    # We search for the number near a % sign or "alc" keyword
    
    norm_text = normalize_text(ocr_text)
    
    # Direct match of the number
    if clean_form_val in norm_text:
        return True
        
    # Fuzzy number match (e.g. 14.5 vs 145 or 14.S)
    # This is complex to do perfectly with simple regex, so we stick to direct substring for now
    # but we can try to match the integer part at least
    try:
        val_float = float(clean_form_val)
        int_val = int(val_float)
        if str(int_val) in norm_text and "%" in ocr_text:
            return True
    except ValueError:
        # Invalid form value, skip fuzzy matching
        pass
        
    return False

def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    Preprocess image to improve OCR accuracy with aggressive enhancement.
    Applies multiple techniques to enhance text readability, especially for low-contrast images.
    """
    # Convert to RGB if needed (handles RGBA, P, etc.)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to grayscale (OCR works better on grayscale)
    img_gray = image.convert('L')
    
    # Convert to OpenCV for better preprocessing
    img_array = np.array(img_gray)
    
    # Apply CLAHE for better contrast (especially for gold on black)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))  # Higher clip limit for low contrast
    img_array = clahe.apply(img_array)
    
    # Aggressive contrast enhancement
    enhancer = ImageEnhance.Contrast(Image.fromarray(img_array))
    img_gray = enhancer.enhance(3.0)  # Triple the contrast for low-contrast images
    
    # Enhance brightness to make text stand out
    enhancer = ImageEnhance.Brightness(img_gray)
    img_gray = enhancer.enhance(1.3)  # More brightness
    
    # Aggressive sharpness enhancement
    enhancer = ImageEnhance.Sharpness(img_gray)
    img_gray = enhancer.enhance(4.0)  # Very high sharpness
    
    # Apply unsharp mask for better edge definition
    img_gray = img_gray.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=2))
    
    # Binarization - convert to pure black and white
    try:
        img_array = np.array(img_gray)
        
        # Try Otsu's method for automatic threshold (better for low contrast)
        _, img_binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Also try inverted (for light text on dark)
        _, img_binary_inv = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Use the one with more text content
        normal_white = np.sum(img_binary == 255)
        inverted_white = np.sum(img_binary_inv == 255)
        
        if inverted_white > normal_white * 1.2:
            img_gray = Image.fromarray(img_binary_inv, mode='L')
        else:
            img_gray = Image.fromarray(img_binary, mode='L')
    except Exception as e:
        # Fallback: simple threshold
        logger.warning(f"Advanced binarization failed, using simple threshold: {e}")
        threshold = 128
        img_gray = img_gray.point(lambda x: 255 if x > threshold else 0, mode='1').convert('L')
    
    # Final denoising (removes small artifacts after binarization)
    img_gray = img_gray.filter(ImageFilter.MedianFilter(size=3))
    
    return img_gray

def deskew_image(img_gray: np.ndarray) -> np.ndarray:
    """
    Deskew image by detecting and correcting rotation angle.
    Uses Hough transform to find text lines and corrects rotation.
    """
    try:
        # Detect edges using Canny
        edges = cv2.Canny(img_gray, CANNY_EDGE_LOW, CANNY_EDGE_HIGH, apertureSize=3)
        
        # Use HoughLines to detect text lines
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is not None and len(lines) > 0:
            # Calculate average angle
            angles = []
            for rho, theta in lines[:HOUGH_LINES_LIMIT]:
                angle = (theta * 180 / np.pi) - 90
                if -45 < angle < 45:  # Only consider reasonable angles
                    angles.append(angle)
            
            if angles:
                avg_angle = np.median(angles)
                if abs(avg_angle) > DESKEW_ANGLE_THRESHOLD:
                    # Rotate image to correct skew
                    (h, w) = img_gray.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
                    img_gray = cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception as e:
        logger.warning(f"Deskew operation failed: {e}")
    
    return img_gray

def preprocess_image_variant_1(image: Image.Image) -> Image.Image:
    """
    Variant 1: Enhanced OpenCV preprocessing with advanced techniques.
    Optimized for low-contrast images (gold on black, etc.)
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Advanced noise reduction
    # Use bilateral filter to reduce noise while preserving edges
    img_gray = cv2.bilateralFilter(img_gray, 9, 75, 75)
    
    # Step 2: Deskew (straighten rotated text)
    img_gray = deskew_image(img_gray)
    
    # Step 3: Enhanced contrast with CLAHE (for low-contrast images)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))  # Higher for gold on black
    img_gray = clahe.apply(img_gray)
    
    # Step 4: Edge enhancement using unsharp masking
    gaussian = cv2.GaussianBlur(img_gray, (0, 0), 2.0)
    img_gray = cv2.addWeighted(img_gray, 1.5, gaussian, -0.5, 0)
    
    # Step 5: Try multiple thresholding methods and pick best
    # Method 1: Adaptive threshold (normal)
    img_binary_normal = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Method 2: Adaptive threshold (inverted - for light text on dark)
    img_binary_inverted = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Method 3: Otsu's threshold (normal)
    _, img_binary_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Method 4: Otsu's threshold (inverted)
    _, img_binary_otsu_inv = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Choose best based on text content (more white pixels = more text)
    candidates = [
        (img_binary_normal, "adaptive normal"),
        (img_binary_inverted, "adaptive inverted"),
        (img_binary_otsu, "otsu normal"),
        (img_binary_otsu_inv, "otsu inverted"),
    ]
    
    best_img = img_binary_normal
    best_name = "adaptive normal"
    best_score = 0
    
    for candidate_img, name in candidates:
        # Score based on text-like features
        white_pixels = np.sum(candidate_img == 255)
        # Prefer images with reasonable amount of text (not too sparse, not too dense)
        text_ratio = white_pixels / candidate_img.size
        if 0.1 < text_ratio < 0.6:  # Reasonable text density
            score = white_pixels * (1 - abs(text_ratio - 0.3))  # Prefer ~30% text
            if score > best_score:
                best_score = score
                best_img = candidate_img
                best_name = name
    
    img_binary = best_img
    
    # Step 6: Morphological operations to clean up
    # Close small gaps in characters
    kernel_close = np.ones((2, 2), np.uint8)
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel_close)
    
    # Remove small noise
    kernel_open = np.ones((2, 2), np.uint8)
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel_open)
    
    # Step 7: Final noise reduction
    # Remove isolated pixels
    img_binary = cv2.medianBlur(img_binary, 3)
    
    # Convert back to PIL
    return Image.fromarray(img_binary, mode='L')

def preprocess_image_variant_2(image: Image.Image) -> Image.Image:
    """
    Variant 2: Enhanced OpenCV preprocessing with advanced denoising and edge preservation.
    Optimized for clear, high-contrast images.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to OpenCV
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Advanced denoising (preserves edges better than Gaussian blur)
    img_gray = cv2.fastNlMeansDenoising(img_gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Step 2: Deskew
    img_gray = deskew_image(img_gray)
    
    # Step 3: CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_gray = clahe.apply(img_gray)
    
    # Step 4: Sharpening using unsharp mask
    gaussian = cv2.GaussianBlur(img_gray, (0, 0), 1.0)
    img_gray = cv2.addWeighted(img_gray, 1.5, gaussian, -0.5, 0)
    
    # Step 5: Otsu's thresholding (automatically finds optimal threshold)
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Also try inverted Otsu
    _, img_binary_inv = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Choose better one
    normal_white = np.sum(img_binary == 255)
    inverted_white = np.sum(img_binary_inv == 255)
    
    if inverted_white > normal_white * 1.2:
        img_binary = img_binary_inv
    
    # Step 6: Morphological cleanup
    kernel = np.ones((2, 2), np.uint8)
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
    
    # Convert back to PIL
    return Image.fromarray(img_binary, mode='L')

def perform_ocr_with_retry(image: Image.Image) -> str:
    """
    Perform OCR with enhanced OpenCV preprocessing and multiple PSM modes using Tesseract.
    Returns the best result based on quality scoring.
    
    Enhanced Preprocessing:
    - 3 preprocessing variants tried in order: Variant 2 (OpenCV Enhanced Otsu), Variant 3 (Standard PIL), Variant 1 (OpenCV Enhanced Adaptive - only if needed)
    - 3 PSM modes (0, 1, 3)
    - Total: Up to 9 OCR attempts (6 from Variant 2+3, 3 from Variant 1 if needed), automatically selects best result
    
    Optimizations:
    - Early exit if high-quality result found (saves time)
    - Variant 1 only tried if high quality not found from Variant 2 and 3
    - No upscaling (uses original image resolution)
    - Advanced OpenCV techniques: deskewing, multi-method thresholding, edge enhancement
    """
    all_results = []
    
    # Score results - prefer longer, more complete results
    def score_result(result_text):
        # Count alphanumeric characters (real text)
        alpha_count = sum(1 for c in result_text if c.isalnum())
        # Count words (more words = better)
        word_count = len(result_text.split())
        # Penalize results with too many special characters
        special_ratio = sum(1 for c in result_text if not c.isalnum() and not c.isspace()) / max(len(result_text), 1)
        
        # Score: length + word count - penalty for too many special chars
        score = len(result_text) * 0.5 + word_count * 10 + alpha_count * 0.3
        if special_ratio > 0.3:  # More than 30% special chars is suspicious
            score *= 0.7
        return score
    
    # Preprocessing variants: Enhanced OpenCV methods
    variants = [
        ("OpenCV Enhanced Adaptive", preprocess_image_variant_1),
        ("OpenCV Enhanced Otsu", preprocess_image_variant_2),
        ("Standard PIL", preprocess_image_for_ocr),
    ]
    
    # PSM modes: Only use 0, 1, 3
    # 0 = Orientation and script detection (OSD) only
    # 1 = Automatic page segmentation with OSD
    # 3 = Fully automatic page segmentation, but no OSD
    psm_modes = [
        (0, "Orientation and script detection (OSD) only"),
        (1, "Automatic page segmentation with OSD"),
        (3, "Fully automatic page segmentation, but no OSD"),
    ]
    
    # Try preprocessing variants in order: Variant 2, then Variant 3, then Variant 1 (only if needed)
    # Variant 2 = OpenCV Enhanced Otsu
    # Variant 3 = Standard PIL
    # Variant 1 = OpenCV Enhanced Adaptive (only if high quality not found)
    variant_2 = ("OpenCV Enhanced Otsu", preprocess_image_variant_2)
    variant_3 = ("Standard PIL", preprocess_image_for_ocr)
    variant_1 = ("OpenCV Enhanced Adaptive", preprocess_image_variant_1)
    
    # First, try Variant 2 and Variant 3
    high_quality_found = False
    for variant_name, preprocess_func in [variant_2, variant_3]:
        try:
            preprocessed = preprocess_func(image)
            
            # Try each PSM mode with this preprocessed image
            for psm_mode, description in psm_modes:
                try:
                    config = f'--psm {psm_mode} --oem 3 -l eng'
                    text = pytesseract.image_to_string(preprocessed, config=config)
                    text = re.sub(r'\n\s*\n+', '\n', text)
                    text = re.sub(r'[ \t]+', ' ', text)
                    text = text.strip()
                    
                    if text and len(text) > 5:
                        all_results.append((text, variant_name, psm_mode, description))
                        
                        # Early exit if we find a very good result
                        tesseract_score = score_result(text)
                        word_count = len(text.split())
                        if len(text) > OCR_MIN_TEXT_LENGTH and word_count > OCR_MIN_WORD_COUNT and tesseract_score > OCR_QUALITY_SCORE_THRESHOLD:
                            high_quality_found = True
                            return text
                except pytesseract.TesseractError as e:
                    logger.debug(f"Tesseract error with PSM {psm_mode}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Unexpected error during OCR (PSM {psm_mode}): {e}")
                    continue
        except Exception as e:
            logger.warning(f"Preprocessing error for variant {variant_name}: {e}")
            continue
    
    # Only try Variant 1 if high quality result not found from Variant 2 and 3
    if not high_quality_found:
        variant_name, preprocess_func = variant_1
        try:
            preprocessed = preprocess_func(image)
            
            # Try each PSM mode with this preprocessed image
            for psm_mode, description in psm_modes:
                try:
                    config = f'--psm {psm_mode} --oem 3 -l eng'
                    text = pytesseract.image_to_string(preprocessed, config=config)
                    text = re.sub(r'\n\s*\n+', '\n', text)
                    text = re.sub(r'[ \t]+', ' ', text)
                    text = text.strip()
                    
                    if text and len(text) > 5:
                        all_results.append((text, variant_name, psm_mode, description))
                        
                        # Early exit if we find a very good result
                        tesseract_score = score_result(text)
                        word_count = len(text.split())
                        if len(text) > OCR_MIN_TEXT_LENGTH and word_count > OCR_MIN_WORD_COUNT and tesseract_score > OCR_QUALITY_SCORE_THRESHOLD:
                            return text
                except pytesseract.TesseractError as e:
                    logger.debug(f"Tesseract error with PSM {psm_mode}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Unexpected error during OCR (PSM {psm_mode}): {e}")
                    continue
        except Exception as e:
            logger.warning(f"Preprocessing error for variant {variant_name}: {e}")
            pass
    
    if not all_results:
        # Fallback to basic OCR
        preprocessed = preprocess_image_for_ocr(image)
        text = pytesseract.image_to_string(preprocessed).strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    # Get best result
    best_result = max(all_results, key=lambda x: score_result(x[0]))
    result_text, _, _, _ = best_result
    return result_text

def find_text_coordinates(image: Image.Image, search_text: str) -> List[Dict[str, int]]:
    """
    Find coordinates of text in image for highlighting.
    Uses pytesseract.image_to_data.
    """
    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        matches = []
        search_words = normalize_text(search_text).split()
        
        if not search_words:
            return []
            
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            word = normalize_text(data['text'][i])
            if word in search_words and len(word) > 2: # Filter out small noise
                matches.append({
                    "x": data['left'][i],
                    "y": data['top'][i],
                    "w": data['width'][i],
                    "h": data['height'][i],
                    "text": data['text'][i]
                })
        return matches
    except Exception as e:
        logger.warning(f"Error finding text coordinates: {e}")
        return []

@app.post("/api/verify")
async def verify_label(
    brand_name: str = Form(...),
    product_type: str = Form(...),
    alcohol_content: str = Form(...),
    net_contents: Optional[str] = Form(None),
    image: UploadFile = File(...)
):
    try:
        # Read image
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        
        logger.info("Starting OCR processing", extra={
            "image_width": img.width,
            "image_height": img.height,
            "image_mode": img.mode
        })
        
        # run OCR
        ocr_text = perform_ocr_with_retry(img)
        
        # clean up form inputs
        brand_name = brand_name.strip() if brand_name else ""
        product_type = product_type.strip() if product_type else ""
        alcohol_content = alcohol_content.strip() if alcohol_content else ""
        net_contents = net_contents.strip() if net_contents else ""
        
        # match fields using fuzzy matching (handles OCR errors)
        brand_match = fuzzy_match(ocr_text, brand_name, threshold=FUZZY_MATCH_THRESHOLD) if brand_name else True
        type_match = fuzzy_match(ocr_text, product_type, threshold=FUZZY_MATCH_THRESHOLD) if product_type else True
        alc_match = validate_alcohol_content(ocr_text, alcohol_content, product_type)
        
        net_match = True
        if net_contents:
            net_match = fuzzy_match(ocr_text, net_contents, threshold=FUZZY_MATCH_THRESHOLD)
            
        # check government warning compliance
        compliance = check_government_warning(ocr_text)
        
        # generate text highlights for UI
        highlights = []
        if brand_match:
            highlights.extend(find_text_coordinates(img, brand_name))
        if alc_match:
            highlights.extend(find_text_coordinates(img, alcohol_content))
            
        # overall verification result
        success = brand_match and type_match and alc_match and net_match and compliance["compliant"]
        
        return JSONResponse({
            "success": success,
            "ocr_text_snippet": ocr_text[:500] + "..." if len(ocr_text) > 500 else ocr_text,
            "results": {
                "brand_name": {"match": brand_match, "value": brand_name},
                "product_type": {"match": type_match, "value": product_type},
                "alcohol_content": {"match": alc_match, "value": alcohol_content},
                "net_contents": {"match": net_match, "value": net_contents} if net_contents else None,
                "compliance": compliance
            },
            "highlights": highlights,
            "image_dimensions": {"width": img.width, "height": img.height}
        })
        
    except Exception as e:
        logger.error(f"Verification error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during verification")

def get_index_html():
    """Serve the React app"""
    index_path = os.path.join("dist", "index.html")
    if os.path.exists(index_path):
        return FileResponse(
            index_path,
            media_type="text/html"
        )
    else:
        # Debug information
        import pathlib
        current_dir = pathlib.Path.cwd()
        dist_exists = os.path.exists("dist")
        files_in_dist = []
        if dist_exists:
            files_in_dist = os.listdir("dist")[:10]  # First 10 files
        
        return JSONResponse({
            "error": "Frontend not built",
            "message": "Run 'npm run build' first",
            "current_dir": str(current_dir),
            "dist_exists": dist_exists,
            "index_path": index_path,
            "files_in_dist": files_in_dist
        }, status_code=500)

@app.get("/")
async def serve_root():
    return get_index_html()

@app.get("/{full_path:path}")
async def serve_app(full_path: str):
    if full_path.startswith("api") or full_path.startswith("assets"):
        raise HTTPException(status_code=404)
    return get_index_html()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

