"""
Unit tests for the screen capture, preprocessing, and OCR change detection pipeline.
"""

from PIL import Image, ImageDraw
import numpy as np
from screen.capture import ScreenCapture
from screen.preprocess import ImagePreprocessor
from screen.ocr import ScreenOCR

def test_screen_capture_frame():
    sc = ScreenCapture()
    frame = sc.capture_frame()
    assert frame is not None
    assert isinstance(frame, Image.Image)
    assert frame.width > 0 and frame.height > 0
    sc.close()

def test_image_preprocessing():
    # Create test image
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "Test Text", fill=(0, 0, 0))

    preprocessor = ImagePreprocessor(to_grayscale=True, contrast_factor=1.5, resize_factor=1.2)
    processed = preprocessor.preprocess(img, crop_region=(0, 0, 150, 80))

    assert processed is not None
    assert processed.mode == "L"
    assert processed.width == int(150 * 1.2)
    assert processed.height == int(80 * 1.2)

def test_ocr_change_detection():
    ocr = ScreenOCR(similarity_threshold=0.90)
    img = Image.new("RGB", (300, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "Sample Problem Text", fill=(0, 0, 0))

    # First extraction
    res1 = ocr.extract_text(img)
    assert res1.has_changed is True

    # Second extraction with same content should flag has_changed=False
    res2 = ocr.extract_text(img)
    assert res2.has_changed is False
    assert res2.similarity_with_previous >= 0.90
