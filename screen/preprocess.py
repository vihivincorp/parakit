"""
Image preprocessing pipeline to improve OCR accuracy for desktop screenshots.
"""

from typing import Optional, Tuple
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from utils.timing import time_it

class ImagePreprocessor:
    """Applies transformations to enhance text clarity for OCR."""

    def __init__(
        self,
        to_grayscale: bool = True,
        contrast_factor: float = 1.5,
        resize_factor: float = 1.0,
        binarize_threshold: Optional[int] = None
    ):
        self.to_grayscale = to_grayscale
        self.contrast_factor = contrast_factor
        self.resize_factor = resize_factor
        self.binarize_threshold = binarize_threshold

    @time_it("image_preprocessing")
    def preprocess(
        self,
        img: Image.Image,
        crop_region: Optional[Tuple[int, int, int, int]] = None
    ) -> Image.Image:
        """
        Preprocesses a raw PIL screenshot.
        
        Args:
            img: Input PIL Image.
            crop_region: Optional (left, top, right, bottom).
            
        Returns:
            Processed PIL Image optimized for OCR.
        """
        processed = img

        # 1. Crop if requested
        if crop_region:
            processed = processed.crop(crop_region)

        # 2. Convert to grayscale
        if self.to_grayscale:
            processed = processed.convert("L")

        # 3. Contrast adjustment
        if self.contrast_factor != 1.0:
            enhancer = ImageEnhance.Contrast(processed)
            processed = enhancer.enhance(self.contrast_factor)

        # 4. Optional Resizing for small fonts
        if self.resize_factor != 1.0:
            new_width = int(processed.width * self.resize_factor)
            new_height = int(processed.height * self.resize_factor)
            processed = processed.resize((new_width, new_height), Image.Resampling.BILINEAR)

        # 5. Optional Adaptive / Fixed Binarization
        if self.binarize_threshold is not None and self.to_grayscale:
            np_img = np.array(processed)
            binary = (np_img > self.binarize_threshold) * 255
            processed = Image.fromarray(binary.astype(np.uint8))

        return processed
