"""
MatteControl node for ComfyUI_Detonate.

All-in-one professional matte refinement tool.
Essential compositor workflow: clean up mattes in a single node.

Reference: Nuke FilterErode + Blur combo, Fusion MatteControl
"""

import torch
import numpy as np
import cv2
from typing import Tuple


class DetonateMatteControl:
    """
    Professional all-in-one matte refinement tool.

    Combines the most common matte operations in proper order:
    1. Contract/Expand (erode/dilate)
    2. Blur edges
    3. Gamma correction
    4. Black/White point clipping

    This is the #1 requested matte tool by compositors - replaces
    chains of 4-5 nodes with a single professional control.

    Features:
    - Erode/Dilate with adjustable size
    - Edge blur (Gaussian)
    - Gamma correction for matte density
    - Black point / White point clipping
    - Proper operation ordering
    - Preview modes for each stage

    Workflow:
    1. Contract/expand to fix edge contamination
    2. Blur to soften edges
    3. Gamma to adjust overall density
    4. Black/white points to clean up values

    Common uses:
    - Clean up keyer output
    - Refine rotoscoped mattes
    - Fix AI-generated masks
    - Prepare mattes for compositing

    Nuke equivalent: FilterErode + Blur + Grade combo
    Fusion equivalent: MatteControl
    """

    CATEGORY = "detonate/matte"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            },
            "optional": {
                "erode_dilate": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Negative = contract (erode), Positive = expand (dilate)",
                }),
                "blur": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Blur amount for softening edges",
                }),
                "gamma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Gamma correction (< 1 = lighter, > 1 = darker)",
                }),
                "black_point": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "display": "slider",
                    "tooltip": "Values below this become black (0.0)",
                }),
                "white_point": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.001,
                    "display": "slider",
                    "tooltip": "Values above this become white (1.0)",
                }),
                "preview_mode": (["Final", "After Erode/Dilate", "After Blur", "After Gamma"], {
                    "default": "Final",
                    "tooltip": "Preview intermediate stages of processing",
                }),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "refine_matte"

    def refine_matte(
        self,
        mask: torch.Tensor,
        erode_dilate: float = 0.0,
        blur: float = 0.0,
        gamma: float = 1.0,
        black_point: float = 0.0,
        white_point: float = 1.0,
        preview_mode: str = "Final"
    ) -> Tuple[torch.Tensor]:
        """
        Refine matte using professional operations on GPU.

        Processing order:
        1. Erode/Dilate (contract/expand edges)
        2. Blur (soften edges)
        3. Gamma (adjust overall density)
        4. Black/White point (clip values)

        Args:
            mask: Input mask [B, H, W]
            erode_dilate: Erode (negative) or dilate (positive) amount
            blur: Gaussian blur radius
            gamma: Gamma correction
            black_point: Values below this become 0.0
            white_point: Values above this become 1.0
            preview_mode: Show intermediate stage or final result

        Returns:
            Refined mask [B, H, W]
        """
        matte = mask.clone()

        # Stage 1: Erode/Dilate
        if abs(erode_dilate) > 0.01:
            amount = erode_dilate
            kernel_size = int(abs(amount) * 2 + 1)
            if kernel_size % 2 == 0: 
                kernel_size += 1
            kernel_size = max(3, kernel_size)
            padding = kernel_size // 2
            
            matte_nchw = matte.unsqueeze(1) # [B, 1, H, W]
            iterations = max(1, int(abs(amount) / 2))
            
            for _ in range(iterations):
                if amount < 0:
                    # Erode
                    matte_nchw = -torch.nn.functional.max_pool2d(
                        -matte_nchw, kernel_size=kernel_size, stride=1, padding=padding
                    )
                else:
                    # Dilate
                    matte_nchw = torch.nn.functional.max_pool2d(
                        matte_nchw, kernel_size=kernel_size, stride=1, padding=padding
                    )
            
            matte = matte_nchw.squeeze(1)

        if preview_mode == "After Erode/Dilate":
            return (matte,)

        # Stage 2: Blur
        if blur > 0.01:
            from ..filter.blur import DetonateBlur
            blur_node = DetonateBlur()
            # DetonateBlur expects [B, H, W, C]. We have [B, H, W]
            matte_bhwc = matte.unsqueeze(-1)
            # blur_node.blur returns a tuple
            matte_bhwc = blur_node.blur(matte_bhwc, blur, blur, blur_alpha=True)[0]
            matte = matte_bhwc.squeeze(-1)

        if preview_mode == "After Blur":
            return (matte,)

        # Stage 3: Gamma
        if abs(gamma - 1.0) > 0.001:
            matte = torch.clamp(matte, 0.0, 1.0)
            matte = torch.pow(matte, gamma)

        if preview_mode == "After Gamma":
            return (matte,)

        # Stage 4: Black/White point clipping
        if abs(black_point) > 0.001 or abs(white_point - 1.0) > 0.001:
            if black_point >= white_point:
                white_point = black_point + 0.001
            matte = torch.clamp((matte - black_point) / (white_point - black_point + 1e-7), 0.0, 1.0)

        # Final result
        return (matte,)
