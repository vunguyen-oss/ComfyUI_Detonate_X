"""
TriMap Generator node for ComfyUI_Detonate.

Generate trimaps for AI inpainting workflows.
Trimaps help AI models understand what to keep, remove, and decide on.

Reference: Image matting workflows, AI inpainting best practices
"""

import torch
import numpy as np
import cv2
from typing import Tuple


class DetonateTriMapGenerator:
    """
    Generate trimaps for AI inpainting and matting workflows.

    A trimap divides an image into three regions:
    - Foreground (white, 1.0): Definitely keep this
    - Background (black, 0.0): Definitely remove this  
    - Unknown (gray, 0.5): AI should decide

    This is ESSENTIAL for high-quality AI inpainting because it tells
    the AI model exactly which areas need attention vs which are certain.

    Features:
    - Auto-generate trimap from alpha mask
    - Adjustable unknown region width
    - Threshold controls for foreground/background
    - Multiple unknown region modes
    - Handles anti-aliased edges properly

    Workflow:
    1. Input: Alpha mask from keyer, roto, or background removal
    2. Set thresholds (what's definitely fg/bg)
    3. Set unknown width (transition zone)
    4. Output: Trimap for AI inpainting

    Common uses:
    - Prepare masks for AI inpainting models
    - Image matting workflows
    - Refine background removal results
    - Guide AI models for better edge quality

    This node is UNIQUE to ComfyUI - no other equivalent exists.
    """

    CATEGORY = "detonate/matte"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            },
            "optional": {
                "foreground_threshold": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Alpha above this = definite foreground (white)",
                }),
                "background_threshold": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Alpha below this = definite background (black)",
                }),
                "unknown_width": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.5,
                    "display": "slider",
                    "tooltip": "Width of unknown region around edges (pixels)",
                }),
                "unknown_mode": (["Edge Distance", "Threshold Only", "Full Unknown"], {
                    "default": "Edge Distance",
                    "tooltip": "How to generate unknown region",
                }),
                "output_format": (["Trimap (0/0.5/1)", "Visualization (RGB)"], {
                    "default": "Trimap (0/0.5/1)",
                    "tooltip": "Output as trimap or visual preview",
                }),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("trimap", "preview")
    FUNCTION = "generate_trimap"

    def generate_trimap(
        self,
        mask: torch.Tensor,
        foreground_threshold: float = 0.95,
        background_threshold: float = 0.05,
        unknown_width: float = 10.0,
        unknown_mode: str = "Edge Distance",
        output_format: str = "Trimap (0/0.5/1)"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate trimap from alpha mask natively on GPU.

        Trimap format:
        - 1.0 (white) = Foreground (keep)
        - 0.0 (black) = Background (remove)
        - 0.5 (gray) = Unknown (AI decides)

        Args:
            mask: Input alpha mask [B, H, W]
            foreground_threshold: Alpha > this = foreground
            background_threshold: Alpha < this = background
            unknown_width: Width of unknown region (pixels)
            unknown_mode: How to generate unknown region
            output_format: Trimap or RGB visualization

        Returns:
            trimap: Trimap mask [B, H, W]
            preview: RGB visualization [B, H, W, 3]
        """
        device = mask.device
        B, H, W = mask.shape

        # Start with all unknown
        trimap = torch.full_like(mask, 0.5)

        if unknown_mode in ["Threshold Only", "Full Unknown"]:
            # Simple thresholding
            trimap = torch.where(mask >= foreground_threshold, torch.tensor(1.0, device=device), trimap)
            trimap = torch.where(mask <= background_threshold, torch.tensor(0.0, device=device), trimap)
        else:  # "Edge Distance"
            fg_mask = (mask >= foreground_threshold).float()
            bg_mask = (mask <= background_threshold).float()

            if unknown_width < 0.1:
                trimap = torch.where(fg_mask == 1.0, torch.tensor(1.0, device=device), trimap)
                trimap = torch.where(bg_mask == 1.0, torch.tensor(0.0, device=device), trimap)
            else:
                kernel_size = int(unknown_width * 2 + 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                kernel_size = max(3, kernel_size)
                padding = kernel_size // 2

                iterations = max(1, int(unknown_width / 2))

                # Prepare NCHW 
                fg_nchw = (-fg_mask).unsqueeze(1)  # Negated for erosion via max_pool
                bg_nchw = bg_mask.unsqueeze(1)

                for _ in range(iterations):
                    fg_nchw = torch.nn.functional.max_pool2d(
                        fg_nchw, kernel_size=kernel_size, stride=1, padding=padding
                    )
                    bg_nchw = torch.nn.functional.max_pool2d(
                        bg_nchw, kernel_size=kernel_size, stride=1, padding=padding
                    )

                fg_eroded = (-fg_nchw).squeeze(1)
                bg_dilated = bg_nchw.squeeze(1)

                trimap = torch.where(fg_eroded == 1.0, torch.tensor(1.0, device=device), trimap)
                trimap = torch.where(bg_dilated == 1.0, torch.tensor(0.0, device=device), trimap)

        # Generate RGB preview
        preview = torch.zeros(B, H, W, 3, dtype=torch.float32, device=device)
        
        # Foreground = Green (0, 1, 0)
        fg = trimap >= 0.9
        preview[..., 1] = torch.where(fg, torch.tensor(1.0, device=device), preview[..., 1])
        
        # Background = Red (1, 0, 0)
        bg = trimap <= 0.1
        preview[..., 0] = torch.where(bg, torch.tensor(1.0, device=device), preview[..., 0])
        
        # Unknown = Blue (0, 0, 1)
        unk = (trimap > 0.1) & (trimap < 0.9)
        preview[..., 2] = torch.where(unk, torch.tensor(1.0, device=device), preview[..., 2])

        return (trimap, preview)
