"""
RotoBezier node for ComfyUI_Detonate.

Interactive Bezier spline drawing for rotoscoping and masking.
Professional-grade rotoscoping tool inspired by Natron's Roto node.

Reference: Natron Roto/RotoPaint, Nuke RotoPaint, Mocha tracking
https://natron.readthedocs.io/en/rb-2.3/devel/PythonReference/NatronEngine/Roto.html
"""

import torch
import numpy as np
import json
from PIL import Image, ImageDraw
import cv2
from typing import Tuple
from ...utils.bezier_utils import discretize_spline


class DetonateRotoBezier:
    """
    Interactive Bezier spline drawing for rotoscoping.

    Professional rotoscoping tool with interactive Bezier curve editing.
    Provides high-quality mask generation from vector splines with
    anti-aliasing and feathering support.

    Features:
    - Interactive Bezier spline drawing (web widget)
    - Smooth cubic Bezier curves (de Casteljau algorithm)
    - Anti-aliased edge rendering
    - Adjustable feathering
    - Multiple spline support
    - Closed and open splines

    Workflow:
    1. Use interactive widget to draw spline shapes
    2. Adjust control points and tangent handles
    3. Set feather amount for soft edges
    4. Generate high-quality anti-aliased masks

    Common uses:
    - Rotoscoping (isolating subjects)
    - Garbage mattes (removing unwanted areas)
    - Vignettes and shape masks
    - Motion graphics shapes
    - Object isolation for compositing

    Natron/Nuke equivalent: Roto / RotoPaint
    """

    CATEGORY = "detonate/matte"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "spline_data": ("STRING", {
                    "default": '{"splines":[]}',
                    "multiline": False,
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                # Feather amount
                "feather": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "display": "slider",
                }),
                # Feather falloff type
                "feather_type": (["Smooth", "Linear", "Gaussian"], {
                    "default": "Smooth",
                }),
                # Anti-aliasing quality
                "antialias_samples": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                }),
                # Global invert
                "invert": ("BOOLEAN", {
                    "default": False,
                }),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "rasterize_splines"

    def rasterize_splines(
        self,
        spline_data: str,
        image: torch.Tensor = None,
        feather: float = 2.0,
        feather_type: str = "Smooth",
        antialias_samples: int = 4,
        invert: bool = False
    ) -> Tuple[torch.Tensor]:
        """
        Rasterize Bezier splines. Supports batches (videos).
        """
        if image is not None:
            B, H, W, C = image.shape
            width, height = W, H
        else:
            B = 1
            width, height = 1280, 720

        # Parse spline data
        try:
            data = json.loads(spline_data)
            splines = data.get('splines', [])
        except json.JSONDecodeError:
            return (torch.zeros(B, height, width, dtype=torch.float32),)

        if len(splines) == 0:
            return (torch.zeros(B, height, width, dtype=torch.float32),)

        # Create mask with supersampling
        aa_factor = max(1, antialias_samples)
        aa_width = int(width * aa_factor)
        aa_height = int(height * aa_factor)

        result_mask = np.zeros((aa_height, aa_width), dtype=np.float32)

        for spline_idx, s_data in enumerate(splines):
            points = s_data.get('points', [])
            closed = s_data.get('closed', False)
            operation = s_data.get('operation', 'add')
            spline_invert = s_data.get('invert', False)

            if len(points) < 2: continue

            # Discretize
            spline_points = discretize_spline(points, closed=closed, samples_per_segment=20)
            if len(spline_points) == 0: continue

            # Scale to supersampled resolution
            spline_points_scaled = spline_points.copy()
            spline_points_scaled[:, 0] *= aa_width
            spline_points_scaled[:, 1] *= aa_height

            polygon_points = [(int(p[0]), int(p[1])) for p in spline_points_scaled]

            # Individual spline mask
            spline_mask_ui8 = np.zeros((aa_height, aa_width), dtype=np.uint8)
            img = Image.fromarray(spline_mask_ui8)
            draw = ImageDraw.Draw(img)

            if closed and len(polygon_points) >= 3:
                draw.polygon(polygon_points, fill=255, outline=255)
            else:
                if len(polygon_points) >= 2:
                    draw.line(polygon_points, fill=255, width=max(1, aa_factor))

            s_mask = np.array(img).astype(np.float32) / 255.0
            if spline_invert: s_mask = 1.0 - s_mask

            if spline_idx == 0 or operation == 'add':
                result_mask = np.maximum(result_mask, s_mask)
            elif operation == 'subtract':
                result_mask = np.maximum(result_mask - s_mask, 0.0)
            elif operation == 'intersect':
                result_mask = result_mask * s_mask

        # Move to GPU
        import comfy.model_management as mm
        device = mm.get_torch_device()
        mask_nchw = torch.from_numpy(result_mask).to(device).unsqueeze(0).unsqueeze(0)

        # Downsample AA
        if aa_factor > 1:
            mask_nchw = torch.nn.functional.interpolate(mask_nchw, size=(height, width), mode='area')

        # Feathering
        if feather > 0.0:
            from ..filter.blur import DetonateBlur
            blur_node = DetonateBlur()
            
            # The blur node expects 3 or 4 channels
            mask_bhwc = mask_nchw.permute(0, 2, 3, 1).repeat(1, 1, 1, 3)

            if feather_type == "Gaussian":
                mask_bhwc = blur_node.blur(mask_bhwc, feather * 2, feather * 2, blur_alpha=False)[0]
            elif feather_type == "Linear":
                mask_bhwc = blur_node.blur(mask_bhwc, feather * 1.5, feather * 1.5, blur_alpha=False)[0]
            else: # Smooth
                mask_bhwc = blur_node.blur(mask_bhwc, feather, feather, blur_alpha=False)[0]
                mask_bhwc = mask_bhwc * mask_bhwc * (3.0 - 2.0 * mask_bhwc)

            # Extract just one channel back to [1, H, W]
            mask_tensor = mask_bhwc[:, :, :, 0]
        else:
            mask_tensor = mask_nchw.squeeze(1)

        # Batch support: Expand 1 frame to B frames
        if B > 1:
            mask_tensor = mask_tensor.expand(B, -1, -1)

        if invert: mask_tensor = 1.0 - mask_tensor
        mask_tensor = torch.clamp(mask_tensor, 0.0, 1.0)

        # Move back to CPU for compatibility with standard ComfyUI nodes
        mask_tensor = mask_tensor.cpu()

        return (mask_tensor,)