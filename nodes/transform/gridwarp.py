"""
GridWarp node for ComfyUI_Detonate.

Mesh-based image warping with control grid.
Professional tool for perspective correction and creative warping.

Reference: Nuke GridWarp, After Effects Mesh Warp, Fusion GridWarp
"""

import torch
import numpy as np
import cv2
from typing import Tuple, List
import json


class DetonateGridWarp:
    """
    Mesh-based image warping with control grid.

    Warps images using a deformable control grid (mesh).
    Essential for perspective correction, lens distortion,
    creative warping, and manual image manipulation.

    Features:
    - Grid-based mesh warping
    - Adjustable grid resolution (4x4 to 32x32)
    - Smooth interpolation (bilinear)
    - JSON grid data for persistence
    - Multiple edge modes

    Workflow:
    1. Define grid resolution (e.g., 8x8)
    2. Set grid point offsets (JSON data)
    3. Choose interpolation quality
    4. Apply warp to image

    Common uses:
    - Perspective correction
    - Lens distortion correction
    - Creative image warping
    - Building/object straightening
    - Puppet warp effects

    Nuke equivalent: GridWarp
    Fusion equivalent: GridWarp
    """

    CATEGORY = "detonate/transform"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "grid_resolution_x": ("INT", {
                    "default": 8,
                    "min": 2,
                    "max": 32,
                    "step": 1,
                    "tooltip": "Horizontal grid resolution (control points)",
                }),
                "grid_resolution_y": ("INT", {
                    "default": 8,
                    "min": 2,
                    "max": 32,
                    "step": 1,
                    "tooltip": "Vertical grid resolution (control points)",
                }),
                "grid_data": ("STRING", {
                    "default": '{"offsets":[]}',
                    "multiline": False,
                    "tooltip": "JSON grid point offsets [{'x': col, 'y': row, 'dx': offset_x, 'dy': offset_y}, ...]",
                }),
            },
            "optional": {
                "edge_mode": (["Clamp", "Black"], {
                    "default": "Clamp",
                    "tooltip": "Edge behavior: Clamp (stretch edges) or Black",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Warp strength multiplier",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output",)
    FUNCTION = "grid_warp"

    def grid_warp(
        self,
        image: torch.Tensor,
        grid_resolution_x: int,
        grid_resolution_y: int,
        grid_data: str,
        edge_mode: str = "Clamp",
        strength: float = 1.0
    ) -> Tuple[torch.Tensor]:
        """
        Apply mesh-based grid warping to image.

        Args:
            image: Input image [B, H, W, C]
            grid_resolution_x: Grid columns (control points)
            grid_resolution_y: Grid rows (control points)
            grid_data: JSON string with grid point offsets
            edge_mode: Edge handling (Clamp/Black)
            strength: Warp strength multiplier

        Returns:
            Warped image [B, H, W, C]
        """
        device = image.device
        B, H, W, C = image.shape

        # Parse grid data
        try:
            data = json.loads(grid_data)
            offsets = data.get('offsets', [])
        except json.JSONDecodeError:
            # Invalid JSON, return unchanged
            return (image,)

        if len(offsets) == 0:
            # No grid deformation, return unchanged
            return (image,)

        # Create initial grid (uniform)
        grid_x = np.linspace(0, W - 1, grid_resolution_x, dtype=np.float32)
        grid_y = np.linspace(0, H - 1, grid_resolution_y, dtype=np.float32)

        # Create mesh grid [rows, cols]
        src_points = np.zeros((grid_resolution_y, grid_resolution_x, 2), dtype=np.float32)
        dst_points = np.zeros((grid_resolution_y, grid_resolution_x, 2), dtype=np.float32)

        for row in range(grid_resolution_y):
            for col in range(grid_resolution_x):
                x = grid_x[col]
                y = grid_y[row]

                src_points[row, col] = [x, y]
                dst_points[row, col] = [x, y]  # Start with identity

        # Apply offsets from grid data
        for offset_data in offsets:
            col = offset_data.get('x', -1)
            row = offset_data.get('y', -1)
            dx = offset_data.get('dx', 0.0) * strength
            dy = offset_data.get('dy', 0.0) * strength

            if 0 <= row < grid_resolution_y and 0 <= col < grid_resolution_x:
                dst_points[row, col, 0] += dx
                dst_points[row, col, 1] += dy

        # Create normalized destination points [-1, 1] for grid_sample
        # dst_points shape: [grid_resolution_y, grid_resolution_x, 2]
        dst_points[..., 0] = (dst_points[..., 0] / max(1, W - 1)) * 2.0 - 1.0
        dst_points[..., 1] = (dst_points[..., 1] / max(1, H - 1)) * 2.0 - 1.0
        
        # Convert to tensor and prepare for interpolation
        # Shape: [1, 2, grid_resolution_y, grid_resolution_x]
        dst_tensor = torch.from_numpy(dst_points).to(device=device, dtype=torch.float32)
        dst_tensor = dst_tensor.permute(2, 0, 1).unsqueeze(0)
        
        # Interpolate the sparse grid of coordinates to dense [1, 2, H, W]
        dense_grid = torch.nn.functional.interpolate(
            dst_tensor, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=True
        )
        
        # Shape for grid_sample: [1, H, W, 2]
        dense_grid = dense_grid.squeeze(0).permute(1, 2, 0).unsqueeze(0)
        
        # Map edge modes
        if edge_mode == "Black":
            padding_mode = 'zeros'
        else:  # Clamp
            padding_mode = 'border'

        # Process each image in batch natively on GPU
        input_img = image.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Expand grid for batch size if needed
        dense_grid = dense_grid.expand(B, -1, -1, -1)
        
        warped = torch.nn.functional.grid_sample(
            input_img,
            dense_grid,
            mode='bilinear',
            padding_mode=padding_mode,
            align_corners=True
        )
        
        output = warped.permute(0, 2, 3, 1)  # [B, H, W, C]

        return (output,)
