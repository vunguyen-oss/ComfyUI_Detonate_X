"""
ColorCurves node for ComfyUI_Detonate.

Professional color grading using curves. Industry-standard tool for
precise tonal control across highlights, midtones, and shadows.

Uses linear interpolation between curve points for fast evaluation.
Supports Master, Red, Green, Blue curve channels.

Improvements over basic curves:
- Curve presets (S-curve, filmic, etc.)
- Monotonic clamping option
- HDR support

Reference: Natron ColorLookup, Blender RGB Curves
"""

import torch
from ...utils import validate_image_tensor, ensure_alpha_channel


class DetonateColorCurves:
    """
    Professional color grading using tone curves.

    Adjust tonal response using curve points that define
    input→output mapping. Essential for precise color grading,
    contrast enhancement, and look development.

    Common uses:
    - Contrast enhancement (S-curve)
    - Shadow/highlight recovery
    - Film emulation and stylized grades
    - Matching footage from different sources
    - Selective tonal control

    Curve Format:
    Curves are defined as strings: "x1,y1;x2,y2;x3,y3"
    Example: "0,0;0.5,0.6;1,1" (boost mids slightly)

    Improvements:
    - Curve presets built-in
    - Fast linear interpolation
    - HDR support (values can exceed 1.0)
    """

    CATEGORY = "detonate/color"

    # Curve presets
    PRESETS = {
        "linear": "0,0;1,1",
        "s_curve": "0,0;0.25,0.2;0.75,0.8;1,1",
        "lift_shadows": "0,0.1;0.5,0.5;1,1",
        "crush_blacks": "0,0;0.1,0;1,1",
        "filmic": "0,0.02;0.18,0.18;0.9,0.95;1,0.98",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "master_curve": ("STRING", {
                    "default": "0,0;1,1",
                    "multiline": False,
                }),
                "red_curve": ("STRING", {
                    "default": "0,0;1,1",
                    "multiline": False,
                }),
                "green_curve": ("STRING", {
                    "default": "0,0;1,1",
                    "multiline": False,
                }),
                "blue_curve": ("STRING", {
                    "default": "0,0;1,1",
                    "multiline": False,
                }),
                "preset": (["none"] + list(cls.PRESETS.keys()), {
                    "default": "none",
                }),
                "clamp_output": ("BOOLEAN", {
                    "default": False,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_curves"

    def apply_curves(
        self,
        image: torch.Tensor,
        master_curve: str,
        red_curve: str,
        green_curve: str,
        blue_curve: str,
        preset: str,
        clamp_output: bool
    ) -> tuple:
        """
        Apply color curves to image.

        Args:
            image: Input tensor [B,H,W,C]
            master_curve: Master curve points (affects all channels)
            red_curve: Red channel curve
            green_curve: Green channel curve
            blue_curve: Blue channel curve
            preset: Apply preset curve to master
            clamp_output: Clamp result to 0-1

        Returns:
            Tuple containing graded image [B,H,W,C]
        """
        validate_image_tensor(image)

        # Ensure RGBA
        image_rgba = ensure_alpha_channel(image)

        rgb = image_rgba[:, :, :, :3]
        alpha = image_rgba[:, :, :, 3:4]

        # Apply preset logic
        # We check if the incoming curves are actually different from default
        # If they are default AND a preset is selected, we use the preset.
        # Otherwise, we trust the interactive curve data from the frontend.
        is_default_master = (master_curve == "0,0;1,1")
        
        if preset != "none" and preset in self.PRESETS and is_default_master:
            master_curve = self.PRESETS[preset]

        # Parse curves
        master_points = self._parse_curve(master_curve)
        red_points = self._parse_curve(red_curve)
        green_points = self._parse_curve(green_curve)
        blue_points = self._parse_curve(blue_curve)

        # Apply master curve first
        rgb = self._apply_curve_to_tensor(rgb, master_points)

        # Apply per-channel curves
        result_r = self._apply_curve_to_tensor(rgb[:, :, :, 0:1], red_points)
        result_g = self._apply_curve_to_tensor(rgb[:, :, :, 1:2], green_points)
        result_b = self._apply_curve_to_tensor(rgb[:, :, :, 2:3], blue_points)

        result_rgb = torch.cat([result_r, result_g, result_b], dim=3)

        # Optional clamping
        if clamp_output:
            result_rgb = torch.clamp(result_rgb, 0.0, 1.0)

        result = torch.cat([result_rgb, alpha], dim=3)

        return (result,)

    def _parse_curve(self, curve_string: str) -> list:
        """
        Parse curve string into list of (x, y) points.

        Format: "x1,y1;x2,y2;x3,y3"

        Args:
            curve_string: Curve definition

        Returns:
            List of (x, y) tuples, sorted by x
        """
        points = []

        try:
            # Split by semicolon
            pairs = curve_string.strip().split(';')

            for pair in pairs:
                if not pair.strip():
                    continue

                # Split by comma
                x_str, y_str = pair.split(',')
                x = float(x_str.strip())
                y = float(y_str.strip())

                points.append((x, y))

            # Sort by x value
            points.sort(key=lambda p: p[0])

            # Ensure we have at least start and end points
            if not points:
                points = [(0.0, 0.0), (1.0, 1.0)]
            elif len(points) == 1:
                # Add endpoints if missing
                if points[0][0] > 0:
                    points.insert(0, (0.0, 0.0))
                if points[-1][0] < 1.0:
                    points.append((1.0, 1.0))

        except (ValueError, IndexError):
            # Parse error, use linear curve
            print(f"Warning: Failed to parse curve '{curve_string}', using linear")
            points = [(0.0, 0.0), (1.0, 1.0)]

        return points

    def _apply_curve_to_tensor(
        self,
        tensor: torch.Tensor,
        curve_points: list
    ) -> torch.Tensor:
        """
        Apply curve to tensor values using vectorized Monotonic Cubic Spline interpolation.
        
        This ensures smooth transitions without the "kinks" of linear interpolation, 
        and prevents overshooting (staying monotonic).

        Args:
            tensor: Input tensor (any shape)
            curve_points: List of (x, y) points

        Returns:
            Mapped tensor (same shape)
        """
        if len(curve_points) < 2:
            return tensor
            
        # Convert points to tensors
        xs = torch.tensor([p[0] for p in curve_points], dtype=tensor.dtype, device=tensor.device)
        ys = torch.tensor([p[1] for p in curve_points], dtype=tensor.dtype, device=tensor.device)
        
        # Sort points by x
        xs, indices = torch.sort(xs)
        ys = ys[indices]
        
        # If we have only 2 points, it's just linear
        if len(xs) == 2:
            # handle extrapolation
            mask_low = tensor <= xs[0]
            mask_high = tensor >= xs[-1]
            mask_mid = ~mask_low & ~mask_high
            
            result = torch.empty_like(tensor)
            result[mask_low] = ys[0]
            result[mask_high] = ys[-1]
            
            dx = xs[1] - xs[0]
            if dx > 1e-7:
                t = (tensor[mask_mid] - xs[0]) / dx
                result[mask_mid] = ys[0] + t * (ys[1] - ys[0])
            else:
                result[mask_mid] = ys[0]
            return result

        # Monotonic Cubic Interpolation (Fritsch-Carlson)
        # 1. Calculate slopes between points
        dx = xs[1:] - xs[:-1]
        dy = ys[1:] - ys[:-1]
        ms = dy / torch.clamp(dx, min=1e-7)  # Slopes between segments
        
        # 2. Calculate tangents at each point
        num_pts = len(xs)
        tangents = torch.zeros(num_pts, dtype=tensor.dtype, device=tensor.device)
        
        # Use average of neighbor slopes correctly for interior points
        tangents[1:-1] = (ms[:-1] + ms[1:]) / 2.0
        tangents[0] = ms[0]
        tangents[-1] = ms[-1]
        
        # 3. Enforce monotonicity (Fritsch-Carlson)
        # Avoid overshoot by scaling tangents if needed
        for i in range(num_pts - 1):
            if ms[i] == 0:
                tangents[i] = 0.0
                tangents[i+1] = 0.0
            else:
                alpha = tangents[i] / ms[i]
                beta = tangents[i+1] / ms[i]
                if alpha**2 + beta**2 > 9.0:
                    tau = 3.0 / torch.sqrt(torch.tensor(alpha**2 + beta**2, device=tensor.device))
                    tangents[i] = tau * alpha * ms[i]
                    tangents[i+1] = tau * beta * ms[i]

        # 4. Interpolate
        result = torch.empty_like(tensor)
        
        # Extrapolation
        result[tensor <= xs[0]] = ys[0]
        result[tensor >= xs[-1]] = ys[-1]
        
        # Piecewise Cubic Hermite Interpolation
        for i in range(num_pts - 1):
            x_i, x_next = xs[i], xs[i+1]
            y_i, y_next = ys[i], ys[i+1]
            m_i, m_next = tangents[i], tangents[i+1]
            
            h = x_next - x_i
            if h <= 1e-7: continue
            
            mask = (tensor > x_i) & (tensor < x_next)
            if not torch.any(mask): continue
            
            t = (tensor[mask] - x_i) / h
            t2 = t * t
            t3 = t2 * t
            
            # Hermite basis functions
            h00 = 2*t3 - 3*t2 + 1
            h10 = t3 - 2*t2 + t
            h01 = -2*t3 + 3*t2
            h11 = t3 - t2
            
            result[mask] = h00 * y_i + h10 * h * m_i + h01 * y_next + h11 * h * m_next
            
        # Handle exact points
        for i in range(num_pts):
            result[tensor == xs[i]] = ys[i]
            
        return result
