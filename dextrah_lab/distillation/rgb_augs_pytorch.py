"""
PyTorch/ONNX-compatible implementations of RGB augmentation operations.
This replaces the Warp kernel implementations with pure PyTorch operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import random
import os
import glob
from torchvision import transforms


class ModifySaturation(nn.Module):
    """PyTorch implementation of modify_saturation_kernel"""

    def forward(self, rgb, gray, saturation, max_pixels):
        """
        Args:
            rgb: (B, 3, H, W) - Input RGB images
            gray: (B, H, W) - Grayscale version
            saturation: (B,) - Saturation factors
            max_pixels: (B, 3) - Maximum pixel values per channel
        Returns:
            rgb_out: (B, 3, H, W) - Saturated RGB images
        """
        # Expand dimensions for broadcasting
        saturation = saturation.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        gray = gray.unsqueeze(1)  # (B, 1, H, W)

        # Apply saturation adjustment: gray + saturation * (rgb - gray)
        rgb_out = gray + saturation * (rgb - gray)

        # Clamp to [0, max_pixels] per channel
        for c in range(3):
            max_val = max_pixels[:, c].view(-1, 1, 1, 1)
            rgb_out[:, c:c+1, :, :] = torch.clamp(rgb_out[:, c:c+1, :, :], 0.0, max_val)

        return rgb_out


class ModifyContrast(nn.Module):
    """PyTorch implementation of modify_contrast_kernel"""

    def forward(self, rgb, avg_brightness, contrast, max_pixels):
        """
        Args:
            rgb: (B, 3, H, W) - Input RGB images
            avg_brightness: (B,) - Average brightness per image
            contrast: (B,) - Contrast factors
            max_pixels: (B, 3) - Maximum pixel values per channel
        Returns:
            rgb_out: (B, 3, H, W) - Contrast-adjusted RGB images
        """
        # Expand dimensions for broadcasting
        avg_brightness = avg_brightness.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        contrast = contrast.view(-1, 1, 1, 1)  # (B, 1, 1, 1)

        # Apply contrast adjustment: avg + contrast * (rgb - avg)
        rgb_out = avg_brightness + contrast * (rgb - avg_brightness)

        # Clamp to [0, max_pixels] per channel
        for c in range(3):
            max_val = max_pixels[:, c].view(-1, 1, 1, 1)
            rgb_out[:, c:c+1, :, :] = torch.clamp(rgb_out[:, c:c+1, :, :], 0.0, max_val)

        return rgb_out


class ModifyBrightness(nn.Module):
    """PyTorch implementation of modify_brightness_kernel"""

    def forward(self, rgb, brightness, max_pixels):
        """
        Args:
            rgb: (B, 3, H, W) - Input RGB images
            brightness: (B,) - Brightness factors
            max_pixels: (B, 3) - Maximum pixel values per channel
        Returns:
            rgb_out: (B, 3, H, W) - Brightness-adjusted RGB images
        """
        # Expand dimensions for broadcasting
        brightness = brightness.view(-1, 1, 1, 1)  # (B, 1, 1, 1)

        # Apply brightness adjustment: rgb * brightness
        rgb_out = rgb * brightness

        # Clamp to [0, max_pixels] per channel
        for c in range(3):
            max_val = max_pixels[:, c].view(-1, 1, 1, 1)
            rgb_out[:, c:c+1, :, :] = torch.clamp(rgb_out[:, c:c+1, :, :], 0.0, max_val)

        return rgb_out


class ModifyHue(nn.Module):
    """PyTorch implementation of modify_hue_kernel"""

    def forward(self, h, hue):
        """
        Args:
            h: (B, H, W) - Hue channel
            hue: (B,) - Hue adjustment values
        Returns:
            h_out: (B, H, W) - Adjusted hue channel
        """
        # Expand dimensions for broadcasting
        hue = hue.view(-1, 1, 1)  # (B, 1, 1)

        # Add hue adjustment
        h_out = h + hue

        # Wrap around [0, 1]
        h_out = torch.where(h_out >= 1.0, h_out - 1.0, h_out)
        h_out = torch.where(h_out < 0.0, h_out + 1.0, h_out)

        return h_out


class Conv2DBlur(nn.Module):
    """PyTorch implementation of conv2d (motion blur) kernel"""

    def forward(self, rgb, kernel, alpha):
        """
        Args:
            rgb: (B, 3, H, W) - Input RGB images
            kernel: (1, kernel_size, kernel_size) - Convolution kernel
            alpha: float - Blending factor
        Returns:
            rgb_out: (B, 3, H, W) - Blurred RGB images
        """
        kernel_size = kernel.shape[1]
        padding = kernel_size // 2

        # Expand kernel to 3 channels (same kernel for R, G, B)
        # Shape: (3, 1, kernel_size, kernel_size)
        kernel_3ch = kernel.unsqueeze(0).repeat(3, 1, 1, 1)

        # Apply depthwise convolution (each channel independently)
        blurred = F.conv2d(rgb, kernel_3ch, padding=padding, groups=3)

        # Clamp and blend
        blurred = torch.clamp(blurred, 0.0, 1.0)
        rgb_out = alpha * blurred + (1.0 - alpha) * rgb

        return rgb_out


def rgb_to_hsv(image):
    """
    Convert RGB to HSV color space.
    Ripped from PyTorch source code (torchvision)

    Args:
        image: (B, 3, H, W) - RGB images
    Returns:
        hsv: (B, 3, H, W) - HSV images
    """
    r, g, _ = image.unbind(dim=-3)

    minc, maxc = torch.aminmax(image, dim=-3)

    eqc = maxc == minc

    channels_range = maxc - minc
    ones = torch.ones_like(maxc)
    s = channels_range / torch.where(eqc, ones, maxc)

    channels_range_divisor = torch.where(eqc, ones, channels_range).unsqueeze_(dim=-3)
    rc, gc, bc = ((maxc.unsqueeze(dim=-3) - image) / channels_range_divisor).unbind(dim=-3)

    mask_maxc_neq_r = maxc != r
    mask_maxc_eq_g = maxc == g

    hg = rc.add(2.0).sub_(bc).mul_(mask_maxc_eq_g & mask_maxc_neq_r)
    hr = bc.sub_(gc).mul_(~mask_maxc_neq_r)
    hb = gc.add_(4.0).sub_(rc).mul_(mask_maxc_neq_r.logical_and_(mask_maxc_eq_g.logical_not_()))

    h = hr.add_(hg).add_(hb)
    h = h.mul_(1.0 / 6.0).add_(1.0).fmod_(1.0)

    return torch.stack((h, s, maxc), dim=-3)


def hsv_to_rgb(img):
    """
    Convert HSV to RGB color space.
    Ripped from PyTorch source code (torchvision)

    Args:
        img: (B, 3, H, W) - HSV images
    Returns:
        rgb: (B, 3, H, W) - RGB images
    """
    h, s, v = img.unbind(dim=-3)
    h6 = h.mul(6)
    i = torch.floor(h6)
    f = h6.sub_(i)
    i = i.to(dtype=torch.int32)

    sxf = s * f
    one_minus_s = 1.0 - s
    q = (1.0 - sxf).mul_(v).clamp_(0.0, 1.0)
    t = sxf.add_(one_minus_s).mul_(v).clamp_(0.0, 1.0)
    p = one_minus_s.mul_(v).clamp_(0.0, 1.0)
    i.remainder_(6)

    vpqt = torch.stack((v, p, q, t), dim=-3)

    select = torch.tensor(
        [[0, 2, 1, 1, 3, 0], [3, 0, 0, 2, 1, 1], [1, 1, 3, 0, 0, 2]],
        dtype=torch.long
    )
    select = select.to(device=img.device, non_blocking=True)

    select = select[:, i]
    if select.ndim > 3:
        select = select.moveaxis(0, -3)

    return vpqt.gather(-3, select)


def get_motion_blur_kernel2d_batched(batch_size, kernel_size, angle_range, direction_range, device):
    """
    Generate batched motion blur kernels.

    Args:
        batch_size: Number of kernels to generate
        kernel_size: Size of the kernel
        angle_range: (min, max) angle range in degrees
        direction_range: (min, max) direction range
        device: torch device
    Returns:
        kernels: (batch_size, kernel_size, kernel_size) - Motion blur kernels
    """
    angles = np.random.uniform(angle_range[0], angle_range[1], size=(batch_size,))
    directions = np.random.uniform(direction_range[0], direction_range[1], size=(batch_size,))

    directions = (np.clip(directions, -1., 1.) + 1.) / 2.

    kernels = np.zeros((batch_size, kernel_size, kernel_size))

    for i in range(batch_size):
        kernels[i, kernel_size // 2, :] = np.linspace(directions[i], 1 - directions[i], kernel_size)

    angles_rad = np.deg2rad(angles)
    cos_angles = np.cos(angles_rad)
    sin_angles = np.sin(angles_rad)
    rotation_matrices = np.array([
        [cos_angles, -sin_angles],
        [sin_angles, cos_angles]
    ]).transpose(2, 0, 1)

    coords = np.stack(np.meshgrid(
        np.arange(kernel_size) - kernel_size // 2,
        np.arange(kernel_size) - kernel_size // 2
    ), axis=-1).reshape(-1, 2).T

    rotated_coords = np.einsum('bij,jk->bik', rotation_matrices, coords)

    x_coords = rotated_coords[:, 0, :].reshape(batch_size, kernel_size, kernel_size) + kernel_size // 2
    y_coords = rotated_coords[:, 1, :].reshape(batch_size, kernel_size, kernel_size) + kernel_size // 2

    x0 = np.floor(x_coords).astype(int).clip(0, kernel_size - 1)
    x1 = np.ceil(x_coords).astype(int).clip(0, kernel_size - 1)
    y0 = np.floor(y_coords).astype(int).clip(0, kernel_size - 1)
    y1 = np.ceil(y_coords).astype(int).clip(0, kernel_size - 1)

    x_diff = x_coords - x0
    y_diff = y_coords - y0

    rotated_kernels = (
        kernels[np.arange(batch_size)[:, None, None], y0, x0] * (1 - x_diff) * (1 - y_diff) +
        kernels[np.arange(batch_size)[:, None, None], y0, x1] * x_diff * (1 - y_diff) +
        kernels[np.arange(batch_size)[:, None, None], y1, x0] * (1 - x_diff) * y_diff +
        kernels[np.arange(batch_size)[:, None, None], y1, x1] * x_diff * y_diff
    )

    rotated_kernels /= rotated_kernels.sum(axis=(1, 2), keepdims=True)

    return torch.from_numpy(rotated_kernels).float().to(device)


class RgbAugPyTorch:
    """PyTorch/ONNX-compatible version of RgbAug class"""

    def __init__(
            self, device, all_env_inds, use_stereo,
            background_cfg, color_cfg, motion_blur_cfg
    ):
        self.device = device
        self.all_env_inds = all_env_inds
        self.num_envs = len(self.all_env_inds)
        self.background_cfg = background_cfg
        self.color_cfg = color_cfg
        self.motion_blur_cfg = motion_blur_cfg

        # Load background images
        img_names = glob.glob(os.path.join(background_cfg["dir"], "*.jpg"))
        self.background_imgs = [
            Image.open(img_name).convert("RGB")
            for img_name in img_names
        ]
        self.background_img_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.use_stereo = use_stereo

        # Initialize augmentation flags
        self.has_background_aug = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        if self.use_stereo:
            self.env_left_backgrounds = torch.zeros(
                (self.num_envs, 3, background_cfg["height"], background_cfg["width"]),
                dtype=torch.float32, device=self.device
            )
            self.env_right_backgrounds = torch.zeros(
                (self.num_envs, 3, background_cfg["height"], background_cfg["width"]),
                dtype=torch.float32, device=self.device
            )
        else:
            self.env_backgrounds = torch.zeros(
                (self.num_envs, 3, background_cfg["height"], background_cfg["width"]),
                dtype=torch.float32, device=self.device
            )

        self.has_color_aug = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.color_aug_params = torch.zeros(
            self.num_envs, 4, dtype=torch.float32, device=self.device
        )

        # Initialize PyTorch modules
        self.modify_saturation = ModifySaturation()
        self.modify_contrast = ModifyContrast()
        self.modify_brightness = ModifyBrightness()
        self.modify_hue = ModifyHue()
        self.conv2d_blur = Conv2DBlur()

        self.reset()

    def reset(self, env_ids=None):
        """Reset augmentation parameters for specified environments"""
        if env_ids is None:
            env_ids = self.all_env_inds

        # Sample random background images
        p = torch.rand(len(env_ids))
        w_background_aug = p < self.background_cfg["aug_prob"]
        env_ids_with_background_aug = env_ids[w_background_aug].flatten()
        env_ids_without_background_aug = env_ids[~w_background_aug].flatten()
        self.has_background_aug[env_ids_with_background_aug] = True
        self.has_background_aug[env_ids_without_background_aug] = False

        if torch.any(w_background_aug):
            selected_imgs = random.sample(
                self.background_imgs,
                w_background_aug.sum().item()
            )
            background_tensors = torch.stack([
                self.background_img_transform(img)
                for img in selected_imgs
            ]).to(self.device)

            top = torch.randint(
                0, 480 - self.background_cfg["height"] + 1, (1,)
            ).item()
            left = torch.randint(
                0, 640 - self.background_cfg["width"] + 1, (1,)
            ).item()

            if self.use_stereo:
                left_background_img = background_tensors[
                    :, :,
                    top:top+self.background_cfg["height"],
                    left:left+self.background_cfg["width"]
                ]
                self.env_left_backgrounds[env_ids_with_background_aug] = left_background_img

                top = torch.randint(
                    0, 480 - self.background_cfg["height"] + 1, (1,)
                ).item()
                left = torch.randint(
                    0, 640 - self.background_cfg["width"] + 1, (1,)
                ).item()

                right_background_img = background_tensors[
                    :, :,
                    top:top+self.background_cfg["height"],
                    left:left+self.background_cfg["width"]
                ]
                self.env_right_backgrounds[env_ids_with_background_aug] = right_background_img
            else:
                background_img = background_tensors[
                    :, :,
                    top:top+self.background_cfg["height"],
                    left:left+self.background_cfg["width"]
                ]
                self.env_backgrounds[env_ids_with_background_aug] = background_img

        # Sample random color jitter params
        p = torch.rand(len(env_ids))
        w_color_aug = p < self.color_cfg["aug_prob"]
        env_ids_with_color_aug = env_ids[w_color_aug].flatten()
        env_ids_without_color_aug = env_ids[~w_color_aug]
        self.has_color_aug[env_ids_with_color_aug] = True
        self.has_color_aug[env_ids_without_color_aug] = False

        if torch.any(w_color_aug):
            self.color_aug_params[env_ids_with_color_aug, 0] = torch.rand(
                env_ids_with_color_aug.shape[0]
            ).to(self.device) * (
                self.color_cfg["saturation_range"][1] -
                self.color_cfg["saturation_range"][0]
            ) + self.color_cfg["saturation_range"][0]

            self.color_aug_params[env_ids_with_color_aug, 1] = torch.rand(
                env_ids_with_color_aug.shape[0]
            ).to(self.device) * (
                self.color_cfg["contrast_range"][1] -
                self.color_cfg["contrast_range"][0]
            ) + self.color_cfg["contrast_range"][0]

            self.color_aug_params[env_ids_with_color_aug, 2] = torch.rand(
                env_ids_with_color_aug.shape[0]
            ).to(self.device) * (
                self.color_cfg["brightness_range"][1] -
                self.color_cfg["brightness_range"][0]
            ) + self.color_cfg["brightness_range"][0]

            self.color_aug_params[env_ids_with_color_aug, 3] = torch.rand(
                env_ids_with_color_aug.shape[0]
            ).to(self.device) * (
                self.color_cfg["hue_range"][1] -
                self.color_cfg["hue_range"][0]
            ) + self.color_cfg["hue_range"][0]

    def apply_background_aug(self, rgb_imgs, masks, env_backgrounds):
        """Apply background augmentation"""
        if torch.any(self.has_background_aug):
            rgb_imgs[self.has_background_aug] = torch.where(
                masks[self.has_background_aug].expand(-1, 3, -1, -1),
                env_backgrounds[self.has_background_aug],
                rgb_imgs[self.has_background_aug]
            )
        return rgb_imgs

    def apply_color_aug(self, rgb_imgs):
        """Apply color augmentation using PyTorch operations"""
        if torch.any(self.has_color_aug):
            aug_rgb_img = rgb_imgs[self.has_color_aug]

            # Saturation
            gray_scale = (
                aug_rgb_img[:, 0, :, :] * 0.299 +
                aug_rgb_img[:, 1, :, :] * 0.587 +
                aug_rgb_img[:, 2, :, :] * 0.114
            )
            max_pixels = torch.amax(aug_rgb_img, dim=(-2, -1))
            aug_rgb_img = self.modify_saturation(
                aug_rgb_img, gray_scale,
                self.color_aug_params[self.has_color_aug, 0],
                max_pixels
            )

            # Contrast
            gray_scale = (
                aug_rgb_img[:, 0, :, :] * 0.299 +
                aug_rgb_img[:, 1, :, :] * 0.587 +
                aug_rgb_img[:, 2, :, :] * 0.114
            )
            avg_brightness = torch.mean(gray_scale, dim=(-2, -1))
            aug_rgb_img = self.modify_contrast(
                aug_rgb_img, avg_brightness,
                self.color_aug_params[self.has_color_aug, 1],
                max_pixels
            )

            # Brightness
            aug_rgb_img = self.modify_brightness(
                aug_rgb_img,
                self.color_aug_params[self.has_color_aug, 2],
                max_pixels
            )

            # Hue
            hsv = rgb_to_hsv(aug_rgb_img)
            hue = hsv[:, 0, :, :]
            s = hsv[:, 1, :, :]
            v = hsv[:, 2, :, :]
            hue = self.modify_hue(hue, self.color_aug_params[self.has_color_aug, 3])
            aug_rgb_img = hsv_to_rgb(torch.stack((hue, s, v), dim=1))

            rgb_imgs[self.has_color_aug] = aug_rgb_img

        return rgb_imgs

    def apply_motion_blur_aug(self, rgb_imgs):
        """Apply motion blur augmentation"""
        p = torch.rand(self.num_envs)
        w_motion_aug = p < self.motion_blur_cfg["aug_prob"]
        env_ids_with_motion_blur_aug = self.all_env_inds[w_motion_aug].flatten()

        if torch.any(w_motion_aug):
            angle = np.random.rand(1) * 180.0
            direction = 2 * (np.random.rand(1) - 0.5)
            kernel_size = np.random.choice(self.motion_blur_cfg["kernel_sizes"])

            # Use batched kernel generation for efficiency
            motion_blur_kernel = get_motion_blur_kernel2d_batched(
                1, kernel_size, (angle, angle), (direction, direction), self.device
            )

            aug_imgs = rgb_imgs[env_ids_with_motion_blur_aug]
            aug_imgs = self.conv2d_blur(aug_imgs, motion_blur_kernel[0], 0.7)
            rgb_imgs[w_motion_aug] = aug_imgs

        return rgb_imgs

    def apply(self, rgb, mask):
        """Apply all augmentations"""
        if self.use_stereo:
            left_img = rgb["left_img"]
            right_img = rgb["right_img"]
            left_mask = mask["left_mask"]
            right_mask = mask["right_mask"]

            left_img = self.apply_background_aug(
                left_img, left_mask, self.env_left_backgrounds
            )
            right_img = self.apply_background_aug(
                right_img, right_mask, self.env_right_backgrounds
            )

            rgb = {
                "left_img": left_img,
                "right_img": right_img
            }
            rgb["left_img"] = self.apply_color_aug(rgb["left_img"])
            rgb["right_img"] = self.apply_color_aug(rgb["right_img"])
            rgb["left_img"] = self.apply_motion_blur_aug(rgb["left_img"])
            rgb["right_img"] = self.apply_motion_blur_aug(rgb["right_img"])
        else:
            img = rgb
            mask_img = mask
            rgb = self.apply_background_aug(img, mask_img, self.env_backgrounds)
            rgb = self.apply_color_aug(rgb)
            rgb = self.apply_motion_blur_aug(rgb)

        return rgb
