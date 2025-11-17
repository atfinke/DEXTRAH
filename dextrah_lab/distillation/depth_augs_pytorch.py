"""
PyTorch/ONNX-compatible implementations of depth augmentation operations.
This replaces the Warp kernel implementations with pure PyTorch operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class AddPixelDropoutAndRandu(nn.Module):
    """
    PyTorch implementation of add_pixel_dropout_and_randu_kernel.

    Note: Dilation logic uses torch.where which is ONNX-compatible but may differ
    slightly from parallel Warp kernel execution.
    """

    def forward(self, depths: torch.Tensor, p_dropout: float, p_randu: float,
                d_min: float, d_max: float, kernel_size: int = 2) -> torch.Tensor:
        """
        Args:
            depths: (B, H, W) - Input depth maps
            p_dropout: float - Probability of dropout
            p_randu: float - Probability of random uniform depth insertion
            d_min: float - Minimum depth value
            d_max: float - Maximum depth value
            kernel_size: int - Size of the dilation kernel for random depth
        Returns:
            depths: (B, H, W) - Augmented depth maps
        """
        batch_size, height, width = depths.shape
        device = depths.device

        # Generate random masks
        rand_dropout = torch.rand(batch_size, height, width, device=device)
        rand_u = torch.rand(batch_size, height, width, device=device)
        rand_u_values = torch.rand(batch_size, height, width, device=device)

        # Apply dropout: set to 0 where rand_dropout <= p_dropout
        dropout_mask = rand_dropout <= p_dropout
        depths[dropout_mask] = 0.0

        # Insert random uniform values
        randu_mask = rand_u <= p_randu
        rand_depth = rand_u_values * (d_max - d_min) + d_min

        # Apply random depth values
        depths[randu_mask] = rand_depth[randu_mask]

        # Dilate random depth values with small kernel
        # Use max pooling to spread the random depth values
        if kernel_size > 1:
            # Create a simple dilation effect by applying average pooling
            # and then filling in nearby pixels with probability 0.25
            rand_prob = torch.rand(batch_size, height, width, device=device)
            dilate_mask = (rand_prob < 0.25) & randu_mask

            # Simple nearest neighbor dilation
            for i in range(1, kernel_size):
                if i < height:
                    depths[:, i:, :] = torch.where(
                        dilate_mask[:, :-i, :],
                        depths[:, :-i, :],
                        depths[:, i:, :]
                    )
                if i < width:
                    depths[:, :, i:] = torch.where(
                        dilate_mask[:, :, :-i],
                        depths[:, :, :-i],
                        depths[:, :, i:]
                    )

        return depths


class AddSticks(nn.Module):
    """
    PyTorch implementation of add_sticks_kernel.

    WARNING: This implementation uses Python loops and .item() calls due to the
    data-dependent nature of stick generation. It will be SLOW compared to the
    original Warp kernel and is NOT ONNX-exportable. Use only during training,
    not in exported models.
    """

    def forward(self, depths: torch.Tensor, p_stick: float, max_stick_len: float,
                max_stick_width: float, d_min: float, d_max: float) -> torch.Tensor:
        """
        Args:
            depths: (B, H, W) - Input depth maps
            p_stick: float - Probability of adding a stick
            max_stick_len: float - Maximum stick length
            max_stick_width: float - Maximum stick width
            d_min: float - Minimum depth value
            d_max: float - Maximum depth value
        Returns:
            depths: (B, H, W) - Augmented depth maps with sticks
        """
        batch_size, height, width = depths.shape
        device = depths.device

        # Generate random values for stick placement
        rand_sticks = torch.rand(batch_size, height, width, device=device)
        rand_sticks_depths = torch.rand(batch_size, height, width, device=device)

        # Find pixels where sticks should be added
        stick_mask = rand_sticks <= p_stick
        stick_indices = torch.nonzero(stick_mask, as_tuple=False)

        if stick_indices.shape[0] > 0:
            for idx in stick_indices:
                b, row, col = idx[0].item(), idx[1].item(), idx[2].item()

                # Generate stick parameters
                stick_width = torch.rand(1, device=device).item() * max_stick_width
                stick_len = torch.rand(1, device=device).item() * max_stick_len + 1.0
                stick_rot = torch.rand(1, device=device).item() * (2 * np.pi)

                # Random depth for this stick
                rand_depth = rand_sticks_depths[b, row, col].item() * (d_max - d_min) + d_min

                # Draw stick line
                for i in range(int(stick_len)):
                    hor_coord = col + i
                    vert_coord = int(i * np.sin(stick_rot)) + row

                    # Bounds checking
                    if hor_coord >= width:
                        hor_coord = width - 1
                    if hor_coord < 0:
                        hor_coord = 0
                    if vert_coord >= height:
                        vert_coord = height - 1
                    if vert_coord < 0:
                        vert_coord = 0

                    depths[b, vert_coord, hor_coord] = rand_depth

                    # Add width to the stick
                    for j in range(1, int(max_stick_width)):
                        # Vertical stick (angle between 45-135 or 225-315 degrees)
                        if (np.pi/4 < stick_rot < 3*np.pi/4) or (5*np.pi/4 < stick_rot < 7*np.pi/4):
                            if vert_coord + j < height:
                                depths[b, vert_coord + j, hor_coord] = rand_depth
                        else:  # Horizontal stick
                            if hor_coord + j < width:
                                depths[b, vert_coord, hor_coord + j] = rand_depth

        return depths


class AddCorrelatedNoise(nn.Module):
    """PyTorch implementation of add_correlated_noise_kernel - fully vectorized"""

    def forward(self, depths: torch.Tensor, sigma_s: float, sigma_d: float,
                d_min: float, d_max: float) -> torch.Tensor:
        """
        Add correlated noise to depth maps based on:
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6907054

        Args:
            depths: (B, H, W) - Input depth maps
            sigma_s: float - Spatial noise standard deviation
            sigma_d: float - Depth noise standard deviation
            d_min: float - Minimum depth value
            d_max: float - Maximum depth value
        Returns:
            noisy_depths: (B, H, W) - Noisy depth maps
        """
        batch_size, height, width = depths.shape
        device = depths.device

        # Generate random noise
        rand_sigma_s_x = sigma_s * torch.randn(batch_size, height, width, device=device)
        rand_sigma_s_y = sigma_s * torch.randn(batch_size, height, width, device=device)
        rand_sigma_d = sigma_d * torch.randn(batch_size, height, width, device=device)

        # Create coordinate grids
        y_coords = torch.arange(height, device=device, dtype=torch.float32).view(1, height, 1).expand(batch_size, height, width)
        x_coords = torch.arange(width, device=device, dtype=torch.float32).view(1, 1, width).expand(batch_size, height, width)

        # Add spatial noise to coordinates
        u = x_coords + rand_sigma_s_x
        v = y_coords + rand_sigma_s_y

        # Bilinear interpolation
        u0 = torch.floor(u).long()
        v0 = torch.floor(v).long()
        u1 = u0 + 1
        v1 = v0 + 1

        # Clamp coordinates
        u0 = torch.clamp(u0, 0, width - 1)
        u1 = torch.clamp(u1, 0, width - 1)
        v0 = torch.clamp(v0, 0, height - 1)
        v1 = torch.clamp(v1, 0, height - 1)

        # Interpolation weights
        fu = u - u0.float()
        fv = v - v0.float()

        w_00 = (1.0 - fu) * (1.0 - fv)
        w_01 = (1.0 - fu) * fv
        w_10 = fu * (1.0 - fv)
        w_11 = fu * fv

        # Gather values and interpolate
        noisy_depths = (
            depths[torch.arange(batch_size).view(-1, 1, 1), v0, u0] * w_00 +
            depths[torch.arange(batch_size).view(-1, 1, 1), v0, u1] * w_01 +
            depths[torch.arange(batch_size).view(-1, 1, 1), v1, u0] * w_10 +
            depths[torch.arange(batch_size).view(-1, 1, 1), v1, u1] * w_11
        )

        # Apply depth quantization noise (simulating stereo camera quantization)
        baseline = 35130.0
        ref = 8.0

        # Avoid division by zero
        noisy_depths = torch.clamp(noisy_depths, min=1e-6)

        denominator = baseline / noisy_depths + rand_sigma_d + 0.5
        noisy_depths = baseline / (torch.round(denominator / ref) * ref)

        # Clamp to valid range
        noisy_depths = torch.clamp(noisy_depths, d_min, d_max)

        return noisy_depths


class AddNormalNoise(nn.Module):
    """PyTorch implementation of add_normal_noise_kernel - fully vectorized"""

    def forward(self, depths: torch.Tensor, sigma_theta: float, cam_matrix: torch.Tensor,
                d_min: float, d_max: float) -> torch.Tensor:
        """
        Add normal-based noise to depth maps.

        Args:
            depths: (B, H, W) - Input depth maps
            sigma_theta: float - Standard deviation of normal noise
            cam_matrix: torch.Tensor (4, 4) - Camera projection matrix
            d_min: float - Minimum depth value
            d_max: float - Maximum depth value
        Returns:
            depths: (B, H, W) - Noisy depth maps
        """
        batch_size, height, width = depths.shape
        device = depths.device

        # Generate random noise
        rand_sigma_theta = sigma_theta * torch.randn(batch_size, height, width, device=device)

        # Create homogeneous pixel coordinates
        y_coords = torch.arange(height, device=device, dtype=torch.float32).view(1, height, 1, 1).expand(batch_size, height, width, 1)
        x_coords = torch.arange(width, device=device, dtype=torch.float32).view(1, 1, width, 1).expand(batch_size, height, width, 1)
        ones = torch.ones(batch_size, height, width, 1, device=device)

        # Create pixel coordinate tensor (B, H, W, 4)
        pixel_coords = torch.cat([x_coords, y_coords, ones, ones], dim=-1)

        # Inverse camera matrix
        cam_matrix_inv = torch.inverse(cam_matrix)

        # Unproject to 3D (B, H, W, 4)
        unprojected = torch.matmul(pixel_coords, cam_matrix_inv.T)

        # Normalize by w coordinate
        x_hat = unprojected[..., 0] / unprojected[..., 3]
        y_hat = unprojected[..., 1] / unprojected[..., 3]
        z_hat = unprojected[..., 2] / unprojected[..., 3]

        # Scale by depth
        d = depths.unsqueeze(-1)
        point_3d = torch.stack([x_hat * depths, y_hat * depths, z_hat * depths], dim=-1)

        # Compute normals from neighboring pixels
        # Handle boundary by padding
        point_3d_padded = F.pad(point_3d.permute(0, 3, 1, 2), (0, 1, 0, 1), mode='replicate').permute(0, 2, 3, 1)

        # Get neighboring points
        point_3d_right = point_3d_padded[:, :-1, 1:, :]
        point_3d_down = point_3d_padded[:, 1:, :-1, :]
        point_3d_center = point_3d_padded[:, :-1, :-1, :]

        # Compute tangent vectors
        x_axis = point_3d_right - point_3d_center
        y_axis = point_3d_down - point_3d_center

        # Normalize
        x_axis = F.normalize(x_axis, p=2, dim=-1)
        y_axis = F.normalize(y_axis, p=2, dim=-1)

        # Compute normal via cross product
        normal = torch.cross(x_axis, y_axis, dim=-1)

        # Perturb 3D points along normal direction
        point_3d_noisy = point_3d + 1000.0 * rand_sigma_theta.unsqueeze(-1) * normal

        # Extract depth from z-coordinate
        depths_noisy = -point_3d_noisy[..., 2] / 1000.0

        # Clamp to valid range
        depths_noisy = torch.clamp(depths_noisy, d_min, d_max)

        return depths_noisy


class DepthAugPyTorch:
    """PyTorch/ONNX-compatible version of DepthAug class"""

    def __init__(self, device):
        self.device = device
        self.kernel_size = 2

        # Initialize PyTorch modules
        self.add_pixel_dropout_and_randu = AddPixelDropoutAndRandu()
        self.add_sticks = AddSticks()
        self.add_correlated_noise = AddCorrelatedNoise()
        self.add_normal_noise = AddNormalNoise()

    def add_pixel_dropout_and_randu_op(self, depths, p_dropout, p_randu, d_min, d_max):
        """
        Add pixel dropout and random uniform depth values.

        Args:
            depths: (B, H, W) - Input depth maps
            p_dropout: float - Dropout probability
            p_randu: float - Random uniform insertion probability
            d_min: float - Minimum depth
            d_max: float - Maximum depth
        """
        return self.add_pixel_dropout_and_randu(
            depths, p_dropout, p_randu, d_min, d_max, self.kernel_size
        )

    def add_sticks_op(self, depths, p_stick, max_stick_len, max_stick_width, d_min, d_max):
        """
        Add random stick artifacts to depth maps.

        Args:
            depths: (B, H, W) - Input depth maps
            p_stick: float - Stick insertion probability
            max_stick_len: float - Maximum stick length
            max_stick_width: float - Maximum stick width
            d_min: float - Minimum depth
            d_max: float - Maximum depth
        """
        return self.add_sticks(
            depths, p_stick, max_stick_len, max_stick_width, d_min, d_max
        )

    def add_correlated_noise_op(self, depths, noisy_depths, sigma_s, sigma_d, d_min, d_max):
        """
        Add correlated noise to depth maps.

        Args:
            depths: (B, H, W) - Input depth maps
            noisy_depths: (B, H, W) - Output depth maps
            sigma_s: float - Spatial noise std
            sigma_d: float - Depth noise std
            d_min: float - Minimum depth
            d_max: float - Maximum depth
        """
        noisy_depths[:] = self.add_correlated_noise(
            depths, sigma_s, sigma_d, d_min, d_max
        )

    def add_normal_noise_op(self, depths, sigma_theta, cam_matrix, d_min, d_max):
        """
        Add normal-based noise to depth maps.

        Args:
            depths: (B, H, W) - Input depth maps
            sigma_theta: float - Normal noise std
            cam_matrix: torch.Tensor (4, 4) - Camera matrix
            d_min: float - Minimum depth
            d_max: float - Maximum depth
        """
        depths[:] = self.add_normal_noise(
            depths, sigma_theta, cam_matrix, d_min, d_max
        )


# Example usage
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    depth_aug = DepthAugPyTorch(device)

    # Dropout and random noise blob parameters
    p_dropout = 0.0125 / 4
    p_randu = 0.0125 / 4
    d_max = 1.5
    d_min = 0.5

    # Random stick parameters
    p_stick = 0.001 / 4
    max_stick_len = 18.0
    max_stick_width = 3.0

    # Correlated noise parameters
    sigma_s = 1.0 / 2
    sigma_d = 1.0 / 6

    # Normal noise parameters
    cam_matrix = torch.eye(4, device=device)
    cam_matrix[0, 0] = 2.2460368
    cam_matrix[1, 1] = 2.9947157
    cam_matrix[2, 3] = -1.0
    cam_matrix[3, 2] = 1.e-3

    sigma_theta = 0.01

    # Create dummy depth map
    depths_raw = torch.rand(1, 240, 320, device=device) * (d_max - d_min) + d_min

    # Create visualizer
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    rendered_img = ax.imshow(depths_raw[0].cpu().numpy(), vmin=0, vmax=1.5, cmap='Greys')
    fig.canvas.draw()
    plt.title("Input")
    plt.show(block=False)

    input('Press ENTER to continue')

    for i in range(1000):
        # Clone raw depths
        depths = torch.clone(depths_raw)

        # This adds correlated noise to depths
        noisy_depths = torch.zeros_like(depths)
        depth_aug.add_correlated_noise_op(
            depths_raw, noisy_depths, sigma_s, sigma_d, d_min, d_max
        )
        depths = noisy_depths

        # Add normal noise
        depth_aug.add_normal_noise_op(depths, sigma_theta, cam_matrix, d_min, d_max)

        # Add pixel dropout and random uniform depth values
        depths = depth_aug.add_pixel_dropout_and_randu_op(
            depths, p_dropout, p_randu, d_min, d_max
        )

        # Add random sticks
        depths = depth_aug.add_sticks_op(
            depths, p_stick, max_stick_len, max_stick_width, d_min, d_max
        )

        # Render
        depth = depths[0].detach().cpu().numpy()
        rendered_img.set_data(depth)
        fig.canvas.draw()
        fig.canvas.flush_events()
