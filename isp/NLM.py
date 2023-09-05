'https://ridiqulous.com/pytorch-non-local-means/'

import torch
import torch.nn as nn
import torch.nn.functional as F
EPSILON = 1e-12

'Non Local Means'
def rgb_to_luminance(rgb_tensor):
    # :param rgb_tensor: torch.Tensor(N, 3, H, W, ...) in [0, 1] range
    # :return: torch.Tensor(N, 1, H, W, ...) in [0, 1] range
    assert rgb_tensor.min() >= 0.0 and rgb_tensor.max() <= 1.0
    return 0.299 * rgb_tensor[:, :1, ...] + 0.587 * rgb_tensor[:, 1:2, ...] + 0.114 * rgb_tensor[:, 2:, ...]

    
def ShiftStack(tensor, window_size):
    """
    Shift n-dim tensor in a local window and generate a stacked (n + 1)-dim tensor with shape (*orig_shapes, w * y), 
    where wx and wy are width and height of the window
    """
    wx, wy = window_size if isinstance(window_size, (list, tuple)) else (window_size, window_size)
    assert wx % 2 == 1 and wy % 2 == 1, 'window size must be odd'
    rx, ry = wx // 2, wy // 2

    shifted_tensors = []
    for x_shift in range(-rx, rx + 1):
        for y_shift in range(-ry, ry + 1):
            shifted_tensors.append(
                torch.roll(tensor, shifts=(y_shift, x_shift), dims=(2, 3))
            )

    return torch.stack(shifted_tensors, dim=-1)


# :param window_size(patch_size / neighbour_size): Int or Tuple(Int, Int) in (win_width, win_height) order
# :param reduction: 'mean' | 'sum' 
# :param tensor: torch.Tensor(N, C, H, W, ...)
# :return: torch.Tensor(N, C, H, W, ...)
def BoxFilter(tensor, window_size, reduction='mean'):
    wx, wy = window_size if isinstance(window_size, (list, tuple)) else (window_size, window_size)
    assert wx % 2 == 1 and wy % 2 == 1, 'window size must be odd'
    rx, ry = wx // 2, wy // 2
    area = wx * wy

    #local_sum将一个位置对应的patch_size * patch_size的位置叠加
    local_sum = torch.zeros_like(tensor)
    for x_shift in range(-rx, rx + 1):
        for y_shift in range(-ry, ry + 1):
            local_sum += torch.roll(tensor, shifts=(y_shift, x_shift), dims=(2, 3))

    return local_sum if reduction == 'sum' else local_sum / area


def begin_NLM(img, h=1, search_window_size=11, patch_size=5):
    img_tensor = img.permute(0, 3, 1, 2) #(BHWC) --> (BCHW)

    img_window_stack = ShiftStack(img_tensor, window_size=search_window_size)

    tmp = (img_tensor.unsqueeze(-1) - img_window_stack) ** 2
    distances = torch.sqrt(torch.relu(BoxFilter(tmp, window_size=patch_size, reduction='mean')))
    
    weights = torch.zeros_like(distances).cuda()
    for b in range(h.shape[0]):
        weights[b,:,:,:,:] = torch.exp(-distances[b,:,:,:,:] / (torch.relu(h[b,:]) + EPSILON))
#     weights = torch.exp(-distances / (torch.relu(h.view(-1, 1, 1, 1, 1)) + EPSILON))

    denoised_img = (weights * img_window_stack).sum(dim=-1) / weights.sum(dim=-1)
    denoised_img = denoised_img.permute(0, 2, 3, 1) #(BCHW) --> (BHWC)

    return torch.clamp(denoised_img, 1e-8, 1)


if __name__ == '__main__':
    import numpy as np
    import cv2
    import time
    
    img = cv2.imread('linghuan.jpg', 1)
    img = np.array(img).astype(np.float32) / 255
    img_tensor = torch.from_numpy(img).unsqueeze(0).cuda()
    print("Input image tensor's shape --->", img_tensor.shape)
    
    h = torch.ones((1), requires_grad = True).cuda()
    
    start = time.time()
    nlm_img = begin_NLM(img_tensor, h=h)
    end = time.time()
    print('NLM Time:', end - start)

    nlm_img = nlm_img * 255
    cv2.imwrite(r'nlm_img.png', nlm_img.cpu().numpy())