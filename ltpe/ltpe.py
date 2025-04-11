import torch, cv2, math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import make_grid

print(torch.cuda.is_available())
class Get_ltpe(nn.Module):
    def __init__(self):
        super(Get_ltpe, self).__init__()
        kernel7 = [[-1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]]
        kernel6 = [[0, -1, 0],
                   [0, 1, 0],
                   [0, 0, 0]]
        kernel5 = [[0, 0, -1],
                  [0, 1, 0],
                  [0, 0, 0]]
        kernel4 = [[0, 0, 0],
                  [0, 1, -1],
                  [0, 0, 0]]
        kernel3 = [[0, 0, 0],
                  [0, 1, 0],
                  [0, 0, -1]]
        kernel2 = [[0, 0, 0],
                  [0, 1, 0],
                  [0, -1, 0]]
        kernel1 = [[0, 0, 0],
                  [0, 1, 0],
                  [-1, 0, 0]]
        kernel0 = [[0, 0, 0],
                  [-1, 1, 0],
                  [0, 0, 0]]

        kernel7 = torch.cuda.FloatTensor(kernel7).unsqueeze(0).unsqueeze(0)
        kernel6 = torch.cuda.FloatTensor(kernel6).unsqueeze(0).unsqueeze(0)
        kernel5 = torch.cuda.FloatTensor(kernel5).unsqueeze(0).unsqueeze(0)
        kernel4 = torch.cuda.FloatTensor(kernel4).unsqueeze(0).unsqueeze(0)
        kernel3 = torch.cuda.FloatTensor(kernel3).unsqueeze(0).unsqueeze(0)
        kernel2 = torch.cuda.FloatTensor(kernel2).unsqueeze(0).unsqueeze(0)
        kernel1 = torch.cuda.FloatTensor(kernel1).unsqueeze(0).unsqueeze(0)
        kernel0 = torch.cuda.FloatTensor(kernel0).unsqueeze(0).unsqueeze(0)

        self.weight_7 = nn.Parameter(data=kernel7, requires_grad=False)
        self.weight_6 = nn.Parameter(data=kernel6, requires_grad=False)
        self.weight_5 = nn.Parameter(data=kernel5, requires_grad=False)
        self.weight_4 = nn.Parameter(data=kernel4, requires_grad=False)
        self.weight_3 = nn.Parameter(data=kernel3, requires_grad=False)
        self.weight_2 = nn.Parameter(data=kernel2, requires_grad=False)
        self.weight_1 = nn.Parameter(data=kernel1, requires_grad=False)
        self.weight_0 = nn.Parameter(data=kernel0, requires_grad=False)

        weight_list = []
        weight_list.append(self.weight_0),weight_list.append(self.weight_1),weight_list.append(self.weight_2),weight_list.append(self.weight_3)
        weight_list.append(self.weight_4),weight_list.append(self.weight_5),weight_list.append(self.weight_6),weight_list.append(self.weight_7)
        self.weights = weight_list
        self.norm = torch.nn.InstanceNorm2d(1)

    def forward(self, x):

        x_gray = (0.3 * x[:, 0] + 0.59 * x[:, 1] + 0.11 * x[:, 2]).unsqueeze(1)
        out = torch.zeros_like(x_gray)

        for j in range(8):
            x_ltpe = F.conv2d(x_gray, self.weights[j], padding=1)
            x_ltpe = (x_ltpe +1)*0.5
            out = out + x_ltpe*(2**j)/255
        out = self.norm(out)
        out = torch.cat((out, out, out), dim=1)

        return out
def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # CHW->HWC, RGB->BGR（因为后续要用cv2保存）
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    return img_np.astype(out_type)
if __name__ == '__main__':

    get_ltpe = Get_ltpe()

    # 单图计算
    img_flie_path = r"D:\AllDataset\OST300\LR\OST_005.png"
    img_flie_save_path = r"D:\Github\imgs\HR\OST_005_ltpe.png"
    img = cv2.imread(img_flie_path)
    img = torch.from_numpy(img)
    img = img.permute(2, 0, 1).unsqueeze(0).float().cuda()
    img_ltpe = get_ltpe(img)
    x_ltpe_img = tensor2img(img_ltpe)
    cv2.imwrite(img_flie_save_path, x_ltpe_img)