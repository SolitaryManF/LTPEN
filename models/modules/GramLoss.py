import torchvision.models as models
import torch
import torch.nn as nn


class Get_vggfeatures(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Get_vggfeatures, self).__init__()
        ### use vgg19 weights to initialize
        vgg_pretrained_features = models.vgg19(pretrained=True).features.cuda()
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        # if not requires_grad:
        for param in self.slice1.parameters():
            param.requires_grad = requires_grad
        for param in self.slice2.parameters():
            param.requires_grad = requires_grad
        for param in self.slice3.parameters():
            param.requires_grad = requires_grad
        for param in self.slice4.parameters():
            param.requires_grad = requires_grad
        for param in self.slice5.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        b,c,h,w = x.shape
        if c==1:
            x = torch.cat((x, x, x), dim=1)
        x = self.slice1(x)
        x_lv1 = x
        x = self.slice2(x)
        x_lv2 = x
        x = self.slice3(x)
        x_lv3 = x
        x = self.slice4(x)
        x_lv4 = x
        x = self.slice5(x)
        x_lv5 = x
        return x_lv1, x_lv2, x_lv3, x_lv4, x_lv5


def gram_matrix(input):

    b, c, h, w = input.size()
    input = input.view(b, c, h * w)
    input_t = input.transpose(1, 2)
    size = input.size()
    size_t = input_t.size()
    input_mean = input.mean(dim=2).view(b, c, 1)
    input_mean_t = input_t.mean(dim=1).view(b, 1, c)
    gram = (input - input_mean.expand(size)).bmm(input_t - input_mean_t.expand(size_t)) / (c * h * w)
    return gram


class GramLoss4(torch.nn.Module):
    def __init__(self):
        super(GramLoss4, self).__init__()
        self.get_vggfeatures = Get_vggfeatures()
        self.l1loss = nn.L1Loss()

    def forward(self, sr, gt):
        sr_features1, sr_features2, sr_features3, sr_features4, _ = self.get_vggfeatures(sr)
        gt_features1, gt_features2, gt_features3, gt_features4, _ = self.get_vggfeatures(gt)

        sr_features1_gram = gram_matrix(sr_features1)
        gt_features1_gram = gram_matrix(gt_features1)
        sr_features2_gram = gram_matrix(sr_features2)
        gt_features2_gram = gram_matrix(gt_features2)
        sr_features3_gram = gram_matrix(sr_features3)
        gt_features3_gram = gram_matrix(gt_features3)
        sr_features4_gram = gram_matrix(sr_features4)
        gt_features4_gram = gram_matrix(gt_features4)

        loss = self.l1loss(sr_features1_gram,gt_features1_gram)+self.l1loss(sr_features2_gram,gt_features2_gram)+\
               self.l1loss(sr_features3_gram,gt_features3_gram)+self.l1loss(sr_features4_gram,gt_features4_gram)
        return loss
