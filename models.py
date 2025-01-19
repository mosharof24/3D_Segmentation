import torch
import torch.nn as nn

################################################################################
# UNet2D
def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding='same'),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=0.1),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding='same'),
        nn.ReLU(inplace=True)
    )
    return conv

# def crop_img(tensor, target_tensor):
#          target_size = target_tensor.size()[2]
#          tensor_size = tensor.size()[2]
#          delta = tensor_size - target_size
#          delta = delta // 2
#          return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]

class UNet2D(nn.Module):
    def __init__(self):
        super(UNet2D, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1, 16)
        self.down_conv_2 = double_conv(16, 32)
        self.down_conv_3 = double_conv(32, 64)
        self.down_conv_4 = double_conv(64, 128)
        self.down_conv_5 = double_conv(128, 256)

        self.up_trans_1 = nn.ConvTranspose2d(in_channels=256, 
                                             out_channels=128,
                                             kernel_size=2,
                                             stride=2,
                                             padding=0)
        
        self.up_conv_1 = double_conv(256, 128)

        self.up_trans_2 = nn.ConvTranspose2d(in_channels=128, 
                                             out_channels=64,
                                             kernel_size=2,
                                             stride=2,
                                             padding=0)
        
        self.up_conv_2 = double_conv(128, 64)

        self.up_trans_3 = nn.ConvTranspose2d(in_channels=64, 
                                             out_channels=32,
                                             kernel_size=2,
                                             stride=2,
                                             padding=0)
        
        self.up_conv_3 = double_conv(64, 32)

        self.up_trans_4 = nn.ConvTranspose2d(in_channels=32, 
                                             out_channels=16,
                                             kernel_size=2,
                                             stride=2,
                                             padding=0)
        
        self.up_conv_4 = double_conv(32, 16)

        self.out = nn.Conv2d(
              in_channels=16,
              out_channels=2,
              kernel_size=1
        )
    
    def forward(self, image):
        # encoder
        x1 = self.down_conv_1(image) #
        # print(x1.size())
        x2 = self.max_pool_2x2(x1)
        # print(x2.size())
        x3 = self.down_conv_2(x2) #
        # print(x3.size())
        x4 = self.max_pool_2x2(x3)
        # print(x4.size())
        x5 = self.down_conv_3(x4) #
        # print(x5.size())
        x6 = self.max_pool_2x2(x5)
        # print(x6.size())
        x7 = self.down_conv_4(x6) #
        # print(x7.size())
        x8 = self.max_pool_2x2(x7)
        # print(x8.size())
        x9 = self.down_conv_5(x8)
        # print(x9.size())

        # decoder
        x = self.up_trans_1(x9)
        # print(x.size())
        # y = crop_img(x7, x)
        # print(y.size())
        x = self.up_conv_1(torch.concat([x, x7], 1))

        x = self.up_trans_2(x)
        # print(x.size())
        # y = crop_img(x5, x)
        # print(y.size())
        x = self.up_conv_2(torch.concat([x, x5], 1))

        x = self.up_trans_3(x)
        # y = crop_img(x3, x)
        x = self.up_conv_3(torch.concat([x, x3], 1))

        x = self.up_trans_4(x)
        # y = crop_img(x1, x)
        x = self.up_conv_4(torch.concat([x, x1], 1))

        x = self.out(x)

        # print(x.size())

#####################################################################################
# UNet3D
def double_conv3D(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=3, padding='same'),
        nn.ReLU(inplace=True),
        nn.Dropout3d(p=0.1),
        nn.Conv3d(out_c, out_c, kernel_size=3, padding='same'),
        nn.ReLU(inplace=True)
    )
    return conv

class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()

        self.max_pool_2x2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv3D(1, 16)
        self.down_conv_2 = double_conv3D(16, 32)
        self.down_conv_3 = double_conv3D(32, 64)
        self.down_conv_4 = double_conv3D(64, 128)
        self.down_conv_5 = double_conv3D(128, 256)

        self.up_trans_1 = nn.ConvTranspose3d(in_channels=256, 
                                             out_channels=128,
                                             kernel_size=2,
                                             stride=2,
                                             padding=0)
        
        self.up_conv_1 = double_conv3D(256, 128)

        self.up_trans_2 = nn.ConvTranspose3d(in_channels=128, 
                                             out_channels=64,
                                             kernel_size=2,
                                             stride=2,
                                             padding=0)
        
        self.up_conv_2 = double_conv3D(128, 64)

        self.up_trans_3 = nn.ConvTranspose3d(in_channels=64, 
                                             out_channels=32,
                                             kernel_size=2,
                                             stride=2,
                                             padding=0)
        
        self.up_conv_3 = double_conv3D(64, 32)

        self.up_trans_4 = nn.ConvTranspose3d(in_channels=32, 
                                             out_channels=16,
                                             kernel_size=2,
                                             stride=2,
                                             padding=0)
        
        self.up_conv_4 = double_conv3D(32, 16)

        self.out = nn.Conv3d(
              in_channels=16,
              out_channels=5,
              kernel_size=1
        )
    
    def forward(self, image):
        # encoder
        x1 = self.down_conv_1(image) #
        # print(x1.size())
        x2 = self.max_pool_2x2(x1)
        # print(x2.size())
        x3 = self.down_conv_2(x2) #
        # print(x3.size())
        x4 = self.max_pool_2x2(x3)
        # print(x4.size())
        x5 = self.down_conv_3(x4) #
        # print(x5.size())
        x6 = self.max_pool_2x2(x5)
        # print(x6.size())
        x7 = self.down_conv_4(x6) #
        # print(x7.size())
        x8 = self.max_pool_2x2(x7)
        # print(x8.size())
        x9 = self.down_conv_5(x8)
        # print(x9.size())

        # decoder
        x = self.up_trans_1(x9)
        # print(x.size())
        # y = crop_img(x7, x)
        # print(y.size())
        x = self.up_conv_1(torch.concat([x, x7], 1))

        x = self.up_trans_2(x)
        # print(x.size())
        # y = crop_img(x5, x)
        # print(y.size())
        x = self.up_conv_2(torch.concat([x, x5], 1))

        x = self.up_trans_3(x)
        # y = crop_img(x3, x)
        x = self.up_conv_3(torch.concat([x, x3], 1))

        x = self.up_trans_4(x)
        # y = crop_img(x1, x)
        x = self.up_conv_4(torch.concat([x, x1], 1))

        x = self.out(x)

        # print(x.size())