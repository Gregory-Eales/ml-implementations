import torch


class DeepFakeDiscriminator(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernal_size):

        super(DeepFakeDiscriminator, self).__init__()

        self.initialize_parameters()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
        self.to(self.device)

    def initialize_parameters(self):

        
        self.conv1 = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1,
                        padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

        self.conv2 = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1,
                        padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

        self.conv3 = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1,
                        padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
