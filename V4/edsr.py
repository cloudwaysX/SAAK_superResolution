import torch
import torch.nn as nn
import math

class _Residual_Block(nn.Module): 
    def __init__(self):
        super(_Residual_Block, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, x): 
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output *= 0.1
        output = torch.add(output,identity_data)
        return output 

class Net(nn.Module):
    def __init__(self,in_channels=1,out_channels=73):
        super(Net, self).__init__()
        
        self.conv_input = nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.residual = self.make_layer(_Residual_Block, 32)

        self.conv_mid = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=256, out_channels=256*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

        self.conv_output = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_input(x)
        residual = out
        out = self.conv_mid(self.residual(out))
        out = torch.add(out,residual)
        out = self.upscale4x(out)
        out = self.conv_output(out)
        return out
 