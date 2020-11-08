import torch

class Discriminator(torch.nn.Module):
  def __init__(self, in_channels, out_filters=1):
    super(Discriminiator, self).__init__()
    self.conv1 = Encoder_Block(in_channels, 64, apply_act=False, apply_batch_norm=False)
    self.conv2 = Encoder_Block(64, 128)
    self.conv3 = Encoder_Block(128, 256)
    self.conv4 = Encoder_Block(256, 512)
    self.conv5 = Encoder_Block(512, out_filters)
    self.activation = torch.nn.Sigmoid()
  
  def forward(self, x, orignal):
    x = torch.cat([x,orignal], 1)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = self.activation(x)

    return x

