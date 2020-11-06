import torch

# Encoder
# C64-C128-C256-C512-C512-C512-C512-C512

# Decoder
# CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
class Generator_Unet(torch.nn.Module):
  def __init__(self, input_channels=3, out_channels=3):
    super(Generator_Unet ,self).__init__()
    # Define layers
    self.conv1 = Encoder_Block(input_channels, 64, apply_act=False, apply_batch_norm=False) # 512
    self.conv2 = Encoder_Block(64, 128) # 256
    self.conv3 = Encoder_Block(128, 256) # 128
    self.conv4 = Encoder_Block(256, 512) # 64
    self.conv5 = Encoder_Block(512, 512) # 32
    self.conv6 = Encoder_Block(512, 512) # 16
    self.conv7 = Encoder_Block(512, 512) # 8
    self.conv8 = Encoder_Block(512, 512, apply_batch_norm=False) #4 -> 2

    self.deconv1 = Decoder_Block(512, 512, apply_dropout=True) # +512 concat
    self.deconv2 = Decoder_Block(1024, 512, apply_dropout=True)
    self.deconv3 = Decoder_Block(1024, 512, apply_dropout=True)
    self.deconv4 = Decoder_Block(1024, 512)
    self.deconv5 = Decoder_Block(1024, 256)
    self.deconv6 = Decoder_Block(512, 128)
    self.deconv7 = Decoder_Block(256, 64)
    self.deconv8 = Decoder_Block(128, out_channels)

    self.activation = torch.nn.Tanh()

  def forward(self, input):
    x1 = self.conv1(input)
    x2 = self.conv2(x1)
    x3 = self.conv3(x2)
    x4 = self.conv4(x3)
    x5 = self.conv5(x4)
    x6 = self.conv6(x5)
    x7 = self.conv7(x6)
    x8 = self.conv8(x7)

    decoded_x1 = self.deconv1(x8)
    # pytorch accepts tensor as NCHW(0123), concatenate channel wise, dim=1
    
    decoded_x1 = torch.cat([decoded_x1, x7], dim=1)

    decoded_x2 = self.deconv2(decoded_x1)
    decoded_x2 = torch.cat([decoded_x2, x6], dim=1)

    decoded_x3 = self.deconv3(decoded_x2)
    decoded_x3 = torch.cat([decoded_x3, x5], dim=1)

    decoded_x4 = self.deconv4(decoded_x3)
    decoded_x4 = torch.cat([decoded_x4, x4], dim=1)

    decoded_x5 = self.deconv5(decoded_x4)
    decoded_x5 = torch.cat([decoded_x5, x3], dim=1)

    decoded_x6 = self.deconv6(decoded_x5)
    decoded_x6 = torch.cat([decoded_x6, x2], dim=1)

    decoded_x7 = self.deconv7(decoded_x6)
    decoded_x7 = torch.cat([decoded_x7, x1], dim=1)

    decoded_x8 = self.deconv8(decoded_x7)

    x = self.activation(decoded_x8)
    return x

