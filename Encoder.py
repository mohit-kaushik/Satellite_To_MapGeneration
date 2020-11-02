import torch

class Encoder_Block(torch.nn.Module):
  def __init__(self, in_filters, out_filters, kernel_size=4, stride=2, padding=1, apply_act=True, apply_batch_norm=True):
    super(Encoder_Block ,self).__init__()
    self.conv_step = torch.nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding)
    self.activation_step = torch.nn.LeakyReLU(0.2)

    # argument is equal to C in NCHW, i.e channels after conv2d
    self.batch_norm = torch.nn.BatchNorm2d(out_filters)
    
    # applying batch norm and activation optionally
    self.apply_activation = apply_act
    self.apply_batch_norm = apply_batch_norm

    # initialize weights
    self.initialize_weights()

  def initialize_weights(self):
    for layer in self.modules():
      if isinstance(layer, torch.nn.Conv2d):
        # mean = 0.0, std=0.2
        torch.nn.init.normal(layer.weight, 0.0, 0.2)
  
  def forward(self, input):
    x = self.conv_step(input)
    if self.apply_batch_norm == True:
      x = self.batch_norm(x)

    if self.apply_activation == True:
      x = self.activation_step(x)
    
    return x

