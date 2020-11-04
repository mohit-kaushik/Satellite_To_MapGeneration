import torch

class Decoder_Block(torch.nn.Module):
  def __init__(self, in_filters, out_filters, kernel_size=4, stride=2, padding=1, apply_dropout=True, apply_batch_norm=True):
    super(Decoder_Block ,self).__init__()
    self.conv_trans = torch.nn.ConvTranspose2d(in_filters, out_filters, kernel_size, stride, padding)
    self.batch_norm = torch.nn.BatchNorm2d(out_filters)
    self.dropout = torch.nn.Dropout(0.5)
    self.apply_batch_norm = apply_batch_norm
    self.activation_step = torch.nn.ReLU()
    self.apply_dropout = apply_dropout

    # initialize weights
    self.initialize_weights()

  def initialize_weights(self):
    for layer in self.modules():
      if isinstance(layer, torch.nn.Conv2d):
        # mean = 0.0, std=0.2
        torch.nn.init.normal(layer.weight, 0.0, 0.2)

  def forward(self, input):
    # convetionally batch_norm is used after activation. But in case of relu its vice versa.
    x = self.conv_trans(input)
    if self.apply_batch_norm == True:
      x = self.batch_norm(x)
    
    x = self.activation_step(x)
    
    if self.apply_dropout == True:
      x = self.dropout(x)

    return x

