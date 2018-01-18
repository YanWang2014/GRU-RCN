import torch
import torch.nn as nn


__all__ = ['RCN', 'RCNCell', 'StackedRCNCell', 'ModifiedRCNCell',
           'GRURCNCellBase', 'ConvGRURCNCell', 'BottleneckGRURCNCell', 
           'Bottleneck']


class RCN(nn.Module):

    def __init__(self, cell):
        super(RCN, self).__init__()
        if not isinstance(cell, RCNCell):
            raise ValueError('cell must be an instance of RCNCell')
        self._cell = cell

    def forward(self, x, h0):
        out, h = self._cell(x[0, :], h0)
        out_seq = [out, ]
        for t in range(1, x.size(0)):
            out, h = self._cell(x[0, :], h)
            out_seq.append(out)
        return torch.stack(out_seq), h


class RCNCell(nn.Module):
    
    def forward(self, x, hidden):
        raise NotImplementedError("function forward(self, x, hidden) \
        is not implemented")


class StackedRCNCell(RCNCell):

    def __init__(self, cells, indices_of_rcn=None):
        super(StackedRCNCell, self).__init__()
        if indices_of_rcn is None:
            indices_of_rcn = [i for i in range(len(cells)) if isinstance(cells[i], RCNCell)]
        self._isrcn = [False, ] * len(cells)
        for i in indices_of_rcn:
            self._isrcn[i] = True
        self._cells = cells

    def forward(self, x, hidden):
        if len(hidden) != len(self._ircn):
            raise ValueError("The seq_len of hidden states does not match the number of RCNCells")
        h = []
        for cell, isrcn in zip(self._cells, self._isrcn):
            if isrcn:
                x, _h = cell(x, hidden[len(h)])
                h.append(_h)
            else:
                x = cell(x)
        return x, h


class _Wrapper(nn.Module):

    def __init__(self, rcn_cell):
        super(_Wrapper, self).__init__()
        if not isinstance(rcn_cell, RCNCell):
            raise ValueError('not an instance of class RCNCell')
        self._cell = rcn_cell
        self.hidden = None

    def forward(self, x):
        if self.hidden is None:
            raise RuntimeError('hidden state has not been set yet')
        out, hidden = self._cell(x, self.hidden)
        self.hidden = hidden
        return out


class ModifiedRCNCell(RCNCell):

    def __init__(self, model):
        super(ModifiedRCNCell, self).__init__()
        self._model = model
        self._rcn_cells = []

    def modify(self, key_seq, rcn_cell):
        unit = self._model
        if isinstance(key_seq, (list, tuple)):
            for key in key_seq[:-1]:
                unit = unit._modules[key]
            key = key_seq[-1]
        else:
            key = key_seq
        new_unit = _Wrapper(rcn_cell)
        unit._modules[key] = new_unit
        self._rcn_cells.append(new_unit)

    def forward(self, x, hidden):
        if len(self._rcn_cells) != len(hidden):
            raise ValueError('size of hidden does not match this module')
        for rcn_cell, h in zip(self._rcn_cells, hidden):
            rcn_cell.hidden = h
        x = self._model(x)
        new_h = [cell.hidden for cell in self._rcn_cells]
        return x, new_h


class GRURCNCellBase(RCNCell):

    def __init__(self, xz, hz, xr, hr, xh, rh):
        super(GRURCNCellBase, self).__init__()
        self._xz = xz
        self._hz = hz
        self._xr = xr
        self._hr = hr
        self._xh = xh
        self._rh = rh

    def forward(self, x, hidden):
        if hidden == None:
            h = torch.sigmoid(self._xz(x)) * torch.tanh(self._xh(x))
            return h, h
        else:
            z = torch.sigmoid(self._xz(x) + self._hz(hidden))
            r = torch.sigmoid(self._xr(x) + self._hr(hidden))
            h_ = torch.tanh(self._xh(x) + self._rh(r * hidden))
            h = (1 - z) * hidden + z * h_
            return h, h


class ConvGRURCNCell(RCNCell):
    r"""GRU-RCN cell using the conventional covolution as the inner operation.

    The convolutions on the hidden states must keep the size unchanged, so 
    we do not provide the arguments to set the stride or padding of these 
    convolutions, but fix stride=1 and padding=(kernel_size - 1) / 2).

    The convolutions on the inputs must make the outputs has the same size as
    hidden states. If the sizes of inputs and hidden states are not the same,
    users must provide the arguments (x_channels, x_kernel_size, x_stride, 
    x_padding) to insure the outputs after the convolutions on inputs to be the 
    same as the hidden states.

    Args: channels, kernel_size, ...
        - **channels**: the number of channels of hidden states. If x_channels 
        is None, it means the channels of inputs and hidden states are the same.
        - **kernel_size**: size of the convolving kernel.
        - **x_channels**, **x_kernel_size**, **x_stride**, **x_padding**:
            parameters of the convolution operation on inputs.
            If None, inputs and hidden states have the same sizes, so the
            convolutions on them have no difference.

    Inputs: x (input), hidden
        - **x** (batch, channel, height, width): tensor containing input features.
        - **hidden** (batch, channel, height, width): tensor containing the hidden
            state for the precious time instance.

    Outputs: output, h
        - **output**: same as h
        - **h**: (batch, hidden_size): tensor containing the next hidden state
    """

    def __init__(self, channels, kernel_size, x_channels=None,
                 x_kernel_size=None, x_stride=None, x_padding=None):
        super(ConvGRURCNCell, self).__init__()

        x_channels = x_channels or channels
        x_kernel_size = x_kernel_size or kernel_size
        x_stride = x_stride or 1
        x_padding = x_padding or ((kernel_size - 1) // 2)

        hz = nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2)
        hr = nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2)
        rh = nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2)

        xz = nn.Conv2d(x_channels, channels, x_kernel_size, x_stride, x_padding)
        xr = nn.Conv2d(x_channels, channels, x_kernel_size, x_stride, x_padding)
        xh = nn.Conv2d(x_channels, channels, x_kernel_size, x_stride, x_padding)

        self._cell = GRURCNCellBase(xz, hz, xr, hr, xh, rh)

    def forward(self, x, hidden):
        return self._cell(x, hidden)


class BottleneckGRURCNCell(RCNCell):
    r"""GRU-RCN cell using a bottleneck block as the inner operation.

    If the sizes of input and hidden state are not the same, users must specify 
    the for channel number and stride for the input-to-hidden convolution, to 
    make output has the same size as the hidden state.

    Args: channels, x_channels, x_stride
        - **channels**: the number of channels of hidden states. If x_channels is 
        None, it means the channels of inputs and hidden states are the same.
        - **x_channels**, **x_stride**:
            parameters of the inputs-to-hidden convolutions.
            If None, inputs and hidden states have the same sizes, so the
            convolutions on them have no difference.

    Inputs: x (input), hidden
        - **x** (batch, channel, height, width): tensor containing input features.
        - **hidden** (batch, channel, height, width): tensor containing the hidden
            state for the precious time instance.

    Outputs: h
        - **h**: (batch, hidden_size): tensor containing the next hidden state
    """

    def __init__(self, channels, x_channels=None, expansion=4, x_stride=None, batch_norm=True):
        super(BottleneckGRURCNCell, self).__init__()

        x_channels = x_channels or channels
        x_stride = x_stride or 1

        hz = Bottleneck(channels, channels, expansion=expansion)
        hr = Bottleneck(channels, channels, expansion=expansion)
        rh = Bottleneck(channels, channels, expansion=expansion)

        xz = Bottleneck(x_channels, channels, expansion=expansion, stride=x_stride)
        xr = Bottleneck(x_channels, channels, expansion=expansion, stride=x_stride)
        xh = Bottleneck(x_channels, channels, expansion=expansion, stride=x_stride)

        self._cell = GRURCNCellBase(xz, hz, xr, hr, xh, rh)

    def forward(self, x, hidden):
        return self._cell(x, hidden)


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, expansion=4, stride=1, batch_norm=True):
        super(Bottleneck, self).__init__()
        planes = out_channels // expansion
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, bias=not batch_norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=not batch_norm)
        self.conv3 = nn.Conv2d(planes, out_channels, kernel_size=1)
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = batch_norm

    def forward(self, x):
        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        return out


'''
# test
if __name__ == '__main__':
    gru_rcn = BottleneckGRURCNCell(64)
    print(gru_rcn)

    from torch.autograd import Variable
    x = Variable(torch.rand(10, 64, 32, 32))
    h = Variable(torch.rand(10, 64, 32, 32))
    gru_rcn(x, h)

    gru_rcn = BottleneckGRURCNCell(64, 32, 2)
    print(gru_rcn)

    x = Variable(torch.rand(10, 32, 64, 64))
    h = Variable(torch.rand(10, 64, 32, 32))
    gru_rcn(x, h)
'''