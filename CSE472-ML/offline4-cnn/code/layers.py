import numpy as np
np.random.seed(0)


class ReLUActivationLayer:

    def __str__(self) -> str:
        return "ReLUActivationLayer"

    def forward(self, input):
        out = np.maximum(input, 0)
        return out

    def backward(self, output, learning_rate):
        return np.where(output > 0, 1, 0)


class MaxPoolingLayer:

    def __str__(self) -> str:
        return "MaxPoolingLayer"

    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
        self.cache = None

    def forward(self, input):

        self.input = input
        batch_size, n_channel, height, width = input.shape

        output_h = int((height - self.pool_size)/self.stride + 1)
        output_w = int((width  - self.pool_size)/self.stride + 1)

        output_shape = (batch_size, n_channel, output_h, output_w)
        output = np.zeros(output_shape)

        for b in range(batch_size):
            for c in range(n_channel):
                for h in range(output_h):
                    for w in range(output_w):
                        output[b, c, h, w] = np.max(input[b, :, h*self.stride :h*self.stride + self.pool_size, w*self.stride : w*self.stride + self.pool_size])

        return output

    def backward(self, output, learning_rate):
        batch_size, n_channel, height, width = output.shape
        input = np.zeros(self.input.shape)

        for b in range(batch_size):
            for c in range(n_channel):
                for h in range(height):
                    for w in range(width):
                        input[b, c, h*self.stride :h*self.stride + self.pool_size, w*self.stride : w*self.stride + self.pool_size] = np.where(input[b, c, h*self.stride :h*self.stride + self.pool_size, w*self.stride : w*self.stride + self.pool_size] == np.max(input[b, c, h*self.stride :h*self.stride + self.pool_size, w*self.stride : w*self.stride + self.pool_size]), output[b, c, h, w], 0)

        return input    


class FlatteningLayer:

    def __str__(self) -> str:
        return "FlatteningLayer"

    def __init__(self) -> None:
        self.input = None

    def forward(self, input):
        self.input = input
        return input.reshape(input.shape[0], -1)

    def backward(self, output, learning_rate):
        return output.reshape(self.input.shape)


class DenseLayer:

    def __str__(self) -> str:
        return "DenseLayer"

    def __init__(self, n_output):
        self.n_output = n_output
        self.weights = None
        self.biases = None
        self.input = None

    def forward(self, input):
        
        self.input = input        
        batch_size, n_input = input.shape

        if self.weights is None:
            self.weights = np.random.randn(n_input, self.n_output) / np.sqrt(n_input)

        if self.biases is None:
            self.biases = np.random.randn(self.n_output)

        output = np.dot(input, self.weights) + self.biases
        return output

    def backward(self, output, learning_rate):

        del_w = np.dot(np.transpose(self.input), output)
        del_b = np.sum(output, axis=0)
        del_u = np.dot(output, np.transpose(self.weights))

        self.weights -= learning_rate * del_w
        self.biases -= learning_rate * del_b
        return del_u


class SoftMaxLayer:

    def __str__(self) -> str:
        return "SoftMaxLayer"

    def forward(self, input):
        val = input - np.max(input, axis=1, keepdims=True)
        val = np.exp(val) / np.exp(val).sum(axis=1, keepdims=True)
        return val

    def backward(self, output, learning_rate):
        return output



class ConvolutionLayer:

    def __str__(self) -> str:
        return "ConvolutionLayer"

    def _get_indices(self, X_shape, HF, WF, stride, pad):

        batch_size, n_channel, height, width = X_shape

        out_h = int((height + 2 * pad - HF) / stride) + 1
        out_w = int((width + 2 * pad - WF) / stride) + 1
    
        # ----Compute matrix of index i----
        level1 = np.repeat(np.arange(HF), WF)
        level1 = np.tile(level1, n_channel)
        everyLevels = stride * np.repeat(np.arange(out_h), out_w)
        i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

        # ----Compute matrix of index j----
        slide1 = np.tile(np.arange(WF), HF)
        slide1 = np.tile(slide1, n_channel)

        everySlides = stride * np.tile(np.arange(out_w), out_h)
        j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)
        d = np.repeat(np.arange(n_channel), HF * WF).reshape(-1, 1)

        return i, j, d


    def _im2col(self, X, HF, WF, stride, pad):

        X_padded = np.pad(X, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')
        i, j, d = self._get_indices(X.shape, HF, WF, stride, pad)
    
        cols = X_padded[:, d, i, j]
        cols = np.concatenate(cols, axis=-1)

        return cols


    def _col2im(self, dX_col, X_shape, HF, WF, stride, pad):

        batch_size, n_channel, height, width = X_shape
        H_padded, W_padded = height + 2 * pad, width + 2 * pad
        X_padded = np.zeros((batch_size, n_channel, H_padded, W_padded))
        
        i, j, d = self._get_indices(X_shape, HF, WF, stride, pad)
        dX_col_reshaped = np.array(np.hsplit(dX_col, batch_size))
        np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)

        if pad == 0:
            return X_padded
        elif type(pad) is int:
            return X_padded[pad:-pad, pad:-pad, :, :]
    

    def __init__(self, n_filter, n_channel_in, kernel_size, stride=1, padding=0):
        self.n_filter = n_filter
        self.k_size = kernel_size
        self.n_channel_in = n_channel_in
        self.stride = stride
        self.padding = padding
        self.cache = None

        # Xavier-Glorot initialization - used for sigmoid, tanh.
        self.W = {
            'val': np.random.randn(self.n_filter, self.n_channel_in, self.k_size, self.k_size) * np.sqrt(1. / (self.k_size)),
            'grad': np.zeros((self.n_filter, self.n_channel_in, self.k_size, self.k_size))
        }
        self.b = {
            'val': np.random.randn(self.n_filter) * np.sqrt(1. / self.n_filter),
            'grad': np.zeros((self.n_filter))
        }

    def forward(self, X):
        batch_size, _, height, width = X.shape

        new_n_channel = self.n_filter
        new_height = int((height + 2 * self.padding - self.k_size)/ self.stride) + 1
        new_width = int((width + 2 * self.padding - self.k_size)/ self.stride) + 1
        
        X_col = self._im2col(X, self.k_size, self.k_size, self.stride, self.padding)
        w_col = self.W['val'].reshape((self.n_filter, -1))
        b_col = self.b['val'].reshape(-1, 1)

        # Perform matrix multiplication.
        out = w_col @ X_col + b_col

        # Reshape back matrix to image.
        out = np.array(np.hsplit(out, batch_size)).reshape((batch_size, new_n_channel, new_height, new_width))
        self.cache = X, X_col, w_col
        return out


    def backward(self, dout, learning_rate):

        X, X_col, w_col = self.cache
        batch_size, _, _, _ = X.shape
        self.b['grad'] = np.sum(dout, axis=(0,2,3))

        # Reshape dout properly.
        dout = dout.reshape(dout.shape[0] * dout.shape[1], dout.shape[2] * dout.shape[3])
        dout = np.array(np.vsplit(dout, batch_size))
        dout = np.concatenate(dout, axis=-1)

        # Perform matrix multiplication between reshaped dout and w_col to get dX_col.
        # Perform matrix multiplication between reshaped dout and X_col to get dW_col.
        dX_col = w_col.T @ dout
        dw_col = dout @ X_col.T

        # Reshape back to image (col2im).
        dX = self._col2im(dX_col, X.shape, self.k_size, self.k_size, self.stride, self.padding)
        self.W['grad'] = dw_col.reshape((dw_col.shape[0], self.n_channel_in, self.k_size, self.k_size))                
        return dX
    