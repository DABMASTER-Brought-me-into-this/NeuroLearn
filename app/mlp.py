import os
import numpy as np
# My MLP Classes
# HyperParameters
### HYPERPARAMETERS
MAX_WORD_LEN = 30
BN_EMA_SF = 0.95
LR = 0.1
N_EMBD = 4
NUM_OF_NEURONS_L1 = 32
NUM_OF_NEURONS_L2 = 32
NUM_OF_NEURONS_L3 = 8

# String Integer Mapping
stoi = {chr(i): i - ord('a') + 1 for i in range(ord('a'), ord('z')+1)}
stoi['.'] = 0 # Padding Char
itos = {ix: char for char, ix in stoi.items()}

# Tokenization Function
def tok_word(word):
  # Array of chars
  letters = list(word)
  enc_letter = lambda letter: stoi[letter] if letter in stoi.keys() else 0
  enc_word = list(map(enc_letter, letters))

  # Padding the Word
  enc_word_len = len(enc_word)
  if enc_word_len < MAX_WORD_LEN:
    enc_word.extend([0] * (MAX_WORD_LEN - enc_word_len))

  return enc_word

# Classes for Neural Network Micrograd Style
class Embedding:
  def __init__ (self, size, n_embd):
    self.n_embd = n_embd
    self.C = np.random.randn(*size, n_embd)
    self.dC = np.zeros_like(self.C)

  def __call__ (self, X):
    self.X = X
    self.out = self.C[self.X]
    return self.out

  def _backwards(self, dup):
    np.add.at(self.dC, self.X.ravel(), dup.reshape(-1, self.n_embd))
    self.dout = dup
    return self.dout

  def _parameters(self):
    return [(self.C, self.dC)]

  def _wipegrad(self):
    self.dC = np.zeros_like(self.C)

class Reshape:
  def __init__ (self, shape):
    self.shape = shape

  def __call__ (self, X):
    self.X = X
    self.out = self.X.reshape(self.shape)
    return self.out

  def _backwards(self, dup):
    self.dout = dup.reshape(self.X.shape)
    return self.dout

  def _parameters(self):
    return []

  def _wipegrad(self):
    pass

class Linear:
  def __init__ (self, size):
    self.W = np.random.randn(*size) * 0.01
    self.B = np.zeros(size[-1])
    self.dW = np.zeros_like(self.W)
    self.dB = np.zeros_like(self.B)

  def __call__ (self, X):
    self.X = X
    self.out = self.X @ self.W + self.B
    return self.out

  def _backwards (self, dup):
    # print(f"deriv of up grad shape: {dup.shape}")
    # print(f"input shape: {self.X.shape}")
    # print(f"weight shape: {self.W.shape}") # I could figure this out, but j for ezier access
    # print(f"bias shape: {self.B.shape}") # same reason
    self.dW += self.X.T @ dup
    self.dB += dup.sum(axis = (tuple(range(self.B.ndim))))
    self.dout = dup @ self.W.T
    return self.dout # The issue driving me insane was I forgot this ONE LINE

  def _parameters(self):
    return [(self.W, self.dW), (self.B, self.dB)]

  def _wipegrad (self):
    self.dW = np.zeros_like(self.W)
    self.dB = np.zeros_like(self.B)

class BatchNorm:
  def __init__(self, size, training=True):
    self.training = training
    self.g = np.ones((size))
    self.b = np.zeros((size))
    self.dg = np.zeros_like(self.g)
    self.db = np.zeros_like(self.b)
    self.running_mean = None
    self.running_var = None

  def __call__ (self, X):
    if self.training:
      # Calculate New Stats
      self.xmean = np.mean(X, axis = 0, keepdims = True)
      self.xvar = np.var(X, axis = 0, keepdims = True)

      # Assign First Run to First Means/Vars
      if self.running_mean is None and self.running_var is None:
        self.running_mean = self.xmean
        self.running_var = self.xvar
      else:
        self.running_mean = BN_EMA_SF * self.running_mean + (1 - BN_EMA_SF) * self.xmean
        self.running_var = BN_EMA_SF * self.running_var + (1 - BN_EMA_SF) * self.xvar
    else:
      self.xmean = self.running_mean
      self.xvar = self.running_var

    self.X = X # Saving Input for BP
    # Batch Norm
    self.raw = (self.X - self.xmean)/np.sqrt(self.xvar + 1e-5)
    self.out = self.g * self.raw + self.b
    return self.out

  def _backwards(self, dup):
    # print(f"Input Type: {type(self.X)}, {self.X.shape}")
    # print(f"Deriv of Up Grad Type: {type(dup)}, {dup.shape}")
    # print(f"Gamma Shape: {self.g.shape}")
    # print(dup)
    self.dg = np.sum(dup * self.raw, axis = tuple(range(0, self.g.ndim, 1)))
    # print(f"Input Gamma (b4 sum): {(dup * self.X).shape}")
    # print(tuple(range(0, self.g.ndim, 1)))
    self.db = np.sum(dup, axis = tuple(range(0, self.b.ndim, 1)))
    # print(f"Input Beta (b4 sum): {(dup).shape}")
    # print(tuple(range(0, self.b.ndim, 1)))
    self.dout = ((dup * self.g) - (dup * self.g).mean(axis=0, keepdims=True) - self.raw * ((dup * self.g) * self.raw).mean(axis=0, keepdims=True))/(np.sqrt(self.xvar + 1e-5))
    return self.dout

  def _parameters(self):
    return [(self.g, self.dg), (self.b, self.db)]

  def _wipegrad(self):
    self.dg = np.zeros_like(self.g)
    self.db = np.zeros_like(self.b)

class LeakyReLu:
  def __call__ (self, X):
    self.X = X # saved for bp
    self.out = np.maximum(0.01 * self.X, self.X)
    return self.out

  def _backwards(self, dup):
    # print(dup)
    self.dout = np.where(self.X >= 0, dup, dup * 0.01)
    return self.dout

  def _parameters(self):
    return []

  def _wipegrad(self):
    pass

class Sigmoid:
  def __call__ (self, X):
    self.X = X # svd for bp
    # print(self.X)
    self.out = (1 + np.e ** -(self.X) + 1e-5) ** -1
    return self.out

  def _backwards(self, dup):
    self.dout = ((np.e ** -self.X) * (1 + np.e ** -(self.X)) ** -2) * dup
    return self.dout

  def _parameters(self):
    return []

  def _wipegrad(self):
    pass

class Model:
  def __init__(self, layers, lr):
    self.layers = layers
    self.lr = lr

  def __call__(self, X):
    self.X = X
    self.out = self.X
    for layer in self.layers:
      self.out = layer(self.out)
      # print(f"{layer.__class__.__name__} {self.layers.index(layer)}: {self.out.mean()}")
    return self.out

  def backwards(self, dup):
    self.dout = dup
    for layer in reversed(self.layers):
      # print(f"B4 {layer.__class__.__name__}: {self.dout}")
      self.dout = layer._backwards(self.dout)
      # print(f"After {layer.__class__.__name__}: {self.dout}")

  def update_parameters(self):
    parameters = []
    for layer in self.layers:
      parameters.extend(layer._parameters())

      # print(layer.__class__.__name__)
      # if len(layer._parameters()) > 0:
      #   for parameter, gradient in layer._parameters():
      #     print(parameter.shape, gradient.shape)

    for p,g in parameters:
      p -= g * self.lr

  def wipegrad(self):
    for layer in self.layers:
      layer._wipegrad()

  def eval_mode(self, mode):
    """
    eval_mode this is to change the model from evaluation to training
    mode: False is evaluation mode. True is training mode.
    """
    # Setting to Eval Mode
    for layer in self.layers:
      if hasattr(layer, 'training'):
        layer.training = mode

  def save_parameters(self, file_name):
    # Grabbing the parameters
    parameters = []
    for layer in self.layers:
      parameters.extend(layer._parameters())

    # Grabbbing the running variables
    running_vars = []
    for layer in self.layers:
      if hasattr(layer, 'training'):
        running_vars.extend([layer.running_mean, layer.running_var])

    # Extracting only the parameters
    para, grad = zip(*parameters)
    # Getting the running variables
    all_para = para + tuple(running_vars)
    np.savez(f'{file_name}.npz', *all_para) # Saving it

  def load_parameters(self, file_name):
    # Setting The Model to Eval
    self.eval_mode(False)

    # Finding the # of Batch Norm Layers
    num_run_var = 0
    for layer in self.layers:
      if layer.__class__.__name__ == 'BatchNorm':
        num_run_var += 2

    # Loading the Data
    data = np.load(f'{file_name}.npz')
    data = list(data.values())

    # Putting em in weights & running_vars
    weights = []
    emas = []
    for weight in data[:-num_run_var]:
      weights.append(weight)
    for ema in data[-num_run_var:]:
      emas.append(ema)

    # Replacing the weights in the model
    for layer in self.layers:
      if hasattr(layer, 'C'):
        layer.C = weights[0]
        weights.pop(0)
      if hasattr(layer, 'W'):
        layer.W = weights[0]
        weights.pop(0)
      if hasattr(layer, 'B'):
        layer.B = weights[0]
        weights.pop(0)
      if hasattr(layer, 'g'):
        layer.g = weights[0]
        weights.pop(0)
      if hasattr(layer, 'b'):
        layer.b = weights[0]
        weights.pop(0)

    # Re-implementing the running variable
    for layer in self.layers:
      if hasattr(layer, 'training'):
        layer.running_mean = emas[0]
        layer.running_var = emas[1]
        emas.pop(0)
        emas.pop(0)

layers = [Embedding((len(stoi),), N_EMBD), Reshape((-1, MAX_WORD_LEN * N_EMBD)),
          Linear((MAX_WORD_LEN * N_EMBD, NUM_OF_NEURONS_L1)), BatchNorm((NUM_OF_NEURONS_L1)), LeakyReLu(),
          Linear((NUM_OF_NEURONS_L1, NUM_OF_NEURONS_L2)), BatchNorm((NUM_OF_NEURONS_L2)), LeakyReLu(),
          Linear((NUM_OF_NEURONS_L2, NUM_OF_NEURONS_L3)), BatchNorm((NUM_OF_NEURONS_L3)), LeakyReLu(),
          Linear((NUM_OF_NEURONS_L3, 1))]

nn = Model(layers, LR)
current_dir = os.path.dirname(__file__)
nn.load_parameters(os.path.join(current_dir, 'NL-WordClassifierMLP.npz'))