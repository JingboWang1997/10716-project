import numpy as npy
from PIL import Image

# -------------------------------
# Weight Initialisation
# -------------------------------

def initialise_weight(in_channel, out_channel):
    """
    """
    W = np.random.randn(in_channel, out_channel).astype(np.float32) * np.sqrt(2.0/(in_channel))
    return W


def initialise_bias(out_channel):
    """
    """
    b = np.zeros(out_channel).astype(np.float32)
    return b


# -------------------------------
# Loss Functions
# -------------------------------

def BCELoss(x, y, derivative=False):
    """
    """
    def _BCE_loss_forward(x, y):
        loss = np.sum(- y * np.log(x + eps) + - (1 - y) * np.log((1 - x) + eps))
        return loss

    def _BCE_loss_derivative(x, y):
        dloss = -y * (1 / (x + eps))
        return dloss
    
    if derivative:
        return _BCE_loss_derivative(x, y)
    else:
        return _BCE_loss_forward(x, y)


def MSELoss(x, y, derivative=False):
    """
    """
    def _MSE_loss_forward(x, y):
        loss = (np.square(y - x)).mean()
        return loss

    def _MSE_loss_derivative(x, y):
        dloss = 2 * (x - y)
        return dloss
    
    if derivative:
        return _MSE_loss_derivative(x, y)
    else:
        return _MSE_loss_forward(x, y)


# -------------------------------
# Activation Functions
# -------------------------------

def sigmoid(x, derivative=False):
    res = 1/(1+np.exp(-x))
    if derivative:
        return res*(1-res)
    return res

def relu(x, derivative=False):
    res = x
    if derivative:
        return 1.0 * (res > 0)
    else:
        return res * (res > 0)   
    
def lrelu(x, alpha=0.01, derivative=False):
    res = x
    if derivative:
        dx = np.ones_like(res)
        dx[res < 0] = alpha
        return dx
    else:
        return np.maximum(x, x*alpha, x)

def tanh(x, derivative=False):
    res = np.tanh(x)
    if derivative:
        return 1.0 - np.tanh(x) ** 2
    return res
  
eps = 10e-8


class Encoder(object):
    def __init__(self, input_channels, layer_size, nz, 
                batch_size=64, lr=1e-3, beta1=0.9, beta2=0.999):
        """
        """
        self.input_channels = input_channels
        self.nz = nz

        self.batch_size = batch_size
        self.layer_size = layer_size

        # Initialise encoder weight
        self.W0 = initialise_weight(self.input_channels, self.layer_size)
        self.b0 = initialise_bias(self.layer_size)

        self.W_mu = initialise_weight(self.layer_size, self.nz)
        self.b_mu = initialise_bias(self.nz)

        self.W_logvar = initialise_weight(self.layer_size, self.nz)
        self.b_logvar = initialise_bias(self.nz)

        # Adam optimiser momentum and velocity
        self.lr = lr
        self.momentum = [0.0] * 6
        self.velocity = [0.0] * 6
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0

    def forward(self, x):
        """
        """
        self.e_input = x.reshape((self.batch_size, -1))
        
        # Dimension check on input
        assert self.e_input.shape == (self.batch_size, self.input_channels)

        self.h0_l = self.e_input.dot(self.W0) + self.b0
        self.h0_a = relu(self.h0_l)

        self.logvar = self.h0_a.dot(self.W_logvar) + self.b_logvar
        self.mu = self.h0_a.dot(self.W_mu) + self.b_mu

        self.rand_sample = np.random.standard_normal(size=(self.batch_size, self.nz))
        self.sample_z = self.mu + np.exp(self.logvar * .5) * self.rand_sample

        return self.sample_z, self.mu, self.logvar

    def optimise(self, grads):
        """
        """
        # ---------------------------
        # Optimise using Adam
        # ---------------------------
        self.t += 1
        # Calculate gradient with momentum and velocity
        for i, grad in enumerate(grads):
            self.momentum[i] = self.beta1 * self.momentum[i] + (1 - self.beta1) * grad
            self.velocity[i] = self.beta2 * self.velocity[i] + (1 - self.beta2) * np.power(grad, 2)
            m_h = self.momentum[i] / (1 - (self.beta1 ** self.t))
            v_h = self.velocity[i] /  (1 - (self.beta2 ** self.t))
            grads[i] = m_h / np.sqrt(v_h + eps)

        grad_W0, grad_b0, grad_W_mu, grad_b_mu, grad_W_logvar, grad_b_logvar = grads

        # Update weights
        self.W0 = self.W0 - self.lr * np.sum(grad_W0, axis=0)
        self.b0 = self.b0 - self.lr * np.sum(grad_b0, axis=0)
        self.W_mu = self.W_mu - self.lr * np.sum(grad_W_mu, axis=0)
        self.b_mu = self.b_mu - self.lr * np.sum(grad_b_mu, axis=0)
        self.W_logvar = self.W_logvar - self.lr * np.sum(grad_W_logvar, axis=0)
        self.b_logvar = self.b_logvar - self.lr * np.sum(grad_b_logvar, axis=0)

        return

    def backward(self, x, grad_dec):
        """
        """
        # ----------------------------------------
        # Calculate gradients from reconstruction
        # ----------------------------------------
        y = np.reshape(x, (self.batch_size, -1))

        db_mu = grad_dec 
        dW_mu = np.matmul(np.expand_dims(self.h0_a, axis=-1), np.expand_dims(grad_dec, axis=1))

        db_logvar = grad_dec * np.exp(self.logvar * .5) * .5 * self.rand_sample
        dW_logvar = np.matmul(np.expand_dims(self.h0_a, axis=-1), np.expand_dims(db_logvar, axis=1))

        drelu = relu(self.h0_l, derivative=True)

        db0 = drelu * (db_mu.dot(self.W_mu.T) + db_logvar.dot(self.W_logvar.T))
        dW0 = np.matmul(np.expand_dims(y, axis=-1), np.expand_dims(db0, axis=1))

        # ----------------------------------------
        # Calculate gradients from K-L
        # ----------------------------------------
        # logvar terms
        dKL_b_logvar = .5 * (np.exp(self.logvar) - 1)
        dKL_W_logvar = np.matmul(np.expand_dims(self.h0_a, axis=-1), np.expand_dims(dKL_b_logvar, axis=1))

        # mu terms
        dKL_b_mu = .5 * 2 * self.mu
        dKL_W_mu = np.matmul(np.expand_dims(self.h0_a, axis=-1), np.expand_dims(dKL_b_mu, axis=1))

        dKL_b0 = drelu * (dKL_b_logvar.dot(self.W_logvar.T) + dKL_b_mu.dot(self.W_mu.T))
        dKL_W0 = np.matmul(np.expand_dims(y, axis=-1), np.expand_dims(dKL_b0, axis=1))

        # Combine gradients for encoder from recon and KL
        grad_b_logvar = dKL_b_logvar + db_logvar
        grad_W_logvar = dKL_W_logvar + dW_logvar
        grad_b_mu = dKL_b_mu + db_mu
        grad_W_mu = dKL_W_mu + dW_mu
        grad_b0 = dKL_b0 + db0
        grad_W0 = dKL_W0 + dW0

        grads = [grad_W0, grad_b0, grad_W_mu, grad_b_mu, grad_W_logvar, grad_b_logvar]

        # Optimise step
        self.optimise(grads)

        return
      
eps = 10e-8


class Decoder(object):
    def __init__(self, input_channels, layer_size, nz, 
                batch_size=64, lr=1e-3, beta1=0.9, beta2=0.999):
        
        self.input_channels = input_channels
        self.nz = nz

        self.batch_size = batch_size
        self.layer_size = layer_size

        # Initialise decoder weight
        self.W0 = initialise_weight(self.nz, self.layer_size)
        self.b0 = initialise_bias(self.layer_size)

        self.W1 = initialise_weight(self.layer_size, self.input_channels)
        self.b1 = initialise_bias(self.input_channels)

        # Adam optimiser momentum and velocity
        self.lr = lr
        self.momentum = [0.0] * 4
        self.velocity = [0.0] * 4
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0

    def forward(self, z):
        self.z = np.reshape(z, (self.batch_size, self.nz))

        self.h0_l = self.z.dot(self.W0) + self.b0
        self.h0_a = relu(self.h0_l)

        self.h1_l = self.h0_l.dot(self.W1) + self.b1
        self.h1_a = sigmoid(self.h1_l)

        self.d_out = np.reshape(self.h1_a, (self.batch_size, self.input_channels))

        return self.d_out

    def optimise(self, grads):
        """
        """
        # ---------------------------
        # Optimise using Adam
        # ---------------------------
        self.t += 1
        # Calculate gradient with momentum and velocity
        for i, grad in enumerate(grads):
            self.momentum[i] = (1 - self.beta1) * grad
            self.velocity[i] = self.beta2 * self.velocity[i] + (1 - self.beta2) * np.power(grad, 2)
            m_h = self.momentum[i] / (1 - (self.beta1 ** self.t))
            v_h = self.velocity[i] /  (1 - (self.beta2 ** self.t))
            grads[i] = m_h / np.sqrt(v_h + eps)

        grad_dW0, grad_db0, grad_dW1, grad_db1 = grads

        # Update weights
        self.W0 = self.W0 - self.lr * np.sum(grad_dW0, axis=0)
        self.b0 = self.b0 - self.lr * np.sum(grad_db0, axis=0)
        self.W1 = self.W1 - self.lr * np.sum(grad_dW1, axis=0)
        self.b1 = self.b1 - self.lr * np.sum(grad_db1, axis=0)

        return

    def backward(self, x, out):
        # ----------------------------------------
        # Calculate gradients from reconstruction
        # ----------------------------------------
        y = np.reshape(x, (self.batch_size, -1))
        out = np.reshape(out, (self.batch_size, -1))

        dL = MSELoss(out, y, derivative=True)
        dSig = sigmoid(self.h1_l, derivative=True)

        dL_dSig = dL * dSig

        grad_db1 = dL_dSig
        grad_dW1 = np.matmul(np.expand_dims(self.h0_a, axis=-1), np.expand_dims(dL_dSig, axis=1))
        
        drelu0 = relu(self.h0_l, derivative=True)

        grad_db0 = grad_db1.dot(self.W1.T) * drelu0
        grad_dW0 = np.matmul(np.expand_dims(self.z, axis=-1), np.expand_dims(grad_db0, axis=1))

        # output gradient to the encoder layer
        grad_dec = grad_db0.dot(self.W0.T)

        grads = [grad_dW0, grad_db0, grad_dW1, grad_db1]

        # Optimiser Step
        self.optimise(grads)

        return grad_dec
      
class VariationalAutoEncoder(object):

    def __init__(self, input_channels, layer_size, nz, 
                batch_size=64, lr=1e-3, beta1=0.9, beta2=0.999):
        
        self.input_channels = input_channels
        self.nz = nz

        self.batch_size = batch_size
        self.layer_size = layer_size

        # Construct encoder module
        self.encoder = Encoder(input_channels, layer_size, nz, 
            batch_size=batch_size, lr=lr, beta1=beta1, beta2=beta2)
        
        # Construct decoder module
        self.decoder = Decoder(input_channels, layer_size, nz,
            batch_size=batch_size, lr=lr, beta1=beta1, beta2=beta2)
        
    def forward(self, x):
        """
        """
        x = x.reshape((self.batch_size, -1))

        # Feed forward encoder - decoder
        sample_z, mu, logvar = self.encoder.forward(x)
        out = self.decoder.forward(sample_z)

        return out, mu, logvar

    def backward(self, x, out):
        """
        """
        grad_dec = self.decoder.backward(x, out)
        self.encoder.backward(x, grad_dec)
