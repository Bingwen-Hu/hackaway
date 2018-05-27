# reference to Tensorflow's implement
import numpy as np

def sigmoid_cross_entropy_loss(labels, logits):
    r"""used when objects is not mutually exclusive. For instance, 
    a picture can have a dog and a cat at the same time
    
    for brevity, let `x = logits`, `z = labels`. The logistic loss is
    
        z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
    
    equivalent formulation:

        max(x, 0)  -  x * z + log(1 + exp(-abs(x)))
    """
    zeros = np.zeros_like(logits, dtype=logits.dtype)
    cond = (logits >= zeros)
    relu_logits = np.where(cond, logits, zeros)             # max(x, 0)
    neg_abs_logits = np.where(cond, -logits, logits)        # -abs(x)
    return relu_logits - logits * labels + np.log1p(np.exp(neg_abs_logits))