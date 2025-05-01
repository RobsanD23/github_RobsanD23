# import torch

# class LinearModel:
    
#     def __init__(self, w =None):
#         self.w = w

#     def score(self, X): 
#         if self.w is None: 
#             self.w = torch.rand((X.size()[1]))

#         return X @ self.w
    
#     def predict(self, X):
#         y_hat = self.score(X) > 0
#         return y_hat.float()
    

# class LogisticRegression(LinearModel):
    
#     def __init__ (self, model = None):
#         if model is not None: 
#             # Copy weights from the linear model
#             self.w = model.w
#         else: 
#             self.w = None

#     def loss(self, X, y):
#         """
#         Compute the logistic loss. The logistic loss is defined as -y_i*log(p_i) - (1-y_i)*log(1-p_i), where p_i = 1/(1+exp(-s_i)). 
#         The final loss is the average of the losses over all data points. 

#         ARGUMENTS: 
#             X, torch.Tensor: the feature matrix. X.size() == (n, p), 
#             where n is the number of data points and p is the 
#             number of features. This implementation always assumes 
#             that the final column of X is a constant column of 1s. 

#         RETURNS: 
#             loss, float: the average logistic loss over all data points.
#         """
#         s = self.score(X)
#         sigma = 1/(1+torch.exp(-s))
#         loss = -torch.mean(y*torch.log(sigma) + (1-y)*torch.log(1-sigma))
#         return loss

    
#     def grad(self, X, y): 
#         s = self.score(X)
#         sigma = 1 / (1 + torch.exp(-s))
#         gradient = X.T @ (sigma - y) / X.size(0)
#         return gradient 
    
#     # def grad_for(self, X, y):
#     #     g = 0
#     #     for idx, x_i in enumerate(X):
#     #         s = self.score(x_i)
#     #         sigma = 1 / (1 + torch.exp(-s))
#     #         g += 
        


    

# class GradientDescentOptimizer(LogisticRegression):
    
#     def __init__(self, model):
#         self.model = model
#         self.w = model.w
#         self.w_prev = model.w

#     def step(self, X, y, alpha, beta):
#         temp = self.w
#         self.w = self.w - alpha * self.grad(X, y) + beta * (self.w - self.w_prev)
#         self.w_prev = temp
        



import torch

class LinearModel:
    def __init__(self, w=None):
        self.w = w

    def score(self, X): 
        if self.w is None: 
            self.w = torch.rand(X.size()[1])
        return X @ self.w  # (n, p) @ (p,) = (n,)

    def predict(self, X):
        y_hat = self.score(X) > 0
        return y_hat.float()
    

class LogisticRegression(LinearModel):
    def __init__(self, base_model=None):
        if base_model is not None:
            # Copy weights from the base model
            self.w = base_model.w
        else:
            self.w = None

    def loss(self, X, y):
        """Compute the logistic loss.
        args:
            X (torch.Tensor): Feature matrix.
            y (torch.Tensor): Target vector.
        returns:
            loss (torch.Tensor): Average logistic loss.
        """
        s = self.score(X)
        sigma = 1 / (1 + torch.exp(-s))
        loss = -torch.mean(y * torch.log(sigma) + (1 - y) * torch.log(1 - sigma))
        return loss
    
    def grad(self, X, y):
        """Compute the gradient of the logistic loss.
        args:
            X (torch.Tensor): Feature matrix.
            y (torch.Tensor): Target vector.
        returns:
            gradient (torch.Tensor): Gradient of the loss with respect to the weights.
        """
        s = self.score(X)
        sigma = 1 / (1 + torch.exp(-s))
        gradient = X.T @ (sigma - y) / X.size(0)
        return gradient


class GradientDescentOptimizer:
    def __init__(self, model):
        self.model = model
        if self.model.w is None:
            raise ValueError("Weight vector is None")
        self.w_prev = self.model.w

    def step(self, X, y, alpha, beta):
        """Perform a gradient descent step with momentum.
        Args:
            X (torch.Tensor): Feature matrix.
            y (torch.Tensor): Target vector.
            alpha (float): Learning rate.
            beta (float): Momentum coefficient.
        """
        temp = self.model.w
        grad = self.model.grad(X, y)
        self.model.w = self.model.w - alpha * grad + beta * (self.model.w - self.w_prev)
        self.w_prev = temp
