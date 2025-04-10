import torch

class LinearModel:
    
    def __init__(self, w):
        self.w = w

    def score(self, X): 
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))

        return self.w @ X
    
    def predict(self, X):
        y_hat = self.score(X) > 0
        return y_hat.float()
    

#class LogisticRegression(LinearModel):
