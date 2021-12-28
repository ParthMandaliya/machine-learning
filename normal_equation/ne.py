import numpy as np

class NormalEquation:

    def __init__(self, alpha=0.0):
        self.__alpha = alpha

    def __compute(self, x, y):
        try:
            '''
            # multiline code
            if self.__alpha > 0.0:
                identity_mat = np.identity(x.shape[1])
                alpha_identity = np.dot(self.__alpha,identity_mat)

            var = np.dot(x.T,x)
            if self.__alpha > 0.0:
                var =  np.add(var, alpha_identity)
            var = np.linalg.inv(var)
            var = np.dot(var,x.T)
            var = np.dot(var,y)
            self.__thetas = var
            '''
            if self.__alpha > 0.0:
                identity_mat = np.identity(x.shape[1])
                alpha_identity = np.dot(self.__alpha,identity_mat)

            var = np.dot(x.T,x)
            if self.__alpha > 0.0:
                var =  np.add(var, alpha_identity)
            var = np.linalg.inv(var)
            var = np.dot(var,x.T)
            var = np.dot(var,y)
            self.__thetas = var
            # one line code
#             self.__thetas = np.dot(np.dot(np.linalg.inv(np.add(np.dot(x.T,x),np.dot(self.__alpha,identity_mat))),x.T),y)
        except Exception as e:
            raise e

    def fit(self, x, y):
        x = np.array(x)
        ones_ = np.ones(x.shape[0])
        x = np.c_[ones_,x]
        y = np.array(y)
        self.__compute(x,y)

    @property
    def intercept_(self):
        return self.__thetas[0]

    @property
    def coef_(self):
        return self.__thetas[1:]
    
    def score(self, x, y):
        y_pred = self.predict(x)
        u = ((y - y_pred) **2).sum()
        v = ((y - y.mean()) **2).sum()
        return (1-(u/v))

    def predict(self, x):
        try:
            x = np.array(x)
            ones_ = np.ones(x.shape[0])
            x = np.c_[ones_,x]
            result = np.dot(x,self.__thetas)
            return result
        except Exception as e:
            raise e