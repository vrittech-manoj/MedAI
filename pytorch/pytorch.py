from torch import nn

class MyModel(nn.Module):
    def __init__(self,in_features,**kwargs):
        super(MyModel,self).__init__(**kwargs)
        self.dense1 = nn.Linear(in_features,16)
        self.act1 = nn.ReLU()
        self.dense2 = nn.Linear(16)