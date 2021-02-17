
import torch


class Encoder(torch.nn.Module):
    def __init__(self, input_dim , out_dim = 64):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim,256)
        self.linear2 = torch.nn.Linear(256,512)
        self.linear3 = torch.nn.Linear(512,256)
        self.linear4 = torch.nn.Linear(256,128)
        self.linear5 = torch.nn.Linear(128,out_dim)
        self.prelu1   = torch.nn.PReLU()
        self.prelu2   = torch.nn.PReLU()
        self.prelu3   = torch.nn.PReLU()
        self.prelu4   = torch.nn.PReLU()
        self.tanh    = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(p=0.0)
        
    def forward(self, x):
        x = self.prelu1(self.linear1(x))
        x = self.dropout(self.prelu2(self.linear2(x)))
        x = self.dropout(self.prelu3(self.linear3(x)))
        x = self.dropout(self.prelu4(self.linear4(x)))
        x = self.tanh(self.linear5(x)/20)*24
        return x.squeeze()
    
class Decoder1(torch.nn.Module):
    def __init__(self,input_dim,output_dim=1):
        super(Decoder1, self).__init__()
        
        self.linear1 = torch.nn.Linear(input_dim,32)
        self.linear2 = torch.nn.Linear(32,16)
        self.linear3 = torch.nn.Linear(16,1)
        self.prelu1   = torch.nn.PReLU()
        self.prelu2   = torch.nn.PReLU()
        self.dropout = torch.nn.Dropout(p=0.0)
    
    def forward(self,x):
        x = self.dropout(self.prelu1(self.linear1(x)))
        x = self.dropout(self.prelu2(self.linear2(x)))
        x = self.linear3(x)
        return x.squeeze()

class Decoder2(torch.nn.Module):
    def __init__(self,input_dim,output_dim=168):
        super(Decoder2, self).__init__()
        

        self.linear1 = torch.nn.Linear(input_dim,64)
        self.linear2 = torch.nn.Linear(input_dim,128)
        self.linear3 = torch.nn.Linear(128,output_dim)
        self.prelu1   = torch.nn.PReLU()
        self.prelu2   = torch.nn.PReLU()
        self.dropout = torch.nn.Dropout(p=0.2)
        self.relu = torch.nn.ReLU()
        
    def forward(self,x):
        x = self.dropout(self.prelu1(self.linear1(x)))
        x = self.dropout(self.prelu2(self.linear2(x)))
        x = self.relu(self.linear3(x))
        return x.squeeze() 