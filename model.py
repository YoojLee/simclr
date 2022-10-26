# base Encoder랑 projection head만 구현하면 됨.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50 # 어떤 backbone을 사용했는지는 모르겠으나 timm에서 가져와서 사용해볼까?
import timm

out_dim_dict = {
    'resnet50': 1000,
}

class SimCLR(nn.Module):
    def __init__(self, model:str='resnet50', projection_dim = 128, return_h = False):
        """
        An architecture of SimCLR framework, which does basically consist of a base encoder and projection head.

        - Args
            model: specify which model to use. 
            in_features: a dimension of a base encoder output.
            projection_dim: a dimension of a projection head.
        """
        super().__init__()

        self.return_h = return_h
        self.encoder = timm.create_model(model, pretrained=False)
        encoder_out_dim = out_dim_dict[model]
        self.projection = nn.Sequential(
            nn.Linear(encoder_out_dim, projection_dim, bias=False), # bias 넣나?
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim, bias=False)
        )
        
    def get_representation(self, x):
        return self.encoder(x)
    
    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)

        if self.return_h:
            return h,z

        return z


class LinearClassifier(nn.Module):
    """
    A Linear Classifier which consists of a single linear layer followed by softmax.
    """
    def __init__(self, model='resnet50', n_class=1000):
        super().__init__()
        self.n_class = n_class
        encoder_out_dim = out_dim_dict[model]
        self.linear = nn.Linear(encoder_out_dim, n_class)
    
    def forward(self, h):
        pred = self.linear(h)

        return pred
        

# train.py에서 call해서 쓸 예정.
def get_cosine_similarity(z):
    """
    Compute pairwise cosine similarities between all samples in X.

    - Args
        z: an output matrix from projection head. A matrix of [2N, D]
    
    - Returns
        sim: A [2N, 2N] matrix containing pairwise cosine similarities. An element at position (i,j) is the cosine similarity between z_i, z_j. Diagonal elements will be all 1.
    """
    # normalize z first.
    z_norm = z / torch.sqrt(torch.sum(z**2, dim=1)).unsqueeze(-1) # broadcasting 해주려면, 뒷부분에 1 추가해줘야함.
    sim = torch.matmul(z_norm, z_norm.T)  # sim: [2N, 2N]

    return sim


if __name__ == "__main__":
    z = torch.randn(20, 128)

    from sklearn.metrics.pairwise import cosine_similarity

    test = torch.round(get_cosine_similarity(z), decimals=4)
    test2 = torch.round(torch.tensor(cosine_similarity(z, dense_output=False)), decimals=4)

    print(test.shape, test2.shape)
    print(test, test2)
    print(torch.sum(test != test2))
