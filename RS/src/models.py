import torch
import torch.nn as nn

class PureMatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=4, max_norm=0.5):
        """
        Pure dot-product MF (no biases). 
        Mandated by Section 4.1 to ensure rigorous DP sensitivity bounds.
        """
        super(PureMatrixFactorization, self).__init__()
        
        self.user_emb = nn.Embedding(num_users, embedding_dim, max_norm=max_norm)
        self.item_emb = nn.Embedding(num_items, embedding_dim, max_norm=max_norm)
        
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_indices, item_indices):
        u = self.user_emb(user_indices)
        i = self.item_emb(item_indices)
        
        prediction = (u * i).sum(dim=1)
        # Localized L2 regularization penalty applied strictly to active features
        l2_penalty = (u.norm(2, dim=1)**2 + i.norm(2, dim=1)**2)
        
        return prediction, l2_penalty