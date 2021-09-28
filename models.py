import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.rnn import LSTMCell

class CNN(nn.Module):
    def __init__(self, state_dim):
        super(CNN, self).__init__()

        # Hyperparameters
        self.state_dim = state_dim  # size of final embeddings
        self.image_size = 64        # height and width of images
        self.in_channels = 1        # channels in inputs (grey-scaled)
        self.kernel_size = 3        # kernel size of convolutions
        self.padding = 0            # padding in conv layers
        self.stride = 2             # stride of conv layers
        self.pool_kernel = 2        # kernel size of max pooling
        self.pool_stride = 2        # stride of max pooling
        self.out_channels1 = 4      # number of channels in conv1
        self.out_channels2 = 8      # number of channels in conv2
        self.num_layers = 2         # number of conv layers

        # Conv layers
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels1, 
                               self.kernel_size, self.stride, self.padding)
        self.maxpool1 = nn.MaxPool2d(self.pool_kernel, self.pool_stride)

        self.conv2 = nn.Conv2d(self.out_channels1, self.out_channels2, 
                               self.kernel_size, self.stride, self.padding)
        self.maxpool2 = nn.MaxPool2d(self.pool_kernel, self.pool_stride)

        # Linear layer
        self.cnn_out_dim = self.calc_cnn_out_dim()
        self.linear = nn.Linear(self.cnn_out_dim, self.state_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Conv 1
        x = self.conv1(x)          # [batch, 4, 31, 31]
        x = self.relu(x)           # [batch, 4, 31, 31]
        x = self.maxpool1(x)       # [batch, 4, 15, 15]

        # Conv 2
        x = self.conv2(x)          # [batch, 8, 7, 7]
        x = self.relu(x)           # [batch, 8, 7, 7]
        x = self.maxpool2(x)       # [batch, 8, 3, 3]

        # Linear
        x = x.view(x.shape[0], -1) # [batch, 72]
        x = self.linear(x)         # [batch, 32]
        
        return x
        
    def calc_cnn_out_dim(self):
        w = self.image_size
        h = self.image_size 
        for l in range(self.num_layers):
            new_w = np.floor(((w - self.kernel_size)/self.stride) + 1)
            new_h = np.floor(((h - self.kernel_size)/self.stride) + 1)
            new_w = np.floor(new_w / self.pool_kernel)
            new_h = np.floor(new_h / self.pool_kernel)
            w = new_w
            h = new_h
        return int(w*h*8)

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.use_images = args.use_images
        self.lesion_p = args.lesion_p
        self.measure_grad_norm = args.measure_grad_norm
        self.n_ctx = 2 # always 2 contexts ("popularity" and "competence")
        
        # Hyperparameters
        self.n_states = 16
        self.state_dim = 32
        self.mlp_in_dim = 3*self.state_dim # (f1 + f2 + context)
        self.hidden_dim = 128
        self.output_dim = 2
        self.analyze = False
        
        # Input embedding (images or one-hot)
        if self.use_images:
            self.face_embedding = CNN(self.state_dim)
        else:
            self.face_embedding = nn.Embedding(self.n_states, self.state_dim)
            nn.init.xavier_normal_(self.face_embedding.weight)    
        self.ctx_embedding = nn.Embedding(self.n_ctx, self.state_dim)
        nn.init.xavier_normal_(self.ctx_embedding.weight)

        # MLP
        self.hidden = nn.Linear(self.mlp_in_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(self.hidden_dim, self.output_dim)
        
        
    def forward(self, f1, f2, ctx):
        # Embed inputs
        f1_embed  = self.face_embedding(f1) # [batch, state_dim]
        f2_embed  = self.face_embedding(f2) # [batch, state_dim]
        ctx_embed = self.ctx_embedding(ctx) # [batch, state_dim]
        
        # Scale context for "lesion" experiments
        ctx_embed = torch.tensor(self.ctx_scale) * ctx_embed
        
        # Save embeddings to measure gradients during analysis
        if self.measure_grad_norm:
            self.f1_embed = f1_embed
            self.f2_embed = f2_embed
            self.ctx_embed = ctx_embed
        
        # MLP
        x = torch.cat([f1_embed, f2_embed, ctx_embed], dim=1) 
        hidd = self.hidden(x)  # [batch, hidden_dim]
        hidd = self.relu(hidd) # [batch, hidden_dim]
        x = self.out(hidd)     # [batch, output_dim]
        return x, hidd

class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()
        self.use_images = args.use_images
        self.ctx_order = args.ctx_order
        self.ctx_scale = args.ctx_scale
        self.measure_grad_norm = args.measure_grad_norm
        self.n_ctx = 2 # always 2 contexts ("popularity" and "competence")

        # Hyperparameters
        self.n_states = 16
        self.state_dim = 32
        self.hidden_dim = 128
        self.output_dim = 2
        self.analyze = False
        
        # Input embedding (images or one-hot)
        if self.use_images:
            self.face_embedding = CNN(self.state_dim)
        else:
            self.face_embedding = nn.Embedding(self.n_states, self.state_dim)
            nn.init.xavier_normal_(self.face_embedding.weight)    
        self.ctx_embedding = nn.Embedding(self.n_ctx, self.state_dim)
        nn.init.xavier_normal_(self.ctx_embedding.weight)

        # LSTM
        self.lstm = nn.LSTM(self.state_dim, self.hidden_dim)

        # Output
        self.out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, f1, f2, ctx):
        # Embed inputs
        f1_embed = self.face_embedding(f1)  # [batch, state_dim]
        f2_embed = self.face_embedding(f2)  # [batch, state_dim]
        ctx_embed = self.ctx_embedding(ctx) # [batch, state_dim]

        # Scale context for "lesion" experiments
        ctx_embed = torch.tensor(self.ctx_scale) * ctx_embed

        # Save embeddings to measure gradients during analysis
        if self.measure_grad_norm:
            self.f1_embed = f1_embed
            self.f2_embed = f2_embed
            self.ctx_embed = ctx_embed

        # LSTM
        if self.ctx_order == 'last':
            x = torch.cat([f1_embed.unsqueeze(0), f2_embed.unsqueeze(0),
                           ctx_embed.unsqueeze(0)], dim=0)
        elif self.ctx_order == 'first':
            x = torch.cat([ctx_embed.unsqueeze(0), f1_embed.unsqueeze(0), 
                           f2_embed.unsqueeze(0)], dim=0)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: [seq_length, batch, hidden_dim]
        # h_n: [1, batch, hidden_dim]
        # c_n: [1, batch, hidden_dim]

        # Output layer (linear)
        if self.analyze: 
            # run output layer on all time steps
            lstm_out = lstm_out.permute(1,0,2)
            # lstm_out: [batch, seq_length, hidden_dim]
            x = self.out(lstm_out)
            # x: [batch, seq_length, output_dim] 
        else:
            # only run output layer on final time step
            x = self.out(h_n.squeeze(0))
            # x: [batch, output_dim] 
        return x, lstm_out

class TruncatedRNN(nn.Module):
    def __init__(self, args):
        super(TruncatedRNN, self).__init__()
        self.use_images = args.use_images
        self.ctx_order = args.ctx_order
        self.ctx_scale = args.ctx_scale
        self.measure_grad_norm = args.measure_grad_norm
        self.n_ctx = 2 # always 2 contexts ("popularity" and "competence")

        # Hyperparameters
        self.n_states = 16
        self.state_dim = 32
        self.hidden_dim = 128
        self.output_dim = 2
        self.analyze = False
        
        # Input embedding (images or one-hot)
        if self.use_images:
            self.face_embedding = CNN(self.state_dim)
        else:
            self.face_embedding = nn.Embedding(self.n_states, self.state_dim)
            nn.init.xavier_normal_(self.face_embedding.weight)
        self.ctx_embedding = nn.Embedding(self.n_ctx, self.state_dim)
        nn.init.xavier_normal_(self.ctx_embedding.weight)

        # Bias context vectors to be far apart
        ctx_bias0 = 1*torch.ones([1, self.state_dim])
        ctx_bias1 = -1*torch.ones([1, self.state_dim])
        ctx_bias = torch.cat([ctx_bias0, ctx_bias1], dim=0)
        self.ctx_embedding.weight.data += ctx_bias

        # LSTM Cell
        self.lstmcell = nn.LSTMCell(self.state_dim, self.hidden_dim)

        # MLP
        self.out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, f1, f2, ctx):
        # Embed inputs
        f1_embed  = self.face_embedding(f1).unsqueeze(0) # [1, batch, state_dim]
        f2_embed  = self.face_embedding(f2).unsqueeze(0) # [1, batch, state_dim]
        ctx_embed = self.ctx_embedding(ctx).unsqueeze(0) # [1, batch, state_dim]
        
        # Scale context for "lesion" experiments
        ctx_embed = torch.tensor(self.ctx_scale) * ctx_embed

        # Save embeddings to measure gradients during analysis
        if self.measure_grad_norm:
            self.f1_embed = f1_embed
            self.f2_embed = f2_embed
            self.ctx_embed = ctx_embed

        # Order of presentation (context first or last)
        if self.ctx_order == 'last':
            x = torch.cat([f1_embed, f2_embed, ctx_embed], dim=0)
        elif self.ctx_order == 'first':
            x = torch.cat([ctx_embed, f1_embed, f2_embed], dim=0)

        # Initialize h0 and c0
        bs = f1_embed.size(1) # batch size
        h_n = torch.zeros([bs, self.hidden_dim]) # [batch, hidden_dim]
        c_n = torch.zeros([bs, self.hidden_dim]) # [batch, hidden_dim]

        # Run LSTM, truncating gradients
        lstm_out = []
        for t in range(len(x)):
            h_n, c_n = self.lstmcell(x[t], (h_n.detach(), c_n.detach())) 
            # h_n/c_n: [batch, hidden_dim]
            lstm_out.append(h_n)
        lstm_out = torch.stack(lstm_out, dim=0) # [seq_len, batch, hidden_dim]

        # Output layer (linear)
        if self.analyze:
            # run output layer on all time steps
            lstm_out = lstm_out.permute(1,0,2)
            # lstm_out: [batch, seq_length, hidden_dim]
            x = self.out(lstm_out)
            # x: [batch, seq_length, output_dim] 
        else:
            # only run output layer on final time step
            x = self.out(h_n)
            # x: [batch, output_dim] 
        return x, lstm_out

class StepwiseMLP(nn.Module):
    def __init__(self, args):
        super(StepwiseMLP,self).__init__()
        self.use_images = args.use_images
        self.ctx_order = args.ctx_order
        self.truncated_mlp = args.truncated_mlp
        self.ctx_scale = args.ctx_scale
        self.measure_grad_norm = args.measure_grad_norm
        self.n_ctx = 2 # always 2 contexts ("popularity" and "competence")

        # Hyperparameters
        self.n_states = 16
        self.state_dim = 32
        self.hidden1_dim = 128
        self.hidden2_dim = 128
        self.mlp_in1_dim = 2*self.state_dim
        self.mlp_in2_dim = self.hidden1_dim+self.state_dim
        self.output_dim = 2
        self.analyze = False
        
        # Input Embedding (images or one-hot)
        if self.use_images:
            self.face_embedding = CNN(self.state_dim)
        else:
            self.face_embedding = nn.Embedding(self.n_states, self.state_dim)
            nn.init.xavier_normal_(self.face_embedding.weight)
        self.ctx_embedding = nn.Embedding(self.n_ctx, self.state_dim)
        nn.init.xavier_normal_(self.ctx_embedding.weight)

        # MLP
        self.hidden1 = nn.Linear(self.mlp_in1_dim, self.hidden1_dim)
        self.hidden2 = nn.Linear(self.mlp_in2_dim, self.hidden2_dim)
        self.out = nn.Linear(self.hidden2_dim, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, f1, f2, ctx):
        # Embed inputs
        f1_embed  = self.face_embedding(f1) # [batch, state_dim]
        f2_embed  = self.face_embedding(f2) # [batch, state_dim]
        ctx_embed = self.ctx_embedding(ctx) # [batch, state_dim]
        
        # Scale context for "lesion" experiments
        ctx_embed = torch.tensor(self.ctx_scale) * ctx_embed

        # Save embeddings to measure gradients during analysis
        if self.measure_grad_norm:
            self.f1_embed = f1_embed
            self.f2_embed = f2_embed
            self.ctx_embed = ctx_embed

        # Hidden 1 (context + face1)
        x = torch.cat([ctx_embed, f1_embed], dim=1) # [batch, 2*state_dim]
        hidd1 = self.hidden1(x) # [batch, hidden1_dim]
        hidd1 = self.relu(hidd1) # [batch, hidden1_dim]
        
        # Truncate gradients
        if self.truncated_mlp=='true':
            x = torch.cat([hidd1.detach(), f2_embed], dim=1) 
            # x: [batch, state_dim+hidden1_dim]
        else:
            x = torch.cat([hidd1, f2_embed], dim=1) 
            # x: [batch, state_dim+hidden1_dim]

        # Hidden 2 (hidden1 + face2)
        hidd2 = self.hidden2(x) # [batch, hidden2_dim]
        hidd2 = self.relu(hidd2) # [batch, hidden2_dim]
        x = self.out(hidd2)  # [batch, output_dim]
        hidd = [hidd1, hidd2]
        return x, hidd

class CognitiveController(nn.Module):
    def __init__(self, args):
        super(CognitiveController, self).__init__()
        self.use_images = args.use_images
        self.ctx_scale = args.ctx_scale
        self.n_ctx = 2 # always 2 contexts ("popularity" and "competence")

        # Hyperparameters
        self.n_states = 16
        self.state_dim = 32
        self.mlp_in_dim = 2*self.state_dim # f1+f2 (context treated separately)
        self.hidden_dim = 128
        msg = "hidden_dim must be divisible by n_ctx"
        assert self.hidden_dim % self.n_ctx == 0, msg
        self.h_dim = self.hidden_dim // self.n_ctx # units per hidden group
        self.output_dim = 2
        self.analyze = False
        

        # Input embedding (images or one-hot)
        if self.use_images:
            self.face_embedding = CNN(self.state_dim)
        else:
            self.face_embedding = nn.Embedding(self.n_states, self.state_dim)
            nn.init.xavier_normal_(self.face_embedding.weight)
            
        self.ctx_embedding = nn.Embedding(self.n_ctx, self.state_dim)
        nn.init.xavier_normal_(self.ctx_embedding.weight)

        # MLP
        self.control = nn.Linear(self.state_dim, self.n_ctx)
        self.linear = nn.Linear(self.mlp_in_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, f1, f2, ctx):
        batch = f1.shape[0]

        # Embed inputs
        f1_embed  = self.face_embedding(f1) # [batch, state_dim]
        f2_embed  = self.face_embedding(f2) # [batch, state_dim]
        ctx_embed = self.ctx_embedding(ctx) # [batch, state_dim]
        
        # Scale context for "lesion" experiments
        ctx_embed = torch.tensor(self.ctx_scale) * ctx_embed

        # Save embeddings to measure gradients during analysis
        if self.measure_grad_norm:
            self.f1_embed = f1_embed
            self.f2_embed = f2_embed
            self.ctx_embed = ctx_embed

        # Hidden
        x = torch.cat([f1_embed, f2_embed], dim=1) # [batch, 2*state_dim]
        hidden = self.relu(self.linear(x)) # [batch, hidden_dim]
        hidden = hidden.view(batch, self.h_dim, self.n_ctx) 
        # hidden: [batch, hidden_dim // n_ctx, n_ctx]

        # Control
        control_signal = self.softmax(self.control(ctx_embed)) # [batch, n_ctx]
        control_signal = control_signal.unsqueeze(1) # [batch, 1, n_ctx]
        hidden = hidden * control_signal # [batch, hidden_dim // n_ctx, n_ctx]
        
        # Output
        hidden = hidden.view(batch,-1) # [batch, hidden_dim]
        output = self.out(hidden) # [batch, output_dim]
        return output, hidden
