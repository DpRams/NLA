import torch 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cuda:1" if torch.cuda.is_available() else "cpu")

class Network(torch.nn.Module):

    def __init__(self, nb_neuro, x_train_scaled, y_train_scaled, **kwargs):

        super().__init__()
        # print(f"參數灌入:{kwargs}")
        # Initialize
        self.linear1 = torch.nn.Linear(x_train_scaled.shape[1], nb_neuro).to(device)
        self.linear2 = torch.nn.Linear(nb_neuro, 1).to(device)
    
        # Stop criteria - threshold
        self.threshold_for_error = eval(kwargs["learning_goal"]) 
        self.threshold_for_lr = eval(kwargs["learning_rate_lower_bound"]) 
        self.tuning_times = eval(kwargs["tuning_times"]) 
        self.regularizing_strength = eval(kwargs["regularizing_strength"])
        
        # Set default now, not open for customization.
        not_used_currently = (kwargs["regularizing_strength"], kwargs["optimizer"])

        # Input data
        self.x = torch.FloatTensor(x_train_scaled).to(device)
        self.y = torch.FloatTensor(y_train_scaled).to(device)

        # Learning rate
        self.learning_rate = eval(kwargs["learning_rate"])

        # Whether the network is acceptable, default as False
        self.acceptable = False

        # Record the experiment result
        self.nb_node_pruned = 0
        self.nb_node = nb_neuro
        
        self.undesired_index = None
        self.message = ""
        
    # Reset the x and y data
    def setData(self, x_train_scaled, y_train_scaled):
        
        self.x = torch.FloatTensor(x_train_scaled).to(device)
        self.y = torch.FloatTensor(y_train_scaled).to(device)
    # Add the new data to the x and y data
    def addData(self, new_x_train, new_y_train):
        
        self.x = torch.cat([self.x, new_x_train.reshape(1,-1)], 0)#.cuda()
        self.y = torch.cat([self.y, new_y_train.reshape(-1,1)], 0)#.cuda()

    # Forward operaion
    def forward(self, reg_strength=0):
        
        h_relu = self.linear1(self.x).clamp(min=0)
        output = self.linear2(h_relu)

        param_val = torch.sum(torch.pow(self.linear2.bias.data, 2)) + \
                    torch.sum(torch.pow(self.linear2.weight.data, 2)) + \
                    torch.sum(torch.pow(self.linear1.bias.data, 2)) + \
                    torch.sum(torch.pow(self.linear1.weight.data, 2))
        reg_term = reg_strength / (
            self.linear1.weight.data.shape[1] + 1 \
            + self.linear1.weight.data.shape[1] * (self.linear1.weight.data.shape[0] + 1)
        ) * param_val

        loss = torch.sqrt(torch.nn.functional.mse_loss(output, self.y)) + reg_term

        return (output, loss)
    
    # Backward operation
    def backward_Adam(self, loss):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()