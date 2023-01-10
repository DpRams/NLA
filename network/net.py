import torch
import copy
from apps import getFreerGpu
from CSI_Modules.modules_s2 import Initialize, Select, Match, Cram, Reorganize

"""
Original
"""

class Network(torch.nn.Module):

    # def __init__(self, nb_neuro, x_train_scaled, y_train_scaled, **kwargs):
    def __init__(self, **model_params):

        super().__init__()
        
        self.model_params = model_params
        self.setting_device()

        # Initialize
        self.linear1 = torch.nn.Linear(self.model_params["inputDimension"], self.model_params["hiddenNode"]).to(self.device)
        self.linear2 = torch.nn.Linear(self.model_params["hiddenNode"], self.model_params["outputDimension"]).to(self.device)

        # Record the experiment result
        self.nb_node_pruned = 0
        self.nb_node = self.model_params["hiddenNode"]

    def setting_device(self):

        FreerGpuId = getFreerGpu.getting_freer_gpu()
        device = torch.device(f"cuda:{FreerGpuId}")
        self.device = device

        return self

    
    # Reset the x and y data
    def setData(self, x_train_scaled, y_train_scaled, only_cpu=False):
        
        if only_cpu:
            self.x = torch.FloatTensor(x_train_scaled).cpu()
            self.y = torch.FloatTensor(y_train_scaled).cpu()

        else:
            self.x = torch.FloatTensor(x_train_scaled).to(self.device)
            self.y = torch.FloatTensor(y_train_scaled).to(self.device)

        # print(self.linear1.weight.data.get_device(), self.linear2.weight.data.get_device())
        # print(self.x.get_device(), self.y.get_device())
    
    
    # Add the new data to the x and y data
    #(目前是都沒在用，但應該也沒差，因為 setData 就可以做到相同操作，只是 train 的過程中，資料新增可能要寫好懂一點)
    def addData(self, new_x_train, new_y_train):
        
        self.x = torch.cat([self.x, new_x_train.reshape(1,-1)], 0)#.cuda()
        self.y = torch.cat([self.y, new_y_train.reshape(-1,1)], 0)#.cuda()

        return self

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

        if self.model_params["lossFunction"] == "RMSE" : loss = torch.sqrt(torch.nn.functional.mse_loss(output, self.y)) + reg_term
        if self.model_params["lossFunction"] == "MSE" : loss = torch.nn.functional.mse_loss(output, self.y) + reg_term
        if self.model_params["lossFunction"] == "CROSSENTROPYLOSS" : loss = torch.nn.CrossEntropyLoss(output, self.y) + reg_term

        return (output, loss)
    
    # Backward operation
    def backward(self, loss):

        if self.model_params["optimizer"] == "Adam" : 
            optimizer = torch.optim.Adam(params=self.parameters(), \
                                            lr=self.model_params["learningRate"], \
                                            betas=self.model_params["betas"], \
                                            eps=self.model_params["eps"], \
                                            weight_decay=self.model_params["weightDecay"])
        if self.model_params["optimizer"] == "AdamW" : 
            optimizer = torch.optim.AdamW(params=self.parameters(), \
                                            lr=self.model_params["learningRate"], \
                                            betas=self.model_params["betas"], \
                                            eps=self.model_params["eps"], \
                                            weight_decay=self.model_params["weightDecay"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
class RIRO(Network):

    def is_initializingNumber_too_big_to_initializing(index_of_data):
        
        if index_of_data == 1 : return True

    def is_learningGoal_too_small_to_cramming(network):
        
        if network == None : return True


class YourCSI_s2():
    def __init__(self, **model_params):
        self.net = yourCSI_s2(**model_params)

    def initializing(self, initial_x, initial_y):
        self.net.initializing(initial_x, initial_y)

    def selecting(self, x_train_scaled, y_train_scaled):
        return self.net.selecting(x_train_scaled, y_train_scaled)

    def matching(self):
        self.net = self.net.matching()

    def cramming(self):
        self.net.cramming()

    # 外層呼叫不到，因此，不須於此層定義。230110 : 可先觀察有無需要，先保留
    # def matching_reorganizing(self):
    #     self.net = self.net.matching_reorganizing()
            
    # def regularizing(self):
    #     self.net = self.net.regularizing()

    def reorganizing(self):
        self.net = self.net.reorganizing()

# TwoLayerNet -> Network_s2(X) Network(O)
class yourCSI_s2(Network): 

    def __init__(self, **model_params):
        super().__init__(**model_params)

    def initializing(self, initial_x, initial_y):

        if self.model_params["initializingRule"] != "Disabled":
            initializing_fn = eval(str("Initialize.")+str(self.model_params["initializingRule"]))
            initializing_fn(self, initial_x, initial_y)
        else:
            print("不啟用 initializing")
            return self

    def selecting(self, x_train_scaled, y_train_scaled):

        if self.model_params["selectingRule"] != "Disabled":
            selecting_fn = eval(str("Select.")+str(self.model_params["selectingRule"]))
            selecting_index = selecting_fn(self, x_train_scaled, y_train_scaled)
            return selecting_index
        else:
            pass
      
    def matching(self):
        
        """
        還沒寫如何因應只選 EU 的可能性
        """
        if self.model_params["matchingRule"] != "Disabled":
            matching_fn = eval(str("Match.")+str(self.model_params["matchingRule"]))
            return matching_fn(self)
        else:
            print("不啟用 matching")
            return self

    def cramming(self):

        if self.model_params["crammingRule"] != "Disabled":
            cramming_fn = eval(str("Cram.")+str(self.model_params["crammingRule"]))
            return cramming_fn(self)
        else:
            print("不啟用 cramming")
            return self

    # 外層呼叫不到，因此，不須於此層定義。230110 : 可先觀察有無需要，先保留
    # def matching_reorganizing(self):

    #     if self.model_params["matchingRule"] != "Disabled":
    #         matching_reorganizing_fn = eval(str("Reorganize.")+str(self.model_params["matchingRule"]))
    #         return matching_reorganizing_fn(self)
    #     else:
    #         pass
         
    # def regularizing(self):

    #     if self.model_params["reorganizingRule"] != "Disabled":
    #         regularizing_fn = eval(str("Reorganize.")+str(self.model_params["regularizingTimes"]))
    #         print(f"str(regularizing_fn) = {eval(str('Reorganize.')+str(self.model_params['regularizingTimes']))}")
    #         print(f"eval(regularizing_fn) = {regularizing_fn}")
    #         return regularizing_fn(self)
    #     else:
    #         pass

    def reorganizing(self):

        if self.model_params["reorganizingRule"] != "Disabled":
            reorganizing_fn = eval(str("Reorganize.")+str(self.model_params["reorganizingRule"]))
            return reorganizing_fn(self)
        else:
            print("不啟用 reorganizing")
            return self
            