import torch
import copy
from apps import getFreerGpu, readingDockerTmp
from CSI_Modules.modules_s2 import Initialize, Select, Match, Cram, Reorganize
import requests


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

        self.acceptable = False
    
    def getting_new_net_size(self):
        self.linear1 = torch.nn.Linear(self.model_params["inputDimension"], self.nb_node).to(self.device)
        self.linear2 = torch.nn.Linear(self.nb_node, self.model_params["outputDimension"]).to(self.device)

    def setting_device(self):

        FreerGpuId = getFreerGpu.getting_freer_gpu()
        if FreerGpuId == -1:
            device = torch.device(f"cpu")
        else:
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
        elif self.model_params["optimizer"] == "AdamW" : 
            optimizer = torch.optim.AdamW(params=self.parameters(), \
                                            lr=self.model_params["learningRate"], \
                                            betas=self.model_params["betas"], \
                                            eps=self.model_params["eps"], \
                                            weight_decay=self.model_params["weightDecay"])
            

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class Hw1(Network):

    def __init__(self, **model_params):

        super().__init__()
        
        self.model_params = model_params
        self.setting_device()

        # Initialize
        self.linear1 = torch.nn.Linear(self.model_params["inputDimension"], self.model_params["hiddenNode"]).to(self.device)
        self.linear2 = torch.nn.Linear(self.model_params["hiddenNode"], self.model_params["outputDimension"]).to(self.device)

        # Weight Initailization
        self.__weightInitializing(self.linear1)
        self.__weightInitializing(self.linear2)

        # Record the experiment result
        self.nb_node_pruned = 0
        self.nb_node = self.model_params["hiddenNode"]

        self.acceptable = False
    
    def __weightInitializing(self, weight):

        if self.model_params["weightInitialization"] == "Xavier normal":
            return torch.nn.init.xavier_normal_(weight)
        elif self.model_params["weightInitialization"] == "Xavier uniform":
            return torch.nn.init.xavier_uniform_(weight)
        elif self.model_params["weightInitialization"] == "Kaiming normal":
            return torch.nn.init.kaiming_normal_(weight)
        elif self.model_params["weightInitialization"] == "Kaiming uniform":
            return torch.nn.init.kaiming_uniform_(weight)

    # Forward operaion
    def forward(self, reg_strength=0):
        
        if self.model_params["regularizationTerm"] == "tanh":
            h_tanh = torch.tanh(self.linear1(self.x))
            output = self.linear2(h_tanh)

        elif self.model_params["regularizationTerm"] == "ReLU":
            h_relu = self.linear1(self.x).clamp(min=0)
            output = self.linear2(h_relu)

        reg_strength = self.model_params["regularizationTerm"]

        param_val = torch.sum(torch.pow(self.linear2.bias.data, 2)) + \
                    torch.sum(torch.pow(self.linear2.weight.data, 2)) + \
                    torch.sum(torch.pow(self.linear1.bias.data, 2)) + \
                    torch.sum(torch.pow(self.linear1.weight.data, 2))
        reg_term = reg_strength * param_val

        if self.model_params["lossFunction"] == "MSE" : loss = torch.nn.functional.mse_loss(output, self.y) + reg_term

        return (output, loss)

    # Backward operation
    def backward(self, loss):

        if self.model_params["optimizer"] == "Adam" : 
            optimizer = torch.optim.Adam(params=self.parameters(), \
                                            lr=self.model_params["learningRate"], \
                                            betas=self.model_params["betas"], \
                                            eps=self.model_params["eps"], \
                                            weight_decay=self.model_params["weightDecay"])
            
        elif self.model_params["optimizer"] == "Gradient descent" : 
            optimizer = torch.optim.SGD(params=self.parameters(), \
                                            lr=self.model_params["learningRate"])
        
        elif self.model_params["optimizer"] == "Momentum" : 
            optimizer = torch.optim.SGD(params=self.parameters(), \
                                            lr=self.model_params["learningRate"], \
                                            momentum=0.9)


        if self.model_params["learningRateDecaySchedule"] == "None" :
            scheduler = None

        elif self.model_params["learningRateDecaySchedule"] == "Cosine" : 
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()
    
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
            try:
                matching_fn = eval(str("Match.")+str(self.model_params["matchingRule"]))
                return matching_fn(self)
            except:
                self = developer_modules(self, the_finding_module_name=self.model_params["matchingRule"])
                return self
            
        else:
            print("不啟用 matching")
            return self

    def cramming(self):

        if self.model_params["crammingRule"] != "Disabled":
            try:
                cramming_fn = eval(str("Cram.")+str(self.model_params["crammingRule"]))
                return cramming_fn(self)
            except:
                self = developer_modules(self, the_finding_module_name=self.model_params["crammingRule"])
                return self
            
        else:
            print("不啟用 cramming")
            return self

    def reorganizing(self):

        if self.model_params["reorganizingRule"] != "Disabled":
            try:
                reorganizing_fn = eval(str("Reorganize.")+str(self.model_params["reorganizingRule"]))
                return reorganizing_fn(self)
            except:
                self = developer_modules(self, the_finding_module_name=self.model_params["reorganizingRule"])
                return self
                
        else:
            print("不啟用 reorganizing")
            return self
            

# def get_module_fn():
    
class tensor_transforming():

    def odct_tensor2list(network):

        old_network = {}
        for dct in network.state_dict().items():
            key, value = dct
            old_network[key] = value.tolist()
        return old_network

    def odct_list2tensor(old_network_weight):

        network_weight = {}
        for dct in old_network_weight.items():
            key, value = dct
            network_weight[key] = torch.FloatTensor(value)
        return network_weight
    
def developer_modules(network, the_finding_module_name=None):
    
    modules_fn = requests.post
    # if the_finding_module_name == "matching-ramsay":
    #     urls = f"http://127.0.0.1:8011/train/matching-ramsay" 
    # elif the_finding_module_name == "reorganizing-ramsay":
    #     urls = f"http://127.0.0.1:8010/train/reorganizing-ramsay" 
    # elif the_finding_module_name == "cramming-ramsay":
    #     urls = f"http://127.0.0.1:8009/train/cramming-ramsay" 
    # else:
    specific_module_dict = readingDockerTmp.getModulesOnDocker(module_name=the_finding_module_name)
    urls = f"http://127.0.0.1:{specific_module_dict['container_port']}/train/{specific_module_dict['module_name']}"  
    old_network = tensor_transforming.odct_tensor2list(network)
    json_data = {"network": old_network, \
                "model_params" : network.model_params, \
                "data" : (network.x.tolist(), network.y.tolist()), \
                "nb_node" : network.nb_node, \
                "nb_node_pruned" : network.nb_node_pruned, \
                "acceptable" : network.acceptable}
    res = modules_fn(url=urls, json=json_data)
    new_network_weight, new_network_model_param, nb_node, nb_node_pruned, acceptable = res.json()["new_network"], res.json()["model_params"], res.json()["nb_node"], res.json()["nb_node_pruned"], res.json()["acceptable"]
    network.acceptable = acceptable
    network.nb_node = nb_node
    network.nb_node_pruned = nb_node_pruned
    network.model_params = new_network_model_param
    network.getting_new_net_size()
    network.load_state_dict(tensor_transforming.odct_list2tensor(new_network_weight), strict=False)

    return network