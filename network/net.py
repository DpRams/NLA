import torch
import copy
from apps import getFreerGpu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

class TwoLayerNet(torch.nn.Module):
    
    def __init__(self, input_size, hidden_size, num_classes):

        super().__init__()
        self.layer_in = torch.nn.Linear(input_size, hidden_size) 
        self.layer_out = torch.nn.Linear(hidden_size, num_classes) 
        self.relu = torch.nn.ReLU()

    def setting_device(self):

        FreerGpuId = getFreerGpu.getting_freer_gpu()
        device = torch.device(f"cuda:{FreerGpuId}")
        self.device = device


    def forward(self, inputs):
        x = self.relu(self.layer_in(inputs))
        x = self.layer_out(x)
        
        return x



# 暫時先別亂動，目前有 import 在用
class Network(torch.nn.Module):

    def __init__(self, nb_neuro, x_train_scaled, y_train_scaled, **kwargs):

        super().__init__()

        self.setting_device()

        # Initialize
        self.linear1 = torch.nn.Linear(x_train_scaled.shape[1], nb_neuro).to(self.device)
        self.linear2 = torch.nn.Linear(nb_neuro, 1).to(self.device)
    
        # Stop criteria - threshold
        self.threshold_for_error = eval(kwargs["learning_goal"]) 
        self.threshold_for_lr = eval(kwargs["learning_rate_lower_bound"]) 
        self.tuning_times = eval(kwargs["tuning_times"]) 
        self.regularizing_strength = eval(kwargs["regularizing_strength"])
        
        # Set default now, not open for customization.
        not_used_currently = (kwargs["regularizing_strength"], kwargs["optimizer"])

        # Input data
        self.x = torch.FloatTensor(x_train_scaled).to(self.device)
        self.y = torch.FloatTensor(y_train_scaled).to(self.device)

        # Learning rate
        self.learning_rate = eval(kwargs["learning_rate"])

        # Whether the network is acceptable, default as False
        self.acceptable = False

        # Record the experiment result
        self.nb_node_pruned = 0
        self.nb_node = nb_neuro
        
        self.undesired_index = None
        self.message = ""
    
    def setting_device(self):

        FreerGpuId = getFreerGpu.getting_freer_gpu()
        device = torch.device(f"cuda:{FreerGpuId}")
        self.device = device

        return self

    
    # Reset the x and y data
    def setData(self, x_train_scaled, y_train_scaled):
        
        self.x = torch.FloatTensor(x_train_scaled).to(self.device)
        self.y = torch.FloatTensor(y_train_scaled).to(self.device)

        return self
    
    
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

        loss = torch.sqrt(torch.nn.functional.mse_loss(output, self.y)) + reg_term

        return (output, loss)
    
    # Backward operation
    def backward_Adam(self, loss):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return self

    
class RIRO(Network):

    def __init__(self, nb_neuro, x_train_scaled, y_train_scaled, **kwargs):
        super().__init__(nb_neuro, x_train_scaled, y_train_scaled, **kwargs)
        print("RIRO Initializing 成功")

    def initializing(self, initial_x, initial_y):
            # Find each minimum output value y
        min_y = min(initial_y)

        # Subtract min_y from each y
        res_y = initial_y - min_y
        
        # Use Linear regression to find the initial W1, b1, W2, b2
        reg = LinearRegression().fit(initial_x, res_y)

        # Set up the initial parameter of the network(through torch.nn.Parameter, we can turn non-trainable tensor into trainable tensor)
        self.linear1.weight = torch.nn.Parameter(torch.FloatTensor(reg.coef_).to(self.device))
        self.linear1.bias = torch.nn.Parameter(torch.FloatTensor(reg.intercept_).to(self.device))
        self.linear2.weight = torch.nn.Parameter(torch.FloatTensor([[1]]).to(self.device))
        self.linear2.bias = torch.nn.Parameter(torch.FloatTensor(min_y).to(self.device))

        self.acceptable = True

        return self

    def selecting(self, x_train_scaled, y_train_scaled):
        residual = []
        temp_network = copy.deepcopy(self)

        # Put each data into network to calculate the loss value
        for i in range(x_train_scaled.shape[0]):
            temp_network.setData(x_train_scaled[i].reshape(1,-1), y_train_scaled[i].reshape(-1, 1))
            residual.append((temp_network.forward()[1].item(),i))
        
        
        # Sort the data according to the residual value from smallest to largest, and save the data index in sorted_index
        sorted_index = [sorted_data[1] for sorted_data in sorted(residual, key = lambda x : x[0])]
        return sorted_index

    def matching(self):
        times_enlarge, times_shrink = 0,0

        # Set up the learning rate of the network
        # network.learning_rate = 1e-3
        self.acceptable = False

        network_pre = copy.deepcopy(self)
        output, loss = self.forward()

        if torch.all(torch.abs(output - self.y) < self.threshold_for_error):
            
            self.acceptable = True
    #         print("Matching Successful")
            return self

        else:

            while times_enlarge + times_shrink < self.tuning_times:
                
                output, loss = self.forward()
                network_pre = copy.deepcopy(self)
                loss_pre = loss
                
                # Backward and check the loss performance of the network with new learning rate
                self.backward_Adam(loss)
                output, loss = self.forward()

                # Confirm whether the loss value of the adjusted network is smaller than the current one
                if loss <= loss_pre and torch.all(torch.abs(output - self.y) < self.threshold_for_error):

                    self.acceptable = True
                    #print("Matching Successful")
                    return self

                elif loss <= loss_pre:
                    
                    # Multiply the learning rate by 1.2
                    times_enlarge += 1
                    self.learning_rate *= 1.2
                else: #loss > loss_pre
                    
                    if self.learning_rate <= self.threshold_for_lr:

                        # If true, set the acceptance of the network as False and return initial_netwrok
                        self.acceptable = False
                        #print("Matching Failed")
                        return network_pre
                    
                    # On the contrary, restore w and adjust the learning rate
                    else:

                        # Restore the parameter of the network
                        self = copy.deepcopy(network_pre)
                        times_shrink += 1
                        self.learning_rate *= 0.7
        

        self.acceptable = False
    #     print("Matching Failed")
        return network_pre

    def matching_for_reorganizing(self):
        times_enlarge, times_shrink = 0, 0

        # Set up the learning rate of the network
        # network.learning_rate = 1e-3
        self.acceptable = False

        network_pre = copy.deepcopy(self)
        output, loss = self.forward()

        if torch.all(torch.abs(output - self.y) < self.threshold_for_error):
            
            self.acceptable = True
    #         print("Matching(Re) Successful")
            return self

        else:

            while times_enlarge + times_shrink < self.tuning_times:
                
                output, loss = self.forward()
                network_pre = copy.deepcopy(self)
                loss_pre = loss
                
                # Backward and check the loss performance of the network with new learning rate
                self.backward_Adam(loss)
                output, loss = self.forward()

                # Confirm whether the loss value of the adjusted network is smaller than the current one
                if loss <= loss_pre and torch.all(torch.abs(output - self.y) < self.threshold_for_error):

                    self.acceptable = True
                    #print("Matching(Re) Successful")
                    return self

                elif loss <= loss_pre:
                    
                    # Multiply the learning rate by 1.2
                    times_enlarge += 1
                    self.learning_rate *= 1.2
                else: #loss > loss_pre
                    
                    if self.learning_rate <= self.threshold_for_lr:

                        # If true, set the acceptance of the network as False and return initial_netwrok
                        self.acceptable = False
                        #print("Matching(Re) Failed")
                        return network_pre
                    
                    # On the contrary, restore w and adjust the learning rate
                    else:

                        # Restore the parameter of the network
                        self = copy.deepcopy(network_pre)
                        times_shrink += 1
                        self.learning_rate *= 0.7
        

        self.acceptable = False
    #     print("Matching(Re) Failed")
        return network_pre 

    def cramming(self):
        """
        其實undesired_index的作用應該可以單純用network.y[-1]來取代，因為每次都在處理最新的一筆資料
        """
        torch.random.manual_seed(0)

        # Find unsatisfied data : k
        network_old = copy.deepcopy(self)
        output, loss = self.forward()

        undesired_index = torch.nonzero(torch.abs(output - self.y) > self.threshold_for_error + 1e-3, as_tuple = False)

        """
        [ [0] [1] [3] ] shape = (3,1), 代表有3筆非0的data。 演算法基礎上應只有一筆，thus undesired_index.shape[0] == 1
        """
        if undesired_index.shape[0] == 1: # if the undesired_index is one and only one.
            
            assert undesired_index[0][0] == len(self.y) - 1, "理應只有最後一筆需要cramming，此處出現異常"

            # Unsatisfied situation
            ## Find the index of the unsatisfied data
            k_data_num = undesired_index[0][0]
            undesired_data = torch.reshape(self.x[k_data_num,:], [1,-1])

            ## Remove the data that does not meet the error term
            left_data = self.x[:k_data_num,:]
            right_data = self.x[k_data_num+1:,:]
            remain_tensor = torch.cat([left_data, right_data], dim=0)
            
            ## Use the random method to find out the gamma and zeta
            while True:
                
                ## Find m-vector gamma: r
                ## Use the random method to generate the gamma that can meet the conditions
                gamma = torch.rand(size = [1, self.x.shape[1]]).to(self.device)
                subtract_undesired_data = torch.sub(remain_tensor, undesired_data)
                matmul_value = torch.mm(gamma, torch.t(subtract_undesired_data))

                if torch.all(matmul_value != 0):
                    break
            
            while True:

                ## Find the tiny value: zeta
                ## Use the random method to generate the zeta that can meet the conditions
                zeta = torch.rand(size=[1]).to(self.device)

                if torch.all(torch.mul(torch.add(zeta, matmul_value), torch.sub(zeta, matmul_value))<0):
                    break
            
            #k_l = undesired_index[0][0]  ### 如果直接用output[-1]，則不需要k_l

            ## The weight of input layer to hidden layer I
            w10 = gamma
            w11 = gamma
            w12 = gamma

            W1_new = torch.cat([w10,w11,w12], dim=0)

            ## The bias of input layer to hidden layer I
            matual_value = torch.mm(gamma, torch.t(undesired_data))

            b10 = torch.sub(zeta, matual_value)
            b11 = -1*matual_value
            b12 = torch.sub(-1*zeta, matual_value)

            b1_new = torch.reshape(torch.cat([b10,b11,b12],0),[3])

            ## The weight of hidden layer I to input layer
            #gap = network.y[k_data_num, k_l] - output[k_data_num, k_l]  ### gap = network.y[-1] - output[-1]
            gap = self.y[-1] - output[-1]

            wo0_value = gap/zeta
            wo1_value = (-2*gap)/zeta
            wo2_value = gap/zeta

            Wo_new = torch.reshape(torch.cat([wo0_value, wo1_value, wo2_value], dim=0), [1,-1])

            ## Add new neuroes to the network
            self.linear1.weight = torch.nn.Parameter(torch.cat([self.linear1.weight.data, W1_new]))
            self.linear1.bias = torch.nn.Parameter(torch.cat([self.linear1.bias.data, b1_new]))
            self.linear2.weight = torch.nn.Parameter(torch.cat([self.linear2.weight.data, Wo_new],dim=1))
            
            self.nb_node += 3

            output, loss = self.forward()

            ## Determine if cramming is succesful and print out the corresponding information
            if torch.all(torch.abs(output - self.y) < self.threshold_for_error):
                self.acceptable = True
                print("Cramming successful")
                self.message = "Cramming successful"
                return self
            else:
                return None

    def regularizing(self):
    
    
        ## Record the number of executions
        times_enlarge, times_shrink = 0, 0

        # ## Set up the learning rate of the network
        # network.learning_rate = 1e-3

        ## Set epoch to 100
        for i in range(self.tuning_times):

            ## Store the parameter of the network
            network_pre = copy.deepcopy(self)
            output, loss = self.forward(self.regularizing_strength)
            loss_pre = loss

            ## Backward operation to optain w'
            self.backward_Adam(loss)
            output, loss = self.forward(self.regularizing_strength)

            ## Confirm whether the adjusted loss value is smaller than the current one
            if loss <= loss_pre:

                ## Identify that all forecast value has met the error term
                if torch.all(torch.abs(output - self.y) < self.threshold_for_error):
                    
                    ## If true, multiply the learning rate by 1.2
                    self.learning_rate *= 1.2
                    times_enlarge += 1

                else:

                    ## Else, restore w and end the process
                    self = copy.deepcopy(network_pre)
                    return(self)

            # If the adjusted loss value is not smaller than the current one
            else:

                ## If the learning rate is greater than the threshold for learning rate
                if self.learning_rate > self.threshold_for_lr:
                    
                    ## Restore the w and multiply the learning rate by 0.7
                    self = copy.deepcopy(network_pre)
                    self.learning_rate *= 0.7
                    times_shrink += 1

                ## If the learning rate is smaller than the threshold for learning rate
                else:
                    
                    ## Restore the w
                    self = copy.deepcopy(network_pre)
    #                 print("Regularizing finished")
                    return(self)
                    
        #print("Regularizing finished")
        return(self)
    
    def reorganizing(self):

        limit = 1

        ## If the number of hidden node has already been 1, do the regularizing then. 
        if self.linear1.bias.shape[0] <= limit:
            self = regularizing(self)
            # self.regularizing()
    #         print("Reorganizing finished")
            return(self)
        
        else:
            
            ## Set up the k = 1, and p = the number of hidden node
            k = 1
            p = self.linear1.weight.data.shape[0]

            while True:

                ## If k > p, end of Process
                if k > p or p <= limit:
    #                 print("Reorganizing finished")
                    return self

                ## Else, Process is ongoing
                else:

                    ## Using the regularizing module to adjust the network
                    self = regularizing(self)
                    # self.regularizing()

                    ## Store the network and w
                    network_pre = copy.deepcopy(self)

                    ## Set up the acceptable of the network as false
                    self.acceptable = False
                    
                
                    ## Ignore the K hidden node
                    self.linear1.weight = torch.nn.Parameter(torch.cat([self.linear1.weight[:k-1],self.linear1.weight[k:]],0))
                    self.linear1.bias = torch.nn.Parameter(torch.cat([self.linear1.bias[:k-1],self.linear1.bias[k:]]))
                    self.linear2.weight = torch.nn.Parameter(torch.cat([self.linear2.weight[:,:k-1],self.linear2.weight[:,k:]],1))

                    
                    ## Using the matching module to adjust the network
                    self = matching_for_reorganizing(self)
                    # self.matching_for_reorganizing()
                    # self.matching()

                    ## If the resulting network is acceptable, this means that the k hidden node can be removed
                    if self.acceptable:

                        self.nb_node -= 1
                        self.nb_node_pruned += 1
                        p-=1

                    ## Else, it means that the k hidden node cannot be removed
                    else:

                        ## Restore the network and w
                        self = copy.deepcopy(network_pre)
                        k+=1


    def is_initializingNumber_too_big_to_initializing(index_of_data):
        
        if index_of_data == 1 : return True

    def is_learningGoal_too_small_to_cramming(network):
        
        if network == None : return True



def regularizing(network):
    
    ## Record the number of executions
    times_enlarge, times_shrink = 0, 0

    # ## Set up the learning rate of the network
    # network.learning_rate = 1e-3

    ## Set epoch to 100
    for i in range(network.tuning_times):

        ## Store the parameter of the network
        network_pre = copy.deepcopy(network)
        output, loss = network.forward(network.regularizing_strength)
        loss_pre = loss

        ## Backward operation to optain w'
        network.backward_Adam(loss)
        output, loss = network.forward(network.regularizing_strength)

        ## Confirm whether the adjusted loss value is smaller than the current one
        if loss <= loss_pre:

            ## Identify that all forecast value has met the error term
            if torch.all(torch.abs(output - network.y) < network.threshold_for_error):
                
                ## If true, multiply the learning rate by 1.2
                network.learning_rate *= 1.2
                times_enlarge += 1

            else:

                ## Else, restore w and end the process
                network = copy.deepcopy(network_pre)
                return(network)

        # If the adjusted loss value is not smaller than the current one
        else:

            ## If the learning rate is greater than the threshold for learning rate
            if network.learning_rate > network.threshold_for_lr:
                
                ## Restore the w and multiply the learning rate by 0.7
                network = copy.deepcopy(network_pre)
                network.learning_rate *= 0.7
                times_shrink += 1

            ## If the learning rate is smaller than the threshold for learning rate
            else:
                
                ## Restore the w
                network = copy.deepcopy(network_pre)
#                 print("Regularizing finished")
                return(network)
                
    #print("Regularizing finished")
    return(network)


def matching_for_reorganizing(network):

    times_enlarge, times_shrink = 0, 0

    # Set up the learning rate of the network
    # network.learning_rate = 1e-3
    network.acceptable = False

    network_pre = copy.deepcopy(network)
    output, loss = network.forward()

    if torch.all(torch.abs(output - network.y) < network.threshold_for_error):
        
        network.acceptable = True
#         print("Matching(Re) Successful")
        return network

    else:

        while times_enlarge + times_shrink < network.tuning_times:
            
            output, loss = network.forward()
            network_pre = copy.deepcopy(network)
            loss_pre = loss
            
            # Backward and check the loss performance of the network with new learning rate
            network.backward_Adam(loss)
            output, loss = network.forward()

            # Confirm whether the loss value of the adjusted network is smaller than the current one
            if loss <= loss_pre and torch.all(torch.abs(output - network.y) < network.threshold_for_error):

                network.acceptable = True
                #print("Matching(Re) Successful")
                return network

            elif loss <= loss_pre:
                
                # Multiply the learning rate by 1.2
                times_enlarge += 1
                network.learning_rate *= 1.2
            else: #loss > loss_pre
                
                if network.learning_rate <= network.threshold_for_lr:

                    # If true, set the acceptance of the network as False and return initial_netwrok
                    network.acceptable = False
                    #print("Matching(Re) Failed")
                    return network_pre
                
                # On the contrary, restore w and adjust the learning rate
                else:

                    # Restore the parameter of the network
                    network = copy.deepcopy(network_pre)
                    times_shrink += 1
                    network.learning_rate *= 0.7
      

    network.acceptable = False
#     print("Matching(Re) Failed")
    return network_pre         
