import torch
import copy
from apps import getFreerGpu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

"""
目前的 function 是用非物件導向設計，要評估過物件導向設計的可能性後，再視情況調整。
"""

class Initialize():

    def Default(network, initial_x, initial_y):
        
        # Find each minimum output value y
        min_y = min(initial_y)

        # Subtract min_y from each y
        res_y = initial_y - min_y
        
        # Use Linear regression to find the initial W1, b1, W2, b2
        reg = LinearRegression().fit(initial_x, res_y)

        # Set up the initial parameter of the network(through torch.nn.Parameter, we can turn non-trainable tensor into trainable tensor)
        network.linear1.weight = torch.nn.Parameter(torch.FloatTensor(reg.coef_).to(network.device))
        network.linear1.bias = torch.nn.Parameter(torch.FloatTensor(reg.intercept_).to(network.device))
        network.linear2.weight = torch.nn.Parameter(torch.FloatTensor([[1]]).to(network.device))
        network.linear2.bias = torch.nn.Parameter(torch.FloatTensor(min_y).to(network.device))

        network.acceptable = True

        return network

class Select():

    def LTS(network, x_train_scaled, y_train_scaled):
        
        residual = []
        temp_network = copy.deepcopy(network)

        # Put each data into network to calculate the loss value
        for i in range(x_train_scaled.shape[0]):
            temp_network.setData(x_train_scaled[i].reshape(1,-1), y_train_scaled[i].reshape(-1, 1))
            residual.append((temp_network.forward()[1].item(),i))
        
        
        # Sort the data according to the residual value from smallest to largest, and save the data index in sorted_index
        sorted_index = [sorted_data[1] for sorted_data in sorted(residual, key = lambda x : x[0])]
        return sorted_index

    # 還沒改成 POS
    def POS(network, x_train_scaled, y_train_scaled):
        
        residual = []
        temp_network = copy.deepcopy(network)

        # Put each data into network to calculate the loss value
        for i in range(x_train_scaled.shape[0]):
            temp_network.setData(x_train_scaled[i].reshape(1,-1), y_train_scaled[i].reshape(-1, 1))
            residual.append((temp_network.forward()[1].item(),i))
        
        
        # Sort the data according to the residual value from smallest to largest, and save the data index in sorted_index
        sorted_index = [sorted_data[1] for sorted_data in sorted(residual, key = lambda x : x[0])]
        return sorted_index

class Match():

    def EU(network):
        pass
    def EU_LG(network):
        pass
    def LG_UA(network):
        pass
    def EU_LG_UA(network):

        times_enlarge, times_shrink = 0,0

        # Set up the learning rate of the network
        # network.learning_rate = 1e-3
        network.acceptable = False

        network_pre = copy.deepcopy(network)
        output, loss = network.forward()

        if torch.all(torch.abs(output - network.y) < network.threshold_for_error):
            
            network.acceptable = True
    #         print("Matching Successful")
            return network

        else:

            while times_enlarge + times_shrink < network.tuning_times:
                
                output, loss = network.forward()
                network_pre = copy.deepcopy(network)
                loss_pre = loss
                
                # Backward and check the loss performance of the network with new learning rate
                network.backward(loss)
                output, loss = network.forward()

                # Confirm whether the loss value of the adjusted network is smaller than the current one
                if loss <= loss_pre and torch.all(torch.abs(output - network.y) < network.threshold_for_error):

                    network.acceptable = True
                    #print("Matching Successful")
                    return network

                elif loss <= loss_pre:
                    
                    # Multiply the learning rate by 1.2
                    times_enlarge += 1
                    network.learning_rate *= 1.2
                else: #loss > loss_pre
                    
                    if network.learning_rate <= network.threshold_for_lr:

                        # If true, set the acceptance of the network as False and return initial_netwrok
                        network.acceptable = False
                        #print("Matching Failed")
                        return network_pre
                    
                    # On the contrary, restore w and adjust the learning rate
                    else:

                        # Restore the parameter of the network
                        network = copy.deepcopy(network_pre)
                        times_shrink += 1
                        network.learning_rate *= 0.7
        

        network.acceptable = False
    #     print("Matching Failed")
        return network_pre 



class Cramming():

    def ri_sro(network):

        torch.random.manual_seed(0)

        # Find unsatisfied data : k
        network_old = copy.deepcopy(network)
        output, loss = network.forward()

        undesired_index = torch.nonzero(torch.abs(output - network.y) > network.threshold_for_error + 1e-3, as_tuple = False)

        """
        [ [0] [1] [3] ] shape = (3,1), 代表有3筆非0的data。 演算法基礎上應只有一筆，thus undesired_index.shape[0] == 1
        """
        if undesired_index.shape[0] == 1: # if the undesired_index is one and only one.
            
            assert undesired_index[0][0] == len(network.y) - 1, "理應只有最後一筆需要cramming，此處出現異常"

            # Unsatisfied situation
            ## Find the index of the unsatisfied data
            k_data_num = undesired_index[0][0]
            undesired_data = torch.reshape(network.x[k_data_num,:], [1,-1])

            ## Remove the data that does not meet the error term
            left_data = network.x[:k_data_num,:]
            right_data = network.x[k_data_num+1:,:]
            remain_tensor = torch.cat([left_data, right_data], dim=0)
            
            ## Use the random method to find out the gamma and zeta
            while True:
                
                ## Find m-vector gamma: r
                ## Use the random method to generate the gamma that can meet the conditions
                gamma = torch.rand(size = [1, network.x.shape[1]]).to(network.device)
                subtract_undesired_data = torch.sub(remain_tensor, undesired_data)
                matmul_value = torch.mm(gamma, torch.t(subtract_undesired_data))

                if torch.all(matmul_value != 0):
                    break
            
            while True:

                ## Find the tiny value: zeta
                ## Use the random method to generate the zeta that can meet the conditions
                zeta = torch.rand(size=[1]).to(network.device)

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
            gap = network.y[-1] - output[-1]

            wo0_value = gap/zeta
            wo1_value = (-2*gap)/zeta
            wo2_value = gap/zeta

            Wo_new = torch.reshape(torch.cat([wo0_value, wo1_value, wo2_value], dim=0), [1,-1])

            ## Add new neuroes to the network
            network.linear1.weight = torch.nn.Parameter(torch.cat([network.linear1.weight.data, W1_new]))
            network.linear1.bias = torch.nn.Parameter(torch.cat([network.linear1.bias.data, b1_new]))
            network.linear2.weight = torch.nn.Parameter(torch.cat([network.linear2.weight.data, Wo_new],dim=1))
            
            network.nb_node += 3

            output, loss = network.forward()

            ## Determine if cramming is succesful and print out the corresponding information
            if torch.all(torch.abs(output - network.y) < network.threshold_for_error):
                network.acceptable = True
                print("Cramming successful")
                network.message = "Cramming successful"
                return network
            else:
                return None

class Reorganize():
        
    def ri_sro(network):
        
        limit = 1

        ## If the number of hidden node has already been 1, do the regularizing then. 
        if network.linear1.bias.shape[0] <= limit:
            network = Match.regularizing(network)
            # network = network.regularizing()
    #         print("Reorganizing finished")
            return(network)
        
        else:
            
            ## Set up the k = 1, and p = the number of hidden node
            k = 1
            p = network.linear1.weight.data.shape[0]

            while True:

                ## If k > p, end of Process
                if k > p or p <= limit:
    #                 print("Reorganizing finished")
                    return(network)

                ## Else, Process is ongoing
                else:

                    ## Using the regularizing module to adjust the network
                    network = Match.regularizing(network)
                    # network = network.regularizing()

                    ## Store the network and w
                    network_pre = copy.deepcopy(network)

                    ## Set up the acceptable of the network as false
                    network.acceptable = False
                    
                
                    ## Ignore the K hidden node
                    network.linear1.weight = torch.nn.Parameter(torch.cat([network.linear1.weight[:k-1],network.linear1.weight[k:]],0))
                    network.linear1.bias = torch.nn.Parameter(torch.cat([network.linear1.bias[:k-1],network.linear1.bias[k:]]))
                    network.linear2.weight = torch.nn.Parameter(torch.cat([network.linear2.weight[:,:k-1],network.linear2.weight[:,k:]],1))

                    
                    ## Using the matching module to adjust the network
                    network = Match.matching_for_reorganizing(network)

                    ## If the resulting network is acceptable, this means that the k hidden node can be removed
                    if network.acceptable:

                        network.nb_node -= 1
                        network.nb_node_pruned += 1
                        p-=1

                    ## Else, it means that the k hidden node cannot be removed
                    else:

                        ## Restore the network and w
                        network = copy.deepcopy(network_pre)
                        k+=1

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
                network.backward(loss)
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
            network.backward(loss)
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
                    return network

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
                    return network
                    
        #print("Regularizing finished")
        return network


def is_initializingNumber_too_big_to_initializing(index_of_data):

    if index_of_data == 1 : return True

def is_learningGoal_too_small_to_cramming(network):
    
    if network == None : return True


"""
------------------
    def __init__(self, network, x_train_scaled, y_train_scaled):
        self.network = network
        self.x_train_scaled = x_train_scaled
        self.y_train_scaled = y_train_scaled

"""

"""
物件導向設計
"""


class Initialize():

    def Default(network, initial_x, initial_y):
        
        # Find each minimum output value y
        min_y = min(initial_y)

        # Subtract min_y from each y
        res_y = initial_y - min_y
        
        # Use Linear regression to find the initial W1, b1, W2, b2
        reg = LinearRegression().fit(initial_x, res_y)

        # Set up the initial parameter of the network(through torch.nn.Parameter, we can turn non-trainable tensor into trainable tensor)
        network.linear1.weight = torch.nn.Parameter(torch.FloatTensor(reg.coef_).to(network.device))
        network.linear1.bias = torch.nn.Parameter(torch.FloatTensor(reg.intercept_).to(network.device))
        network.linear2.weight = torch.nn.Parameter(torch.FloatTensor([[1]]).to(network.device))
        network.linear2.bias = torch.nn.Parameter(torch.FloatTensor(min_y).to(network.device))

        network.acceptable = True

        return network

class Select():

    def LTS(network, x_train_scaled, y_train_scaled):
        
        residual = []
        temp_network = copy.deepcopy(network)

        # Put each data into network to calculate the loss value
        for i in range(x_train_scaled.shape[0]):
            temp_network.setData(x_train_scaled[i].reshape(1,-1), y_train_scaled[i].reshape(-1, 1))
            residual.append((temp_network.forward()[1].item(),i))
        
        
        # Sort the data according to the residual value from smallest to largest, and save the data index in sorted_index
        sorted_index = [sorted_data[1] for sorted_data in sorted(residual, key = lambda x : x[0])]
        return sorted_index

    # 還沒改成 POS
    def POS(network, x_train_scaled, y_train_scaled):
        
        residual = []
        temp_network = copy.deepcopy(network)

        # Put each data into network to calculate the loss value
        for i in range(x_train_scaled.shape[0]):
            temp_network.setData(x_train_scaled[i].reshape(1,-1), y_train_scaled[i].reshape(-1, 1))
            residual.append((temp_network.forward()[1].item(),i))
        
        
        # Sort the data according to the residual value from smallest to largest, and save the data index in sorted_index
        sorted_index = [sorted_data[1] for sorted_data in sorted(residual, key = lambda x : x[0])]
        return sorted_index

class Match():

    def EU(network):
        pass
    def EU_LG(network):
        pass
    def LG_UA(network):
        pass
    def EU_LG_UA(network):

        times_enlarge, times_shrink = 0,0

        # Set up the learning rate of the network
        # network.learning_rate = 1e-3
        network.acceptable = False

        network_pre = copy.deepcopy(network)
        output, loss = network.forward()

        if torch.all(torch.abs(output - network.y) < network.threshold_for_error):
            
            network.acceptable = True
    #         print("Matching Successful")
            return network

        else:

            while times_enlarge + times_shrink < network.tuning_times:
                
                output, loss = network.forward()
                network_pre = copy.deepcopy(network)
                loss_pre = loss
                
                # Backward and check the loss performance of the network with new learning rate
                network.backward(loss)
                output, loss = network.forward()

                # Confirm whether the loss value of the adjusted network is smaller than the current one
                if loss <= loss_pre and torch.all(torch.abs(output - network.y) < network.threshold_for_error):

                    network.acceptable = True
                    #print("Matching Successful")
                    return network

                elif loss <= loss_pre:
                    
                    # Multiply the learning rate by 1.2
                    times_enlarge += 1
                    network.learning_rate *= 1.2
                else: #loss > loss_pre
                    
                    if network.learning_rate <= network.threshold_for_lr:

                        # If true, set the acceptance of the network as False and return initial_netwrok
                        network.acceptable = False
                        #print("Matching Failed")
                        return network_pre
                    
                    # On the contrary, restore w and adjust the learning rate
                    else:

                        # Restore the parameter of the network
                        network = copy.deepcopy(network_pre)
                        times_shrink += 1
                        network.learning_rate *= 0.7
        

        network.acceptable = False
    #     print("Matching Failed")
        return network_pre 



class Cramming():

    def ri_sro(network):

        torch.random.manual_seed(0)

        # Find unsatisfied data : k
        network_old = copy.deepcopy(network)
        output, loss = network.forward()

        undesired_index = torch.nonzero(torch.abs(output - network.y) > network.threshold_for_error + 1e-3, as_tuple = False)

        """
        [ [0] [1] [3] ] shape = (3,1), 代表有3筆非0的data。 演算法基礎上應只有一筆，thus undesired_index.shape[0] == 1
        """
        if undesired_index.shape[0] == 1: # if the undesired_index is one and only one.
            
            assert undesired_index[0][0] == len(network.y) - 1, "理應只有最後一筆需要cramming，此處出現異常"

            # Unsatisfied situation
            ## Find the index of the unsatisfied data
            k_data_num = undesired_index[0][0]
            undesired_data = torch.reshape(network.x[k_data_num,:], [1,-1])

            ## Remove the data that does not meet the error term
            left_data = network.x[:k_data_num,:]
            right_data = network.x[k_data_num+1:,:]
            remain_tensor = torch.cat([left_data, right_data], dim=0)
            
            ## Use the random method to find out the gamma and zeta
            while True:
                
                ## Find m-vector gamma: r
                ## Use the random method to generate the gamma that can meet the conditions
                gamma = torch.rand(size = [1, network.x.shape[1]]).to(network.device)
                subtract_undesired_data = torch.sub(remain_tensor, undesired_data)
                matmul_value = torch.mm(gamma, torch.t(subtract_undesired_data))

                if torch.all(matmul_value != 0):
                    break
            
            while True:

                ## Find the tiny value: zeta
                ## Use the random method to generate the zeta that can meet the conditions
                zeta = torch.rand(size=[1]).to(network.device)

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
            gap = network.y[-1] - output[-1]

            wo0_value = gap/zeta
            wo1_value = (-2*gap)/zeta
            wo2_value = gap/zeta

            Wo_new = torch.reshape(torch.cat([wo0_value, wo1_value, wo2_value], dim=0), [1,-1])

            ## Add new neuroes to the network
            network.linear1.weight = torch.nn.Parameter(torch.cat([network.linear1.weight.data, W1_new]))
            network.linear1.bias = torch.nn.Parameter(torch.cat([network.linear1.bias.data, b1_new]))
            network.linear2.weight = torch.nn.Parameter(torch.cat([network.linear2.weight.data, Wo_new],dim=1))
            
            network.nb_node += 3

            output, loss = network.forward()

            ## Determine if cramming is succesful and print out the corresponding information
            if torch.all(torch.abs(output - network.y) < network.threshold_for_error):
                network.acceptable = True
                print("Cramming successful")
                network.message = "Cramming successful"
                return network
            else:
                return None

class Reorganize():
        
    def ri_sro(network):
        
        limit = 1

        ## If the number of hidden node has already been 1, do the regularizing then. 
        if network.linear1.bias.shape[0] <= limit:
            network = Match.regularizing(network)
            # network = network.regularizing()
    #         print("Reorganizing finished")
            return(network)
        
        else:
            
            ## Set up the k = 1, and p = the number of hidden node
            k = 1
            p = network.linear1.weight.data.shape[0]

            while True:

                ## If k > p, end of Process
                if k > p or p <= limit:
    #                 print("Reorganizing finished")
                    return(network)

                ## Else, Process is ongoing
                else:

                    ## Using the regularizing module to adjust the network
                    network = Match.regularizing(network)
                    # network = network.regularizing()

                    ## Store the network and w
                    network_pre = copy.deepcopy(network)

                    ## Set up the acceptable of the network as false
                    network.acceptable = False
                    
                
                    ## Ignore the K hidden node
                    network.linear1.weight = torch.nn.Parameter(torch.cat([network.linear1.weight[:k-1],network.linear1.weight[k:]],0))
                    network.linear1.bias = torch.nn.Parameter(torch.cat([network.linear1.bias[:k-1],network.linear1.bias[k:]]))
                    network.linear2.weight = torch.nn.Parameter(torch.cat([network.linear2.weight[:,:k-1],network.linear2.weight[:,k:]],1))

                    
                    ## Using the matching module to adjust the network
                    network = Match.matching_for_reorganizing(network)

                    ## If the resulting network is acceptable, this means that the k hidden node can be removed
                    if network.acceptable:

                        network.nb_node -= 1
                        network.nb_node_pruned += 1
                        p-=1

                    ## Else, it means that the k hidden node cannot be removed
                    else:

                        ## Restore the network and w
                        network = copy.deepcopy(network_pre)
                        k+=1

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
                network.backward(loss)
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
            network.backward(loss)
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
                    return network

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
                    return network
                    
        #print("Regularizing finished")
        return network