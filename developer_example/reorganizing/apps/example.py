def reorganizing(network):
    
    limit = 1

    ## If the number of hidden node has already been 1, do the regularizing then. 
    if network.linear1.bias.shape[0] <= limit:
        network = regularizing(network)
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
                network = regularizing(network)
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
                network = matching_for_reorganizing(network)

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
    # network.model_params["learningRate"] = 1e-3
    network.acceptable = False

    network_pre = copy.deepcopy(network)
    output, loss = network.forward()

    if torch.all(torch.abs(output - network.y) < network.model_params["learningGoal"]):
        
        network.acceptable = True
#         print("Matching(Re) Successful")
        return network

    else:

        while times_enlarge + times_shrink < network.model_params["matchingTimes"]:
            
            output, loss = network.forward()
            network_pre = copy.deepcopy(network)
            loss_pre = loss
            
            # Backward and check the loss performance of the network with new learning rate
            network.backward(loss)
            output, loss = network.forward()

            # Confirm whether the loss value of the adjusted network is smaller than the current one
            if loss <= loss_pre and torch.all(torch.abs(output - network.y) < network.model_params["learningGoal"]):

                network.acceptable = True
                #print("Matching(Re) Successful")
                return network

            elif loss <= loss_pre:
                
                # Multiply the learning rate by 1.2
                times_enlarge += 1
                network.model_params["learningRate"] *= 1.2
            else: #loss > loss_pre
                
                if network.model_params["learningRate"] <= network.model_params["matchingLearningRateLowerBound"]:

                    # If true, set the acceptance of the network as False and return initial_netwrok
                    network.acceptable = False
                    #print("Matching(Re) Failed")
                    return network_pre
                
                # On the contrary, restore w and adjust the learning rate
                else:

                    # Restore the parameter of the network
                    network = copy.deepcopy(network_pre)
                    times_shrink += 1
                    network.model_params["learningRate"] *= 0.7
    

    network.acceptable = False
#     print("Matching(Re) Failed")
    return network_pre       

def regularizing(network):
    
    ## Record the number of executions
    times_enlarge, times_shrink = 0, 0

    # ## Set up the learning rate of the network
    # network.model_params["learningRate"] = 1e-3

    ## Set epoch to 100
    for i in range(network.model_params["matchingTimes"]):

        ## Store the parameter of the network
        network_pre = copy.deepcopy(network)
        output, loss = network.forward(network.model_params["regularizingStrength"])
        loss_pre = loss

        ## Backward operation to optain w'
        network.backward(loss)
        output, loss = network.forward(network.model_params["regularizingStrength"])

        ## Confirm whether the adjusted loss value is smaller than the current one
        if loss <= loss_pre:

            ## Identify that all forecast value has met the error term
            if torch.all(torch.abs(output - network.y) < network.model_params["learningGoal"]):
                
                ## If true, multiply the learning rate by 1.2
                network.model_params["learningRate"] *= 1.2
                times_enlarge += 1

            else:

                ## Else, restore w and end the process
                network = copy.deepcopy(network_pre)
                return network

        # If the adjusted loss value is not smaller than the current one
        else:

            ## If the learning rate is greater than the threshold for learning rate
            if network.model_params["learningRate"] > network.model_params["regularizingLearningRateLowerBound"]:
                
                ## Restore the w and multiply the learning rate by 0.7
                network = copy.deepcopy(network_pre)
                network.model_params["learningRate"] *= 0.7
                times_shrink += 1

            ## If the learning rate is smaller than the threshold for learning rate
            else:
                
                ## Restore the w
                network = copy.deepcopy(network_pre)
#                 print("Regularizing finished")
                return network
                
    #print("Regularizing finished")
    return network