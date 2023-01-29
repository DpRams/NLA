def matching(network):

    times_enlarge, times_shrink = 0,0

    network.acceptable = False

    network_pre = copy.deepcopy(network)
    output, loss = network.forward()

    if torch.all(torch.abs(output - network.y) < network.model_params["learningGoal"]):
        
        network.acceptable = True
#         print("Matching Successful")
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
                #print("Matching Successful")
                return network

            elif loss <= loss_pre:
                
                # Multiply the learning rate by 1.2
                times_enlarge += 1
                network.model_params["learningRate"] *= 1.2
            else: #loss > loss_pre
                
                if network.model_params["learningRate"] <= network.model_params["matchingLearningRateLowerBound"]:

                    # If true, set the acceptance of the network as False and return initial_netwrok
                    network.acceptable = False
                    #print("Matching Failed")
                    return network_pre
                
                # On the contrary, restore w and adjust the learning rate
                else:

                    # Restore the parameter of the network
                    network = copy.deepcopy(network_pre)
                    times_shrink += 1
                    network.model_params["learningRate"] *= 0.7
    

    network.acceptable = False
#     print("Matching Failed")
    return network_pre