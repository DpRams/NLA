def cramming(network):

    torch.random.manual_seed(0)

    # Find unsatisfied data : k
    network_old = copy.deepcopy(network)
    output, loss = network.forward()

    undesired_index = torch.nonzero(torch.abs(output - network.y) > network.model_params["learningGoal"] + 1e-3, as_tuple = False)

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
        if torch.all(torch.abs(output - network.y) < network.model_params["learningGoal"]):
            network.acceptable = True
            print("Cramming successful")
            network.message = "Cramming successful"
            return network
        else:
            return None