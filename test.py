import torch
import copy
from apps import getFreerGpu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from network.net import TwoLayerNet, Network
from module.modules import Initialize, Select, Match, Cramming, Reorganize

"""
暫時存放YourCSI而已，無實用
"""

class YourCSI():
    def __init__(self, **model_params):
        self.net = yourCSI(**model_params)

    def initializing(self, initial_x, initial_y):
        self.net.initializing(self, initial_x, initial_y)

    def selecting(self, x_train_scaled, y_train_scaled):
        self.net.selecting(self, x_train_scaled, y_train_scaled)

    def matching(self):
        self.net = self.net.matching(self)

    def cramming(self):
        self.net.cramming(self)

    def matching_reorganizing(self):
        self.net = self.net.matching_reorganizing(self)
            
    def regularizing(self):
        self.net = self.net.regularizing(self)

    def reoranizing(self):
        self.net = self.net.reoranizing(self)

# TwoLayerNet -> Network
class yourCSI(Network): 

    def __init__(self, **model_params):
        super().__init__(**model_params)

    def initializing(self, initial_x, initial_y):
        Initialize.Default(self, initial_x, initial_y)

    def selecting(self, x_train_scaled, y_train_scaled):
        sorted_index = Select.LTS(self, x_train_scaled, y_train_scaled)
        return sorted_index

    def matching(self):
        # matching_fn = "EU_LG"
        # eval(matching_fn)(self)
        return Match.EU_LG_UA(self)

    def cramming(self):
        return Cramming.ri_sro(self)

    def matching_reorganizing(self):
        return Reorganize.ri_sro(self)
            
    def regularizing(self):
        return Reorganize.regularizing(self)

    def reoranizing(self):
        return Reorganize.ri_sro(self)

network = YourCSI()
print(network)