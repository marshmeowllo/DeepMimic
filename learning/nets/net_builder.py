import learning.nets.fc_2layers_1024units as fc_2layers_1024units
import learning.nets.fc_2layers_gated_1024units as fc_2layers_gated_1024units

def build_net(net_name, input_tfs, reuse=False):
    # net_name: the name of the neural network architecture.
    # input_tfs: The input tensors for the network.
    # reuse: A Boolean flag indicating whether to reuse variables, useful in TensorFlow if you want to share weights or replicate parts of the network.

    net = None

    if (net_name == fc_2layers_1024units.NAME):
        net = fc_2layers_1024units.build_net(input_tfs, reuse)
    elif (net_name == fc_2layers_gated_1024units.NAME):
        net = fc_2layers_gated_1024units.build_net(input_tfs, reuse)
    else:
        assert False, 'Unsupported net: ' + net_name
    
    return net