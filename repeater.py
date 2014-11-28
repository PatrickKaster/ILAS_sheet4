#!/usr/bin/env python3
import subprocess
import time
import profile
from backpropagation import *

eta_values = [6, 3, 2, 1, 0.5, 0.4, 0.3, 0.25, 0.2]
for value in eta_values:
    print("============== Run with eta={} ==============".format(value))
    nn = NeuronalNetwork(2,[2],1) 
    examples = generate_examples(xor,2)
    print(nn)
    profile.run('nn.backward_propagation(examples,value,100000,0)')
    print(nn)
    for example in examples:
        print(example)
        nn.forward_propagation(example)
        print(nn.output_nodes[0].output_value)

    
