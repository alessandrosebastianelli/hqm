import sys
sys.path += ['.', './hqm/']
import numpy as np
import pennylane as qml


from hqm.circuits.flexiblecircuit import FlexibleCircuit


if __name__ == "__main__":

    config = {
                'F' : [
                        ['H', 'CNOT-1'], #Q0
                        ['H', 'CNOT-2'], #Q1
                        ['H', 'CNOT-0']  #Q2
                ],
                'U' : [
                        2*['RY', 'CNOT-1', 'RY'], #Q0
                        2*['RY', 'CNOT-2', 'RY'], #Q1
                        2*['RY', 'CNOT-0', 'RY']  #Q2
                ],
                'M' : [True, True, False]
            }


    fc = FlexibleCircuit(config=config)

    inputs  = [0, 0, 0]
    print(fc.weight_shape['weights'])
    weights = np.arange(fc.weight_shape['weights'])
    print(weights) 

    drawer = qml.draw(fc.circuit)
    print(drawer(inputs, weights))