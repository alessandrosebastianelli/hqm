from types import FunctionType
from itertools import chain
import pennylane as qml
import numpy as np
import sys

sys.path += ['.', './utils/']

from .circuit import QuantumCircuit

class FlexibleCircuit(QuantumCircuit):
    '''
        This class implements a torch/keras quantum layer using the flexible circuit. 
    '''
    
    def __init__(self, config : dict, dev : qml.devices = None, encoding : str = 'angle') -> None:
        '''
            FlexibleCircuit constructor. 

                                       ^ --> 
                                       | <--  
                    ___       ___       ___
            |0> ---|   | --- |   | --- |   | --- M
            |0> ---| E | --- | F | --- | U | --- M
            |0> ---| . | --- |   | --- |   | --- M
            |0> ---| . | --- |   | --- |   | --- M
            |0> ---|___| --- |___| --- |___| --- M

            Where E is the encoding layer, F is a fixed layer and U is a configurable
            and repeating layer. The configuration can be changed via a dictionary. 
            For instance, for a 3 qubits, 2 layers and full measurement circuit:

            config = {
                'F' : [
                        ['H', 'CNOT-2'], #Q1
                        ['H', 'CNOT-3'], #Q2
                        ['H', 'CNOT-1']  #Q3
                ],
                'U' : [
                        2*['RY', 'CNOT-2', 'RY'], #Q1
                        2*['RY', 'CNOT-3', 'RY'], #Q2
                        2*['RY', 'CNOT-1', 'RY']  #Q3
                ],
                'M' : [True, True, True]
            }

            will result in
                            *===== F ====*======== U1 =========*======== U2 ==========*= M =*
                    ___              
            |0> ---|   | --- H - X ----- | - Ry - X ----- | - Ry - Ry - X ----- | - Ry - M1
            |0> ---| E | --- H - | - X - | - Ry - | - X - | - Ry - Ry - | - X - | - Ry - M2
            |0> ---|___| --- H ----- | - X - Ry ----- | - X - Ry - Ry ----- | - X - Ry - M3

                        
            Parameters:  
            -----------
            - n_qubits : int  
                number of qubits for the quantum circuit  
            - n_layers : int  
                number of layers for the U circuit 
            - config : dict
                dictionary that configures F and U circuit      
            - dev : qml.device  
                PennyLane device on wich run quantum operations (dafault : None). When None it will be set
                to 'default.qubit'  
            - encoding : str  
                string representing the type of input data encoding in quantum circuit, can be 'angle' or 'amplitude'
            
            Returns:  
            --------  
            Nothing, a RealAmplitudesCircuit object will be created.  
        '''
        super().__init__(n_qubits=n_qubits, n_layers=n_layers, dev=dev)

        if encoding not in ['angle', 'amplitude']: raise(f"encoding can be angle or amplitude, found {encoding}")
        if 'F' not in config.keys(): raise(f'Config does not contain configuration for circuit F component, found {config.keys()}')
        if 'U' not in config.keys(): raise(f'Config does not contain configuration for circuit U component, found {config.keys()}')
        if 'M' not in config.keys(): raise(f'Config does not contain configuration for circuit M component, found {config.keys()}')

        self.config       = config
        self.encoding     = encoding
        self.n_qubits     = np.shape(config['U'])[0]


        self.weight_shape = {"weights": (self.__calculate_weights(config))}
        self.circuit      = self.circ(self.dev, self.n_qubits, self.config, self.encoding)

    def __calculate_weights(self, config):
        '''
            Calculates the numer of rotational gates to infer the weights shape.

            Parameters:
            -----------
            - config : dict
                dictionary that configures F and U circuit      
            
            Returns:  
            --------  
            ct : int
                counts of R gates.
        '''
        
        ct = 0
        for el in list(chain(*config['V'])):
            if 'R' in el:
                ct += 1
        
        for el in list(chain(*config['U'])):
            if 'R' in el:
                ct += 1

        return ct

    @staticmethod
    def circ(dev : qml.devices, n_qubits : int, config: dict, encoding : str) -> FunctionType:
        '''
            FlexibleCircuit static method that implements the quantum circuit.  

            Parameters:  
            -----------  
            - dev : qml.device  
                PennyLane device on wich run quantum operations (dafault : None). When None it will be set  
                to 'default.qubit'   
            - n_qubits : int  
                number of qubits for the quantum circuit 
            - config : dict
                dictionary that configures F and U circuit  
            - encoding : str  
                string representing the type of input data encoding in quantum circuit, can be 'angle' or 'amplitude'
            
            Returns:  
            --------  
            - qnode : qml.qnode  
                the actual PennyLane circuit   
        '''
        @qml.qnode(dev)
        def qnode(inputs : np.ndarray, weights : np.ndarray) -> list:
            '''
                PennyLane based quantum circuit composed of an angle embedding, fixed and configurable layers.

                Parameters:  
                -----------  
                - inputs : np.ndarray  
                    array containing input values (can came from previous torch/keras layers or quantum layers)  
                - weights : np.ndarray  
                    array containing the weights of the circuit that are tuned during training, the shape of this
                    array depends on circuit's layers and qubits.   
                
                Returns:  
                --------  
                - measurements : list  
                    list of values measured from the quantum circuits  
            '''

            # E component
            if encoding == 'angle':     qml.AngleEmbedding(inputs, wires=range(n_qubits))
            if encoding == 'amplitude': qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True)

            # V component
            V = config['V']

            for i in range(V.shape[0]):
                for j in range(V.shape[1]):
                    pass
            
            # U Component
            U = config['U']
            for i in range(U.shape[0]):
                for j in range(U.shape[1]):
                    pass

            # M component
            measurements = []
            for i in range(n_qubits): 
                if config['M'][i]:
                    measurements.append(qml.expval(qml.PauliZ(wires=i)))

            return measurements
    
        return qnode