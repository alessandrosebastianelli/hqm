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
                        ['H', 'CNOT-1'], #Q0  
                        ['H', 'CNOT-2'], #Q1  
                        ['H', 'CNOT-0']  #Q2  
                ],  
                'U' : [  
                        2*['RY', 'CNOT-1', 'RY'], #Q0  
                        2*['RY', 'CNOT-2', 'RY'], #Q1  
                        2*['RY', 'CNOT-0', 'RY']  #Q2  
                ],  
                'M' : [True, True, True]  
            }  

            will result in  
                            *===== F ====*======== U1 =========*======== U2 ==========*= M =*  
                    ___              
            |0> ---|   | --- H - X ----- | - Ry - X ----- | - Ry - Ry - X ----- | - Ry - M0  
            |0> ---| E | --- H - | - X - | - Ry - | - X - | - Ry - Ry - | - X - | - Ry - M1  
            |0> ---|___| --- H ----- | - X - Ry ----- | - X - Ry - Ry ----- | - X - Ry - M2  

                        
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
        super().__init__(n_qubits=np.shape(config['U'])[0], n_layers=1, dev=dev)

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
        for el in list(chain(*config['F'])):
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

            ct = 0
            # V component
            V = config['F']
            for j in range(np.shape(V)[1]):
                for i in range(np.shape(V)[0]):
                    ct = decode_gates(key=V[i][j], qubit=i, weights=weights, ct=ct)

            # U Component
            U = config['U']
            for j in range(np.shape(U)[1]):
                for i in range(np.shape(U)[0]):
                    ct = decode_gates(key=U[i][j], qubit=i, weights=weights, ct=ct)

            # M component
            measurements = []
            for i in range(n_qubits): 
                if config['M'][i]:
                    measurements.append(qml.expval(qml.PauliZ(wires=i)))

            return measurements
    
        return qnode
    
def decode_gates(key : str, qubit : int, weights : np.ndarray, ct : int):
    '''
        Decode string into qml gate

        Parameters:  
            -----------  
            - key : str
                string representing gate
            - qubit : int 
                to which qubit apply the gate
            - weights : np.ndarray  
                array containing the weights of the circuit that are tuned during training, the shape of this
                array depends on circuit's layers and qubits. 
            - ct : int
                counter that keeps track of weight position

            Returns:  
            --------  
            Nothing
    '''

    if key == 'H':
        qml.Hadamard(wires=qubit)
    if key == 'RY':
        qml.RY(weights[ct], wires=qubit)
        ct += 1
    if key == 'RX':
        qml.RX(weights[ct], wires=qubit)
        ct += 1
    if key == 'RZ':
        qml.RZ(weights[ct], wires=qubit)
        ct += 1
    if 'CNOT' in key:
        qx = int(key.split('-')[-1])
        qml.CNOT(wires=[qubit, qx])

    return ct