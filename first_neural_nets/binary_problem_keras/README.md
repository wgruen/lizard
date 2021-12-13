This machine is using keras. In order to do experiments, there is a YAML configuration file as input.
The main challenge is the design of the machine. 

For this simple problem, the input data is provided in an array.


#How to use
Train:
train_machine.py-i logical_or.configuration.keras.yaml


Training will save the machine and the machine can be loaded to run it with the trained parameters.

Run:
run_machine.py -i logical_or.configuration.keras.yaml


#Problem Logical OR3
The example logical_or.* is about training the machine for a logical OR

How to use
train_machine.py-i logical_or.configuration.keras.yaml
run_machine.py -i logical_or.configuration.keras.yaml

Looks like this training set delivers the expected results
0.04  0.03  0.07  0.01
0.09  0.03  0.95  0.99
0.04  0.96  0.01  0.99
0.98  0.04  0.02  0.99


##This machine seems to work pretty well for that problem
The yaml file show the machine layout

#Problem Logical XOR2
The example logical_or.* is about training the machine for a logical OR

How to use
train_machine.py-i logical_xor2.configuration.keras.yaml
run_machine.py -i logical_xor2.configuration.keras.yaml

Looks like this training set delivers the expected results
0.04  0.03  0.01
0.04  0.96  0.99
0.98  0.02  0.99
0.96  0.97  0.01

##This machine seems to work pretty well for that problem
The yaml file show the machine layout with two neurons in the hidden layer


    
    
    


