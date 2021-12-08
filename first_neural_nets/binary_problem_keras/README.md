This machine is using keras. In order to do experiments, there is a YAML configuration file as input.
The main challenge is the design of the machine. 

For this simple problem, the input data is provided in an array.


The example logical_or.* is about training the machine for a logical OR

Train:
train_machine.py-i logical_or.configuration.keras.yaml


Training will save the machine and the machine can be loaded to run it with the trained parameters.

Run:
run_machine.py -i logical_or.configuration.keras.yaml



