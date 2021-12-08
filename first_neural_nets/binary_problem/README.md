I wrote a machine with one hidden layer. In order to do experiments, there is a YAML configuration file as input.
The machine has a configurable number of Input, Hidden, and Output neurons. Currenlty only one layer of hidden neurons is supported!

For this simple problem, the input data is provided in an array. The machine which does learning, backpropogation, etc happens is in a library.
The app will load the machine from the checkpoint (dump) file and then use the input to test the machine.


Example_1 is about training the machine for a logical OR

Train:
train_machine.py-i logical_or.configuration.yaml


Training will save the machine and the machine can be loaded to run it with new data.

Run:
run_machine.py -i logical_or.configuration.yaml



