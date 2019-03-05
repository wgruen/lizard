There are two machines in this directory. One machine with just one neuron and another with 7 neurons.

a)
The machine with just one neuron was copied from the web and slightly modified. This simple machine is not able to classify the input of [0,0,0] properly.

Considering new situation [0 0 0] ->  (array([0.5]), array([0.]))
Considering new situation [0 0 1] ->  (array([0.00316459]), array([0.]))
Considering new situation [0 1 0] ->  (array([0.99998549]), array([1.]))
Considering new situation [0 1 1] ->  (array([0.99544905]), array([1.]))
Considering new situation [1 0 0] ->  (array([0.5884634]), array([1.]))
Considering new situation [1 0 1] ->  (array([0.00451896]), array([0.]))
Considering new situation [1 1 0] ->  (array([0.99998985]), array([1.]))
Considering new situation [1 1 1] ->  (array([0.99681298]), array([1.]))


b)
I believed that a machine with at least one hidden layer would have a better chance to classify that pattern properly. The second machine was written, based on the simple machine's code one day. It has 7 neurons, 3 for input, 3 hidden and 1 output neuron. This more complex machine is capable of classifying [0,0,0] properly.

Considering new situation [0 0 0] ->  [0.09700252]
Considering new situation [0 0 1] ->  [0.09700536]
Considering new situation [0 1 0] ->  [0.90413788]
Considering new situation [0 1 1] ->  [0.9041399]
Considering new situation [1 0 0] ->  [0.09832533]
Considering new situation [1 0 1] ->  [0.0983282]
Considering new situation [1 1 0] ->  [0.90507061]
Considering new situation [1 1 1] ->  [0.90507262]


c)
My next project may be a machine which detects a pattern in a byte stream. Similar the way sonar or radar has to  find and correlate the response/reflection in a noisy environment.


