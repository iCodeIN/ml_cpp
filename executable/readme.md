# nn

**nn** is a command-line utility offering neural networks in a linux-style fashion. 

## training

In training mode, **nn** allows you to pipe an input file, specify network parameters (or a pre-existing network) and train the network using gradient descent (using a specified learning rate decay schedule).

    --size "2,4,4,1"
    Allows you to specify the dimensions of each layer in the neural network
    Starting at the input layer (2 neurons in this example), all the way to the output
    layer (a single neuron).
    There is no default value (since the number of neurons in the input and output layer depends on the
    data you are feeding the network)
    
    --iterations 1024
    Allows you to specify the number of iterations to go through when training
    the neural network. The default value is 1024.
    
    --i input_file
    Allows you to specify an existing neural network
    By doing so, you can (continue) train(ing) an existing network
    
    --o output_file
    Allows you to specify a file location, where to output the weights of the
    neural network once the program has finished training
