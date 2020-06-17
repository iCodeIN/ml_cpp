# nn

**nn** is a command-line utility offering neural networks in a linux-style fashion. 

## parameters

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
         
## example usage

1. Create an example data file containing the XOR gate logic


    0   0   0
    0   1   1
    1   0   1
    1   1   0
    
    
2. Train a neural network on the example data

   cat example_xor | ./nn -train -size "2,3,1" -iterations 100 -o "nn_xor_100" 
   
   
3. You should now have a file nn_xor_100 containing the weights of the neural network

    2           3
    0.72142     0.195587        0.424347
    0.595779    0.000991821     0.72287
    3           1
    0.72142
    0.195587
    0.424347
    

4. Now you can evaluate the performance (determine the loss) using your pre-trained network


    cat example_xor | ./nn -loss -i "nn_xor_100"
    [[0.287842]]
    
5. Let's train the network a bit more  to see whether the performance improves


    cat example_xor | ./nn -train -size "2,3,1" -iterations 1000 -i "nn_xor_100" -o "nn_xor_1000"
    cat example_xor | ./nn -loss -i "nn_xor_1000"
    [[0.132258]]
    
6. Finally, let's see the network's predictions for the output

    [[0.0196405]
     [0.499367]
     [0.978964]
     [0.500722]]
