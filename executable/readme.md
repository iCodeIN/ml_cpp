# nn

**nn** is a command-line utility offering neural networks in a linux-style fashion. 

## example usage

### 1. Create an example data file containing the XOR gate logic

    0   0   0
    0   1   1
    1   0   1
    1   1   0
    
### 2. Train a neural network on the example data

   cat example_xor | ./nn -train -size "2,3,1" -iterations 100 -o "nn_xor_100" 
   
   
### 3. You should now have a file nn_xor_100 containing the weights of the neural network

    2           3
    0.72142     0.195587        0.424347
    0.595779    0.000991821     0.72287
    3           1
    0.72142
    0.195587
    0.424347
    
### 4. Now you can evaluate the performance (determine the loss) using your pre-trained network

    cat example_xor | ./nn -loss -i "nn_xor_100"
    [[0.287842]]
    
### 5. Let's train the network a bit more  to see whether the performance improves

    cat example_xor | ./nn -train -size "2,3,1" -iterations 1000 -i "nn_xor_100" -o "nn_xor_1000"
    cat example_xor | ./nn -loss -i "nn_xor_1000"
    [[0.132258]]
    
### 6. Finally, let's see the network's predictions for the output

    [[0.0196405]
     [0.499367]
     [0.978964]
     [0.500722]]
