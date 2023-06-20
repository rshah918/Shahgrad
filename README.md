# Shahgrad
Shahgrad is a mathematical expression engine and ML library designed for building neural networks from scratch. It provides the ability to create and run mathematical expressions, backprop gradients through the expression graphs, and includes a lightweight neural net library. The library also offers visualization capabilities to explore the mathematical expression graph of any neural net.

Made this to better understand how pytorch/tensorflow work under the hood.
## Features

* Mathematical Expression Engine: Create and manipulate mathematical expressions.
* Gradient Backpropagation: Backprop gradients through any mathematical expression.
* Neural Net Library: Implement neural networks using the mathematical expressions as building blocks.
* Expression Graph Visualization: Visualize the expression graph of neural networks for better understanding.

## Gradient Engine demo

```//Demonstration of the gradient engine
    //inputs
    Value x1 = Value(2);
    Value x2 = Value(3);
    Value w1 = Value(-3);
    Value w2 = Value(1);
    Value bias = Value(6.8814);

    //build expression
    Value x1w1 = x1* w1;
    x1w1.label = "x1w1";
    Value x2w2 = x2*w2;
    x2w2.label = "x2w2";
    Value x1w1x2w2 = (x1w1 + x2w2);
    Value x1w1x2w2b = x1w1x2w2 + bias;
    x1w1x2w2b.label = "x1w1 + x2w2 + bias";
    Value res = x1w1x2w2b;

    res.label = "output";

    //init root node gradient to 1
    res.grad = 1.0;

    //backprop gradients and visualize
    res.backprop();
    res.visualizeGraph();
```

##### expression_graph.png
<img width="410" alt="Screenshot 2023-06-19 at 7 06 57 PM" src="https://github.com/rshah918/Shahgrad/assets/20956909/3e686cbd-77a3-46b6-b4f7-94a6bde2a0f2">

### Neuron Demo

```
    Neuron* n = new Neuron(5);//instantiate neuron with input size of 5
    //input vector
    vector<Value*> inputs;
    inputs.push_back(new Value(1.0));
    inputs.push_back(new Value(2.0));
    inputs.push_back(new Value(3.0));
    inputs.push_back(new Value(4.0));
    inputs.push_back(new Value(5.0));
    //label all the inputs
    for(int i = 0; i < inputs.size(); i++){
        inputs[i]->label = "input";
    }
    /forward pass inputs through the neuron
    n->forward(inputs);
    //backprop gradients
    n->backward();
    //visualize the neuron's mathematical expression graph
    n->out.visualizeGraph();
```
##### expression_graph.png
<img width="467" alt="Screenshot 2023-06-19 at 7 10 00 PM" src="https://github.com/rshah918/Shahgrad/assets/20956909/f7cb5164-f5d4-4846-b8b6-d703a365205e">
