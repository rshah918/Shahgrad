# Shahgrad
Shahgrad is ann autograd engine and ML library designed for building neural networks from scratch. It provides the ability to create and run mathematical expressions, backprop gradients through the expression graphs, and includes a lightweight neural net library. The library also offers visualization capabilities to explore the mathematical expression graph of any neural net.

Made this to better understand how pytorch/tensorflow work under the hood.
## Features

* Autograd Engine: Create and manipulate mathematical expressions.
* Gradient Backpropagation: Backprop gradients through any mathematical expression.
* Neural Net Library: Implement neural networks using the mathematical expressions as building blocks.
* Expression Graph Visualization: Visualize the expression graph of neural networks for better understanding.

## Gradient Engine demo

```
    //Demonstration of the gradient engine
    //inputs
    Value a = Value(2);
    Value b = Value(3);
    Value c = Value(-3);
    Value d = Value(1);

    //expression
    Value a_times_b = a * b;
    a_times_b.label = "a times b";
    Value c_times_d = c * d;
    c_times_d.label = "c times d";
    Value res = a_times_b + c_times_d;
    res.label = "output";

    res.grad = 1.0; //init root node grad to 1

    //backprop and visualize
    res.backprop();
    res.visualizeGraph();
```

##### expression_graph.png
<img width="430" alt="Screenshot 2023-06-19 at 9 33 07 PM" src="https://github.com/rshah918/Shahgrad/assets/20956909/84ef14ab-915d-4ee4-ad3a-afa6580b2a90">


### Neuron Demo
#### Forward pass inputs through a neuron, and view its expression graph and gradients: 
```
    //instantiate neuron with input size of 5
    Neuron* n = new Neuron(5);
    //create input vector
    vector<Value*> inputs;
    for(int i = 0; i < inputs.size(); i++){
        inputs.push_back(new Value(i+1));
        //label all the inputs
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
