# Shahgrad
Shahgrad is an autograd engine and ML library designed for building neural networks from scratch. It provides the ability to create and run mathematical expressions, backprop gradients through the expression graphs, and includes a lightweight neural net library. The library also offers visualization capabilities to explore the mathematical expression graph of any neural net.

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
 /*
    Demo of a single neuron forward/backward pass + visualization
    */
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
    //Instantiate Neuron and forward pass
    int input_vector_length = inputs.size();
    Neuron* n = new Neuron(input_vector_length);
    n->forward(inputs);
    //Set output gradient and backpropagate
    n->out.grad = 1.0;
    n->backward();
    n->out.visualizeGraph();
```
##### expression_graph.png
<img width="973" alt="Screenshot 2023-11-11 at 9 06 31 AM" src="https://github.com/rshah918/Shahgrad/assets/20956909/e7898c6c-4ed5-4cce-947c-429982b79eec">

