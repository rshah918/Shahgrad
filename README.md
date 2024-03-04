# Shahgrad
Shahgrad is an autograd engine and ML library useful for building neural networks from scratch. It provides the ability to create and execute mathematical expressions, backprop gradients through the expression graphs, and includes a lightweight neural net library. The library also offers visualization capabilities to explore the mathematical expression graph of any arbitrary neural net. 

For quick start, MNIST training code is included in `/MNIST`:
    
1: `cd MNIST`

2: `make` 

3: `./a.out`

<img width="800" alt="Screenshot 2024-02-11 at 3 06 00 PM" src="https://github.com/rshah918/Shahgrad/assets/20956909/57b4e1c7-b830-430d-92db-e728f136939d">


Made this to better understand how pytorch/tensorflow work under the hood. 
## Features

* Autograd Engine: Create and manipulate mathematical expressions.
* Gradient Backpropagation: Backprop gradients through any mathematical expression.
* Neural Net Library: Implement neural networks using the mathematical expression graphs as building blocks.
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
#### Forward pass inputs through a neuron, view its expression graph and gradients, and perform backprop: 
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

### Neural Net Demo
#### Create arbitrary neural nets by leveraging the shahgrad gradient engine:
```
    /*
    Demo that trains a small neural network to fit against y=x^2 .
    */
    //create x_train
    int input_vector_length = 1;
    vector<vector<Value*> > X_train;
    for(float i = 0; i < 1; i += 0.001){
        X_train.push_back(vector<Value*>(1, new Value(i)));
    }

    //create y_train
    vector<vector<float> > Y_train;
    for(float i = 0; i < 1; i += 0.001){
        Y_train.push_back(vector<float>(1, i * i));
    }
    cout << Y_train[0].size() << endl;
    //create model
    Model m = * new Model(input_vector_length);
    m.add_layer("linear", 6, "exp");
    m.add_layer("linear", 1, "exp");
    m.compile(X_train[0]);
    //train and visualize
    m.train(X_train, Y_train,30, learning_rate = 0.001, "mean_squared_error");
    m.layers.back()->visualizeGraph();
    //forward pass entire training set and save NN outputs
    std::ofstream outputFile("output_values.txt");
    for(int i=1;i<X_train.size();i++){
        m.forward(X_train[i]);
        outputFile << m.outputs[0]->data << "\n";
    }
    outputFile.close();
    //plot NN outputs against y=x^2
    system("python3 plot.py");
```

<img width="584" alt="Screenshot 2024-03-03 at 7 17 24 PM" src="https://github.com/rshah918/Shahgrad/assets/20956909/022107a2-4d0c-4073-a539-3474693b76f2">


<img width="958" alt="Screenshot 2024-03-03 at 7 16 50 PM" src="https://github.com/rshah918/Shahgrad/assets/20956909/5bd3708a-f412-415d-b5cd-8608bb215365">


