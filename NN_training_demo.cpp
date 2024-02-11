#include <iostream>
#include <vector>
#include <random>
#include "shahgrad.cpp"
using namespace std;

void expression_engine_demo(){
    //Demonstration of the gradient engine
    //inputs
    Value a = Value(2);
    Value b = Value(3);
    Value c = Value(-3);
    Value d = Value(1);

    //expression
    Value * a_times_b = a * &b;
    a_times_b->label = "a times b";
    Value * c_times_d = c * &d;
    c_times_d->label = "c times d";
    Value * res = *a_times_b + c_times_d;
    res->label = "output";

    res->grad = 1.0; //init root node grad to 1

    //backprop and visualize
    res->backprop();
    res->visualizeGraph();
}

void single_neuron_demo(){
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
    n->forward();
    //Set output gradient and backpropagate
    n->out->grad = 1.0;
    n->backward();
    n->out->visualizeGraph();
}

void linear_layer_demo(){
    /*
    Demo of a Linear layer
    */
    vector<Value*> inputs;
    //create input vector
    int input_vector_length = 2;
    for(int i = 0; i < input_vector_length; i++){
        inputs.push_back(new Value(i + 1.0));
        inputs[i]->label = "input"; //label all the input nodes for visualization purposes
    }
    int output_size = 3;
    Linear l = Linear(input_vector_length, output_size);
    l.forward();
    cout << (l.outputs[0]->data) << endl;
    //initialize output gradients to 1
    vector<float> out_grads;
    for(int i = 0; i < output_size;i++){
        out_grads.push_back(1.0/(output_size-1));
    }
    l.backward(out_grads);
    l.visualizeGraph();
}

int main(){
    /*
    Demo that trains a small neural network to learn an exp function.
    */
     //create x_train
    int input_vector_length = 1;
    vector<vector<Value*> > X_train;
    for(int i = 0; i < 1000; i++){
        vector<Value*> input;
        input.push_back(new Value(i));
        X_train.push_back(input);
    }
    //create y_train
    vector<vector<float> > Y_train;
    for(int i = 0; i < 1000; i++){
        vector<float> input;
        input.push_back(i * i);
        cout << input[0] << endl;
        Y_train.push_back(input);
    }
    //create model
    Model m = * new Model(input_vector_length);
    m.add_layer("linear", 1, "");
    m.add_layer("linear", 1, "");
    m.compile(X_train[0]);
    //train and visualize
    m.train(X_train, Y_train,40, learning_rate = 0.0000000001, "mean_squared_error"); //disgustingly low lr, results in exploding grad otherwise
    m.layers.back()->visualizeGraph();
    return 0;
}