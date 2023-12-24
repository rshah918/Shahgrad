#include <iostream>
#include <vector>
#include <numeric>
#include<math.h>
#include <fstream>
#include <unordered_set>
using namespace std;

class Value{
    /* 
        This object represents a node in a mathematical expression. It allows you to backprop gradients through the expression, useful for 
        building neural nets on top. Intermediate nodes are labeled by "operation" and tells you which mathematical operation created it.
        The "label" field is intended to label nodes as a weight, input, or bias mainly for visualization purposes.
        Call Value.visualize_graph() to visualize the entire mathematical expression graph as a png.
    */
    public:
        float data;
        vector<Value*> prev;//contain pointers to child nodes
        vector<Value*> next;
        string operation;
        string label;
        float grad = 0.0;//very important to init gradients to zero, dont rely on the compiler to do this for you
        float learning_rate = 0.0001;

        Value(float data){
            this->data = data;
        }
        Value* operator+(Value * operand) {
            Value * out = new Value(this->data + operand->data); // Create an output node, store in heap so it doesnt get garbage collected
            out->operation = "+";
            // add operands to output node as child nodes
            out->prev.push_back(this);
            out->prev.push_back(operand);
            //save reference of output node in the child node's vector of nextnodes
            operand->next.push_back(out);
            this->next.push_back(out);
            return out;
        }
        Value *operator+=(Value* operand) {
            //This operation is needed to accumulate the sum of X nodes into 1 node, as opposed to creating an output node for each pair of operands
            this->operation = "+";
            this->data = operand->data + this->data;
            // add operands to output node as child nodes
            this->prev.push_back(operand);
            //save reference of output node in the child node's vector of nextnodes
            operand->next.push_back(this);
            return this;
        }

        Value *operator*(Value* operand){
            Value * out = new Value(this->data * operand->data);
            out->operation = "*";
            //add operands to output node as child nodes
            out->prev.push_back(this);
            out->prev.push_back(operand);
            //save reference of output node in the child node's vector of nextnodes
            operand->next.push_back(out);
            this->next.push_back(out);
            return  out;
        }
        Value tanh(){
            float x = this->data;
            //tanh
            float t = (pow(2.718, (2.0*x)) - 1)/(pow(2.718,(2.0*x)) + 1);
            Value out = Value(t);
            out.operation = "tanh";
            //add operands to output node as child nodes
            out.prev.push_back(this);
            return out;
        }
        void backward(){
            //calculate the child nodes gradients depending on the op
            if (this->operation=="+"){
                //addition distributes gradient to both operands
                for(int i = 0; i < this->prev.size(); i++){
                    this->prev[i]->grad += this->grad;
                }
            }
            else if (this->operation=="*"){
                //parent node's gradient * other operand
                this->prev[0]->grad += this->grad * this->prev[1]->data;
                this->prev[1]->grad += this->grad * this->prev[0]->data;
            }
            else if (this->operation == "tanh"){
                this->prev[0]->grad += (1 - pow(this->data,2)) * this->grad;
            }
        }
        void update_weight(){
            //only update the data field if its a weight node
            if(this->label == "weight"){
                this->data -= this->learning_rate * this->grad;
            }
        }
        void backprop(){
            //level order traversal through the expression graph to backprop gradients through the expression
            //Karpathy overcomplicated things by using topological sort imo
            vector<Value*> queue;
            queue.insert(queue.begin(), this);
            while (queue.size() > 0){
                Value* root = queue[0];
                for(int i = 0; i < root->prev.size(); i++){
                    Value* child = root->prev[i];
                    bool unique = true;
                    //make sure child isnt already in the queue
                    for (int j = 0; j < queue.size(); j++){
                        if(queue[j] == child){
                            unique= false;
                            break;
                        }
                    }
                    if(unique==true){
                        queue.push_back(child);
                    }
                }
                //backward pass current node
                root->backward();
                //update weights
                root->update_weight();
                queue.erase(queue.begin() + 0);
            }
        }
        void zero_grad(){
            //recursively erase gradients through the expression. 
            //Gradients are already initialized to zero, so no need to call this before training
            vector<Value*> queue;
            queue.insert(queue.begin(), this);
            //level order traversal backwards through the expression graph
            while (queue.size() > 0){
                Value* root = queue[0];
                for(int i = 0; i < root->prev.size(); i++){
                    Value* child = root->prev[i];
                    bool unique = true;
                    //make sure child isnt already in the queue
                    for (int j = 0; j < queue.size(); j++){
                        if(queue[i] == child){
                            unique= false;
                            break;
                        }
                    }
                    if(unique==true){
                        queue.push_back(child);
                    }
                }
                //backward pass current node
                root->grad = 0.0;
                queue.erase(queue.begin() + 0);
            }
        }

        void generateDotFile(const std::string& dotFilePath) {
            //This is a helper function for the mathematical expression graph visualizer. It generates a DOT file which gets converted to an image
                //chatgpt wrote this function and visualizeGraph() lolol. I'm too lazy to learn graphViz
            Value* root = this;
            std::ofstream dotFile(dotFilePath);
            if (!dotFile) {
                std::cerr << "Failed to open the DOT file for writing." << std::endl;
                return;
            }

            std::unordered_set<Value*> visited;
            std::vector<Value*> queue;
            queue.push_back(root);

            dotFile << "digraph ExpressionGraph {" << std::endl;
            while (!queue.empty()) {
                Value* node = queue.front();
                queue.erase(queue.begin());

                // Skip if already visited
                if (visited.find(node) != visited.end())
                    continue;
                visited.insert(node);

                // Determine the node color based on its label
                std::string nodeColor = "none";  // Default color for unknown nodes
                if (node->label == "input")
                    nodeColor = "lightblue";
                else if (node->label == "weight")
                    nodeColor = "lightgreen";

                // Write node information
                dotFile << "  " << reinterpret_cast<uintptr_t>(node) << " [label=\"";
                dotFile << "addr: " << node << '\n';
                dotFile << "data: " << node->data;
                if (!node->label.empty())
                    dotFile << "\\nlabel: " << node->label;
                if (!node->operation.empty())
                    dotFile << "\\nop: " << node->operation;
                dotFile << "\n grad: " << node->grad;
                dotFile << "\", shape=box, style=filled, fillcolor=" << nodeColor << "];" << std::endl;

                // Write edge information
                for (Value* prevNode : node->prev) {
                    dotFile << "  " << reinterpret_cast<uintptr_t>(prevNode) << " -> " << reinterpret_cast<uintptr_t>(node) << ";" << std::endl;
                    queue.push_back(prevNode);
                }
            }
            dotFile << "}" << std::endl;

            dotFile.close();
        }

        void visualizeGraph() {
            //visualizes the mathematical expression graph as a png. Input nodes and weight nodes are color coded.
            Value* root = this;
            //create the DOT file
            std::string dotFilePath = "expression_graph.dot";
            generateDotFile(dotFilePath);
            // Generate the image using GraphViz and save as a png
            std::string outputFilePath = "expression_graph.png";
            std::string command = "dot -Tpng " + dotFilePath + " -o " + outputFilePath;
            std::system(command.c_str());

            std::cout << "Graph visualization saved as '" << outputFilePath << "'" << std::endl;
        }
};

class Neuron{
    /*
        Inplements a neuron as a mathematical expression. 
        Performs the operation Sum(weight_i * input_i) during a forward pass, and backprops 
        a gradient through the neuron during the backward pass.
    */
    public:
        int input_size = 0;
        vector<Value*> weights;
        Value out = Value(0.0);

            // Deep copy constructor
    Neuron(const Neuron& other) : input_size(other.input_size), out(other.out) {
        // Copy weights with new instances
        for (Value* weight : other.weights) {
            weights.push_back(new Value(*weight));
        }
        this->out = * new Value(other.out.data);
    }
        Neuron(int input_size){
            this->input_size = input_size;
            //initialize all weights to 1.0 for now. Change later so its random
            for(int i = 0; i < input_size; i++){
                this->weights.push_back(new Value(1.0));
                weights[i]->label = "weight";
                weights[i]->grad = 0.0;
            }
        }
        void forward(vector<Value*> & input){
            //perform a forward pass: Sum(weight vector * input vector)
            for (int i = 0; i < input.size(); i++) {
                //Use += to accumulate the sum of each weight[i]*input[i] product directly into out. Dont use +, as it creates intermediate 
                    //nodes for each pair of products. This causes the expression graph to become too unbalanced, and screws up gradient
                    //accumulation during backprop. This bug took me a month to solve, and only shows up when backpropping through larger multi-layer NN's
                this->out += (*input[i] * weights[i]);
            }
            this->out.label = "output";
        }

        void backward(){
            cout << "out gradient: " << this->out.grad << endl;
            out.backprop();
        }
};

class Layer{
    public:
        int input_size;
        vector<Value*> outputs;
        void forward(vector<Value*> & inputs);
};

class Linear: public Layer{
    /*
    Created a fully connected layer of neurons. Feeds an input vector through all the neurons in the layer during the forward pass, and backprops 
    gradients through the layer during the backward pass.
    */
    public:
        int input_size;
        int output_size;
        vector<Neuron*> neurons;
        vector<Value*> outputs;

        Linear(int input_size, int output_size){
            this->input_size = input_size;
            this->output_size = output_size;
            for(int i = 0; i < output_size; i++){
                neurons.push_back(new Neuron(input_size));
                outputs.push_back(&(neurons[i]->out));
            }
        }
        
        void forward(vector<Value*> & inputs){
            /*
            Each forward pass builds intermediate Value nodes inside each neuron (for each input*weight product), which need to persist to ensure 
            expression graph connectivity for backprop. Subsequent forward pass builds new intermediate nodes, thus resulting in a 
            memory leak from previous runs. Need to perform a deep copy for each neuron in order to garbage collect the old
            intermediate nodes
            */
            vector<Neuron*> new_neurons; 
            vector<Value*> new_outputs; 
            //populate new neuron and output vectors with deep copies
            for (int i = 0; i < neurons.size(); i++) {
                new_neurons.push_back(new Neuron(*neurons[i]));// Deep copy
                new_outputs.push_back( &(new_neurons[i]->out));
            }
            // Replace the original neurons with the deep-copies
            neurons.swap(new_neurons);
            outputs.swap(new_outputs);
            // Delete the old neurons to free up memory
            for (int i = 0; i < new_neurons.size(); i++) {
                delete new_neurons[i];
            }
            //Now do the actual forward pass
            for(int i = 0; i < neurons.size(); i++){
                Neuron* n = neurons[i];
                n->forward(inputs);
            }
        }
        void backward(vector<float> grads){
            //accumulate gradients and pass to output neurons.
            for(int i = 0; i < this->outputs.size(); i++){
                for(int j = 0; j < grads.size(); j++){
                    outputs[i]->grad += grads[j];
                }
            }  
            //backward pass gradients through each neuron
            vector<Value*> queue = this->outputs;
            //level order traversal backwards through the expression graph
            while (queue.size() > 0){
                Value* root = queue[0];
                for(int i = 0; i < root->prev.size(); i++){
                    Value* child = root->prev[i];
                    bool unique = true;
                    //make sure child isnt already in the queue
                    for (int j = 0; j < queue.size(); j++){
                        if(queue[j] == child){
                            unique= false;
                            break;
                        }
                    }
                    if(unique==true){
                        queue.push_back(child);
                    }
                }
                //backward pass current node
                root->backward();
                //update weight
                root->update_weight();
                queue.erase(queue.begin() + 0);
            }
        }
        void visualizeGraph(){
            //create a dummy tail node containing pointers to all output nodes for the visualizer. Visualizer can only traverse the graph backwards starting from a single node.
            Value dummy_tail = Value(0.0);
            dummy_tail.label = "dummy tail node";
            for(int i = 0; i < outputs.size(); i++){
                dummy_tail.prev.push_back(outputs[i]);
            }
            dummy_tail.visualizeGraph();
        }
};

class Model{
    /*
    Slap together any combination of layers to form a neural net.
    */
    public:
        vector<Linear*> layers;
        int input_size;
        int num_layers = layers.size();

        void add_layer(string layer_name, int output_size){
            if(layer_name == "linear"){
                if(layers.size() == 0){
                    Linear * new_layer = new Linear(this->input_size, output_size);
                    layers.push_back(new_layer);
                }
                else{
                    Linear * new_layer = new Linear(layers.back()->outputs.size(), output_size);
                    layers.push_back(new_layer);
                
                }
            }
        }

        vector<Value*> forward(vector<Value*> & input_vector){
            /*
            iteratively forward pass through each layer in the model.
            */
            for(int i = 0; i < layers.size(); i++){
                //forward pass current layer
                layers[i]->forward(input_vector);
                //output vector of current layer becomes the input vector of the next layer
                input_vector = (layers[i]->outputs);
                //print the output vector
                if(i==layers.size()-1){
                    vector<Value*> out;
                    cout << "Layer " << i+1 << " outputs: " << endl;
                    for(int j = 0; j < input_vector.size(); j++){
                        cout << input_vector[j]->data << endl;;
                        out.push_back(input_vector[j]);
                    }
                    return out;
                }
            }
            
        }
        void backward(vector<float> out_grads){
            /*
            backprop gradients throughout the entire neural net's expression graph
            */
            Linear * last_layer = layers.back();
            last_layer->backward(out_grads);
            last_layer->visualizeGraph();
        }
        float mean_squared_error(vector<float> & true_output, vector<Value*> & NN_output){
            float MSE = 0.0;
            for(int i = 0; i < true_output.size();i++){
                MSE += (true_output[i] - NN_output[i]->data) * (true_output[i] - NN_output[i]->data);
            }
            return MSE/true_output.size();
        }
        float mean_squared_error_derivative(vector<float> & true_output, vector<Value*> & NN_output){
            float MSE_derivative = 0.0;
            for(int i = 0; i < true_output.size();i++){
                MSE_derivative += 2.0 * (true_output[i] - NN_output[i]->data);
            }
            return MSE_derivative/true_output.size();
        }
        void train(vector<Value*> & X_train, vector<float> & Y_train, int num_epochs=1){
            cout << "Starting Training..." << endl;
            for(int i = 0; i < num_epochs; i++){
                //1: forward pass
                vector<Value*> NN_out = this->forward(X_train);
                //2: calculate loss
                float MSE = mean_squared_error(Y_train, NN_out);
                vector<float> loss;
                loss.push_back(mean_squared_error_derivative(Y_train, NN_out));
                //3: backprop gradients and update weights
                this->backward(loss);
                cout << "banana" << endl;
                //4: zero out gradients
                Value dummy_tail = Value(0.0);
                for(int j = 0; j < NN_out.size(); j++){
                    NN_out[j]->zero_grad();
                }
                cout << "banana" << endl;
            }
        }
};

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
    n->forward(inputs);
    //Set output gradient and backpropagate
    n->out.grad = 1.0;
    n->backward();
    n->out.visualizeGraph();
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
    l.forward(inputs);
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
    int input_vector_length = 2;

    //create input vector
    vector<Value*> inputs;
    for(int i = 0; i < input_vector_length; i++){
        inputs.push_back(new Value(1.0));
        inputs[i]->label = "input"; //label all the input nodes for visualization purposes
    }

    Value * res = *inputs[0] + inputs[1];
    res-> visualizeGraph();
    cout << inputs[0]->next[0] << endl;
    cout << (inputs[1]->next[0]) << endl;
    cout << res << endl;
    return 0;
    /*
    Segfault occurs when visualizing the graph after doing 2 subsequent forward passes
        verify graph connectivity
        implement a model.compile and make the graph bidirectional
            -signficant refactoring 
    Alright lets take a page out of Capital One's book and break up this big task into smaller "stories"
    -add vector containing pointers to next nodes
        -DONE
    -update operators to add child nodes to the nextNode vector
    -update neuron.forward to forward pass results through the graph 
    -update model.compile to make sure nextNode vectors at layer outputs are properly populated
    */

};