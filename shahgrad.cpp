#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>
#include <fstream>
#include <unordered_set>
#include <ctime>
using namespace std;

float learning_rate; //using a global var is actually the cleaner approach here. Alternative is to propagate it through all the layers of abstraction, which is messier.

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
        Value* next;
        string operation;
        string label;
        float grad = 0.00;

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
            operand->next = out;
            this->next = out;
            return out;
        }
        Value *operator+=(Value* operand) {
            //This operation is needed to accumulate the sum of X nodes into 1 node, as opposed to creating an output node for each pair of operands
            this->operation = "+";
            this->data = operand->data + this->data;
            // add operands to output node as child nodes
            this->prev.push_back(operand);
            //save reference of output node in the child node's vector of nextnodes
            operand->next = this;
            return this;
        }

        Value *operator*(Value* operand){
            Value * out = new Value(this->data * operand->data);
            out->operation = "*";
            //add operands to output node as child nodes
            out->prev.push_back(this);
            out->prev.push_back(operand);
            //save reference of output node in the child node's vector of nextnodes
            operand->next = out;
            this->next = out;
            return  out;
        }
        void inorderTraversal() {
            // Perform an inorder traversal to forward propagate inputs 
                for (Value* child : prev) {
                    child->inorderTraversal(); // Traverse left child
                }
                // Forward propagate based on the operation
                if (this->operation == "+") {
                    this->data = 0;//zero out as its the additive identity
                    for(int i = 0;i < this->prev.size();i++){
                        this->data += prev[i]->data;
                    }
                }
                if (this->operation == "*") {
                    this->data = 1;//set to 1 as its the multiplicative identity
                    for(int i = 0;i < this->prev.size();i++){
                        this->data *= prev[i]->data;
                    }
                }
                if (this->operation == "sigmoid"){
                    this->data = 1/(1+pow(2.71828, -1*prev[0]->data));
                }
                if (this->operation == "relu"){
                    float out_value = prev[0]->data;
                    if (out_value < 0.0){
                        out_value = 0.00000001 * out_value;
                    }
                    this->data = out_value;
                }
                if(this->operation == "tanh"){
                    float x = prev[0]->data;
                    this->data = (pow(2.718, (2.0*x)) - 1)/(pow(2.718,(2.0*x)) + 1);
                }
                if(this->operation == "exp"){
                    float x = prev[0]->data;
                    this->data = (pow(2.718, (x)));
                }
                for (Value* child : prev) {
                    child->inorderTraversal(); // Traverse right child
                }
        }
        Value *tanh(){
            float x = this->data;
            //tanh
            float t = (pow(2.718, (2.0*x)) - 1)/(pow(2.718,(2.0*x)) + 1);
            Value * out = new Value(t);
            out->operation = "tanh";
            out->prev.push_back(this);
            this->next = out;
            return out;
        }
        Value *sigmoid(){
            Value * out = new Value(1/(1+pow(2.71828, -1*this->data)));
            out->operation = "sigmoid";
            out->prev.push_back(this);
            this->next = out;
            return out;
        }
         Value *exp(){
            Value * out = new Value((pow(2.71828, this->data)));
            out->operation = "exp";
            out->prev.push_back(this);
            this->next = out;
            return out;
        }
        Value *relu(){
            float out_value = this->data;
            if(out_value < 0){
                out_value = out_value * 0.00000001; //leaky relu to prevent vanishing gradient
            }
            Value * out = new Value(out_value);
            out->operation = "relu";
            out->prev.push_back(this);
            this->next = out;
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
            else if (this->operation == "sigmoid"){
                this->prev[0]->grad += (this->data * (1-this->data)) * this->grad;
            }
            else if (this->operation == "exp"){
                cout << this->grad << endl;
                cout << this->data << endl;
                cout << "___" << endl;
                this->prev[0]->grad += (this->data) * this->grad;
            }
            else if (this->operation == "relu"){
                if (this->data > 0){
                    this->prev[0]->grad += this->grad;
                }
                else{
                    this->prev[0]->grad += this->grad * 0.00000001;
                }
            }
        }
        void update_weight(){
            //only update the data field if its a weight node
            if(this->label == "weight"){
                this->data -= learning_rate * this->grad;
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
        Value * out = new Value(0.0);
        string activation = "";

        Neuron(int input_size){
            this->input_size = input_size;
            //randomly initialize weights
            srand(std::time(0));
            for(int i = 0; i < input_size; i++){
                std::random_device rd; // obtain a random number from hardware
                std::mt19937 gen(rd()); // seed the generator
                std::uniform_int_distribution<> distr(0, 1); // define the range
                double r = distr(gen);
                this->weights.push_back(new Value(r));
                weights[i]->label = "weight";
                weights[i]->grad = 0.0;
            }
        }
        void compile(vector<Value*> input, string activation=""){
            //builds the expression graph, by constructing the intermediate nodes sitting in between the output and weights/inputs
            for (int i = 0; i < input.size(); i++) {
                //Use += to accumulate the sum of each weight[i]*input[i] product directly into out. Dont use +, as it creates intermediate 
                //nodes for each pair of products. This causes the expression graph to become too unbalanced, and screws up gradient
                //accumulation during backprop. This bug took me a month to solve, and only shows up when backpropping through larger multi-layer NN's
                *this->out += (*input[i] * weights[i]);
            }
            if (activation == "sigmoid"){
                this->out =  this->out->sigmoid();
            }
            else if(activation == "tanh"){
                this->out = this->out->tanh();
            }
            else if(activation == "relu"){
                this->out = this->out->relu();
            }
            else if(activation == "exp"){
                this->out = this->out->exp();
            }
        }
        void forward(){
            //perform a forward pass: Sum(weight vector * input vector)
            this->out->inorderTraversal();
            this->out->label = "output";
        }

        void backward(){
            out->backprop();
        }
};

class Layer{
    public:
        int input_size;
        int output_size;
        vector<Value*> outputs;
        virtual void forward(vector<Value*> inputs){}
        virtual void forward(){}
        virtual void backward(vector<float> grads, bool accumulate=true){}
        virtual void compile(vector<Value*> input_vector){}
        virtual void visualizeGraph(){}
};

class Softmax: public Layer{
    public:
        Softmax(int input_size){
            this->input_size = input_size;
            this->output_size = input_size;
        }
        void compile(vector<Value*> inputs) override {
            for (Value* v : inputs) {
                Value* softmax_output = new Value(0.0);
                softmax_output->operation = "softmax";
                softmax_output->prev.push_back(v);
                v->next = softmax_output;
                outputs.push_back(softmax_output);
            }
        }
        void forward() override {
            //get max value
            float max = 0.0;
            for (int i = 0; i < outputs.size();i++) {
                if (outputs[i]->prev[0]->data > max){
                    max = outputs[i]->prev[0]->data;
                }
            }
            float sum = 0.0;
            // Calculate exponentials and sum
            for (int i = 0; i < outputs.size();i++) {
                float exp_value = std::exp(outputs[i]->prev[0]->data - max);
                outputs[i]->data = exp_value;
                sum += exp_value;
            }
            // Normalize by dividing each exponential by the sum
            for (int i = 0; i < outputs.size();i++) {
                outputs[i]->data = outputs[i]->data / sum;
            }
            // Kickstart forward pass process for previous layers
            for (int i = 0; i < outputs.size();i++) {
                outputs[i]->inorderTraversal();
            }
        }
        void backward(vector<float> grads, bool accumulate=false) override {
            for(int i = 0; i < outputs.size(); i++){
                outputs[i]->grad = grads[i];
                outputs[i]->prev[0]->grad = grads[i];
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
        void visualizeGraph() override {
            //create a dummy tail node containing pointers to all output nodes for the visualizer. Visualizer can only traverse the graph backwards starting from a single node.
            Value dummy_tail = Value(0.0);
            dummy_tail.label = "dummy tail node";
            for(int i = 0; i < outputs.size(); i++){
                dummy_tail.prev.push_back(outputs[i]);
            }
            dummy_tail.visualizeGraph();
        }
};
class Linear: public Layer{
    /*
    Created a fully connected layer of neurons. Feeds an input vector through all the neurons in the layer during the forward pass, and backprops 
    gradients through the layer during the backward pass.
    */
    public:
        vector<Neuron*> neurons;
        string activation = "";

        Linear(int input_size, int output_size, string activation=""){
            this->input_size = input_size;
            this->output_size = output_size;
            this->activation = activation;
            for(int i = 0; i < output_size; i++){
                neurons.push_back(new Neuron(input_size));
            }
        }
        void compile(vector<Value*> inputs) override {
            for(Neuron* n:neurons){
                n->compile(inputs, this->activation);
                outputs.push_back(n->out);
            }
        }
        void forward() override {
            for(Value* output:outputs){
                output->inorderTraversal();
            }
        }
        void backward(vector<float> grads, bool accumulate=true) override {
            //accumulate gradients and pass to output neurons.
            for(int i = 0; i < this->outputs.size(); i++){
                if(accumulate == true){
                    for(int j = 0; j < grads.size(); j++){
                        outputs[i]->grad += grads[j];
                    }
                }
                else{
                    outputs[i]->grad = grads[i];
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
        void visualizeGraph() override {
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
        vector<Layer*> layers;
        vector<Value*> inputs;
        vector<Value*> outputs;
        int input_size;
        int num_layers = layers.size();

        Model(int input_size){
            this->input_size = input_size;
        }

        void add_layer(string layer_name, int output_size=1, string activation=""){
            if(layer_name == "linear"){
                if(layers.size() == 0){
                    Linear * new_layer = new Linear(this->input_size, output_size, activation);
                    layers.push_back(new_layer);
                }
                else{
                    Linear * new_layer = new Linear(layers.back()->output_size, output_size, activation);
                    layers.push_back(new_layer);
                }
            }
            else if(layer_name == "softmax"){
                if(layers.size() == 0){
                    Softmax * new_layer = new Softmax(this->input_size);
                    layers.push_back(new_layer);
                }
                else{
                    Softmax * new_layer = new Softmax(layers.back()->output_size);
                    layers.push_back(new_layer);
                }
            }
        }
        void compile(vector<Value*> input_vector){
            //make sure layers are conjoined properly
            //iteratively build each layer's expression graph
            this->inputs = input_vector;
            for(int i = 0; i < layers.size(); i++){
                //first layer is a special case, as it needs to be conjoined with the input vector
                if(i==0){
                    layers[i]->compile(input_vector);
                }
                else{
                    layers[i]->compile(layers[i-1]->outputs);
                }
            }
            this->outputs = layers.back()->outputs;
        }
        vector<Value*> forward(vector<Value*> input_vector){
            //inputs are embedded into the NN expression graph, so their "data" field needs to be updated in-place
            for(int i = 0; i < input_vector.size(); i++){
                this->inputs[i]->data = input_vector[i]->data;
            }
            //forward prop is implemented as an inorder traversal, so initiate inference from the last layer in NN
            layers.back()->forward(); 
            return outputs;
        }
        void backward(vector<float> out_grads){
            /*
            backprop gradients throughout the entire neural net's expression graph
            */
            Layer * last_layer = layers.back();
            //generally, the gradient of each layers output node is the sum of all the gradients in the next layer's input nodes. 
            //This is not the case of the last layer in the NN, as we want to manually set each output gradient. That special case is handled here
            last_layer->backward(out_grads, false);
        }
        void print_vector(vector<float> vec){
            //utility function to print out float vectors. Why isn't this built in std:vector??? 
            cout << " | ";
            for(int i = 0;i < vec.size(); i++){
                cout << vec[i] << " | ";
            }
            cout << endl;
        }
        void print_vector(vector<Value*> vec){
            //utility function to print out Value vectors
            cout << " | ";
            for(int i = 0;i < vec.size(); i++){
                cout << vec[i]->data << " | ";
            }
            cout << endl;
        }
        float mean_squared_error(vector<float> true_output, vector<Value*> NN_output){
            if (NN_output.size() != true_output.size()) {
                std::cerr << "Error: NN output vector and true output vector must have the same size." << std::endl;
            }
            float MSE = 0.0;
            for(int i = 0; i < true_output.size();i++){
                MSE += (true_output[i] - NN_output[i]->data) * (true_output[i] - NN_output[i]->data);
            }
            MSE = MSE/true_output.size();
            cout << "MSE: " << MSE << endl;
            return MSE;
        }
        vector<float> mean_squared_error_derivative(vector<float> true_output, vector<Value*> NN_output){
            if (NN_output.size() != true_output.size()) {
                std::cerr << "Error: NN output vector and true output vector must have the same size." << std::endl;
            }
            vector<float> MSE_derivative;
            for(int i = 0; i < true_output.size();i++){
                MSE_derivative.push_back((2.0 * (NN_output[i]->data - true_output[i]))/true_output.size());
            }
            return MSE_derivative;
        }
        float categoricalCrossEntropy(const std::vector<float>& true_probs, const std::vector<Value*>& predicted_probs) {
            if (predicted_probs.size() != true_probs.size()) {
                std::cerr << "Error: NN output vector and true output vector must have the same size." << std::endl;
            }
            float CCE;
            for (int i = 0; i < predicted_probs.size(); i++) {
                // Avoid log(0) by adding a small epsilon
                double epsilon = 1e-8;
                CCE += true_probs[i] * std::log10(predicted_probs[i]->data + epsilon);
            }
            CCE = (-1 * (CCE/predicted_probs.size()));
            cout << "CCE loss: " << CCE << endl;
            return CCE;
        }

        vector<float> categoricalCrossEntropyDerivative(const std::vector<float>& true_probs, const std::vector<Value*>& predicted_probs) {
            if (predicted_probs.size() != true_probs.size()) {
                std::cerr << "Error: NN output vector and true output vector must have the same size." << std::endl;
            }
            vector<float> derivative;
            for (int i = 0; i < predicted_probs.size(); i++) {
                derivative.push_back(predicted_probs[i]->data - true_probs[i]);
            }
            return derivative;
        }
        void train(vector<vector<Value*> > X_train, vector<vector<float> > & Y_train, int num_epochs=1, float learning_rate=0.01, string loss_function="mean_squared_error"){
            cout << "Starting Training..." << endl;
            learning_rate = ::learning_rate;//set the global learning rate var. Entire expression graph will use it
            for(int i = 0; i < num_epochs; i++){
                for(int j = 0; j < X_train.size();j++){
                    vector<float> true_output = Y_train[j];
                    vector<Value*> input_vector = X_train[j];
                     //1: forward pass
                    vector<Value*> NN_out = this->forward(input_vector);
                    cout << "Predicted: ";
                    print_vector(NN_out);
                    cout << "Correct:   ";
                    print_vector(true_output);
                    //2: calculate loss
                    float loss;
                    vector<float> loss_derivative;
                    if (loss_function == "mean_squared_error"){
                            loss = mean_squared_error(true_output, NN_out);
                            loss_derivative = mean_squared_error_derivative(true_output, NN_out);
                    }
                    else if(loss_function == "categorical_cross_entropy"){
                        loss = categoricalCrossEntropy(true_output, NN_out);
                        loss_derivative = categoricalCrossEntropyDerivative(true_output, NN_out);
                    }
                    //3: backprop gradients and update weights
                    this->backward(loss_derivative);
                    //4: zero out gradients
                    Value dummy_tail = Value(0.0);
                    for(int j = 0; j < NN_out.size(); j++){
                        NN_out[j]->zero_grad();
                    }
                }
            }
        }
};
