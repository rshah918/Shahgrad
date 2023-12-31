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
        Value* next;
        string operation;
        string label;
        float grad = 0.0;//very important to init gradients to zero, dont rely on the compiler to do this for you
        float learning_rate = 0.1;

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
            if (this != nullptr) {
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
                for (Value* child : prev) {
                    child->inorderTraversal(); // Traverse right child
                }
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
        Value * out = new Value(0.0);
        string activation = "";

        Neuron(int input_size){
            this->input_size = input_size;
            //initialize all weights to 1.0 for now. Change later so its random
            for(int i = 0; i < input_size; i++){
                this->weights.push_back(new Value(0.01));
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
            float sum = 0.0;
            // Calculate exponentials and sum
            for (int i = 0; i < outputs.size();i++) {
                float exp_value = std::exp(outputs[i]->prev[0]->data);
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
        float mean_squared_error(vector<float> true_output, vector<Value*> NN_output){
            float MSE = 0.0;
            for(int i = 0; i < true_output.size();i++){
                MSE += (true_output[i] - NN_output[i]->data) * (true_output[i] - NN_output[i]->data);
            }
            return MSE/true_output.size();
        }
        float mean_squared_error_derivative(vector<float> true_output, vector<Value*> NN_output){
            float MSE_derivative = 0.0;
            for(int i = 0; i < true_output.size();i++){
                MSE_derivative += 2.0 * (NN_output[i]->data - true_output[i]);
            }
            return MSE_derivative/true_output.size();
        }
        vector<float> categoricalCrossEntropy(const std::vector<float>& true_probs, const std::vector<Value*>& predicted_probs) {
            vector<float> loss;
            if (predicted_probs.size() != true_probs.size()) {
                std::cerr << "Error: Input vectors must have the same size." << std::endl;
            }
            for (size_t i = 0; i < predicted_probs.size(); ++i) {
                // Avoid log(0) by adding a small epsilon
                double epsilon = 1e-8;
                loss.push_back(-true_probs[i] * std::log(predicted_probs[i]->data + epsilon));
            }

            return loss;
        }

        vector<float> categoricalCrossEntropyDerivative(const std::vector<float>& true_probs, const std::vector<Value*>& predicted_probs) {
            vector<float> derivative;
            if (predicted_probs.size() != true_probs.size()) {
                std::cerr << "Error: Input vectors must have the same size." << std::endl;
            }
            for (size_t i = 0; i < predicted_probs.size(); ++i) {
                derivative.push_back(predicted_probs[i]->data - true_probs[i]);
            }

            return derivative;
        }
        void train(vector<vector<Value*> > X_train, vector<vector<float> > & Y_train, int num_epochs=1){
            cout << "Starting Training..." << endl;
            for(int i = 0; i < num_epochs; i++){
                for(int j = 0; j < X_train.size();j++){
                    vector<float> true_output = Y_train[j];
                    vector<Value*> input_vector = X_train[j];
                     //1: forward pass
                    vector<Value*> NN_out = this->forward(input_vector);
                    //2: calculate loss
                    vector<float> CCE = categoricalCrossEntropy(true_output, NN_out);
                    cout << "CCE: ";
                    for(float loss:CCE){
                        cout << " " << loss;
                    }
                    cout << endl;
                    vector<float> loss = categoricalCrossEntropyDerivative(true_output, NN_out);
                    //3: backprop gradients and update weights
                    this->backward(loss);
                    //4: zero out gradients
                    Value dummy_tail = Value(0.0);
                    for(int j = 0; j < NN_out.size(); j++){
                        NN_out[j]->zero_grad();
                    }
                }
                layers.back()->visualizeGraph();
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
// int main(){
//     //create x_train
//     int input_vector_length = 2;
//     vector<vector<Value*> > X_train;
//     for(int i = 0; i < 100; i++){
//         vector<Value*> input;
//         input.push_back(new Value(i));
//         input.push_back(new Value(i));
//         X_train.push_back(input);
//     }
//     //create y_train
//         vector<vector<float> > Y_train;
//     for(int i = 0; i < 100; i++){
//         vector<float> input;
//         input.push_back(0);
//         input.push_back(1);
//         Y_train.push_back(input);
//     }
//     Model m = * new Model(input_vector_length);
//     m.add_layer("linear", 2, "");
//     m.add_layer("linear", 5, "");
//     m.add_layer("linear", 2, "");
//     m.add_layer("linear", 2, "");
//     m.add_layer("linear", 2, "");
//     m.add_layer("linear", 2, "");
//     m.add_layer("linear", 2, "");
//     m.add_layer("softmax");
//     m.compile(X_train[0]);
//     m.layers.back()->visualizeGraph();
//     //m.train(X_train, Y_train,5);
//     return 0;
//     /*
//     Segfault occurs when visualizing the graph after doing 2 subsequent forward passes
//         verify graph connectivity
//         implement a model.compile and make the graph bidirectional
//             -signficant refactoring 
//     -add vector containing pointers to next nodes
//         -DONE
//     -update operators to add child nodes to the nextNode vector
//         -DONE
//     -update neuron.forward to forward pass results through the graph 
//         -DONE
//     -update linear.forward 
//         -DONE
//     -implement linear.compile
//         -DONE
//     -update model.compile to make sure nextNode vectors at layer outputs are properly populated
//         -DONE
//     -update model.forward and verify proper functionality for multi layer NN's
//         -DONE
//     -training loop
//         -DONE
//     Ok im gonna train MNIST using shahgrad. Need to implement sigmoid and softmax. I might have to implement some optimizers 
//     -Sigmoid
//         -DONE
//     -update linear.compile to update layer outputs with softmax
//         -DONE
//     -softmax backward pass
//         -DONE
//     -categorical crossentropy
//         -validate calculation
//         -validate derivative calculation
//         -DONE
//     -Sigmoid
//         -Ok theres a strange issue. L1.out is populated during linear.compile(), but then magically gets emptied when I try accessing that
//         data member outside the class. Look closely at linear.compile and model.compile. Figure out why data members are being erased
//         -try reading other data members outside the class. If its indeed being wiped, maybe its happening to all data members
//         -Swap order of layers
//             -Oh yea the problem def has to do with data members being wiped. probably a parent/child class issue
//         -Fixed, forgot to add the "virtual" and "override" labels
//     Other bugs fixed:
//         -Moved softmax into its own layer
//         -Fixed polymorphism issues
//         -Softmax derivative and sigmoid calculations had typos
//         -Gradients wouldnt flow past the sigmoid layer
//         -Softmax would disconnect the expression graph
//         -Categorical crossentropy derivative calculation had a mistake
//     Ok now I can go ahead with MNIST
//         -DONE
//     Training is super slow. Need to parallelize. Main priority is to minimize codebase complexity and lines of code. 
//         */
//     Never mind, found a smaller MNIST with 10x10 images. Training speed is reasonable. Wont optimize performance as this library is 
//         meant to optimize for verbosity and simplicity
//      Validation loop
//      Model save/load
//      Clean up repo and try to reduce lines of code
// };