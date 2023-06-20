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
        vector<Value*> prev;
        string operation;
        string label;
        float grad = 0.0;
   
        Value(float data){
            this->data = data;
        }

        Value operator+(Value& op) {
            Value out = *(new Value(this->data + op.data)); // Create an output node, store in heap so it doesnt get garbage collected
            out.operation = "+";
            // add operands to output node as child nodes
            out.prev.push_back(this);
            out.prev.push_back(&op);
            return out;
        }

        Value operator*(Value& operand){
            Value out = *(new Value(this->data * operand.data));
            out.operation = "*";
            //add operands to output node as child nodes
            out.prev.push_back(this);
            out.prev.push_back(&operand);
            return out;
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
        void backprop(){
            //recursively backprop gradients through the expression
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
                    if(unique){
                        queue.push_back(child);
                    }
                }
                //backward pass current node
                root->backward();
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

        Neuron(int input_size){
            this->input_size = input_size;
            //initialize all weights to 1.0 for now. Change later so its random
            for(int i = 0; i < input_size; i++){
                this->weights.push_back(new Value(1.0));
                weights[i]->label = "weight";
            }
        }

        void forward(vector<Value*> & input){
            //perform a forward pass: Sum(weight vector * input vector)
            Value * running_sum = new Value(0.0);
            for(int i = 0; i < input.size(); i++){
                Value *temp =  new Value(*input[i] * *weights[i]); //wi * xi
                running_sum = new Value(*temp + *running_sum);//add the product to the running sum
            }
            this->out = *running_sum;
        }

        void backward(){
            out.backprop();
        }
};

class Layer{
    public:
        int input_size;
        vector<Value*> outputs;
};

class Linear: public Layer{
    /*
    Created a fully connected layer of neurons. Feeds an input vector through all the neurons in the layer during the forward pass, and backprops 
    gradients through the layer during the backward pass.
    */
    public:
        int input_size;
        vector<Neuron*> neurons;
        vector<Value*> outputs;

        Linear(int input_size, int output_size){
            this->input_size = input_size;
            for(int i = 0; i < output_size; i++){
                neurons.push_back(new Neuron(input_size));
                outputs.push_back(&neurons[i]->out);
            }
        }
        
        void forward(vector<Value*> & inputs){
            for(int i = 0; i < neurons.size(); i++){
                Neuron* n = neurons[i];
                n->forward(inputs);
            }
        }
        void backward(vector<float> grads){
            //accumulate gradients and pass to output neurons.
            for(int i = 0; i < outputs.size(); i++){
                for(int j = 0; j < grads.size(); j++){
                    outputs[i]->grad += grads[j];
                }
                //backward pass gradients through each neuron
                neurons[i]->backward();
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
    -make sure shapes match up
    -make sure gradients pass between layers properly
    -
    */
    public:
        vector<Layer*> layers;
        int input_size;

        void add_layer(string layer_name, int output_size){
            if(layer_name == "linear"){
                if(layers.size() == 0){
                    Layer * new_layer = new Linear(this->input_size, output_size);
                    layers.push_back(new_layer);
                }
                else{
                    Layer * new_layer = new Linear(layers[layers.size()-1]->outputs.size()-1, output_size);
                    layers.push_back(new_layer);
                }
            }
        }

        void view_layers(){
        }

        void forward(){

        }

        void backward(){

        }
};

int main(){
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
    /*
    Demo of a single neuron forward/backward pass + visualization
    */
    Neuron* n = new Neuron(2);
    vector<Value*> inputs;
    inputs.push_back(new Value(1.0));
    inputs.push_back(new Value(2.0));
    // inputs.push_back(new Value(3.0));
    // inputs.push_back(new Value(4.0));
    // inputs.push_back(new Value(5.0));
    //label all the inputs
    for(int i = 0; i < inputs.size(); i++){
        inputs[i]->label = "input";
    }
    n->forward(inputs);
    n->backward();
    n->out.visualizeGraph();
    /*
    Demo of a Linear layer
    */
    Linear l = Linear(2, 2);
    l.forward(inputs);
    cout << (l.outputs[0]->data) << endl;
    vector<float> out_grads;
    out_grads.push_back(0.5);
    out_grads.push_back(0.5);
    l.backward(out_grads);
    l.visualizeGraph();
    return 0;
};


