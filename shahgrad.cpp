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
            Value * out = new Value(this->data + op.data); // Create an output node, store in heap so it doesnt get garbage collected
            out->operation = "+";
            // add operands to output node as child nodes
            out->prev.push_back(this);
            out->prev.push_back(&op);
            return *out;
        }

        Value operator*(Value& operand){
            Value * out = new Value(this->data * operand.data);
            out->operation = "*";
            //add operands to output node as child nodes
            out->prev.push_back(this);
            out->prev.push_back(&operand);
            return * out;
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
                weights[i]->grad = 0.0;
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
            for(int i = 0; i < this->outputs.size(); i++){
                for(int j = 0; j < grads.size(); j++){
                    outputs[i]->grad += grads[j];
                }
                //backward pass gradients through each neuron
                this->neurons[i]->backward();
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

        void view_layers(){
        }

        void forward(vector<Value*> & input_vector){
            /*
            iteratively forward pass through each layer in the model.
            */
            for(int i = 0; i < layers.size(); i++){
                //forward pass current layer
                layers[i]->forward(input_vector);
                //output vector of current layer becomes the input vector of the next layer
                input_vector = (layers[i]->outputs);
                //print the output vector
                for(int j = 0; j < input_vector.size(); j++){
                    cout << input_vector[j]->data << endl;
                }
            }
            
        }

        void backward(vector<float> out_grads){
            /*
            backprop gradients throughout the entire neural net's expression graph
            Alright. Gradient backprop is wrong when num_layers > 1. Addition is increasing gradients by 1 for some reason. and mul is also wrong...
            */
            Linear * last_layer = layers[1];
            last_layer->backward(out_grads);
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
}

void single_neuron_demo(){
    /*
    Demo of a single neuron forward/backward pass + visualization
    */
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
    int input_vector_length = inputs.size();
    Neuron* n = new Neuron(input_vector_length);
    n->forward(inputs);
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
    int output_size = 1;
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
        inputs.push_back(new Value(i + 1.0));
        inputs[i]->label = "input"; //label all the input nodes for visualization purposes
    }
    int output_size = 2;
    /*
    Demo of a model
    */
    Model m = Model();
    m.input_size = input_vector_length;
    m.add_layer("linear", output_size);
    m.add_layer("linear", output_size);
    m.forward(inputs);
    vector<float> out_grads;
    for(int i = 0; i < output_size;i++){
        out_grads.push_back(0.5);
    }
    m.backward(out_grads);
    m.layers[1]->visualizeGraph();
    return 0;
};


