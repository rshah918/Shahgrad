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
            for(int i = 0; i < input_size; i++){
                this->weights.push_back(new Value(1.0));
                weights[i]->label = "weight";
            }
            cout << "weight vector length "<< this->weights.size() << endl;
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

        void backward(float grad = 1.0){
            out.grad = grad;
            out.backprop();
        }
};

class Linear{
    /* TO DO:
    Created a fully connected layer of neurons. Feeds an input vector through all the neurons in the layer during the forward pass, and backprops 
    gradients throughout the layer during the backward pass.
    */
};

int main(){
    //Demonstration of the gradient engine
    //inputs
    Value x1 = Value(2);
    Value x2 = Value(3);
    Value w1 = Value(-3);
    Value w2 = Value(1);
    Value bias = Value(6.8814);

    //expression
    Value x1w1 = x1* w1;
    x1w1.label = "x1w1";
    Value x2w2 = x2*w2;
    x2w2.label = "x2w2";
    Value x1w1x2w2 = (x1w1 + x2w2);
    Value x1w1x2w2b = x1w1x2w2 + bias;
    x1w1x2w2b.label = "x1w1 + x2w2 + bias";
    Value res = x1w1x2w2b;

    res.label = "output";

    res.grad = 1.0; //init root node grad to 1

    //backprop and visualize
    res.backprop();
    res.visualizeGraph();

    Neuron* n = new Neuron(5);
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
    n->forward(inputs);
    n->backward();
    //n->out.visualizeGraph();

    return 0;
};


