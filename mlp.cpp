#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>

using namespace std;


struct Connection
{
    double weight;
    double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;

class Neuron{
    public:
        Neuron(unsigned numOutputs, unsigned myIndex);
        void setOutputVal(double val) {m_outputVal = val;}
        double getOutputVal() const {return m_outputVal;}
        void feedForward(const Layer &prevLayer);
        void calcOutputGradients(double targetVal);
        void calcHiddenGradients(const Layer &nextLayer);
        void updateInputWeights(Layer &prevLayer);
    private:
        static double transferFunction(double x);
        static double transferFunctionDerivative(double x);
        static double randomWeight(void) {return rand() / double(RAND_MAX);}
        double sumDOW(const Layer &nextLayer) const;
        double m_outputVal;
        vector<Connection> m_outputWeights;
        unsigned m_myIndex;
        double m_gradient;
        static double eta;
        static double alpha;
};

double Neuron::eta = .15;
double Neuron::alpha = .5;

Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
    for (unsigned c = 0; c < numOutputs; ++c){
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;
}

double Neuron::transferFunction(double x){
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x){
    // Quick approximation of derivative

    return 1.0 - x * x;
}

void Neuron::feedForward(const Layer &prevLayer){
    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer

    double sum = 0.0;
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::transferFunction(sum);
}

void Neuron::calcOutputGradients(double targetVal){
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
};

void Neuron::calcHiddenGradients(const Layer &nextLayer){
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
};

double Neuron::sumDOW(const Layer &nextLayer) const{
    double sum = 0.0;

    //Sum our contributions of the errors at the nodes we feed.

    for (unsigned n = 0; n < nextLayer.size() - 1; ++n){
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
    
    return sum;
};

void Neuron::updateInputWeights(Layer &prevLayer){
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer

    for (unsigned n = 0; n < prevLayer.size(); ++n){
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
        double newDeltaWeight =
            eta *
            neuron.getOutputVal() *
            m_gradient +
            alpha *
            oldDeltaWeight;
        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

class Net {
    public:
        Net(const vector<unsigned> &topology);
        void feedForward(const vector<double> &inputVals);
        void backProp(const vector<double> &targetVals);
        void getResults(vector<double> &resultVals) const;
    private:
        vector <Layer> m_layers;
        double m_error;
        double m_recentAverageError;
        double m_recentAverageSmoothingFactor;
};

Net::Net(const vector<unsigned> &topology){
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == numLayers - 1 ? 0: topology[layerNum + 1];
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
        }
    m_layers.back().back().setOutputVal(1.0);
    }
}

void Net::feedForward(const vector<double> &inputVals){
    // Check the num of inputVals euqal to neuronnum expect bias

    assert(inputVals.size() == m_layers[0].size() - 1);

    // Assign {latch} the input values into the input neurons

    for (unsigned i = 0; i < inputVals.size(); ++i) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum){
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

void Net::backProp(const vector<double> &targetVals){
    // Calculate overall net error (RMS method)

    Layer &outputLayer = m_layers.back();
    m_error = 0.0;
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1;
    m_error = sqrt(m_error);

    // Implement a recent average measurement

    m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0); 

    // Calculate output layer gradients

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // Calculate gradients on hidden layers

    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // For all layers from outputs to first hidden layer, update connection weights

    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Net::getResults(vector<double> &resultVals) const{
    resultVals.clear();

    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

int main() {
    // How to use
    // Substitute with actual training data

    vector <unsigned> topology;
    Net myNet(topology);

    vector<double> inputVals;
    myNet.feedForward(inputVals);

    vector <double> targetVals;
    myNet.backProp(targetVals);

    vector <double> resultVals;
    myNet.getResults(resultVals);
}