package MSc;

public class Main {

    public static void main(String[] args) {
        int inputLayer = 4;
        int hiddenLayer = 2;
        int outputLayer = 4;
        double[] pattern1 = {1, 0, 0, 0};
        double[] pattern2 = {0, 1, 0, 0};
        double[] pattern3 = {0, 0, 1, 0};
        double[] pattern4 = {0, 0, 0, 1};
        double lowestWeight = -0.5;
        double highestWeight = 0.5;
        double trainingStep = 0.005;
        double biasValue = 0.83;

        Neuron n = new Neuron(inputLayer, hiddenLayer, outputLayer, biasValue);
        n.performingTraining(trainingStep, lowestWeight, highestWeight, pattern1, pattern2, pattern3, pattern4);
    }

}
