package MSc;

/**
 * Created by kmo on 17/07/18.
 */

public class TrainingPattern {
    private double[] outputs;
    private double[] errorOfOutput;
    private double[] errorOfHidden;
    private double[] hiddens;
    private static double[] inputs;
    private static double[] hiddenWeights;
    private static double[] outputWeights;
    private static double[] biasWeights;
    private static int numberOfHiddens;
    private static int numberOfInputs;
    private static int numberOfOutputs;
    private static double bias;

    public TrainingPattern(double[] outputNeurons, double[] outputError, double[] hiddenError, double[] hiddenLayer, double[] inputNeurons, double[] weightsOfHiddens, double[] weightsOfOutputs, double[] weightsOfBias, int hiddensNumber, int inputsNumber, int outputsNumber, double biasValue){
        outputs = outputNeurons;
        errorOfOutput = outputError;
        errorOfHidden = hiddenError;
        hiddens = hiddenLayer;
        inputs = inputNeurons;
        hiddenWeights = weightsOfHiddens;
        outputWeights = weightsOfOutputs;
        biasWeights = weightsOfBias;
        numberOfHiddens = hiddensNumber;
        numberOfInputs = inputsNumber;
        numberOfOutputs = outputsNumber;
        bias = biasValue;
    }

    private static double[] setArrayToZero(double[] array, int size)
    {
        int i;
        for (i = 0; i < size; i++)
        {
            array[i] = 0;
        }
        return array;
    }

    private static double[] calculateHiddenLayer(double[] hiddenLayer, double[] inputNeurons, double[] weightsOfHiddens, double[] weightsOfBias)
    {
        int i;
        int j;

        for (i = 0; i < numberOfHiddens; i++)
        {
            for (j = 0; j < numberOfInputs; j++)
            {
                hiddenLayer[i] += inputNeurons[j] * weightsOfHiddens[j + 4 * i];
            }
            hiddenLayer[i] += bias * weightsOfBias[i];
        }
        return hiddenLayer;
    }

    private static double[] calculateOutputNeurons(double[] outputNeurons, double[] hiddenLayer, double[] weightsOfOutputs, double[] weightsOfBias)
    {
        int i;
        int j;

        for (i = 0; i < numberOfOutputs; i++)
        {
            for (j = 0; j < numberOfHiddens; j++)
            {
                outputNeurons[i] += hiddenLayer[j] * weightsOfOutputs[j + 2 * i];
            }
            outputNeurons[i] += bias * weightsOfBias[i + 2];
        }
        return outputNeurons;
    }

    public void propagateTrainingPattern()
    {
        this.outputs = setArrayToZero(outputs, numberOfOutputs);
        this.errorOfOutput = setArrayToZero(errorOfOutput, numberOfOutputs);
        this.hiddens = setArrayToZero(hiddens, numberOfHiddens);
        this.errorOfHidden = setArrayToZero(errorOfHidden, numberOfHiddens);

        this.hiddens = calculateHiddenLayer(hiddens, inputs, hiddenWeights, biasWeights);
        this.outputs = calculateOutputNeurons(outputs, hiddens, outputWeights, biasWeights);
    }
}


