package MSc;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Created by Krzysztof Olszewski on 17/06/18.
 */
public class Neuron {

    private static int inputs;
    private static int hiddens;
    private static int outputs;
    private static double bias;

    public Neuron(int inputLayer, int hiddenLayer, int outputLayer, double biasValue) {
        inputs = inputLayer;
        outputs = outputLayer;
        hiddens = hiddenLayer;
        bias = biasValue;
    }

    private static void printHidden(double[] hiddenLayer, int patternNumber) {
        int i;
        System.out.print("Hidden layers of pattern number ");
        System.out.print(patternNumber);
        System.out.print(":");
        System.out.print("\n");
        for (i = 0; i < hiddens; i++) {
            System.out.printf("%.3f", hiddenLayer[i]);
            System.out.print(", ");
            System.out.print("\t");
        }
        System.out.print("\n");
    }

    private static void printOutputs(double[] outputNeurons, int patternNumber) {
        int i;
        System.out.print("Outputs of pattern number ");
        System.out.print(patternNumber);
        System.out.print(":");
        System.out.print("\n");
        for (i = 0; i < outputs; i++) {
            System.out.printf("%.3f", outputNeurons[i]);
            System.out.print(", ");
            System.out.print("\t");
        }
        System.out.print("\n");
    }

    private static boolean compareArrays(double[] array1, double[] array2) {
        int i;
        for (i = 0; i < 4; i++) {
            if (array1[i] < array2[i] - 0.005 || array1[i] > array2[i] + 0.005) {
                return false;
            }
        }
        return true;
    }

    private static double[] randomizeArray(double[] array, double lowest, double highest, int size) {
        int i;
        for (i = 0; i < size; i++) {
            array[i] = ThreadLocalRandom.current().nextDouble(lowest, highest);
        }
        return array;
    }

    private static void setArrayToZero(double[] array, int size) {
        int i;
        for (i = 0; i < size; i++) {
            array[i] = 0;
        }
    }

    private static void calculateHiddenLayer(double[] hiddenLayer, double[] inputNeurons, double[] weightsOfHiddens, double[] weightsOfBias) {
        int i;
        int j;

        for (i = 0; i < hiddens; i++) {
            for (j = 0; j < inputs; j++) {
                hiddenLayer[i] += inputNeurons[j] * weightsOfHiddens[j + 4 * i];
            }
            hiddenLayer[i] += bias * weightsOfBias[i];
        }
        //return hiddenLayer;
    }

    private static void calculateOutputNeurons(double[] outputNeurons, double[] hiddenLayer, double[] weightsOfOutputs, double[] weightsOfBias) {
        int i;
        int j;

        for (i = 0; i < outputs; i++) {
            for (j = 0; j < hiddens; j++) {
                outputNeurons[i] += hiddenLayer[j] * weightsOfOutputs[j + 2 * i];
            }
            outputNeurons[i] += bias * weightsOfBias[i + 2];
        }
        //return outputNeurons;
    }

    public void propagateTrainingPattern(double[] outputNeurons, double[] outputError, double[] hiddenError, double[] hiddenLayer, double[] inputNeurons, double[] weightsOfHiddens, double[] weightsOfOutputs, double[] weightsOfBias) {
        setArrayToZero(outputNeurons, outputs);
        setArrayToZero(outputError, outputs);
        setArrayToZero(hiddenLayer, hiddens);
        setArrayToZero(hiddenError, hiddens);

        calculateHiddenLayer(hiddenLayer, inputNeurons, weightsOfHiddens, weightsOfBias);
        calculateOutputNeurons(outputNeurons, hiddenLayer, weightsOfOutputs, weightsOfBias);
    }

    private static void calculateOutputError(double[] outputNeurons, double[] outputError, double[] inputNeurons) {
        int i;
        for (i = 0; i < outputs; i++) {
            outputError[i] = outputNeurons[i] * ((double) 1 - outputNeurons[i]) * (inputNeurons[i] - outputNeurons[i]);
        }
//        return outputError;
    }

    private static void calculateHiddenError(double[] outputError, double[] hiddenError, double[] hiddenLayer, double[] weightsOfOutputs, double[] weightsOfBias) {
        int i;
        double errorWeight = 0;

        for (i = 0; i < outputs; i++) {
            errorWeight += (weightsOfOutputs[i * 2] + weightsOfOutputs[i * 2 + 1] + weightsOfBias[i + 2]) * outputError[i];
        }
        for (i = 0; i < hiddens; i++) {
            hiddenError[i] = hiddenLayer[i] * (1 - hiddenLayer[i]) * errorWeight;
        }
//        return hiddenError;
    }

    private void calculateErrors(double[] outputNeurons, double[] outputError, double[] hiddenError, double[] hiddenLayer, double[] inputNeurons, double[] weightsOfOutputs, double[] weightsOfBias) {
        calculateOutputError(outputNeurons, outputError, inputNeurons);
        calculateHiddenError(outputError, hiddenError, hiddenLayer, weightsOfOutputs, weightsOfBias);
    }

    private static int[] calculateWeightsOfHiddens(double[] hiddenError, double[] inputNeurons, double[] weightsOfHiddens, double[] weightsOfBias, double trainingStep, int i, int[] indicator) {
        if (i >= inputs) {
            weightsOfHiddens[i] = weightsOfHiddens[i] + trainingStep * hiddenError[1] * inputNeurons[i - inputs];
            if (indicator[0] == 0) {
                weightsOfBias[1] = weightsOfBias[1] + trainingStep * hiddenError[1] * inputNeurons[i - inputs];
                indicator[0]++;
            }
        } else {
            weightsOfHiddens[i] = weightsOfHiddens[i] + trainingStep * hiddenError[0] * inputNeurons[i];
            if (indicator[1] == 0) {
                weightsOfBias[0] = weightsOfBias[0] + trainingStep * hiddenError[1] * inputNeurons[i];
                indicator[1]++;
            }
        }
        return indicator;
    }

    private static void calculateWeightsOfOutputs(double[] outputError, double[] hiddenLayer, double[] weightsOfOutputs, double[] weightsOfBias, double trainingStep, int i) {
        if (i == 0 || i == 2 || i == 4 || i == 6) {
            weightsOfOutputs[i] = weightsOfOutputs[i] + trainingStep * outputError[i / 2] * hiddenLayer[0];
            weightsOfBias[i / 2 + 2] = weightsOfBias[i / 2 + 2] + trainingStep * outputError[i / 2] * bias;
        } else {
            weightsOfOutputs[i] = weightsOfOutputs[i] + trainingStep * outputError[(i - 1) / 2] * hiddenLayer[1];
        }
    }

    private static void calculateWeights(double[] outputError, double[] hiddenError, double[] hiddenLayer, double[] inputNeurons, double[] weightsOfHiddens, double[] weightsOfOutputs, double trainingStep, double[] weightsOfBias) {
        int i;
        int[] indicator = {0, 0};
        for (i = 0; i < hiddens * 4; i++) {
            indicator = calculateWeightsOfHiddens(hiddenError, inputNeurons, weightsOfHiddens, weightsOfBias, trainingStep, i, indicator);
            calculateWeightsOfOutputs(outputError, hiddenLayer, weightsOfOutputs, weightsOfBias, trainingStep, i);
        }
    }

    private int calculatePattern(double[] outputNeurons, double[] outputError, double[] hiddenError, double[] hiddenLayer, int patternOccurence, double[] inputNeurons, double[] weightsOfHiddens, double[] weightsOfOutputs, double[] weightsOfBias, double trainingStep) {
        int i;
        double e = 2.718281828459;
//        TrainingPattern t = new TrainingPattern(outputNeurons, outputError, hiddenError, hiddenLayer, inputNeurons, weightsOfHiddens, weightsOfOutputs, weightsOfBias, hiddens, inputs, outputs, bias);
//        t.propagateTrainingPattern();
        propagateTrainingPattern(outputNeurons, outputError, hiddenError, hiddenLayer, inputNeurons, weightsOfHiddens, weightsOfOutputs, weightsOfBias);
        for (i = 0; i < hiddens; i++) {
            hiddenLayer[i] = 1 / (1 + Math.pow(e, -hiddenLayer[i]));
        }

        for (i = 0; i < outputs; i++) {
            outputNeurons[i] = 1 / (1 + Math.pow(e, -outputNeurons[i]));
        }
        calculateErrors(outputNeurons, outputError, hiddenError, hiddenLayer, inputNeurons, weightsOfOutputs, weightsOfBias);
        calculateWeights(outputError, hiddenError, hiddenLayer, inputNeurons, weightsOfHiddens, weightsOfOutputs, trainingStep, weightsOfBias);

        patternOccurence++;
        return patternOccurence;
    }

    public void performingTraining(double trainingStep, double lowestWeight, double highestWeight, double[] pattern1, double[] pattern2, double[] pattern3, double[] pattern4) {
        int patternChoice;
        int steps = 0;

        int pattern1Occurences;
        int pattern2Occurences;
        int pattern3Occurences;
        int pattern4Occurences;
        long tStart = System.currentTimeMillis();
        long tEnd;
        long tDelta;
        double elapsedMinutes;

        double[] weightsOfBias = new double[hiddens + outputs];
        double[] weightsOfHiddens = new double[hiddens * 4];
        double[] weightsOfOutputs = new double[outputs * 2];

        double[] outputNeurons1 = new double[outputs];
        double[] outputNeurons2 = new double[outputs];
        double[] outputNeurons3 = new double[outputs];
        double[] outputNeurons4 = new double[outputs];

        double[] hiddenLayer1 = new double[hiddens];
        double[] hiddenLayer2 = new double[hiddens];
        double[] hiddenLayer3 = new double[hiddens];
        double[] hiddenLayer4 = new double[hiddens];

        double[] outputError = new double[outputs];
        double[] hiddenError = new double[hiddens];

        weightsOfBias = randomizeArray(weightsOfBias, lowestWeight, highestWeight, hiddens + outputs);
        weightsOfHiddens = randomizeArray(weightsOfHiddens, lowestWeight, highestWeight, hiddens * 4);
        weightsOfOutputs = randomizeArray(weightsOfOutputs, lowestWeight, highestWeight, outputs * 2);

        while (!compareArrays(outputNeurons1, pattern1) || !compareArrays(outputNeurons2, pattern2) || !compareArrays(outputNeurons3, pattern3) || !compareArrays(outputNeurons4, pattern4)) {
            pattern1Occurences = 0;
            pattern2Occurences = 0;
            pattern3Occurences = 0;
            pattern4Occurences = 0;
            steps++;

            while (pattern1Occurences == 0 || pattern2Occurences == 0 || pattern3Occurences == 0 || pattern4Occurences == 0) {
                patternChoice = ThreadLocalRandom.current().nextInt(1, 5);
                if (patternChoice == 1) {
                    pattern1Occurences = calculatePattern(outputNeurons1, outputError, hiddenError, hiddenLayer1, pattern1Occurences, pattern1, weightsOfHiddens, weightsOfOutputs, weightsOfBias, trainingStep);
                } else if (patternChoice == 2) {
                    pattern2Occurences = calculatePattern(outputNeurons2, outputError, hiddenError, hiddenLayer2, pattern2Occurences, pattern2, weightsOfHiddens, weightsOfOutputs, weightsOfBias, trainingStep);
                } else if (patternChoice == 3) {
                    pattern3Occurences = calculatePattern(outputNeurons3, outputError, hiddenError, hiddenLayer3, pattern3Occurences, pattern3, weightsOfHiddens, weightsOfOutputs, weightsOfBias, trainingStep);
                } else if (patternChoice == 4) {
                    pattern4Occurences = calculatePattern(outputNeurons4, outputError, hiddenError, hiddenLayer4, pattern4Occurences, pattern4, weightsOfHiddens, weightsOfOutputs, weightsOfBias, trainingStep);
                } else {
                    System.out.print("Something went wrong with the randomization\n");
                }
            }

            if ((steps % 1000000 == 0 && steps != 0) || steps == 1) {
                tEnd = System.currentTimeMillis();
                tDelta = tEnd - tStart;
                elapsedMinutes = tDelta / 1000.0;

                System.out.print("Hidden: ");
                System.out.print(steps);
                System.out.print("\n");
                printHidden(hiddenLayer1, 1);
                printHidden(hiddenLayer2, 2);
                printHidden(hiddenLayer3, 3);
                printHidden(hiddenLayer4, 4);
                System.out.print("\n");

                System.out.print("Epoch: ");
                System.out.print(steps);
                System.out.print("\n");
                printOutputs(outputNeurons1, 1);
                printOutputs(outputNeurons2, 2);
                printOutputs(outputNeurons3, 3);
                printOutputs(outputNeurons4, 4);
                System.out.print("\n");
                System.out.print(elapsedMinutes);
            }
        }

        System.out.print("Hidden: ");
        System.out.print(steps);
        System.out.print("\n");
        printOutputs(outputNeurons1, 1);
        printOutputs(outputNeurons2, 2);
        printOutputs(outputNeurons3, 3);
        printOutputs(outputNeurons4, 4);
    }

}
