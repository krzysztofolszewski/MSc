//
// Created by Krzysztof Olszewski on 29/08/18.
// Copyright (c) 2018 ___FULLUSERNAME___. All rights reserved.
//

#import "neuron.h"

@implementation neuron

double randomDouble(double a, double b) {
    double random = ((double) rand()) / (double) RAND_MAX;
    double diff = b - a;
    double r = random * diff;
    return a + r;
}

int compareArrays(double *array1, double *array2) {
    int i;

    for (i = 0; i < 4; i++) {
        if (array1[i] < array2[i] - 0.005 || array1[i] > array2[i] + 0.005) {
            return 0;
        }
    }
    return 1;
}

double randomizeArray(double *array, double lowest, double highest, int size) {
    int i;
    for (i = 0; i < size; i++) {
        array[i] = randomDouble(lowest, highest);
    }
    return *array;
}

void printOutputs(double *outputNeurons, int patternNumber) {
    int i;
    NSLog(@"Outputs of pattern number: %i\n", patternNumber);
    for (i = 0; i < 4; i++) {
        NSLog(@"%.3f\t", outputNeurons[i]);
    }
}

void printHidden(double *hiddenLayer, int patternNumber) {
    int i;
    NSLog(@"Hidden layers of pattern number: %i\n", patternNumber);
    for (i = 0; i < 2; i++) {
        NSLog(@"%.3f,\t", hiddenLayer[i]);
    }
}

void setArrayToZero(double *array, int size) {
    int i;
    for (i = 0; i < size; i++) {
        array[i] = 0;
    }
}

void calculateHiddenLayer(double *hiddenLayer, double *inputNeurons, double *weightsOfHiddens, double *weightsOfBias) {
    int i;
    int j;

    for (i = 0; i < 2; i++) {
        for (j = 0; j < 4; j++) {
            hiddenLayer[i] += inputNeurons[j] * weightsOfHiddens[j + 4 * i];
        }
        hiddenLayer[i] += 0.82 * weightsOfBias[i];
    }
}

void calculateOutputNeurons(double *outputNeurons, double *hiddenLayer, double *weightsOfOutputs, double *weightsOfBias) {
    int i;
    int j;

    for (i = 0; i < 4; i++) {
        for (j = 0; j < 2; j++) {
            outputNeurons[i] += hiddenLayer[j] * weightsOfOutputs[j + 2 * i];
        }
        outputNeurons[i] += 0.82 * weightsOfBias[i + 2];
    }
}

void propagateTrainingPattern(double *outputNeurons, double *outputError, double *hiddenError, double *hiddenLayer, double *inputNeurons, double *weightsOfHiddens, double *weightsOfOutputs, double *weightsOfBias) {
    setArrayToZero(outputNeurons, 4);
    setArrayToZero(outputError, 4);
    setArrayToZero(hiddenLayer, 2);
    setArrayToZero(hiddenError, 2);

    calculateHiddenLayer(hiddenLayer, inputNeurons, weightsOfHiddens, weightsOfBias);
    calculateOutputNeurons(outputNeurons, hiddenLayer, weightsOfOutputs, weightsOfBias);
}

void calculateOutputError(double *outputNeurons, double *outputError, double *inputNeurons) {
    int i;
    for (i = 0; i < 4; i++) {
        outputError[i] = outputNeurons[i] * ((double) 1 - outputNeurons[i]) * (inputNeurons[i] - outputNeurons[i]);
    }
}

void calculateHiddenError(double *outputError, double *hiddenError, double *hiddenLayer, double *weightsOfOutputs, double *weightsOfBias) {
    int i;
    double errorWeight = 0;

    for (i = 0; i < 4; i++) {
        errorWeight += (weightsOfOutputs[i * 2] + weightsOfOutputs[i * 2 + 1] + weightsOfBias[i + 2]) * outputError[i];
    }
    for (i = 0; i < 2; i++) {
        hiddenError[i] = hiddenLayer[i] * (1 - hiddenLayer[i]) * errorWeight;
    }
}

void calculateErrors(double *outputNeurons, double *outputError, double *hiddenError, double *hiddenLayer, double *inputNeurons, double *weightsOfOutputs, double *weightsOfBias) {
    calculateOutputError(outputNeurons, outputError, inputNeurons);
    calculateHiddenError(outputError, hiddenError, hiddenLayer, weightsOfOutputs, weightsOfBias);
}

int calculateWeightsOfHiddens(double *hiddenError, double *inputNeurons, double *weightsOfHiddens, double *weightsOfBias, double trainingStep, int i, int *indicator) {
    if (i >= 4) {
        weightsOfHiddens[i] = weightsOfHiddens[i] + trainingStep * hiddenError[1] * inputNeurons[i - 4];
        if (indicator[0] == 0) {
            weightsOfBias[1] = weightsOfBias[1] + trainingStep * hiddenError[1] * inputNeurons[i - 4];
            indicator[0]++;
        }
    } else {
        weightsOfHiddens[i] = weightsOfHiddens[i] + trainingStep * hiddenError[0] * inputNeurons[i];
        if (indicator[1] == 0) {
            weightsOfBias[0] = weightsOfBias[0] + trainingStep * hiddenError[1] * inputNeurons[i];
            indicator[1]++;
        }
    }
    return *indicator;
}

void calculateWeightsOfOutputs(double *outputError, double *hiddenLayer, double *weightsOfOutputs, double *weightsOfBias, double trainingStep, int i) {
    if (i == 0 || i == 2 || i == 4 || i == 6) {
        weightsOfOutputs[i] = weightsOfOutputs[i] + trainingStep * outputError[i / 2] * hiddenLayer[0];
        weightsOfBias[i / 2 + 2] = weightsOfBias[i / 2 + 2] + trainingStep * outputError[i / 2] * 0.82;
    } else {
        weightsOfOutputs[i] = weightsOfOutputs[i] + trainingStep * outputError[(i - 1) / 2] * hiddenLayer[1];
    }
}

void calculateWeights(double *outputError, double *hiddenError, double *hiddenLayer, double *inputNeurons, double *weightsOfHiddens, double *weightsOfOutputs, double trainingStep, double *weightsOfBias) {
    int i;
    int indicator[2] = {0, 0};
    for (i = 0; i < 2 * 4; i++) {
        *indicator = calculateWeightsOfHiddens(hiddenError, inputNeurons, weightsOfHiddens, weightsOfBias, trainingStep, i, indicator);
        calculateWeightsOfOutputs(outputError, hiddenLayer, weightsOfOutputs, weightsOfBias, trainingStep, i);
    }
}

+ (int)calculatePattern:(double *)outputNeurons :(double *)outputError :(double *)hiddenError :(double *)hiddenLayer :(int)patternOccurences :(double *)inputNeurons :(double *)weightsOfHiddens :(double *)weightsOfOutputs :(double *)weightsOfBias :(double)trainingStep {
    int i;
    double e = 2.718281828459;
    propagateTrainingPattern(outputNeurons, outputError, hiddenError, hiddenLayer, inputNeurons, weightsOfHiddens, weightsOfOutputs, weightsOfBias);
    for (i = 0; i < 2; i++) {
        hiddenLayer[i] = 1 / (1 + pow(e, -hiddenLayer[i]));
    }

    for (i = 0; i < 4; i++) {
        outputNeurons[i] = 1 / (1 + pow(e, -outputNeurons[i]));
    }
    calculateErrors(outputNeurons, outputError, hiddenError, hiddenLayer, inputNeurons, weightsOfOutputs, weightsOfBias);
    calculateWeights(outputError, hiddenError, hiddenLayer, inputNeurons, weightsOfHiddens, weightsOfOutputs, trainingStep, weightsOfBias);

    patternOccurences++;
    return patternOccurences;
}

- (void)performingTraining:(double)trainingStep :(double)lowestWeight :(double)highestWeight :(double *)pattern1 :(double *)pattern2 :(double *)pattern3 :(double *)pattern4 {
    int patternChoice;
    int steps = 0;

    int pattern1Occurences = 0;
    int pattern2Occurences = 0;
    int pattern3Occurences = 0;
    int pattern4Occurences = 0;

    double weightsOfBias[6] = {0, 0, 0, 0, 0, 0};
    double weightsOfHiddens[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    double weightsOfOutputs[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    double outputNeurons1[4] = {0, 0, 0, 0};
    double outputNeurons2[4] = {0, 0, 0, 0};
    double outputNeurons3[4] = {0, 0, 0, 0};
    double outputNeurons4[4] = {0, 0, 0, 0};

    double hiddenLayer1[2] = {0, 0};
    double hiddenLayer2[2] = {0, 0};
    double hiddenLayer3[2] = {0, 0};
    double hiddenLayer4[2] = {0, 0};

    double outputError[4] = {0, 0, 0, 0};
    double hiddenError[2] = {0, 0};

//    *weightsOfBias = setArrayToZero(weightsOfBias, hiddens + outputs);
    NSLog(@"%f", weightsOfBias[0]);

    *weightsOfBias = randomizeArray(weightsOfBias, lowestWeight, highestWeight, 6);
    *weightsOfHiddens = randomizeArray(weightsOfHiddens, lowestWeight, highestWeight, 8);
    *weightsOfOutputs = randomizeArray(weightsOfOutputs, lowestWeight, highestWeight, 8);

    NSLog(@"%f", weightsOfBias[0]);

    while (!compareArrays(outputNeurons1, pattern1) || !compareArrays(outputNeurons2, pattern2) || !compareArrays(outputNeurons3, pattern3) || !compareArrays(outputNeurons4, pattern4)) {
        pattern1Occurences = 0;
        pattern2Occurences = 0;
        pattern3Occurences = 0;
        pattern4Occurences = 0;
        steps++;

        while (pattern1Occurences == 0 || pattern2Occurences == 0 || pattern3Occurences == 0 || pattern4Occurences == 0) {
            patternChoice = rand() % 4 + 1;
            if (patternChoice == 1) {
                pattern1Occurences = [neuron calculatePattern:outputNeurons1 :outputError :hiddenError :hiddenLayer1 :pattern1Occurences :pattern1 :weightsOfHiddens :weightsOfOutputs :weightsOfBias :trainingStep];
            } else if (patternChoice == 2) {
                pattern2Occurences = [neuron calculatePattern:outputNeurons2 :outputError :hiddenError :hiddenLayer2 :pattern2Occurences :pattern2 :weightsOfHiddens :weightsOfOutputs :weightsOfBias :trainingStep];
            } else if (patternChoice == 3) {
                pattern3Occurences = [neuron calculatePattern:outputNeurons3 :outputError :hiddenError :hiddenLayer3 :pattern3Occurences :pattern3 :weightsOfHiddens :weightsOfOutputs :weightsOfBias :trainingStep];
            } else if (patternChoice == 4) {
                pattern4Occurences = [neuron calculatePattern:outputNeurons4 :outputError :hiddenError :hiddenLayer4 :pattern4Occurences :pattern4 :weightsOfHiddens :weightsOfOutputs :weightsOfBias :trainingStep];
            } else {
                NSLog(@"Something went wrong with the randomization");
            }
        }

        if ((steps % 1000000 == 0 && steps != 0) || steps == 1) {
            NSLog(@"Hidden: %i", steps);
            printHidden(hiddenLayer1, 1);
            printHidden(hiddenLayer2, 2);
            printHidden(hiddenLayer3, 3);
            printHidden(hiddenLayer4, 4);

            NSLog(@"Epoch: %i", steps);
            printOutputs(outputNeurons1, 1);
            printOutputs(outputNeurons2, 2);
            printOutputs(outputNeurons3, 3);
            printOutputs(outputNeurons4, 4);
        }
    }

    NSLog(@"Solution: %i", steps);
    printOutputs(outputNeurons1, 1);
    printOutputs(outputNeurons2, 2);
    printOutputs(outputNeurons3, 3);
    printOutputs(outputNeurons4, 4);
}

@end