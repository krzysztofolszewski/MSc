//
// Created by Krzysztof Olszewski on 30/08/18.
// Copyright (c) 2018 ___FULLUSERNAME___. All rights reserved.
//

import Foundation

class neuron {
    let inputs: Int
    let hiddens: Int
    let outputs: Int
    let bias: Double

    init(inputs: Int, hiddens: Int, outputs: Int, bias: Double) {
        self.inputs = inputs
        self.hiddens = hiddens
        self.outputs = outputs
        self.bias = bias
    }

    func printHidden(hiddenLayer: [Double], patternNumber: Int) {
        print("Hidden layers of pattern number", patternNumber)
        for i in 0 ..< hiddens {
            print(hiddenLayer[i])
        }
    }

    func printOutputs(outputNeurons: [Double], patternNumber: Int) {
        print("Outputs of pattern number", patternNumber)
        for i in 0 ..< outputs {
            print(outputNeurons[i])
        }
    }

    func compareArrays(array1: [Double], array2: [Double]) -> Bool {
        for i in 0 ..< 4 {
            if (array1[i] < array2[i] - 0.005 || array1[i] > array2[i] + 0.005) {
                return false;
            }
        }
        return true;
    }

    func randomizeArray(lowest: Double, highest: Double, size: Int) -> [Double] {
        var array = [Double](count: size, repeatedValue: 0)

        for i in 0 ..< size {
            array[i] = drand48() * (highest - lowest) + lowest
        }
        return array;
    }

    func setArrayToZero(size: Int) -> [Double] {
        var array = [Double](count: size, repeatedValue: 0)

        for i in 0 ..< size {
            array[i] = 0
        }
        return array;
    }

    func calculateHiddenLayer(hiddenLayer: [Double], inputNeurons: [Double], weightsOfHiddens: [Double], weightsOfBias: [Double]) -> [Double] {
        var hiddenLayerValues: [Double] = hiddenLayer

        for i in 0 ..< hiddens {
            for j in 0 ..< inputs {
                hiddenLayerValues[i] += inputNeurons[j] * weightsOfHiddens[j + 4 * i]
            }
            hiddenLayerValues[i] += bias * weightsOfBias[i]
        }
        return hiddenLayerValues
    }

    func calculateOutputNeurons(outputNeurons: [Double], hiddenLayer: [Double], weightsOfOutputs: [Double], weightsOfBias: [Double]) -> [Double] {
        var outputNeuronsValues: [Double] = outputNeurons

        for i in 0 ..< outputs {
            for j in 0 ..< hiddens {
                outputNeuronsValues[i] += hiddenLayer[j] * weightsOfOutputs[j + 2 * i]
            }
            outputNeuronsValues[i] += bias * weightsOfBias[i + 2]
        }
        return outputNeuronsValues
    }

    func propagateTrainingPattern(inout outputNeurons: [Double], inout outputError: [Double], inout hiddenError: [Double], inout hiddenLayer: [Double], inputNeurons: [Double], weightsOfHiddens: [Double], weightsOfOutputs: [Double], weightsOfBias: [Double]) {
        outputNeurons = setArrayToZero(outputs)
        outputError = setArrayToZero(outputs)
        hiddenLayer = setArrayToZero(hiddens)
        hiddenError = setArrayToZero(hiddens)

        hiddenLayer = calculateHiddenLayer(hiddenLayer, inputNeurons: inputNeurons, weightsOfHiddens: weightsOfHiddens, weightsOfBias: weightsOfBias)
        outputNeurons = calculateOutputNeurons(outputNeurons, hiddenLayer: hiddenLayer, weightsOfOutputs: weightsOfOutputs, weightsOfBias: weightsOfBias)
    }

    func calculateOutputError(outputNeurons: [Double], inout outputError: [Double], inputNeurons: [Double]) {
        for i in 0 ..< outputs {
            outputError[i] = outputNeurons[i] * (1.0 - outputNeurons[i]) * (inputNeurons[i] - outputNeurons[i])
        }
        //        return outputError;
    }

    func calculateHiddenError(outputError: [Double], inout hiddenError: [Double], hiddenLayer: [Double], weightsOfOutputs: [Double], weightsOfBias: [Double]) {
        var errorWeight = 0.0

        for i in 0 ..< outputs {
            errorWeight += (weightsOfOutputs[i * 2] + weightsOfOutputs[i * 2 + 1] + weightsOfBias[i + 2]) * outputError[i];
        }
        for i in 0 ..< hiddens {
            hiddenError[i] = hiddenLayer[i] * (1 - hiddenLayer[i]) * errorWeight;
        }
        //        return hiddenError;
    }

    func calculateErrors(outputNeurons: [Double], inout outputError: [Double], inout hiddenError: [Double], hiddenLayer: [Double], inputNeurons: [Double], weightsOfOutputs: [Double], weightsOfBias: [Double]) {
        calculateOutputError(outputNeurons, outputError: &outputError, inputNeurons: inputNeurons)
        calculateHiddenError(outputError, hiddenError: &hiddenError, hiddenLayer: hiddenLayer, weightsOfOutputs: weightsOfOutputs, weightsOfBias: weightsOfBias)
    }

    func calculateWeightsOfHiddens(hiddenError: [Double], inputNeurons: [Double], inout weightsOfHiddens: [Double], inout weightsOfBias: [Double], trainingStep: Double, i: Int, inout indicator: [Int]) -> [Int] {
        if (i >= inputs) {
            weightsOfHiddens[i] = weightsOfHiddens[i] + trainingStep * hiddenError[1] * inputNeurons[i - inputs];
            if (indicator[0] == 0) {
                weightsOfBias[1] = weightsOfBias[1] + trainingStep * hiddenError[1] * inputNeurons[i - inputs];
                indicator[0] += indicator[0];
            }
        } else {
            weightsOfHiddens[i] = weightsOfHiddens[i] + trainingStep * hiddenError[0] * inputNeurons[i];
            if (indicator[1] == 0) {
                weightsOfBias[0] = weightsOfBias[0] + trainingStep * hiddenError[1] * inputNeurons[i];
                indicator[1] += indicator[1];
            }
        }
        return indicator;
    }

    func calculateWeightsOfOutputs(outputError: [Double], hiddenLayer: [Double], inout weightsOfOutputs: [Double], inout weightsOfBias: [Double], trainingStep: Double, i: Int) {
        if (i == 0 || i == 2 || i == 4 || i == 6) {
            weightsOfOutputs[i] = weightsOfOutputs[i] + trainingStep * outputError[i / 2] * hiddenLayer[0]
            weightsOfBias[i / 2 + 2] = weightsOfBias[i / 2 + 2] + trainingStep * outputError[i / 2] * bias
        } else {
            weightsOfOutputs[i] = weightsOfOutputs[i] + trainingStep * outputError[(i - 1) / 2] * hiddenLayer[1]
        }
    }

    func calculateWeights(outputError: [Double], hiddenError: [Double], hiddenLayer: [Double], inputNeurons: [Double], inout weightsOfHiddens: [Double], inout weightsOfOutputs: [Double], trainingStep: Double, inout weightsOfBias: [Double]) {
        var indicator: [Int] = [0, 0]

        for i in 0 ..< hiddens * 4 {
            indicator = calculateWeightsOfHiddens(hiddenError, inputNeurons: inputNeurons, weightsOfHiddens: &weightsOfHiddens, weightsOfBias: &weightsOfBias, trainingStep: trainingStep, i: i, indicator: &indicator)
            calculateWeightsOfOutputs(outputError, hiddenLayer: hiddenLayer, weightsOfOutputs: &weightsOfOutputs, weightsOfBias: &weightsOfBias, trainingStep: trainingStep, i: i)
        }
    }

    func calculatePattern(inout outputNeurons: [Double], inout outputError: [Double], inout hiddenError: [Double], inout hiddenLayer: [Double], patternOccurence: Int, inputNeurons: [Double], inout weightsOfHiddens: [Double], inout weightsOfOutputs: [Double], inout weightsOfBias: [Double], trainingStep: Double) -> Int {
        let e = 2.718281828459;
        var patternOccurences = patternOccurence

        propagateTrainingPattern(&outputNeurons, outputError: &outputError, hiddenError: &hiddenError, hiddenLayer: &hiddenLayer, inputNeurons: inputNeurons, weightsOfHiddens: weightsOfHiddens, weightsOfOutputs: weightsOfOutputs, weightsOfBias: weightsOfBias);
        for i in 0 ..< hiddens {
            hiddenLayer[i] = 1 / (1 + pow(e, -hiddenLayer[i]));
        }

        for i in 0 ..< outputs {
            outputNeurons[i] = 1 / (1 + pow(e, -outputNeurons[i]));
        }
        calculateErrors(outputNeurons, outputError: &outputError, hiddenError: &hiddenError, hiddenLayer: hiddenLayer, inputNeurons: inputNeurons, weightsOfOutputs: weightsOfOutputs, weightsOfBias: weightsOfBias);
        calculateWeights(outputError, hiddenError: hiddenError, hiddenLayer: hiddenLayer, inputNeurons: inputNeurons, weightsOfHiddens: &weightsOfHiddens, weightsOfOutputs: &weightsOfOutputs, trainingStep: trainingStep, weightsOfBias: &weightsOfBias);

        patternOccurences += 1
        return patternOccurences
    }

    func performingTraining(trainingStep: Double, lowestWeight: Double, highestWeight: Double, pattern1: [Double], pattern2: [Double], pattern3: [Double], pattern4: [Double]) {
        var patternChoice: Int
        var steps = 0

        var pattern1Occurences: Int
        var pattern2Occurences: Int
        var pattern3Occurences: Int
        var pattern4Occurences: Int
//        long tStart = System.currentTimeMillis();
//        long tEnd;
//        long tDelta;
//        double elapsedMinutes;

        var weightsOfBias = [Double](count: hiddens + outputs, repeatedValue: 0)
        var weightsOfHiddens = [Double](count: hiddens * 4, repeatedValue: 0)
        var weightsOfOutputs = [Double](count: outputs * 2, repeatedValue: 0)

        var outputNeurons1 = [Double](count: outputs, repeatedValue: 0)
        var outputNeurons2 = [Double](count: outputs, repeatedValue: 0)
        var outputNeurons3 = [Double](count: outputs, repeatedValue: 0)
        var outputNeurons4 = [Double](count: outputs, repeatedValue: 0)

        var hiddenLayer1 = [Double](count: hiddens, repeatedValue: 0)
        var hiddenLayer2 = [Double](count: hiddens, repeatedValue: 0)
        var hiddenLayer3 = [Double](count: hiddens, repeatedValue: 0)
        var hiddenLayer4 = [Double](count: hiddens, repeatedValue: 0)

        var outputError = [Double](count: outputs, repeatedValue: 0)
        var hiddenError = [Double](count: hiddens, repeatedValue: 0)

        weightsOfBias = randomizeArray(lowestWeight, highest: highestWeight, size: hiddens + outputs)
        weightsOfHiddens = randomizeArray(lowestWeight, highest: highestWeight, size: hiddens * 4)
        weightsOfOutputs = randomizeArray(lowestWeight, highest: highestWeight, size: outputs * 2)

        while (!compareArrays(outputNeurons1, array2: pattern1) || !compareArrays(outputNeurons2, array2: pattern2) || !compareArrays(outputNeurons3, array2: pattern3) || !compareArrays(outputNeurons4, array2: pattern4)) {
            pattern1Occurences = 0;
            pattern2Occurences = 0;
            pattern3Occurences = 0;
            pattern4Occurences = 0;
            steps += 1;

            while (pattern1Occurences == 0 || pattern2Occurences == 0 || pattern3Occurences == 0 || pattern4Occurences == 0) {
                patternChoice = Int(arc4random_uniform(4)) + 1
                if (patternChoice == 1) {
                    pattern1Occurences = calculatePattern(&outputNeurons1, outputError: &outputError, hiddenError: &hiddenError, hiddenLayer: &hiddenLayer1, patternOccurence: pattern1Occurences, inputNeurons: pattern1, weightsOfHiddens: &weightsOfHiddens, weightsOfOutputs: &weightsOfOutputs, weightsOfBias: &weightsOfBias, trainingStep: trainingStep)
                } else if (patternChoice == 2) {
                    pattern2Occurences = calculatePattern(&outputNeurons2, outputError: &outputError, hiddenError: &hiddenError, hiddenLayer: &hiddenLayer2, patternOccurence: pattern2Occurences, inputNeurons: pattern2, weightsOfHiddens: &weightsOfHiddens, weightsOfOutputs: &weightsOfOutputs, weightsOfBias: &weightsOfBias, trainingStep: trainingStep)
                } else if (patternChoice == 3) {
                    pattern3Occurences = calculatePattern(&outputNeurons3, outputError: &outputError, hiddenError: &hiddenError, hiddenLayer: &hiddenLayer3, patternOccurence: pattern3Occurences, inputNeurons: pattern3, weightsOfHiddens: &weightsOfHiddens, weightsOfOutputs: &weightsOfOutputs, weightsOfBias: &weightsOfBias, trainingStep: trainingStep)
                } else if (patternChoice == 4) {
                    pattern4Occurences = calculatePattern(&outputNeurons4, outputError: &outputError, hiddenError: &hiddenError, hiddenLayer: &hiddenLayer4, patternOccurence: pattern4Occurences, inputNeurons: pattern4, weightsOfHiddens: &weightsOfHiddens, weightsOfOutputs: &weightsOfOutputs, weightsOfBias: &weightsOfBias, trainingStep: trainingStep)
                } else {
                    print("Something went wrong with the randomization\n");
                }
            }

            if ((steps % 1000000 == 0 && steps != 0) || steps == 1) {
//                tEnd = System.currentTimeMillis();
//                tDelta = tEnd - tStart;
//                elapsedMinutes = tDelta / 1000.0;

                print("Hidden:", steps);
                printHidden(hiddenLayer1, patternNumber: 1);
                printHidden(hiddenLayer2, patternNumber: 2);
                printHidden(hiddenLayer3, patternNumber: 3);
                printHidden(hiddenLayer4, patternNumber: 4);
                print("\n");

                print("Epoch:", steps);
                printOutputs(outputNeurons1, patternNumber: 1);
                printOutputs(outputNeurons2, patternNumber: 2);
                printOutputs(outputNeurons3, patternNumber: 3);
                printOutputs(outputNeurons4, patternNumber: 4);
                print("\n");
//                System.out.print(elapsedMinutes);
            }
        }

        print("Hidden:", steps);
        print(steps);
        printOutputs(outputNeurons1, patternNumber: 1);
        printOutputs(outputNeurons2, patternNumber: 2);
        printOutputs(outputNeurons3, patternNumber: 3);
        printOutputs(outputNeurons4, patternNumber: 4);
    }
}