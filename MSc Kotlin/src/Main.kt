/**
 * Created by kmo on 29/08/18.
 */

import java.util.concurrent.ThreadLocalRandom

object Main {

    @JvmStatic fun main(args: Array<String>) {
        val inputLayer = 4
        val hiddenLayer = 2
        val outputLayer = 4
        val pattern1 = doubleArrayOf(1.0, 0.0, 0.0, 0.0)
        val pattern2 = doubleArrayOf(0.0, 1.0, 0.0, 0.0)
        val pattern3 = doubleArrayOf(0.0, 0.0, 1.0, 0.0)
        val pattern4 = doubleArrayOf(0.0, 0.0, 0.0, 1.0)
        val lowestWeight = -0.5
        val highestWeight = 0.5
        val trainingStep = 0.005
        val biasValue = 0.83

        val n = Neuron(inputLayer, hiddenLayer, outputLayer, biasValue)
        n.performingTraining(trainingStep, lowestWeight, highestWeight, pattern1, pattern2, pattern3, pattern4)
    }

}

class Neuron(inputLayer: Int, hiddenLayer: Int, outputLayer: Int, biasValue: Double) {

    init {
        inputs = inputLayer
        outputs = outputLayer
        hiddens = hiddenLayer
        bias = biasValue
    }

    fun propagateTrainingPattern(outputNeurons: DoubleArray, outputError: DoubleArray, hiddenError: DoubleArray, hiddenLayer: DoubleArray, inputNeurons: DoubleArray, weightsOfHiddens: DoubleArray, weightsOfOutputs: DoubleArray, weightsOfBias: DoubleArray) {
        setArrayToZero(outputNeurons, outputs)
        setArrayToZero(outputError, outputs)
        setArrayToZero(hiddenLayer, hiddens)
        setArrayToZero(hiddenError, hiddens)

        calculateHiddenLayer(hiddenLayer, inputNeurons, weightsOfHiddens, weightsOfBias)
        calculateOutputNeurons(outputNeurons, hiddenLayer, weightsOfOutputs, weightsOfBias)
    }

    private fun calculateErrors(outputNeurons: DoubleArray, outputError: DoubleArray, hiddenError: DoubleArray, hiddenLayer: DoubleArray, inputNeurons: DoubleArray, weightsOfOutputs: DoubleArray, weightsOfBias: DoubleArray) {
        calculateOutputError(outputNeurons, outputError, inputNeurons)
        calculateHiddenError(outputError, hiddenError, hiddenLayer, weightsOfOutputs, weightsOfBias)
    }

    private fun calculatePattern(outputNeurons: DoubleArray, outputError: DoubleArray, hiddenError: DoubleArray, hiddenLayer: DoubleArray, patternOccurence: Int, inputNeurons: DoubleArray, weightsOfHiddens: DoubleArray, weightsOfOutputs: DoubleArray, weightsOfBias: DoubleArray, trainingStep: Double): Int {
        var numberOfPatterns = patternOccurence
        var i: Int = 0
        val e = 2.718281828459
        //        TrainingPattern t = new TrainingPattern(outputNeurons, outputError, hiddenError, hiddenLayer, inputNeurons, weightsOfHiddens, weightsOfOutputs, weightsOfBias, hiddens, inputs, outputs, bias);
        //        t.propagateTrainingPattern();
        propagateTrainingPattern(outputNeurons, outputError, hiddenError, hiddenLayer, inputNeurons, weightsOfHiddens, weightsOfOutputs, weightsOfBias)
        while (i < hiddens) {
            hiddenLayer[i] = 1 / (1 + Math.pow(e, -hiddenLayer[i]))
            i++
        }

        i = 0
        while (i < outputs) {
            outputNeurons[i] = 1 / (1 + Math.pow(e, -outputNeurons[i]))
            i++
        }
        calculateErrors(outputNeurons, outputError, hiddenError, hiddenLayer, inputNeurons, weightsOfOutputs, weightsOfBias)
        calculateWeights(outputError, hiddenError, hiddenLayer, inputNeurons, weightsOfHiddens, weightsOfOutputs, trainingStep, weightsOfBias)

        numberOfPatterns++
        return numberOfPatterns
    }

    fun performingTraining(trainingStep: Double, lowestWeight: Double, highestWeight: Double, pattern1: DoubleArray, pattern2: DoubleArray, pattern3: DoubleArray, pattern4: DoubleArray) {
        var patternChoice: Int
        var steps = 0

        var pattern1Occurences: Int
        var pattern2Occurences: Int
        var pattern3Occurences: Int
        var pattern4Occurences: Int
        val tStart = System.currentTimeMillis()
        var tEnd: Long
        var tDelta: Long
        var elapsedMinutes: Double

        var weightsOfBias = DoubleArray(hiddens + outputs)
        var weightsOfHiddens = DoubleArray(hiddens * 4)
        var weightsOfOutputs = DoubleArray(outputs * 2)

        val outputNeurons1 = DoubleArray(outputs)
        val outputNeurons2 = DoubleArray(outputs)
        val outputNeurons3 = DoubleArray(outputs)
        val outputNeurons4 = DoubleArray(outputs)

        val hiddenLayer1 = DoubleArray(hiddens)
        val hiddenLayer2 = DoubleArray(hiddens)
        val hiddenLayer3 = DoubleArray(hiddens)
        val hiddenLayer4 = DoubleArray(hiddens)

        val outputError = DoubleArray(outputs)
        val hiddenError = DoubleArray(hiddens)

        weightsOfBias = randomizeArray(weightsOfBias, lowestWeight, highestWeight, hiddens + outputs)
        weightsOfHiddens = randomizeArray(weightsOfHiddens, lowestWeight, highestWeight, hiddens * 4)
        weightsOfOutputs = randomizeArray(weightsOfOutputs, lowestWeight, highestWeight, outputs * 2)

        while (!compareArrays(outputNeurons1, pattern1) || !compareArrays(outputNeurons2, pattern2) || !compareArrays(outputNeurons3, pattern3) || !compareArrays(outputNeurons4, pattern4)) {
            pattern1Occurences = 0
            pattern2Occurences = 0
            pattern3Occurences = 0
            pattern4Occurences = 0
            steps++

            while (pattern1Occurences == 0 || pattern2Occurences == 0 || pattern3Occurences == 0 || pattern4Occurences == 0) {
                patternChoice = ThreadLocalRandom.current().nextInt(1, 5)
                if (patternChoice == 1) {
                    pattern1Occurences = calculatePattern(outputNeurons1, outputError, hiddenError, hiddenLayer1, pattern1Occurences, pattern1, weightsOfHiddens, weightsOfOutputs, weightsOfBias, trainingStep)
                } else if (patternChoice == 2) {
                    pattern2Occurences = calculatePattern(outputNeurons2, outputError, hiddenError, hiddenLayer2, pattern2Occurences, pattern2, weightsOfHiddens, weightsOfOutputs, weightsOfBias, trainingStep)
                } else if (patternChoice == 3) {
                    pattern3Occurences = calculatePattern(outputNeurons3, outputError, hiddenError, hiddenLayer3, pattern3Occurences, pattern3, weightsOfHiddens, weightsOfOutputs, weightsOfBias, trainingStep)
                } else if (patternChoice == 4) {
                    pattern4Occurences = calculatePattern(outputNeurons4, outputError, hiddenError, hiddenLayer4, pattern4Occurences, pattern4, weightsOfHiddens, weightsOfOutputs, weightsOfBias, trainingStep)
                } else {
                    print("Something went wrong with the randomization\n")
                }
            }

            if (steps % 1000000 == 0 && steps != 0 || steps == 1) {
                tEnd = System.currentTimeMillis()
                tDelta = tEnd - tStart
                elapsedMinutes = tDelta / 1000.0

                print("Hidden: ")
                print(steps)
                print("\n")
                printHidden(hiddenLayer1, 1)
                printHidden(hiddenLayer2, 2)
                printHidden(hiddenLayer3, 3)
                printHidden(hiddenLayer4, 4)
                print("\n")

                print("Epoch: ")
                print(steps)
                print("\n")
                printOutputs(outputNeurons1, 1)
                printOutputs(outputNeurons2, 2)
                printOutputs(outputNeurons3, 3)
                printOutputs(outputNeurons4, 4)
                print("\n")
                print(elapsedMinutes)
            }
        }

        print("Hidden: ")
        print(steps)
        print("\n")
        printOutputs(outputNeurons1, 1)
        printOutputs(outputNeurons2, 2)
        printOutputs(outputNeurons3, 3)
        printOutputs(outputNeurons4, 4)
    }

    companion object {

        private var inputs: Int = 0
        private var hiddens: Int = 0
        private var outputs: Int = 0
        private var bias: Double = 0.0

        private fun printHidden(hiddenLayer: DoubleArray, patternNumber: Int) {
            var i: Int = 0
            print("Hidden layers of pattern number ")
            print(patternNumber)
            print(":")
            print("\n")
            while (i < hiddens) {
                System.out.printf("%.3f", hiddenLayer[i])
                print(", ")
                print("\t")
                i++
            }
            print("\n")
        }

        private fun printOutputs(outputNeurons: DoubleArray, patternNumber: Int) {
            var i: Int = 0
            print("Outputs of pattern number ")
            print(patternNumber)
            print(":")
            print("\n")
            while (i < outputs) {
                System.out.printf("%.3f", outputNeurons[i])
                print(", ")
                print("\t")
                i++
            }
            print("\n")
        }

        private fun compareArrays(array1: DoubleArray, array2: DoubleArray): Boolean {
            var i: Int = 0

            while (i < 4) {
                if (array1[i] < array2[i] - 0.005 || array1[i] > array2[i] + 0.005) {
                    return false
                }
                i++
            }
            return true
        }

        private fun randomizeArray(array: DoubleArray, lowest: Double, highest: Double, size: Int): DoubleArray {
            var i: Int = 0
            while (i < size) {
                array[i] = ThreadLocalRandom.current().nextDouble(lowest, highest)
                i++
            }
            return array
        }

        private fun setArrayToZero(array: DoubleArray, size: Int): DoubleArray {
            var i: Int = 0
            while (i < size) {
                array[i] = 0.0
                i++
            }
            return array
        }

        private fun calculateHiddenLayer(hiddenLayer: DoubleArray, inputNeurons: DoubleArray, weightsOfHiddens: DoubleArray, weightsOfBias: DoubleArray) {
            var i: Int = 0
            var j: Int

            while (i < hiddens) {
                j = 0
                while (j < inputs) {
                    hiddenLayer[i] += inputNeurons[j] * weightsOfHiddens[j + 4 * i]
                    j++
                }
                hiddenLayer[i] += bias * weightsOfBias[i]
                i++
            }
            //return hiddenLayer;
        }

        private fun calculateOutputNeurons(outputNeurons: DoubleArray, hiddenLayer: DoubleArray, weightsOfOutputs: DoubleArray, weightsOfBias: DoubleArray) {
            var i: Int = 0
            var j: Int

            while (i < outputs) {
                j = 0
                while (j < hiddens) {
                    outputNeurons[i] += hiddenLayer[j] * weightsOfOutputs[j + 2 * i]
                    j++
                }
                outputNeurons[i] += bias * weightsOfBias[i + 2]
                i++
            }
            //return outputNeurons;
        }

        private fun calculateOutputError(outputNeurons: DoubleArray, outputError: DoubleArray, inputNeurons: DoubleArray) {
            var i: Int = 0
            while (i < outputs) {
                outputError[i] = outputNeurons[i] * (1.toDouble() - outputNeurons[i]) * (inputNeurons[i] - outputNeurons[i])
                i++
            }
            //        return outputError;
        }

        private fun calculateHiddenError(outputError: DoubleArray, hiddenError: DoubleArray, hiddenLayer: DoubleArray, weightsOfOutputs: DoubleArray, weightsOfBias: DoubleArray) {
            var i: Int = 0
            var errorWeight = 0.0

            while (i < outputs) {
                errorWeight += (weightsOfOutputs[i * 2] + weightsOfOutputs[i * 2 + 1] + weightsOfBias[i + 2]) * outputError[i]
                i++
            }
            i = 0
            while (i < hiddens) {
                hiddenError[i] = hiddenLayer[i] * (1 - hiddenLayer[i]) * errorWeight
                i++
            }
            //        return hiddenError;
        }

        private fun calculateWeightsOfHiddens(hiddenError: DoubleArray, inputNeurons: DoubleArray, weightsOfHiddens: DoubleArray, weightsOfBias: DoubleArray, trainingStep: Double, i: Int, indicator: IntArray): IntArray {
            if (i >= inputs) {
                weightsOfHiddens[i] = weightsOfHiddens[i] + trainingStep * hiddenError[1] * inputNeurons[i - inputs]
                if (indicator[0] == 0) {
                    weightsOfBias[1] = weightsOfBias[1] + trainingStep * hiddenError[1] * inputNeurons[i - inputs]
                    indicator[0]++
                }
            } else {
                weightsOfHiddens[i] = weightsOfHiddens[i] + trainingStep * hiddenError[0] * inputNeurons[i]
                if (indicator[1] == 0) {
                    weightsOfBias[0] = weightsOfBias[0] + trainingStep * hiddenError[1] * inputNeurons[i]
                    indicator[1]++
                }
            }
            return indicator
        }

        private fun calculateWeightsOfOutputs(outputError: DoubleArray, hiddenLayer: DoubleArray, weightsOfOutputs: DoubleArray, weightsOfBias: DoubleArray, trainingStep: Double, i: Int) {
            if (i == 0 || i == 2 || i == 4 || i == 6) {
                weightsOfOutputs[i] = weightsOfOutputs[i] + trainingStep * outputError[i / 2] * hiddenLayer[0]
                weightsOfBias[i / 2 + 2] = weightsOfBias[i / 2 + 2] + trainingStep * outputError[i / 2] * bias
            } else {
                weightsOfOutputs[i] = weightsOfOutputs[i] + trainingStep * outputError[(i - 1) / 2] * hiddenLayer[1]
            }
        }

        private fun calculateWeights(outputError: DoubleArray, hiddenError: DoubleArray, hiddenLayer: DoubleArray, inputNeurons: DoubleArray, weightsOfHiddens: DoubleArray, weightsOfOutputs: DoubleArray, trainingStep: Double, weightsOfBias: DoubleArray) {
            var i: Int = 0
            var indicator = intArrayOf(0, 0)
            while (i < hiddens * 4) {
                indicator = calculateWeightsOfHiddens(hiddenError, inputNeurons, weightsOfHiddens, weightsOfBias, trainingStep, i, indicator)
                calculateWeightsOfOutputs(outputError, hiddenLayer, weightsOfOutputs, weightsOfBias, trainingStep, i)
                i++
            }
        }
    }

}
