//
//  main.swift
//  MSC Swift
//
//  Created by Krzysztof Olszewski on 30/08/18.
//  Copyright (c) 2018 Krzysztof Olszewski. All rights reserved.
//

import Foundation

var inputLayer = 4
var hiddenLayer = 2
var outputLayer = 4
var pattern1:[Double] = [1, 0, 0, 0]
var pattern2:[Double] = [0, 1, 0, 0]
var pattern3:[Double] = [0, 0, 1, 0]
var pattern4:[Double] = [0, 0, 0, 1]
var lowestWeight = -0.5
var highestWeight = 0.5
var trainingStep = 0.005
var biasValue = 0.83

let n = neuron(inputs: inputLayer, hiddens: hiddenLayer, outputs: outputLayer, bias: biasValue)
n.performingTraining(trainingStep, lowestWeight: lowestWeight, highestWeight: highestWeight, pattern1: pattern1, pattern2: pattern2, pattern3: pattern3, pattern4:pattern4)
