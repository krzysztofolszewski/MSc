//
//  main.m
//  MSc_Objective-C
//
//  Created by Krzysztof Olszewski on 29/08/18.
//  Copyright (c) 2018 Krzysztof Olszewski. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "neuron.h"

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        int inputLayer = 4;
        int hiddenLayer = 2;
        int outputLayer = 4;
        double pattern1[4] = {1, 0, 0, 0};
        double pattern2[4] = {0, 1, 0, 0};
        double pattern3[4] = {0, 0, 1, 0};
        double pattern4[4] = {0, 0, 0, 1};
        double lowestWeight = -0.5;
        double highestWeight = 0.5;
        double trainingStep = 0.0005;
        double biasValue = 0.82;

        neuron *n1 = [[neuron alloc] init];
        n1.inputs = inputLayer;
        n1.hiddens = hiddenLayer;
        n1.outputs = outputLayer;
        n1.bias = biasValue;
        [n1 performingTraining:trainingStep :lowestWeight :highestWeight :pattern1 :pattern2 :pattern3 :pattern4];
    }

    return 0;
}