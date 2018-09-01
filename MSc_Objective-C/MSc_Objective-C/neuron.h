//
// Created by Krzysztof Olszewski on 29/08/18.
// Copyright (c) 2018 ___FULLUSERNAME___. All rights reserved.
//

#import <Foundation/Foundation.h>


@interface neuron : NSObject {
    int inputs;
    int hiddens;
    int outputs;
    double bias;
}
@property int inputs;
@property int hiddens;
@property int outputs;
@property double bias;

- (void)performingTraining:(double)trainingStep :(double)lowestWeight :(double)highestWeight :(double *)pattern1 :(double *)pattern2 :(double *)pattern3 :(double *)pattern4;

@end