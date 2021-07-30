# AdamPSO 
A Particle Swarm Optimization Algorithm with Adaptive Moment Estimation. The algorithm and the experimental results detail in [[pdf]](/AdamPSO.pdf).

## Overview

<img align="right" width="300" src="/fig/f5_dim.jpg">
AdamPSO is a Particle Swarm Optimization Algorithm with Adaptive Moment Estimation (Adam) method for single objective black-box optimization. In this approach, the learning rate in each dimension is independently adjusted in a self-adaptive manner. As a result, it improves the performance of the conventional PSO algorithm in some classic benchmarking functions.

## Guideline for running the code 

Run `testEA.m` for reproducing the experimental results.

Modify `configurations.m` if you want to customize the testing cases.

The implementation of AdamPSO is in `optimisers/EA.m`.

## Reference

The testing framework is based on [Dr. Liu and Mr. Pei's repo](https://github.com/SUSTech-EC2021/Assignment1).
