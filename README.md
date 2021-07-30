# AdamPSO [[pdf]](/AdamPSO.pdf)
A Particle Swarm Optimization Algorithm with Adaptive Moment Estimation

## Overview

<img align="right" width="300" src="/fig/f5_dim.jpg">
AdamPSO is a Particle Swarm Optimization Algorithm with Adaptive Moment Estimation (Adam) method for single objective black-box optimization. In this approach, the learning rate in each dimension is independently adjusted in a self-adaptive manner. As a result, it improves the performance of the conventional PSO algorithm in some classic benchmarking functions.

## Guideline for running the code 

Run `testEA.m` for validating the results in [[pdf]](/AdamPSO.pdf).

Modify `configurations.m` if you want to customize the testing cases.

The implementation of AdamPSO is in `optimisers/EA.m`.

## Reference

This repo is based on [Dr. Liu and Mr. Pei's code](https://github.com/SUSTech-EC2021/Assignment1).
