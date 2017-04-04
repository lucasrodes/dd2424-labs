# Report


## Exercise 1

 1. Order code + set an index of the files
 2. Skeleton of the report
    - Intro
    - Problem
    - Loss fct plot and weight matrix images + corresponding accuracies
        - [0, 40, 100, .1] (1), Accuracy = 22.55 %
        - [0, 40, 100, .01] (2), Accuracy = 36.65 %
        - [0.1, 40, 100, .01] (3), Accuracy = 33.37 %
        - [1, 40, 100, .01] (4), Accuracy = 21.92 %
    - Effect of increasing regularization + importance of learning rate

## Exercise 2

2.1: Optimization

A) Working on Setting (2)

+ permute samples after each epoch!
Accuracy = 38.34 %
[WE PRESERVE THIS]

+ Center data
Accuracy = 39.34 %
[WE PRESERVE THIS]

+ We increase training set, include 9000 samples from 'data_batch_2.mat' (aka validation set)
Accuracy = 40.47 %
[WE PRESERVE THIS]

B) Working on Setting (1)

+ Introduce decrease factor 0.9 for the learning rate
Accuracy = 38.85 %
[WE PRESERVE THIS]

+ Center data
Accuracy = 39.32 %   Accuracy = 39.37 %
[WE PRESERVE THIS]

+ permute samples after each epoch!
Accuracy = 39.37 %
[WE PRESERVE THIS]

+ We increase training set, include 9000 samples from 'data_batch_2.mat' (aka validation set)
Accuracy = 40.13 %
[WE PRESERVE THIS]

* Analysis of jitter addition
- Include picture
- Noise increases generalization.


Not satisfactory
----------------

- 0.9 LR decrease after each epoch
Accuracy = 37.40 %
[FORGET BOUT THIS]

Notes: No good lambda found!
