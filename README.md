# Classic-Hidden-Markov-Model
Python Implementation of HMM

The code for HMM includes the following functions and class –

1. HMM Class – Class to store all the model parameters and the functions to train and evaluate the model.

2. Initialize() – Initializes the A, B and pi matrices using random. The elements of A and pi are close to 1/N and elements of B are close to 1/M.

3. alphaPass() – Computes the Forward Pass with scaling.

4. betaPass() – Computes the Backward Pass with scaling.

5. gamma() – Computes Gammas and digammas using the α and β matrices calculated above.

6. reEstimate() – Re-estimates the HMM parameters, I.e. A, B and pi matrices based on gammas and digammas.

7. computeLog() – Computes new log probability score of the observation sequence.

8. control() – Controls the iterations and the entire flow of the HMM algorithm.

9. preprocess() – Pre-processes the text data. Maps each alphabet in the text to number (0,1,2,…25) and each word space “ “ to 26. This creates the Observation Sequence (O) to train the model. We discard any other character encountered like numbers, special characters.
