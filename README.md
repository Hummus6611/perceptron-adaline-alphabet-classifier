# Perceptron and ADALINE Alphabet Classifier

## Title
Perceptron and ADALINE Alphabet Classifier

## Overview
This MATLAB project reconstructs an assignment that compares **Perceptron** and **ADALINE** learning on alphabet patterns loaded from `alphabet.mat`, with a focus on distinguishing the letters **E** and **F**.

It also includes the extra-credit XOR backpropagation example shown in the uploaded PDF.

## What the script does
- Loads `alphabet.mat`
- Trains a perceptron to separate **E** and **F**
- Tests a modified **E** pattern with the bottom row removed
- Trains three ADALINE variants with different learning rates
- Compares their responses on the modified pattern
- Includes a small 2–2–1 XOR neural network using sigmoid activations

## Required file
- `alphabet.mat`

Place `alphabet.mat` in the same folder before running the script.

## Files
- `perceptron_adaline_alphabet_classifier.m` — main MATLAB script

## Notes
The original PDF output shows unstable values and logical-scalar errors in parts of the ADALINE section. This reconstructed version keeps the same assignment idea while organizing it into a cleaner runnable form.

## Suggested repo description
MATLAB implementation of perceptron and ADALINE letter classification using alphabet patterns, plus a small XOR backpropagation example.
