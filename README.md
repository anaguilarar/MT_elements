## Element concentration prediction using a multi-target regression

This repository contains the code necessary to reproduce the analysis presented in the paper titled ['A multi-target regressor method to predict element concentrations in tomato leaves using hyperspectral imaging and machine learning models'](https://spj.science.org/doi/full/10.34133/plantphenomics.0146).

The repository focuses on:

- Spectral Data Processing: Includes code for preprocessing and transforming hyperspectral data, extracting relevant spectral features, and preparing the data for regression analysis.

- Element Concentration Prediction using Single-Target and Multi-Target Regression: Provides code implementations for predicting element concentrations in tomato leaves using both single-target regression, where each element is predicted individually, and multi-target regression, which includes the relationships between element concentrations.
  
- Regression Results Comparison: Offers code for evaluating and comparing the performance of different regression models, assessing prediction accuracies, and generating visualizations for result analysis.

<figure>
    <p align="center">
        <img src="figures/mt_pipeline.jpg" alt="alt" width="350" height="480">
    </p>
    <figcaption>Multi-target regression based on a chaining strategy</figcaption>
</figure>

### Usage

We provide google colabs to facilitate the understanding and implementation of the methods described in the paper The notebook includes detailed explanations, code snippets, and example data to guide you through the process.

To get started, please refer to the following notebook:

- Multi-Target prediction for element concentration: The [Multi-Target prediction for element concentration notebook](https://github.com/anaguilarar/MT_elements/blob/main/Multi-Target%20prediction%20for%20element%20concentration.ipynb) explains the process that was done to get element concentration predictions.







