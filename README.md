Title: Deep learning for parametric surrogate modelling of 2D mantle convection

![alt text](https://github.com/agsiddhant/ForwardSurrogate_Mars_2D/main/Fig_concept.pdf)


Abstract: Mantle convection, the buoyancy-driven creeping flow of silicate rocks in the interior of terrestrial planets like Earth or Mars, plays a fundamental role in the long-term thermal evolution of these bodies. Yet, key parameters and initial conditions of the partial differential equations governing mantle convection are poorly constrained. This of often requires a large sampling of the parameter space to determine which combinations can ultimately satisfy certain observational constraints. Traditionally, 1D models based on scaling laws used to parameterized convective heat transfer, have been used to tackle the computational bottleneck of high-fidelity forward runs in 2D or 3D. However, these are limited to one or two parameters and predict limited information such as the heat flux. Recent machine learning studies have shown that neural networks (NN) trained using a large number of 2D simulations can overcome this limitation and reliably predict in time the entire 1D laterally-averaged temperature profile of complex models (Agarwal 2020). We now extend that approach to predict the full 2D temperature field, which contains more information in the form of convection structures such as hot plumes and cold downwellings. Using a dataset of 10,525 2D simulations of the thermal evolution of the mantle of a Mars-like planet, we show that deep learning techniques can produce reliable parametric surrogates of the underlying partial differential equations. We first compress the temperature fields by a factor of 142 and then use NN and long-short term memory networks (LSTM) to predict the compressed fields. On average, the NN predictions are 99.29% and the LSTM predictions are 99.21% accurate with respect to unseen simulations. Proper orthogonal decomposition of the LSTM and NN predictions shows that despite a lower mean absolute relative accuracy, LSTMs capture the flow dynamics better than NNs.

Keywords: surrogate modelling, deep learning, mantle convection

Funding/grant number: We acknowledge the support of the Helmholtz Einstein International Berlin Research School in Data Science (HEIBRiDS). We also acknowledge the North-German Supercomputing Alliance (HLRN) for providing HPC resources (project id: bep00087). This work was also funded by the German Ministry for Education and Research as BIFOLD - Berlin Institute for the Foundations of Learning and Data (ref. 01IS18025A and ref 01IS18037A)

Date project Started: 01.02.2020

Date files were created: 2020/2021

Date the files were last updated: 20.03.2021

Version: 0.1

Author information:
Siddhant Agarwal
agsiddhant@gmail.com
ORCID: 0000-0002-0840-2114
Affiliation: German Aerospace Center (DLR), Berlin Institute of Technology 

Collaborators: 
Nicola Tosi, German Aerospace Center (DLR)
Pan Kessel, Berlin Institute of Technology 
Doris Breuer, German Aerospace Center (DLR) 
Grégoire Montavon, Berlin Institute of Technology 

License: This project is licensed under the terms of the MIT license.

Directory structure:

File Formats:
Jupyter notebooks (.ipynb files) are all written in python (v3.7.3). Loading trained machine learning models (.hdf5) requires keras (v2.3.1) and tensorflow (v1.14.0).

Links:
Not published yet.

Data Source and Methods: The results are obtained from a dataset consists of 10,525 mantle convection simulations run for a Mars-like planet on a 2D quarter-cylindrical grid. A detailed description of the setup is available on https://doi.org/10.1093/gji/ggaa234 (Agarwal 2020). This respository does not contain the dataset itself as it is over 10 TB large, but provides some trained machine learning models which can be used to predict the spatio-temporal thermal evolution of a Mars-like planet. 

Directory structure: simulateMars.ipynb can be used to calculate the 2D convective thermal evolution of Mars. User needs to only provide five input parameters. 
More advanced users can dig into processData/ to examine the data processing routines, train/ and utils/ to see how machine learning models were trained and loadTrainedModels/ for verifying the results obtained (such as error/relative accuracy plots).
