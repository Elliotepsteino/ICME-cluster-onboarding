# ICME Cluster Onboarding
This is a quick tutorial used as teaching material in the Stanford Course CME 218, Applied Data Science, A hands-on project course for graduate students working on machine learning and data science projects.

## Getting Started
The first step is to get access to the ICME cluster. Message Brian Tempero and ask him to add you to the ICME-GPU cluster. Some basic information about the cluster can be found [here](https://icme.stanford.edu/get-involved/resources/hpc-resources).

Check that you have access by ssh'ing into the cluster. You can do this by typing the following command into your terminal:
```
ssh <username>@icme-gpu.stanford.edu
```
where `<username>` is your Stanford username. A message similar to "The authenticity of the host 'icme-gpu.stanford.edu can't be established", are you sure you want to continue..." will appear. Type "yes" and hit enter. You will then be prompted to enter your password. Enter your Stanford password and hit enter. You should now be logged into the cluster.

You are currently on the login-node of the cluster. To run computation, you will need to transfer to a compute-node. To do this, type the following command into your terminal:
```
srun --partition=V100 --gres=gpu:1 --pty bash
```
The partiton is which GPU to use. You must use either the V100, k80, or CME partition. The gres is the number of GPUs you want to use. The pty bash is to open a bash terminal. 

To get two GPUs on the k80 node type:
```
srun --partition=k80 --gres=gpu:2 --pty bash
```
We will now train a basic Neural Network on the MNIST dataset. 

On the GPU node, type the following command to clone this repository:
```
git clone https://github.com/Elliotepsteino/ICME-cluster-onboarding.git
```
This will create a folder called ICME-cluster-onboarding. Navigate into this folder by typing:
```
cd ICME-cluster-onboarding
```




