# ICME Cluster Onboarding
Tutorial for getting set up on the ICME cluster.
This is used in the Stanford Course CME 218, Applied Data Science, a hands-on project course for graduate students working on machine learning and data science projects.
## Point of Contact
If something in this guide is unclear, please contact me at epsteine@stanford.edu!

## Logging on to the cluster
The first step is to get access to the ICME cluster. All registered students should 
have been added to the cluster. Let me know if this is not the case. Some basic information about the cluster can be found [here](https://icme.stanford.edu/get-involved/resources/hpc-resources).

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
You can check that you are on the compute node by running 
```
nvidia-smi
```
This will show you which GPUs are available. If you get an error, you are most likely still on the login node.


## Managing package dependencies
When running code on the cluster, you may have package dependencies (such as Numpy or Pytorch) that you need to install.
To manager your dependencies, I recommend to use Anaconda.
Anaconda is a package manager for Python. Using Anaconda allows us to for example have different packages 
installed in different environment, which can be very useful if you are working on multiple projects 
with conflicting package dependencies. 

To make sure you are downloading anaconda for the right operating system, check the operating system on the server with 
```
cat /etc/os-release
```
From this, I can see that I am running ubuntu 22 LTS and I should hence use the Linux version of anaconda. Find the link address for the Anaconda installer for python 3.11 and download it to the server with
```
wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh
```

Then install anaconda with the following command:
```
bash Anaconda-latest-Linux-x86_64.sh
```
Follow the prompted instructions with all defaults.

After this is done, go ahead and close and reopen the terminal.

If the installation works correctly, the command 
```
conda
```
should give a menu of options. 

We can go ahead and delete the installer:
```
rm Anaconda3-2023.07-2-Linux-x86_64.sh
```
My go-to cheat-sheet for anaconda is [here](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)

Following the cheat cheet, let's create an anaconda environment.
```
conda create --name cme_218 python=3.9
```
This will take a few min. 

Now, activate the environment with 
```
conda activate cme_218
```
Packages you install will now be installed on the cme_218 environment. 

You can now list all your environments with 
```
conda env list
```
List the installed packages with 
```
conda list
```
Let's add matplotlib:

```
conda install matplotlib
```
Let's add pytorch and it's dependencies. Start by checking the cuda version on the cluster, this can be done with 
```
nvidia-smi
```
From this, I can see that the cuda version is 11.7.
Now we can install pytorch with the following command (see https://pytorch.org/get-started/locally/)
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```
To install with cuda version 11.3 instead (on k80 partition, CUDA 11.3 is compatible with CUDA 11.4), use
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
This will take a few minutes.

## Connect to VS Code
If you want to modify the code, you can either use a text editor such as emacs (https://www.gnu.org/software/emacs/) or you can use VSCode. 

I recommend using VSCode. VSCode can be downloaded here: (https://code.visualstudio.com/)

Once VSCode is installed, you will need to install a few extensions to connect to the cluster. 

You need to install the Remote - SSH extension. Press the extensions button in the left sidebar and use the search bar. 

You will also need to install the Remote - SSH: Editing Configuration Files extension and te Remote Explorer extension.

Also install the Pylance and Python extensions for linting and debugging tools. 

I also recommend to install GitHub Copilot, which you can get for free as a student.

Now, to connect to the cluster, press command-shift-p and type "Remote-SSH: Connect to Host". Then type:
```
<username>@icme-gpu.stanford.edu
```
where `<username>` is your Stanford username. You will then need to enter your password twice. 

You can now open the code and make changes from VSCode.

In addition to editing code, VS Code allows you to view images, render markdown files, and connect to a terminal.

You can open a terminal inside vs code by clicking on the terminal bar on the top left and selecting new terminal.

## Getting data on the Cluster
To get data on the cluster, you can use the `scp` command. This command allows you to transfer files between your local computer and the cluster.
Download the images from the ICME-cluster-onboarding folder in the files on the CME 218 Canvas page. 

Then upload the images to the cluster by typing the following commands into your local terminal:

```
scp /local_path_to_image/mnist_image.png <username>@icme-gpu.stanford.edu:~/ICME-cluster-onboarding/
```
```
scp /local_path_to_image/mnist_image_rotated.png <username>@icme-gpu.stanford.edu:~/ICME-cluster-onboarding/
```
where `<username>` is your Stanford username. This will upload the image to the ICME-cluster-onboarding folder on the cluster.

You can use the -r option if you want to move a directory instead of a file onto the cluster, for example

```
scp -r ./directory_to_move/ <username>@icme-gpu.stanford.edu:~/ICME-cluster-onboarding/
```

To download a file from the cluster, type the following command into your local terminal:
```
scp <username>@icme-gpu.stanford.edu:/file_path/<filename> /local_path/
```
where `<username>` is your Stanford username, local_path is the path to where you want to save the file on your local computer, and filename is the name of the file you want to download. You can download the file to the current path by typing `.` as the /local_path/.
## Screen
Screen is a linux program that allows you to run multiple programs at once. It also allows you to run programs in the background, so that you can close the terminal without stopping the program. This is really useful if you have jobs that take a long time to run. 

You can create a screen session by typing:
```
screen -S <screen_name>
```
where `<screen_name>` is the name of your screen session. You can name it anything you want. 

Typing this will create a screen session and you will be in the screen session. You can now run programs in the screen session.

To detach from the screen session, type:
```
control-a d
```
This takes you back to the terminal. To see all the screen sessions, type:
```
screen -list
```
This will show you all the screen sessions. To reattach to an existing screen session, type:
```
screen -r <screen_name>
```
where `<screen_name>` is the name of your screen session. For example, you can use different conda environments in different screen sessions.
To remove a screen session, first attach to it,
and then do:
```
control-a k
```
Sometimes, it's also useful to split your screen into multiple windows, for example if you want to monitor GPU usage while running a program. From within a screen session, you can split the screen vertically by typing:
```
control-a |
```
To switch between the different windows, type:
```
control-a tab
```
Once on the new tab, you can create a terminal by typing:
```
control-a c
```

## Train a Neural Network on the cluster
We will now train a basic Neural Network on the MNIST dataset. 

Type the following command to clone this repository:
```
git clone https://github.com/Elliotepsteino/ICME-cluster-onboarding.git
```
This will create a folder called ICME-cluster-onboarding. Navigate into this folder by typing:
```
cd ICME-cluster-onboarding
```
Have a quick look at the code in the mnist_pytorch_example.py code. This code trains a simple neural network on the MNIST dataset. The MNIST dataset contains handwritten digits and the goal is to classify the digits correctly.
Let's create a screen session to run the training script in. Type:
```
screen -S CME_218
```
Let's make sure the conda environment we created earlier is activated, as the script uses PyTorch. Type:
```
conda activate cme_218
```
Let's split the screen so we can monitor the GPU usage while training the model. Type:
```
control-a |
```
followed by: 
```
control-a tab
```
and 
```
control-a c
```
To create a new terminal in the split window. Let's run
```
nvidia-smi -l 1
```
This will show the GPU usage, refreshed every second.
Navigate back to the original window and run the training script by typing:
```
python mnist_pytorch_example.py --epochs 2 --save-model
```
This will download the mnist dataset and train the model for 2 epochs and save the trained model weights. You should see the loss decreasing and arond 1.7 GB of GPU memory being used. The MNIST data will be downloaded in the home directory.

After the training is done, you can run inference on the images you uploaded. You may need to change the image name in the inference_single_image.py file. 
```
python inference_single_image.py
```
What is the most likely class for the up-side down seven?
What about the correctly oriented seven?
## Other useful commands
You can change which GPU you are using to run a script by typing:
```
CUDA_VISIBLE_DEVICES=<gpu_id> 
```
If you requested two GPUs, you can use <gpu_id> either 0,1 or 0 or 1.

To print all the environment variables, type:
```
printenv
```
This will show which GPUs are available, among other things. 

Render markdown file in VSCode: 
```
command-shift-v
```

Use 
```
htop
```
To show which programs are currently running. Sometimes, the memory on GPUs is not released properly after a program is done running. You can then kill the relevant processes with F9 in htop. Exit htop with F10.

Scrolling in the terminal while in a screen session can be done with 
```
control-a esc
```
Exit the scrolling by pressing esc again.


Debug in python by setting breakpoints, by the breakpoint() function, this allows you to step through the code and inspect variables more efficiently than using print statements.

### Visualization
To track performance of ML jobs as they run, it's useful to use logging packages like Weights and Biases (https://wandb.ai/site).

### Running bash scripts
Bash scripts is a useful way to run multiple jobs at once. There is a batch script to run the mnist example on two different GPU nodes, with different learning rate. Run the bash script with 
```
bash run_mnist.sh
```
This will run the mnist example on two different GPU nodes, with different learning rate.
Remember that you must be on a compute node with two GPUs to run this script. 
### Author
Elliot Epstein