# ICME Cluster Onboarding
Tutorial for getting set up on the ICME cluster.
This is used in the Stanford Course CME 218, Applied Data Science, a hands-on project course for graduate students working on machine learning and data science projects.
## Point of Contact
If something in this guide is unclear, please contact Elliot Epstein at epsteine@stanford.edu!

## Logging on to the cluster
The first step is to get access to the ICME cluster. All registered students should 
have been added to the cluster. Let me know if this is not the case.


If you are using Windows, it's recommended to first install windows subsystem for linux. This
can be installed here: https://learn.microsoft.com/en-us/windows/wsl/install

To use ssh to connect to the cluster you will need to be on the Stanford VPN.

Follow the guide [here](https://uit.stanford.edu/service/vpn) to set up the VPN. You will need to authenticate with
one duo mobile to get on to the VPN. 
Now check that you have access by ssh'ing into the cluster. 
You can do this by typing the following command into your terminal:
```
ssh <username>@icme-course-login.stanford.edu
```
where `<username>` is your Stanford username. A message similar to "The authenticity of the host 'icme-course-login.stanford.edu can't be established", are you sure you want to continue..." will appear. Type "yes" and hit enter. You will then be prompted to enter your password. Enter your Stanford password and hit enter. You should now be logged into the cluster.

If you get an error where nothing happens when you try to ssh in to the cluster, you may not be connected to the VPN.

You are currently on the login-node of the cluster. To run computation, you will need to transfer to a compute-node. To do this, type the following command into your terminal:
```
srun -p gpu-pascal --gres=gpu:1 --pty bash
```
The partiton is which GPU to use. You must use either the gpu-pascal, gpu-volta, or gpu-turing partition. The gres is the number of GPUs you want to use. The pty bash is to open a bash terminal. 

To get two GPUs use the gpu-volta or gpu-turing paritions:
```
srun -p gpu-volta --gres=gpu:2 --pty bash
```
You can check that you are on the gpu node by running 
```
nvidia-smi
```
This will show you which GPUs are available. If you get an error, you are most likely still on the login node.

Exit to the login node by running 
```
control-a d
```

If you don't need access to GPUs, you can submit to CPU nodes instead:
```
srun --pty bash
```

## Managing package dependencies
When running code on the cluster, you may have package dependencies (such as Numpy or Pytorch) that you need to install. To manager your dependencies, I recommend to use Anaconda. Anaconda is a package manager for Python. Using Anaconda allows us to for example have different packages installed in different environment, which can be very useful if you are working on multiple projects with conflicting package dependencies.

Conda is already installed on the cluster. It's needs to be initiated first 
by running 

```
eval "$(/opt/ohpc/pub/compiler/anancoda3/2023.09-0/bin/conda shell.bash hook)"
conda init
```


If the installation works correctly, the command
```
conda
```
should give a menu of options.

My go-to cheat-sheet for anaconda is [here](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)

Following the cheat cheet, let's create an anaconda environment.
```
conda create --name cme_218 python=3.9
```
This will take a few min.

Now, activate the environment with
```
source activate cme_218
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
Let's make sure we are on the Pascal partition. We can now add pytorch and it's dependencies. Start by checking the cuda version on the cluster, this can be done with
```
nvidia-smi
```
From this, I can see that the cuda version on the Pascal partition is 12.2. The most recent stable the release of Pytorch use CUDA 11.8, so let's use that version. Now we can install pytorch with the following command (see https://pytorch.org/get-started/locally/)
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
This will take more then 10 minutes.
You can check that pytorch was successfully installed with GPU support by starting a python terminal
```
python3
```
and then importing torch and seeing if a GPU is available
```
import torch
torch.cuda.is_available()
```
This should output True. Exit the python termial with 
```
control-d
```

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
<username>@icme-course-login.stanford.edu
```
where `<username>` is your Stanford username. You will then need to enter your password twice.

You can now open the code and make changes from VSCode.

In addition to editing code, VS Code allows you to view images, render markdown files, and connect to a terminal.

You can open a terminal inside vs code by clicking on the terminal bar on the top left and selecting new terminal.
## Clone the repo
Type the following command to clone the repository:
```
git clone https://github.com/Elliotepsteino/ICME-cluster-onboarding.git
```
This will create a folder called ICME-cluster-onboarding. Navigate into this folder by typing:
```
cd ICME-cluster-onboarding
```
## Getting data on the cluster
To get data on the cluster, you can use the `scp` command. This command allows you to transfer files between your local computer and the cluster.
Download the images from the ICME-cluster-onboarding folder in the files on the CME 218 Canvas page (in case you don't have access to the CME 218 canvas page, I have included the images in the /images folder on the repo)

Then upload the images to the cluster by typing the following commands into your local terminal:

```
scp /local_path_to_image/mnist_image.png <username>@icme-course-login.stanford.edu:~/ICME-cluster-onboarding/
```
```
scp /local_path_to_image/mnist_image_rotated.png <username>@icme-course-login.stanford.edu:~/ICME-cluster-onboarding/
```
where `<username>` is your Stanford username. This will upload the image to the ICME-cluster-onboarding folder on the cluster. If you use the images in the /image folder. Move them to the ICME-cluster-onboarding folder with the follwing command from the ICME-cluster-onboarding directory:

```
mv ./images/mnist_image_rotated.png .
mv ./images/mnist_image.png .
```

You can use the -r option if you want to move a directory instead of a file onto the cluster, for example

```
scp -r ./directory_to_move/ <username>@icme-course-login.stanford.edu:~/ICME-cluster-onboarding/
```

To download a file from the cluster, type the following command into your local terminal:
```
scp <username>@icme-course-login.stanford.edu:/file_path/<filename> /local_path/
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

Have a quick look at the code in the mnist_pytorch_example.py code. This code trains a simple neural network on the MNIST dataset. The MNIST dataset contains handwritten digits and the goal is to classify the digits correctly.

First, log onto a GPU node
```
srun -p gpu-pascal --gres=gpu:1 --pty bash
```

Let's create a screen session to run the training script in. Type:
```
screen -S CME_218
```
Let's make sure the conda environment we created earlier is activated, as the script uses PyTorch. Type:
```
source activate cme_218
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
### Acknowledgment
Thanks to Steve Jones and Brian Tempero for helpful tips on the cluster management.
