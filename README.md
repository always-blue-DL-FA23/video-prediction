# Project Name

## Description
This project installs 

## Prerequisites
- Access to an HPC environment with Slurm job scheduling.
- Python environment (specify the version if necessary).

## Installation
1. **Install Dependencies**: Navigate to the project directory and install the required packages listed in `requirements.txt`.

    ```
    cd [project-directory]
    pip install -r requirements.txt
    ```
2. From the project directory, create the following structure:
    ```
        mkdir ../outs
        mkdir ../outs/models ../outs/images ../outs/logs
    ```
    This will create folders that are not committed to git, that have all the output information of the python file

## Running the Script
1. You can run the script directly if you are not using the HPC Slurm environment. However, for HPC environments, you will need to submit it as a Slurm job.

    First, set-up a conda environment, and while logged into an instance with GPUs, perform pip install using requirements.txt as shown above.
    Then, traverse to the above mentioned outs directory, and modify relevant details in job.slurm. Submit job using:
    ```
        sbatch ../video-prediction/job.slurm
    ```
    job.slurm triggers the main.py file, which sequentially trains our SimVP and UNet models

2. To further train the generated Unet model, modify line *306* of main_semseg_2.py, then run:
    ```
        sbatch ../video-prediction/job_semseg.slurm
    ```


### Direct Execution
These scripts can also be directly executed with python, but the above directory structure is important for execution.
