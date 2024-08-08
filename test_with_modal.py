from typing import Optional
from modal import Image, gpu, App, Mount, Secret
import subprocess
import os
import sys

image = Image.debian_slim().pip_install_from_requirements("dev-requirements.txt")
app = App(name="test-auto_hookpoint")

#mount the tests directory
tests = Mount.from_local_dir(".", remote_path="/root/project")
#run all tests on modal
# this enables us to run tests on a GPU 
@app.function(
    gpu=gpu.T4(count=1),  
    image=image,
    mounts=[tests]
)
def run_tests():
    # Add the project directory to the Python path
    project_dir = "/root/project"
    sys.path.append(project_dir)
    # Change to the project directory
    os.chdir(project_dir)

    print("in run_tests os.listdir()", os.listdir())
    print("in run_tests os.getcwd()", os.getcwd())

    # Run the tests
    subprocess.run(["pytest", "tests"], check=True)

@app.function(
    gpu=gpu.T4(count=1),  
    image=image,
    mounts=[tests],  
    secrets=[Secret.from_name('my-wandb-secret')] #type: ignore
)
def run_examples(
    filter_names : list[str] = [], 
    target_file : Optional[str] = None
):
    project_dir = "/root/project"
    sys.path.append(project_dir)
    # Change to the project directory
    os.chdir(project_dir)
    
    if target_file is not None:
        files_to_run = [target_file]
    else:
        files_to_run = [file for file in os.listdir("examples") if file.endswith(".py") and file not in filter_names]
    
    for file_name in files_to_run:
        print("running example:", file_name)
        subprocess.run(["python", f"examples/{file_name}"])
        print("success")

@app.local_entrypoint()
def main():
    run_examples.remote(target_file="sae_lens_example.py")
    
