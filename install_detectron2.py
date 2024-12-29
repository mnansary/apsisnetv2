import os
import sys
import subprocess
import distutils.core

# Clone the Detectron2 repository
def clone_repo(repo_url, dest_dir):
    if not os.path.exists(dest_dir):
        subprocess.check_call(['git', 'clone', repo_url, dest_dir])
    else:
        print(f"Repository already exists at {dest_dir}")

# Install Detectron2 dependencies
# Install Detectron2 dependencies
def install_dependencies(setup_file_path):
    dist = distutils.core.run_setup(setup_file_path)
    dependencies = dist.install_requires  # Do not add quotes here
    subprocess.check_call(['python', '-m', 'pip', 'install', *dependencies])

# Update the Python path to include Detectron2
def update_sys_path(detectron2_path):
    abs_path = os.path.abspath(detectron2_path)
    if abs_path not in sys.path:
        sys.path.insert(0, abs_path)

def main():
    repo_url = 'https://github.com/facebookresearch/detectron2'
    dest_dir = './detectron2'
    setup_file = os.path.join(dest_dir, 'setup.py')

    # Clone the repository
    clone_repo(repo_url, dest_dir)

    # Install the dependencies
    install_dependencies(setup_file)

    # Update sys.path
    update_sys_path(dest_dir)

if __name__ == "__main__":
    main()
