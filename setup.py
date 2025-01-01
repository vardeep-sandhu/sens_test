from setuptools import setup, find_packages

setup(
    name="trajectory_prediction_project",
    version="0.1",
    author="Your Name",
    description="A project for ego vehicle trajectory prediction using LiDAR scans.",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "torchaudio",
        "pytorch-lightning",
        "spconv-cu120", 
        "tqdm",
	    "lightning",
        "scipy",  
        "matplotlib",
        "numba",
        "python-box",
        "pyyaml"
    ],
    python_requires=">=3.7",
)
