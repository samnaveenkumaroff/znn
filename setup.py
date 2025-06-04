from setuptools import setup, find_packages

setup(
    name="znn",
    version="0.1.0",
    description="Zero Neural Network - A minimal autograd engine inspired by micrograd",
    author="Sam Naveen Kumar",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "matplotlib",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)
