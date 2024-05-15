from setuptools import setup, find_packages

setup(
    name="kaconv",
    version="0.1.0",
    packages=find_packages(include=['model_zoo.kaconv']),
    install_requires=[
        "torch>=2.3.0",
        "torchvision>=1.18.0",
        "numpy",
        "pandas",
        "tqdm",
    ],
    author="Xiangbo Gao",
    author_email="xiangbogaobarry@gmail.com",
    description="KA-Conv: Kolmogorov-Arnold Convolutional Networks with Various Basis Functions",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/XiangboGaoBarry/KA-Conv",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)