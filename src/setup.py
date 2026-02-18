import setuptools

setuptools.setup(
    name="bayesgm", 
    version="1.0.0",
    author="Qiao Liu",
    author_email="qiao.liu@yale.edu",
    description="A toolkit for AI-driven Bayesian Generative Modeling",
    long_description="bayesgm is a toolkit providing a AI-driven Bayesian generative modeling framework for various Bayesian inference tasks in complex, high-dimensional data. The framework is built upon Bayesian principles combined with modern AI models, enabling flexible modeling of complex dependencies with principled uncertainty estimation. Currently, the bayesgm package includes two model families: BGM and CausalBGM.",
    long_description_content_type="text/markdown",
    url="https://github.com/liuq-lab/bayesgm",
    packages=setuptools.find_packages(),
    install_requires=[
   'numpy==1.24.2',
   'tensorflow==2.10.0',
   'tensorflow-probability==0.18.0',
   'pyyaml',
   'scikit-learn',
   'pandas',
   'tqdm',
   'python-dateutil'
    ],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7, <3.11',
    entry_points={
    'console_scripts': [
        'bayesgm = bayesgm.cli.cli:main',
        'causalBGM = bayesgm.cli.cli:main_causalbgm',
    ]},
)
