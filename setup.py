from setuptools import setup, find_packages

setup(
    name='Leviathan',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0,<2.0",
        "pandas>=2.0.0,<3.0",
        "dill>=0.3.7",
        "openai>=1.0.0,<2.0",
        "aisuite>=0.1.0",
        "python-dotenv>=1.0.0,<2.0",
        "pyyaml>=6.0.0,<7.0",
        "matplotlib>=3.7.0,<4.0",
        "seaborn>=0.12.0,<1.0",
        "fastapi>=0.100.0,<1.0",
        "uvicorn>=0.20.0,<1.0",
        "httpx>=0.24.0,<1.0",
        "cryptography>=41.0.0,<44.0",
        "requests>=2.31.0,<3.0",
    ],
    author='Danyang Chen, Chenyu Li',
    description='Leviathan: simulator for the evolution of human society',
    python_requires='>=3.10',
)
