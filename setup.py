from setuptools import setup, find_packages

setup(
    name="shelfscale",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "requests",
        "plotly",
        "dash",
        "scikit-learn",
        "fuzzywuzzy",
        "python-Levenshtein",
        "openpyxl"
    ],
    author="ShelfScale Team",
    author_email="info@shelfscale.org",
    description="A standardized data product for nutrition and sustainability metrics at basket level",
    long_description=open("README.md").read() if hasattr(open("README.md"), "read") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/shelfscale/shelfscale",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "shelfscale=shelfscale.main:main",
        ],
    },
) 