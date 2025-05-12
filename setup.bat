@echo off
echo Setting up Product Weight Project environment...

:: Create a virtual environment using Python's venv
echo Creating virtual environment...
python -m venv product_weight_env

:: Activate the virtual environment
echo Activating environment...
call product_weight_env\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install core dependencies
echo Installing core packages...
pip install pandas numpy scikit-learn

:: Install additional dependencies
echo Installing additional packages...
pip install fuzzywuzzy python-Levenshtein tabula-py PyPDF2 tqdm

:: Install visualization tools
echo Installing visualization packages...
pip install matplotlib seaborn plotly dash

:: Success message
echo.
echo Environment setup complete!
echo To activate this environment in the future, run: product_weight_env\Scripts\activate.bat
echo.
echo For future use, remember to activate the environment before running any scripts.
echo.
pause