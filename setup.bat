@echo off
echo ===================================================
echo    Product Weight Project Environment Setup
echo ===================================================

:: Move to a clean directory outside the current path
cd ..
mkdir product_weight_clean_env
cd product_weight_clean_env

:: Create virtual environment
echo Creating clean virtual environment...
python -m venv venv

:: Activate the environment
call venv\Scripts\activate.bat
echo Environment activated

:: Install all required packages
echo Installing packages...
pip install pandas==2.0.3 numpy==1.24.3 scikit-learn==1.2.2
pip install fuzzywuzzy python-Levenshtein tabula-py PyPDF2 tqdm
pip install matplotlib seaborn plotly dash

:: Create a run script
echo Creating run script...
cd ..
cd Product_Weight_Project_Build

echo @echo off > run_project.bat
echo cd .. >> run_project.bat
echo cd product_weight_clean_env >> run_project.bat
echo call venv\Scripts\activate.bat >> run_project.bat
echo cd .. >> run_project.bat
echo cd Product_Weight_Project_Build >> run_project.bat
echo python shelfscale\main.py >> run_project.bat
echo pause >> run_project.bat

echo ===================================================
echo Setup complete! 
echo.
echo To run your project, use: run_project.bat
echo ===================================================
pause