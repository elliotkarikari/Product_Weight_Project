@echo off 
cd .. 
cd product_weight_clean_env 
call venv\Scripts\activate.bat 
cd .. 
cd Product_Weight_Project_Build 
python shelfscale\main.py 
pause 
