@echo off
REM PaperWoCode Setup Script for Windows
REM This script sets up the conda environment and runs the project

echo PaperWoCode Setup Script for Windows

REM Check if conda is installed
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Conda is not installed. Please install conda first.
    echo Visit https://docs.conda.io/en/latest/miniconda.html for installation instructions.
    exit /b 1
)

REM Check if environment already exists and remove it if it does
conda env list | findstr /C:"paperwocode" >nul
if %ERRORLEVEL% EQU 0 (
    echo Conda environment 'paperwocode' already exists. Removing it to ensure a clean setup...
    conda env remove -n paperwocode
)

REM Create the conda environment
echo Creating conda environment from environment.yml...
conda env create -f environment.yml
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to create conda environment from environment.yml.
    echo Trying alternative setup method...
    
    REM Create a basic environment and install dependencies manually
    conda create -n paperwocode python=3.10 -y
    if %ERRORLEVEL% NEQ 0 (
        echo Error: Failed to create conda environment.
        exit /b 1
    )
    
    REM Install dependencies
    echo Installing dependencies...
    call conda activate paperwocode
    conda install -y numpy pandas matplotlib scikit-learn pytorch torchvision torchaudio -c pytorch
    pip install "markitdown[all]~=0.1.0a1" anthropic openai PyPDF2 tqdm pyyaml tiktoken
) else (
    REM Activate the environment
    echo Activating conda environment...
    call conda activate paperwocode
)

REM Install the package in development mode
echo Installing PaperWoCode package in development mode...
pip install -e .

REM Check if API keys are set
if "%ANTHROPIC_API_KEY%"=="" (
    echo ANTHROPIC_API_KEY is not set.
    set /p anthropic_key="Enter your Anthropic API key: "
    setx ANTHROPIC_API_KEY "%anthropic_key%"
    set ANTHROPIC_API_KEY=%anthropic_key%
    echo ANTHROPIC_API_KEY has been set as a system environment variable.
)

if "%OPENAI_API_KEY%"=="" (
    echo OPENAI_API_KEY is not set.
    set /p openai_key="Enter your OpenAI API key: "
    setx OPENAI_API_KEY "%openai_key%"
    set OPENAI_API_KEY=%openai_key%
    echo OPENAI_API_KEY has been set as a system environment variable.
)

REM Create output directory
mkdir output\workflows 2>nul

REM Run the project
echo.
echo Setup complete! You can now run the project with:
echo conda activate paperwocode
echo python main.py [arxiv_url]
echo.
echo For example:
echo python main.py https://arxiv.org/abs/2411.16905
