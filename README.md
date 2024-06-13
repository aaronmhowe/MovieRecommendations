# Movie Recommendations
A simple recommender system that uses item-item collaborative filtering to recommend movies to users based on user-submitted ratings.

## Instructions for Running the Program
1. Install Python 3
    - This program's interpreter is Python 3+, I am using Python 3.10.12, you can download and install Python 3 from this [link](https://www.python.org/downloads/source/).
        - Note: That link takes you to the Python 3 release page for Linux/UNIX Operating Systems. I highly recommend this program is ran on such system.
2. Install PIP: PIP is required for this program.
    - Windows Installation Guide:
        1. On your system search, search for Windows Powershell, Git Bash, or CMD. Right-click the result and click 'Run as Administrator'
        2. type and enter this command:
            `curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py`
            PIP should download to your system.
        3. To install, enter this command:
            `python get-pip.py`
        4. Verify PIP installed by entering:
            `python -m pip help`
            Entering this command should display the location on your system of where PIP is installed
        5. Add a PATH variable for PIP
            1. Open the Windows Search, and type and enter "Environment Variables"
            2. System Properties should open, at the bottom of the window, click "Environment Variables".
            3. This will open a list of environment variables, double-click 'Path', or single-click and then click 'Edit'
            4. Click 'New', and then add the directory of where PIP is installed on your system. This directory should've been displayed from entering into your commmand prompt: `python -m pip help`
            5. Click 'OK' and the changes should save.
        6. Open a clean CMD, Bash, or Powershell, then type and enter `pip help`. This should display the same location information from step 4. You might have to instead enter `pip3 help`. If you're having issues, it might be wise to add the directory where your Python3 installation is located to the same PATH variables from step 5.
    - Linux/UNIX Installation Guide:
        1. In your system terminal, type `wget https://bootstrap.pypa.io/get-pip.py`
            This will download the installer.
        2. Install PIP to your system by typing and entering: `python3 ./get-pip.py`
            PIP should now be installed.

### Linux/UNIX Only
3. Configure the Shell Script.
    - The shell script located in MovieRecommendations/src will need a couple commands entered into the terminal before it will be an executable.
        1. CD into /src (`cd path/to/file/MovieRecommendations/src`)
        2. Type and enter `sudo chmod 777 run_code.sh` and enter your password when prompted.
        3. run_code.sh should now be executable. Type and enter `sudo ./run_code.sh` to run the shell script.
4. Alternatively, in your terminal, cd into /src (`cd path/to/file/MovieRecommendations/src`), and then type and enter `python3 Recommender.py`. This does the exact same thing as the shell script.

## Sources Used
- Execution Timer Information: https://www.geeksforgeeks.org/how-to-check-the-execution-time-of-python-script/# 

- Understanding Cosine Similarity: https://builtin.com/machine-learning/cosine-similarity

- Foundational Template for Programming Item-Item Collaborative Filtering: https://medium.com/geekculture/overview-of-item-item-collaborative-filtering-recommendation-system-64ee15b24bb8 

- My own notes that I wrote using various sources of which I no longer have the links to.
