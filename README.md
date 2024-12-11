# Bowling Advisor Chatbot
## Overview
This project implements an AI-powered bowling advisor that analyzes game data and provides personalized recommendations using LangChain and OpenAI's language models through a Streamlit interface.
## Features
* Real-time analysis of bowling statistics
* Interactive chat interface for personalized advice
* Analysis of speed patterns (fast, medium, slow)
* Analysis of angle patterns (straight, left, right)
* Visual data presentation with expandable sections
## Prerequisites
* Python 3.9+
* OpenAI API key
* Excel file with bowling data
## Installation
```bash
Clone the repository
git clone [your-repo-url]
Navigate to project directory
Install required packages
pip install streamlit pandas openpyxl langchain openai
```
## Data Format
The Excel file should contain the following columns:
* Run
* player
* speed 1
* speed 2
* angle 1
* angle 2
* pins 1
* pins 2
* total pins
## Usage
1. Place your Excel file in the project directory
2. Run the Streamlit app:
```bash
streamlit run app.py
```
3. Enter your OpenAI API key in the sidebar
4. Start chatting with the bowling advisor!
## How It Works
1. The application reads bowling data from the local Excel file
2. Statistical analysis is performed on speeds and angles
3. LangChain processes the data and user queries
4. The chatbot provides personalized advice based on the analysis
## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
## License
This project is licensed under the MIT License - see the LICENSE file for details