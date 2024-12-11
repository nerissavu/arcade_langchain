import streamlit as st
import pandas as pd
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os

# Set page config
st.set_page_config(page_title="Bowling Advisor", layout="wide")

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.messages = []

# Function to analyze bowling patterns
def analyze_bowling_patterns(df):
    stats = {
        'speed_patterns': {},
        'angle_patterns': {}
    }
    
    # Analyze first throws for all three speed categories
    for speed in ['fast', 'medium', 'slow']:
        stats['speed_patterns'][speed] = df[df['speed 1'] == speed]['pins 1'].mean()
        if pd.isna(stats['speed_patterns'][speed]):
            stats['speed_patterns'][speed] = 0.0
    
    # Analyze angles
    for angle in ['straight', 'left', 'right']:
        stats['angle_patterns'][angle] = df[df['angle 1'] == angle]['pins 1'].mean()
        if pd.isna(stats['angle_patterns'][angle]):
            stats['angle_patterns'][angle] = 0.0
    
    # Add counts for more context
    stats['speed_counts'] = {}
    stats['angle_counts'] = {}
    
    for speed in ['fast', 'medium', 'slow']:
        stats['speed_counts'][speed] = len(df[df['speed 1'] == speed])
    
    for angle in ['straight', 'left', 'right']:
        stats['angle_counts'][angle] = len(df[df['angle 1'] == angle])
    
    return stats

# Main title
st.title("🎳 Bowling Advisor Bot")

# Sidebar for API key
with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
    if openai_api_key:
        os.environ['OPENAI_API_KEY'] = openai_api_key
        st.session_state.api_key_provided = True
    else:
        st.warning("Please enter your OpenAI API key to enable the chatbot.")
        st.session_state.api_key_provided = False

# Load data from local Excel file
try:
    bowling_data = pd.read_excel('arcadedata.xlsx')  # Make sure this file is in the same directory as your script
    
    # Show data in an expander
    with st.expander("View Bowling Data"):
        st.dataframe(bowling_data)

    # Analyze patterns
    stats = analyze_bowling_patterns(bowling_data)

    # Display stats in an expander
    with st.expander("View Bowling Statistics"):
        st.write("Speed Patterns (Average Pins):", stats['speed_patterns'])
        st.write("Speed Counts:", stats['speed_counts'])
        st.write("Angle Patterns (Average Pins):", stats['angle_patterns'])
        st.write("Angle Counts:", stats['angle_counts'])

    # Only show chat interface if API key is provided
    if st.session_state.api_key_provided:
        # Create the prompt template
        template = """You are an expert bowling advisor analyzing real bowling data.

        Current bowling statistics:
        Speed patterns (average pins for first throw):
        {speed_stats}
        Number of throws at each speed:
        {speed_counts}
        
        Angle patterns (average pins for first throw):
        {angle_stats}
        Number of throws at each angle:
        {angle_counts}

        Chat history:
        {chat_history}
        
        Human: {input}
        Assistant: Based on the data, here's my bowling advice:"""

        prompt = PromptTemplate(
            input_variables=["chat_history", "input", "speed_stats", "angle_stats", "speed_counts", "angle_counts"],
            template=template
        )

        # Initialize the language model and chain
        llm = OpenAI(temperature=0.7)
        chain = LLMChain(llm=llm, prompt=prompt)

        # Chat interface
        st.subheader("Chat with the Bowling Advisor")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask for bowling advice..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response with loading indicator
            with st.chat_message("assistant"):
                with st.spinner('Thinking...'):
                    # Format chat history
                    chat_history = "\n".join([
                        f"{'Human' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                        for msg in st.session_state.messages[:-1]  # Exclude the current message
                    ])
                    
                    # Get response from chain
                    response = chain.run({
                        "input": prompt,
                        "chat_history": chat_history,
                        "speed_stats": str(stats['speed_patterns']),
                        "angle_stats": str(stats['angle_patterns']),
                        "speed_counts": str(stats['speed_counts']),
                        "angle_counts": str(stats['angle_counts'])
                    })
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

        # Clear chat history button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

except FileNotFoundError:
    st.error("Error: Could not find 'arcadedata.xlsx'. Make sure the file is in the same directory as this script.")
except Exception as e:
    st.error(f"Error loading data: {str(e)}")