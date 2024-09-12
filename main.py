import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pandasai import SmartDataframe, SmartDatalake
from pandasai import Agent
from langchain_groq.chat_models import ChatGroq

# Load environment variables (if you have .env file for API keys)
load_dotenv()

# Set up the ChatGroq LLM
llm = ChatGroq(
    model_name='llama3-70b-8192',
    api_key=st.secrets["apikey"]  # Replace with your actual API key
)
# Streamlit App
st.set_page_config(page_title="Data Science for Everyone", page_icon="📊")
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
# Title and logo
st.image("logoo.png", width=150)  # Replace "logo.png" with the path to your logo file
st.title('Data Science for Everyone')

# Step 1: Ask user to choose between SmartDataframe, SmartDatalake, or Agent
option = st.radio("Choose how you want to work with the data:",
                  ("SmartDataframe", "SmartDatalake", "Agent"))

# Step 2: Based on selection, show different options
if option == "SmartDataframe":
    # Directly show uploader for one CSV file
    uploaded_file = st.file_uploader("Upload a CSV file for SmartDataframe", type="csv")
    
    if uploaded_file is not None:
        # Load the data into a DataFrame
        data = pd.read_csv(uploaded_file)
        st.write("## Uploaded Data")
        st.dataframe(data)

        # Use SmartDataframe
        df = SmartDataframe(data, config={'llm': llm, "verbose": True})
        st.write("## You are working with SmartDataframe")

        # Ask a question or describe the dataset
        question = st.text_input("Ask a question about the data:")
        if question:
            response = df.chat(question)
            st.write("## Detailed Insights")
            st.write(response)

        if st.button("Describe the dataset"):
            description = df.chat("Describe the dataset and their rows and columns in a sentence.")
            st.write("## Dataset Description")
            st.write(description)

        if st.button("What Does the Dataset tell about?"):
            # Generate impact insights
            prompt = (
                "Analyze the dataset and explain how the insights can impact stakeholders. "
                "Discuss potential benefits, risks, and strategic implications based on the data."
            )
            ans = df.chat(prompt)
            st.write("## Wait! Here are more insights")
            st.write(ans)

elif option in ["SmartDatalake", "Agent"]:
    # Ask how many DataFrames the user wants to work with
    num_dataframes = st.number_input(f"How many DataFrames do you want to work with for {option}?", min_value=1, step=1)
    
    if num_dataframes:
        dataframes = []
        for i in range(int(num_dataframes)):
            # Show uploader for multiple CSV files
            uploaded_file = st.file_uploader(f"Upload CSV file {i+1}", type="csv")
            
            if uploaded_file is not None:
                df_data = pd.read_csv(uploaded_file)
                st.write(f"## Uploaded Data for DataFrame {i+1}")
                st.dataframe(df_data)
                dataframes.append(df_data)

        if len(dataframes) == num_dataframes:
            if option == "SmartDatalake":
                df = SmartDatalake(dataframes, config={"llm": llm, "verbose": True})
                st.write(f"## You are working with SmartDatalake using {num_dataframes} DataFrames")
            elif option == "Agent":
                df = Agent(dataframes, config={"llm": llm, "verbose": True})
                st.write(f"## You are working with an Agent using {num_dataframes} DataFrames")

            # Ask a question or describe the dataset
            question = st.text_input("Ask a question about the data:")
            if question:
                response = df.chat(question)
                st.write("## Detailed Insights")
                st.write(response)

            if st.button("Describe the dataset"):
                description = df.chat("Describe the dataset and their rows and columns in a sentence.")
                st.write("## Dataset Description")
                st.write(description)

            if st.button("What Does the Dataset tell about?"):
                # Generate impact insights
                prompt = (
                    "Analyze the dataset and explain how the insights can impact stakeholders. "
                    "Discuss potential benefits, risks, and strategic implications based on the data."
                )
                ans = df.chat(prompt)
                st.write("## Wait! Here are more insights")
                st.write(ans)

# Data visualization options
st.sidebar.title("Visualizations")
visualization_type = st.sidebar.selectbox("Select Visualization Type", ["None", "Univariate", "Bivariate"])

if uploaded_file is not None:
    if visualization_type == "Univariate":
        x_axis = st.sidebar.selectbox("Select Axis", data.columns)
        if st.sidebar.button("Bar Chart"):
            fig, ax = plt.subplots(figsize=(10, 6))
            data[x_axis].value_counts().plot(kind='bar', ax=ax)
            ax.set_xlabel(x_axis)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Bar Chart of {x_axis}')
            st.pyplot(fig)
        if st.sidebar.button("Line Chart"):
            fig, ax = plt.subplots(figsize=(10, 6))
            data[x_axis].plot(kind='line', ax=ax)
            ax.set_xlabel(x_axis)
            ax.set_ylabel('Values')
            ax.set_title(f'Line Chart of {x_axis}')
            st.pyplot(fig)
        if st.sidebar.button("Histogram"):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(data[x_axis], bins=20)
            ax.set_xlabel(x_axis)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Histogram of {x_axis}')
            st.pyplot(fig)

    elif visualization_type == "Bivariate":
        x_axis = st.sidebar.selectbox("X-axis", data.columns)
        y_axis = st.sidebar.selectbox("Y-axis", data.columns)
        if st.sidebar.button("Bar Chart"):
            fig, ax = plt.subplots(figsize=(12, 8))
            data[[x_axis, y_axis]].set_index(x_axis).plot(kind='bar', ax=ax)
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.set_title(f'Bar Chart of {y_axis} by {x_axis}')
            st.pyplot(fig)
        if st.sidebar.button("Line Chart"):
            fig, ax = plt.subplots(figsize=(12, 8))
            data[[x_axis, y_axis]].set_index(x_axis).plot(kind='line', ax=ax)
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.set_title(f'Line Chart of {y_axis} by {x_axis}')
            st.pyplot(fig)
        if st.sidebar.button("Scatter Plot"):
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.scatter(data[x_axis], data[y_axis])
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.set_title(f'Scatter Plot of {y_axis} vs {x_axis}')
            st.pyplot(fig)
else:
    st.write("Please upload a CSV file to get started.")
