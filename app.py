import streamlit as st
import google.generativeai as genai
import os
import json
import re
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

# Load environment variables
load_dotenv()

# Define Pydantic model for output validation
class OutputSchema(BaseModel):
    framework_name: str
    reasoning: str
    criteria_values: dict

# Streamlit UI
st.title("Flow Wise")
st.markdown("### Find the Most Suitable AI Framework for Your Use Case")
st.markdown("Select a detailed description of your use case from the dropdown or enter your own:")

# Load configuration from config.json
try:
    with open('config.json') as config_file:
        config_data = json.load(config_file)
    use_case_list = config_data['use_case_list']
except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
    st.error(f"Error loading config.json: {e}")
    st.stop()

# Dropdown for pre-populated use cases
selected_use_case = st.selectbox(
    'Select a Use Case Description:',
    options=use_case_list,
    index=0
)

# Text input for custom use case
custom_use_case = st.text_input("Or enter your own use case description:")

# Determine the final use case description
use_case_description = custom_use_case if custom_use_case else selected_use_case

if st.button("Submit"):
    genai.configure(api_key=os.environ["API_KEY"])
    
    system_prompt = config_data['system_prompt']
    
    try:
        print("Sending request with prompt:", system_prompt)
        response = genai.GenerativeModel('gemini-1.5-flash-latest').generate_content({
            'parts': [{
                'text': system_prompt.replace('{use_case_description}', use_case_description)
            }]
        })
        
        # Clean the response
        cleaned_response = re.sub(r'^```json\s*|\s*```$', '', response.text, flags=re.MULTILINE)
        
        # Attempt to parse the JSON, handling potential errors
        response_data = None
        try:
            response_data = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            st.error(f"JSON Decode Error: {e}")
            
        if response_data:
            try:
                validated_response = OutputSchema(**response_data)
            except ValidationError as ve:
                print(f"Validation Error: {ve}")
                st.error(f"Validation Error: {ve}")
                # Attempt to extract data manually if validation fails
                validated_response = OutputSchema(
                    framework_name=response_data.get('framework_name', ''),
                    reasoning=response_data.get('reasoning', ''),
                    criteria_values=response_data.get('criteria_values', {})
                )
        else:
            validated_response = None
        
        if validated_response:
            # Display the response in a table format without the sequence number column
            st.markdown("### Recommended Framework")
            st.dataframe({
                "Framework": [validated_response.framework_name],
                "Reasoning": [validated_response.reasoning]
            })
            
            st.markdown("### Criteria Values")
            criteria_table = {
                "Criterion": list(validated_response.criteria_values.keys()),
                "Value": list(validated_response.criteria_values.values())
            }
            st.dataframe(criteria_table)
    except Exception as e:
        st.error(f"Error processing request: {str(e)}")