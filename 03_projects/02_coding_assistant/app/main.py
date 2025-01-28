import gradio as gd
import requests
import json
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage, AIMessage

BASE_URL = "http://localhost:11434/api/chat"

headers = {
    "Content-Type": "application/json"
}

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def serialize_messages(messages):
    serialized = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            role = "system"
        elif isinstance(msg, HumanMessage):
            role = "user"  # Changed from "human" to "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            role = "unknown"
        
        serialized.append({
            "role": role,
            "content": msg.content
        })
    return serialized

def extract_ai_response(response_data):
    """
    Extracts the AI response content from the API response.
    """
    # Possible fields that might contain the AI response
    possible_fields = ["response", "reply", "message", "data"]

    for field in possible_fields:
        if field in response_data:
            # If the field is 'message' and it's a dict, extract 'content'
            if field == "message" and isinstance(response_data[field], dict):
                return response_data[field].get("content", "No content in message.")
            elif field == "data" and isinstance(response_data[field], dict):
                # Adjust based on your API's structure
                return response_data[field].get("response_text", "No response_text in data.")
            else:
                return response_data[field]
    
    # If none of the expected fields are present
    return "No recognizable response field in the API output."

def chatbot(input_text):
    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Please respond to the user's input."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input_text}")  # Changed from "human" to "user"
    ])
    
    # Retrieve chat history from memory
    chat_history = memory.chat_memory.messages
    
    # Format messages with both input_text and chat_history
    messages = prompt.format_messages(input_text=input_text, chat_history=chat_history)
    
    # Add user message to memory
    memory.chat_memory.add_messages(messages=messages)
    
    # Serialize messages for JSON
    serialized_messages = serialize_messages(messages)
    
    # Prepare data for API request
    data = {
        "model": "abbasikoder",
        "messages": serialized_messages,
        "stream": False
    }
    
    # Debug: Print the payload being sent
    print("Request Payload:")
    print(json.dumps(data, indent=2))
    
    # Send POST request to the API
    try:
        response = requests.post(BASE_URL, headers=headers, data=json.dumps(data))  # Changed from 'data' to 'json'
        response.raise_for_status()  # Raise an exception for HTTP errors
    except requests.exceptions.RequestException as e:
        # If the response has content, print it for debugging
        if response is not None and response.text:
            print("Response Content:")
            print(response.text)
        return f"Request failed: {e}"
    
    # Parse the JSON response
    try:
        response_data = response.json()
    except json.JSONDecodeError:
        print("Failed to parse JSON response.")
        print("Response Text:")
        print(response.text)
        return "Failed to parse response from the API."
    
    # Debug: Print the entire response data
    print("API Response Data:")
    print(json.dumps(response_data, indent=2))
    
    # Add AI response to memory
    ai_response = extract_ai_response(response_data)
    memory.chat_memory.add_ai_message(ai_response)
    
    return ai_response

# Define the Gradio interface
interface = gd.Interface(
    fn=chatbot, 
    inputs=gd.Textbox(lines=7, placeholder="Enter Your Prompt Here..."),
    outputs="text", 
    title="Chatbot"
)

# Launch the interface
interface.launch()