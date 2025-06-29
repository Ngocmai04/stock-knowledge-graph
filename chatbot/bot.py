import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st
from chatbot.utils import write_message
from chatbot.agent import generate_response


# tag::setup[]
# Page Config to configure the title and icon used on the page
st.set_page_config("Ebert", page_icon=":movie_camera:")
# end::setup[]

# tag::session[]
# Set up Session State to store chat messages (cap nhat va luu tru trang thai)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm the GraphAcademy Chatbot!  How can I help you?"},
    ]
# end::session[]



# tag::submit[]
# Submit handler
def handle_submit(message):
    """
    Submit handler:

    You will modify this method to talk with an LLM and provide
    context using data from Neo4j.
    """

    # Handle the response
    with st.spinner('Thinking...'):
        # Call the agent
        response = generate_response(message)
        write_message('assistant', response)
        
# end::submit[]

# tag::chat[]
# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    write_message('user', prompt)

    # Generate a response
    handle_submit(prompt)
# end::chat[]
