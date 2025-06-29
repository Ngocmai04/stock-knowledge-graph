



import streamlit as st
# from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
import uuid

def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def write_message(role, content, save=True):
   '''
   this is ahelper function that saves a messages to the session state and
   then writes a message to the UI
   '''
   
   #append the message to the session state
   if save:
       st.session_state.messages.append({"role": role, "content": content})
       
       
    #write the message to the UI
   with st.chat_message(role):
       st.markdown(content)
