@@ -0,0 +1,38 @@
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from lang import get_response


st.set_page_config(page_title="Oceanova")
st.title("FloatChat")

# Initial message
if "messages" not in st.session_state:

    st.session_state.messages = [
        AIMessage(content="Hello, I am a bot. How can I help you?")
    ]


for message in st.session_state.messages:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)

prompt = st.chat_input("Say Something")

if prompt is not None and prompt != "":
    if not isinstance(st.session_state.messages[-1], HumanMessage):
        st.session_state.messages.append(HumanMessage(content=prompt))
        message = st.chat_message("user")
        message.write(f"{prompt}")

    if not isinstance(st.session_state.messages[-1], AIMessage):
        with st.chat_message("assistant"):
            print(prompt, type(st.session_state.messages))
            response = st.write_stream(get_response(prompt))
        st.session_state.messages.append(AIMessage(content=response))