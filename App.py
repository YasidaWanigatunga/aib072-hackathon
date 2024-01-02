import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from RagModel import getChain

# Set Streamlit app title
st.title(f"Product Recommendation Chatbot :panda_face:")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Need info? Drop your question here!"):
    try:
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get response from the model
        chain = getChain()
        response = chain(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response['result'])

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response['result']})

    except Exception as e:
        # Handle exceptions gracefully and display an error message
        with st.chat_message("assistant"):
            st.error(f"An error occurred: {e}")

        # Add the error message to the chat history
        st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})