import streamlit as st
from main import agent_executor #Importing Working Agent


st.set_page_config(page_title="AI Financial Research Agent",page_icon="📈") 

st.title("📈 Autonomous Financial Research Agent")
st.markdown("Ask me about stock prices, company news, or deep financial research!")


# Initialize chat history in streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



if prompt := st.chat_input("What would you like to research?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    
     # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent_executor.invoke({"input": prompt})
            answer = response["output"]
            st.markdown(answer)
    
    
      # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": answer})