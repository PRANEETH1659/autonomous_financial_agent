import streamlit as st
from main import agent_executor

st.set_page_config(
    page_title="AI Financial Research Agent",
    page_icon="📈"
)

st.title("📈 Autonomous Financial Research Agent")
st.markdown(
    "Ask me about stock prices, company news, company comparisons, or financial research."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to research?"):

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Researching..."):

            try:
                response = agent_executor.invoke(
                    {"input": prompt}
                )

                answer = response.get(
                    "output",
                    "No response generated."
                )

            except Exception as e:
                answer = f"Error: {str(e)}"

            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )