from agent import create_app
from langchain_core.messages import HumanMessage

import streamlit as st
import asyncio

from langchain_core.messages import HumanMessage
from agent import create_app

app = create_app()

ai_avatar = "imgs/AI_Avatar.png"

#Deploy app on streamlit
@st.cache_resource
def run_async_task(async_func, *args):
    loop = None
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(async_func(*args))
    except:
        if loop is not None:
            loop.close()
        loop = asyncio.new_event_loop()
        loop.run_until_complete(async_func(*args))
    finally:
        if loop is not None:
            loop.close()

async def main():
    st.title("📚🌸 Lina Novel ")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        else:
            with st.chat_message("assistant", avatar=ai_avatar):
                st.markdown(msg.content)

    user_input = st.chat_input("Chat with me here")

    if user_input:
        user_msg = HumanMessage(content=user_input)
        st.session_state.chat_history.append(user_msg)

        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            result = app.invoke({"messages": st.session_state.chat_history})
            assistant_msg = result["messages"][-1]
            st.session_state.chat_history.append(assistant_msg)

            with st.chat_message("assistant", avatar=ai_avatar):
                st.markdown(assistant_msg.content)

        except Exception as e:
            error_message = str(e)
            if "trigger content" in error_message.lower():
                fallback_msg = "Sorry, something went wrong. Please try a different prompt."
                st.session_state.chat_history.append(HumanMessage(content=fallback_msg))
                with st.chat_message("assistant", avatar=ai_avatar):
                    st.markdown(fallback_msg)
            else:
                st.error(f"Unexpected error: {error_message}")


def run_agent():
    print("\n=== NOVEL READING IMOUTO ===")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ['break', 'quit']:
            break 
        print("\nOni-chan: ", user_input)
        messages = [HumanMessage(content=user_input)]

        result = app.invoke({"messages" : messages}, {"recursion_limit": 100})

        print("\nImouto: ",result['messages'][-1].content)

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main())
