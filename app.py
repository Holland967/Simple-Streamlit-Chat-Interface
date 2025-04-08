from typing import Tuple, Dict, List
from dotenv import load_dotenv
import streamlit as st
import requests
import json
import os
import io

# Load the .env content
load_dotenv()

@ st.cache_data
def get_basic_params() -> Tuple[str, str, str]:
    api_key: str = os.getenv("API_KEY")
    url: str = os.getenv("URL")
    model = "your-model-id"
    return api_key, url, model

# Initialize headers
@st.cache_data
def init_headers(api_key: str) -> Dict:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    return headers

# Rendering dialogue
@st.fragment
def display_conversation(session_state: List) -> None:
    for msg in session_state:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Chatbot interaction logic
@st.fragment
def chat_with_llm(
    query: str,
    session_state: List,
    inst: str | None,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    frequency_penalty: float,
    presence_penalty: float,
    url: str,
    headers: Dict
) -> None:
    # Cache the current user's message
    session_state.append({"role": "user", "content": query})
    # Display the current user's message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Organize the messages parameter
    messages = [{"role": "system", "content": inst}] if inst is not None else []
    messages += session_state

    # Prepare the model message in advance
    buffer = io.StringIO()
    with st.chat_message("assistant"):
        placeholder = st.empty()
    
    # Organize the payload parameter
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stream": True
    }

    # Get and handle response
    try:
        response = requests.request(
            method="POST",
            url=url,
            headers=headers,
            json=payload,
            stream=True
        )
        if response.status_code == 200:
            # Process the model response
            for chunk in response.iter_lines():
                if not chunk:
                    continue
                decoded_chunk: str = chunk.decode("utf-8")
                if decoded_chunk.startswith("data: [DONE]"):
                    break
                if decoded_chunk.startswith("data:"):
                    json_chunk = json.loads(decoded_chunk.split("data:")[1].strip())
                    if not json_chunk["choices"]:
                        continue
                    delta = json_chunk["choices"][0]["delta"]
                    if "content" in delta and delta["content"] is not None:
                        buffer.write(delta["content"])
                        placeholder.markdown(buffer.getvalue())
            
            # Cache the model's reply
            content = buffer.getvalue()
            session_state.append({"role": "assistant", "content": content})
            buffer.close()
            st.rerun()
        else:
            st.warning(f"**{response.status_code}**:\n\n{response.text}")
    except Exception as e:
        st.error(f"**Response Error**:\n\n{e}")

def main() -> None:
    api_key, url, model = get_basic_params()

    # Initialize headers
    headers: Dict = init_headers(api_key=api_key)

    # Just a simple title
    st.title("AI Chatbot Demo", anchor=False)

    # Initialize session_state
    if "content" not in st.session_state:
        st.session_state.content = []
    
    # Create a button to clear the conversation
    @st.fragment
    def clear_conversation() -> None:
        if st.button("Clear", "_clear", type="primary"):
            st.session_state.content = []
            st.rerun()
    
    # Sidebar
    with st.sidebar:
        clear_conversation()
        system_prompt: str = st.text_area("System Prompt", "", key="_inst")
        max_tokens: int = st.slider("Max Tokens", 1, 4096, 4096, 1, key="_tokens")
        temperature: float = st.slider("Temperature", 0.00, 2.00, 0.70, 0.01, key="_temp")
        top_p: float = st.slider("Top P", 0.01, 1.00, 0.95, 0.01, key="_topp")
        top_k: int = st.slider("Top K", 1, 100, 50, 1, key="_topk")
        frequency_penalty: float = st.slider("Frequency Penalty", -2.00, 2.00, 0.00, 0.01, key="_freq")
        presence_penalty: float = st.slider("Presence Penalty", -2.00, 2.00, 0.00, 0.01, key="_pres")

    
    display_conversation(st.session_state.content)

    # User message input box
    if query := st.chat_input("Say something...", key="_query"):
        if not system_prompt:
            chat_with_llm(
                query=query,
                session_state=st.session_state.content,
                inst=None,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                url=url,
                headers=headers
            )
        else:
            chat_with_llm(
                query=query,
                session_state=st.session_state.content,
                inst=system_prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                url=url,
                headers=headers
            )

if __name__ == "__main__":
    main()
