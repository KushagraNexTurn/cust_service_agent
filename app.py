import os
import streamlit as st
from typing import List

from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.agent import ReActAgent

st.set_page_config(page_title="Customer Service ReAct Agent", page_icon="🤖", layout="wide")
st.title("🤖 Customer Service ReAct Agent")

def get_groq_api_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")

groq_api_key = get_groq_api_key()
if not groq_api_key:
    st.error("GROQ_API_KEY is not set. Add it to .streamlit/secrets.toml or set an environment variable.")
    st.stop()

with st.sidebar:
    st.header("Configuration")
    model_name = st.selectbox(
        "Groq Model",
        [
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "mixtral-8x7b-32768",
            "llama3-8b-8192",
            "llama3-70b-8192",
        ],
        index=0,
    )
    temperature = st.slider("LLM temperature", 0.0, 1.0, 0.1, 0.1)
    chunk_size = st.slider("Chunk size (chars)", min_value=256, max_value=2048, value=1024, step=128)
    support_pdf = st.text_input(
        "Customer Support PDF",
        value="Customer Service.pdf",
        help="Place this file next to app.py or provide a full path.",
    )
    init_clicked = st.button("Initialize Agent", type="primary")

def build_agent(groq_key: str, model_name: str, temperature: float, chunk_size: int, support_pdf_path: str) -> ReActAgent:
    Settings.llm = Groq(model=model_name, api_key=groq_key, temperature=temperature)
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    splitter = SentenceSplitter(chunk_size=chunk_size)
    support_docs = SimpleDirectoryReader(input_files=[support_pdf_path]).load_data()
    support_nodes = splitter.get_nodes_from_documents(support_docs)
    support_index = VectorStoreIndex(support_nodes)
    support_query_engine = support_index.as_query_engine()

    def get_order_items(order_id: int) -> List[str]:
        order_items = {
            1001: ["Laptop", "Mouse"],
            1002: ["Keyboard", "HDMI Cable"],
            1003: ["Laptop", "Keyboard"],
        }
        return order_items.get(order_id, [])

    def get_delivery_date(order_id: int) -> str:
        delivery_dates = {
            1001: "10-Jun",
            1002: "12-Jun",
            1003: "08-Jun",
        }
        return delivery_dates.get(order_id, "No delivery date found")

    def get_item_return_days(item: str) -> int:
        item_returns = {
            "Laptop": 30,
            "Mouse": 15,
            "Keyboard": 15,
            "HDMI Cable": 5,
        }
        return item_returns.get(item, 45)

    order_item_tool = FunctionTool.from_defaults(fn=get_order_items)
    delivery_date_tool = FunctionTool.from_defaults(fn=get_delivery_date)
    return_policy_tool = FunctionTool.from_defaults(fn=get_item_return_days)

    support_tool = QueryEngineTool.from_defaults(
        query_engine=support_query_engine,
        name="customer_support",
        description="Customer support policies and contact information. Use this for general support questions, policies, and contact details.",
    )

    agent = ReActAgent(
        tools=[order_item_tool, delivery_date_tool, return_policy_tool, support_tool],
        llm=Settings.llm,
        verbose=True,
        system_prompt=(
            "You are a helpful customer service agent. "
            "Use the available tools to help customers with their orders, deliveries, returns, and support questions. "
            "Always be polite and helpful in your responses."
        ),
    )
    return agent

if init_clicked:
    try:
        if not os.path.exists(support_pdf):
            st.warning(f"PDF not found at: {support_pdf}. Make sure the file exists.")
        st.session_state.agent = build_agent(groq_api_key, model_name, temperature, chunk_size, support_pdf)
        st.success("Agent initialized successfully.")
    except Exception as e:
        st.error(f"Initialization failed: {e}")

st.subheader("Ask a customer service question")
examples = [
    "What is the return policy for order number 1001?",
    "What is the delivery date for order number 1002?",
    "Provide customer support contact info.",
]
query = st.text_input("Your question", value=examples)

ex_cols = st.columns(len(examples))
for i, ex in enumerate(examples):
    if ex_cols[i].button(ex):
        query = ex
        st.experimental_rerun()

if st.button("Run"):
    if "agent" not in st.session_state:
        st.error("Agent is not initialized. Click Initialize Agent in the sidebar.")
    elif not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Agent thinking..."):
            try:
                handler = st.session_state.agent.run(query)
                for _ in handler.stream_events():
                    pass
                answer = handler.result()
                st.success("Done")
                st.markdown("Answer:")
                st.write(str(answer))
            except Exception as e:
                st.error(f"Query failed: {e}")
                st.info("If iteration-limit errors occur, retry with a more specific question or reduce temperature.")
