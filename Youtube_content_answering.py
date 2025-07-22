import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import yt_dlp
from pytube import YouTube
from speechbrain.inference import EncoderDecoderASR
import streamlit as st

import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

class LlamaContextQA:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            use_auth_token=True
        )
        self.context = None
        self.history = []  # Stores list of {"question": ..., "answer": ..., "timestamp": ...}

    def set_context(self, context, append=False):
        """Set or append to context."""
        if append and self.context:
            self.context += "\n" + context.strip()
        else:
            self.context = context.strip()

    def clear_context(self):
        """Remove current context and history."""
        self.context = None
        self.history = []

    def ask_question(self, question, max_new_tokens=150):
        """Ask a question using current context and store history."""
        if not self.context:
            return "No context has been set."

        prompt = f"""[INST] <<SYS>>
            You are a helpful assistant.
            <</SYS>>
            Given the following content, answer the question concisely.

            Content:
            {self.context}

            Question:
            {question}
            [/INST]
            """

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean up output
        answer_start = response.find("Question:")
        if answer_start != -1:
            start= response.find("[/INST]")
            answer = response[(start + 7):].strip()
        else:
            answer = response.strip()

        # Save to history
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer
        })

        return answer

    def save_session(self, filepath="qa_session.json"):
        """Save context and Q&A history to a JSON file."""
        data = {
            "context": self.context,
            "history": self.history
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_session(self, filepath="qa_session.json"):
        """Load context and Q&A history from a JSON file."""
        try:
            print(filepath)
            with open(filepath, "r") as f:
                data = json.load(f)
            self.context = data.get("context", "")
            self.history = data.get("history", [])
        except Exception as e:
            raise RuntimeError(f"Failed to load session: {e}")

    def print_history(self):
        """Print Q&A history in a readable format."""
        if not self.history:
            print("No question-answer history found.")
            return

        print("Conversation History:")
        for i, entry in enumerate(self.history, 1):
            print(f"\n Question {i} [{entry['timestamp']}]:")
            print(f"   Question: {entry['question']}")
            print(f"   Answer: {entry['answer']}")


# DISPLAY Q&A HISTORY
def display_history(qa, height=0):
    """Display Q&A history in the sidebar in a scrollable, expandable format."""
    with st.sidebar:
        st.subheader("📜 Q&A History")

        if not qa.history:
            st.info("📭 No history yet.")
            return

        st.markdown(f"<div style='height: {height}px; overflow-y: auto;'>", unsafe_allow_html=True)

        for i, entry in enumerate(qa.history, 1):
            with st.expander(f"Q{i} — {entry['timestamp']}"):
                st.markdown(f"**Q:** {entry['question']}")
                st.markdown(f"**A:** {entry['answer']}")

        st.markdown("</div>", unsafe_allow_html=True)

# STREAMLIT UI
st.set_page_config(page_title="🦙 LLaMA Contextual Q&A", layout="centered")
st.title("🦙 LLaMA Contextual Q&A Assistant")

# Initialize session state
if "qa" not in st.session_state:
    st.session_state.qa = LlamaContextQA("meta-llama/Llama-2-7b-chat-hf")
qa = st.session_state.qa

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")

    # Save session with user-defined filename
    save_filename = st.text_input("Save As:", value="qa_session.json")
    if st.button("💾 Save Session"):
        if save_filename:
            try:
                qa.save_session(save_filename)
                st.success(f"Session saved to `{save_filename}`.")
            except Exception as e:
                st.error(f"Failed to save: {e}")
        else:
            st.warning("Please enter a filename.")

    # Load session from file uploader
    # uploaded_file = st.file_uploader("Load Session (.json)", type="json")
    uploaded_file = st.text_input("Load Session (.json)", value="qa_session.json")
    # if uploaded_file is not None:
    if st.button("💾 Load Session"):
        try:
            qa.load_session(uploaded_file)
            st.success("Session loaded from uploaded file.")
        except Exception as e:
            st.error(f"Failed to load: {e}")

    # Clear session
    if st.button("Clear Context & History"):
        qa.clear_context()
        st.warning("Context and history cleared.")

# Set or update context
with st.expander("Set or Update Context"):
    new_context = st.text_area("Enter context (you can update this at any time):", height=150)
    append = st.checkbox("Append to existing context", value=False)
    if st.button("Apply Context"):
        if new_context.strip():
            qa.set_context(new_context, append=append)
            st.success("Context updated.")
        else:
            st.warning("Please enter some context.")

# Ask a question
st.subheader("Ask a Question")
question = st.text_input("Type your question here:")
if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("💬 Thinking... generating answer..."):
            answer = qa.ask_question(question)
            qa.save_session()  # Auto-save after each question
        st.success("Answer:")
        st.write(answer)

# Always show scrollable history in sidebar
display_history(qa) 



