import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random
import time

# Page Configuration
st.set_page_config(
    page_title="Math Meme Repair 🔧",
    page_icon="🛠️",
    layout="centered"
)

# Title & Subtitle
st.title("🛠️ Math Meme Repair")
st.caption("Fixing broken math... one meme at a time!")

# Sidebar Controls
st.sidebar.header("🔧 App Settings")
temperature = st.sidebar.slider("Creativity (Temperature)", 0.1, 1.0, 0.7)

# Load model and tokenizer
@st.cache_resource
def load_model():
    with st.spinner("Loading Math Meme Repair GPT-2 Model..."):
        tokenizer = GPT2Tokenizer.from_pretrained("hassanhaseen/MathMemeRepairGPT2")
        model = GPT2LMHeadModel.from_pretrained("hassanhaseen/MathMemeRepairGPT2")
    return tokenizer, model

tokenizer, model = load_model()

# Fun Error Rating Generator
def generate_error_rating():
    sass_levels = [
        "90% sass, 10% patience",
        "70% confusion, 30% redemption",
        "80% teacher vibes, 20% sarcasm",
        "100% fix-it mode",
        "50% shock, 50% fix",
        "99% judgment, 1% encouragement"
    ]
    return random.choice(sass_levels)

# User Input
st.subheader("🔍 Enter Your Wrong Math Meme / Statement")
user_input = st.text_area("Paste a wrong math meme here... (Example: 8 ÷ 2(2+2) = 1)")

# Generate Fix Button
if st.button("🛠️ Repair the Meme!"):
    if user_input.strip() == "":
        st.warning("Please enter a math meme first!")
    else:
        with st.spinner("Fixing your math meme..."):
            # Give GPT-2 a prompt
            prompt = f"<|startoftext|>Incorrect: {user_input} Correct:"
            inputs = tokenizer(prompt, return_tensors="pt")

            outputs = model.generate(
                **inputs,
                max_length=100,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=temperature
            )

            # Decode the result
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the "Correct" part (after 'Correct:')
            if "Correct:" in result:
                correct_part = result.split("Correct:")[-1].strip()
            else:
                correct_part = "Oops! Something went wrong..."

        time.sleep(1)  # Pause for spinner effect

        # Display error rating and corrected answer
        st.success("✅ Math Meme Repaired!")
        st.markdown(f"**🧮 Error Rating:** {generate_error_rating()}")

        # Reveal the corrected explanation
        with st.expander("Click to reveal the correct explanation:"):
            st.info(correct_part)

# Footer with hover effect
st.markdown("---")
st.markdown(
    """
    <style>
        .footer {
            text-align: center;
            font-size: 14px;
            color: #888888;
        }
        .footer span {
            position: relative;
            cursor: pointer;
            color: #FF4B4B;
        }
        .footer span::after {
            content: "Hassan Haseen & Sameen Muzaffar";
            position: absolute;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            background-color: #333;
            color: #fff;
            padding: 5px 10px;
            border-radius: 8px;
            white-space: nowrap;
            opacity: 0;
            transition: opacity 0.3s;
            pointer-events: none;
            font-size: 12px;
        }
        .footer span:hover::after {
            opacity: 1;
        }
    </style>

    <div class='footer'>
        Created with ❤️ by <span>Team CodeRunners</span>
    </div>
    """,
    unsafe_allow_html=True
)
