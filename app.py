import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import openai
import os
import base64
from PIL import Image
import io

# ------------------ Config ------------------ #
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path)
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="DIY Advisor Bot", layout="wide")

# ------------------ Image Processing Helper ------------------ #
def compress_image(image_bytes, max_size=(800, 800), quality=85):
    """Compress image to reduce token usage"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        
        # Resize if too large
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Save as JPEG with compression
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=quality, optimize=True)
        return output.getvalue()
    except Exception as e:
        print(f"Error compressing image: {e}")
        return image_bytes

# ------------------ GPT‑4o Helper Functions ------------------ #

def generate_project_plan(image_b64: str, tools: str, experience: str, question: str) -> str:
    prompt = f"""
DIY project assistant. User: {tools} tools, {experience} level. Goal: {question}

Create:
1. Required tools
2. Step-by-step instructions
3. Key warnings
4. YouTube search suggestion

Be concise and practical.
"""
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ] + (
                        [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_b64}}] if image_b64 else []
                    ),
                }
            ],
            max_tokens=600,  # Further reduced
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating plan: {str(e)}"


def answer_quick_start(image_b64: str, question: str) -> str:
    prompt = f"""
You are a DIY quick‑help assistant. Look at the image and answer the question concisely.
Question: {question}
"""
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ] + (
                        [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_b64}}] if image_b64 else []
                    ),
                }
            ],
            max_tokens=400,  # Reduced from 600
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting answer: {str(e)}"


def create_conversation_summary(chat_history: list, max_messages_to_keep: int = 8) -> tuple:
    """
    Summarize old conversation while keeping recent messages intact.
    Returns (summary_text, recent_messages)
    """
    if len(chat_history) <= max_messages_to_keep:
        return None, chat_history
    
    # Split into old and recent messages
    old_messages = chat_history[:-max_messages_to_keep]
    recent_messages = chat_history[-max_messages_to_keep:]
    
    # Create summary of old messages (text only)
    old_text_messages = [msg for msg in old_messages if msg["role"] != "user_image"]
    
    if not old_text_messages:
        return None, recent_messages
    
    # Create a simple summary
    summary_parts = []
    for msg in old_text_messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        summary_parts.append(f"{role}: {content}")
    
    summary = "Previous conversation summary:\n" + "\n".join(summary_parts[-6:])  # Keep last 6 old messages in summary
    return summary, recent_messages


def get_image_references(chat_history: list) -> dict:
    """Extract all images from chat history with descriptions"""
    images = {}
    image_count = 0
    
    for i, msg in enumerate(chat_history):
        if msg["role"] == "user_image":
            image_count += 1
            images[f"image_{image_count}"] = {
                "base64": msg["content"],
                "position": i,
                "description": f"Image {image_count} uploaded during conversation"
            }
    
    return images


def answer_quick_followup(chat_history: list, image_b64: str, question: str, new_image_b64: str = None) -> str:
    # Get conversation summary and recent messages
    summary, recent_messages = create_conversation_summary(chat_history, max_messages_to_keep=6)
    
    # Get all image references
    image_refs = get_image_references(chat_history)
    
    # Build system message with context
    system_content = "You are a DIY assistant. Keep responses concise and helpful."
    
    if summary:
        system_content += f"\n\n{summary}"
    
    if image_refs:
        img_descriptions = [f"- {key}: {info['description']}" for key, info in image_refs.items()]
        system_content += f"\n\nAvailable images in this conversation:\n" + "\n".join(img_descriptions)
        system_content += "\nYou can reference these previous images when relevant."

    system_msg = {
        "role": "system",
        "content": system_content,
    }

    # Build user message content (text + images)
    user_content = [{"type": "text", "text": question}]

    # Include original image if available and no new image
    if image_b64 and not new_image_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + image_b64}
        })

    # Include new image if provided
    if new_image_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + new_image_b64}
        })

    # Only include text messages from recent history (no images to save tokens)
    text_only_recent = [msg for msg in recent_messages if msg["role"] != "user_image"]
    
    messages = [system_msg] + text_only_recent + [
        {
            "role": "user",
            "content": user_content
        }
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=400,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting follow-up answer: {str(e)}"
    system_msg = {
        "role": "system",
        "content": "You are a DIY assistant. Keep responses concise and helpful.",
    }

    # Build user message content (text + images)
    user_content = [{"type": "text", "text": question}]

    # Only include the new image if provided, skip original image to save tokens
    if new_image_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + new_image_b64}
        })

    # Include more chat history - adjust this number based on your needs
    # Filter out image messages from history to save tokens
    text_only_history = [msg for msg in chat_history[-6:] if msg["role"] != "user_image"]
    
    messages = [system_msg] + text_only_history + [
        {
            "role": "user",
            "content": user_content
        }
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=300,  # Further reduced
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting follow-up answer: {str(e)}"


def answer_project_followup(chat_history: list, image_b64: str, tools: str, experience: str, question: str) -> str:
    """Enhanced project follow-up with better memory"""
    # Get conversation summary and recent messages
    summary, recent_messages = create_conversation_summary(chat_history, max_messages_to_keep=8)
    
    # Build enhanced system message
    system_content = f"You are a DIY project assistant. User has {tools} tools and {experience} experience. Keep responses concise but thorough."
    
    if summary:
        system_content += f"\n\n{summary}"

    system_msg = {
        "role": "system",
        "content": system_content,
    }

    # Build user message content - include original image for project context
    user_content = [{"type": "text", "text": question}]
    
    if image_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + image_b64}
        })

    # Include text-only recent messages
    text_only_recent = [msg for msg in recent_messages if msg["role"] != "user_image"]
    
    messages = [system_msg] + text_only_recent + [
        {
            "role": "user",
            "content": user_content
        }
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=400,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting project follow-up: {str(e)}"


# ------------------ Session State ------------------ #
if "project_data" not in st.session_state:
    st.session_state.project_data = None
if "project_history" not in st.session_state:
    st.session_state.project_history = []
if "quick_data" not in st.session_state:
    st.session_state.quick_data = None

# ------------------ UI ------------------ #
st.markdown("<h1 style='text-align: center;'>🔧 DIY Advisor Bot</h1>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align: center; font-size: 1.1rem;'>"
    "Smart help for real‑world repairs. Upload a photo, ask a question, get guidance."
    "</p>",
    unsafe_allow_html=True
)

# --- Center the native st.tabs() bar ---
st.markdown(
    """
    <style>
    /* Target the tab bar inside any stTabs container */
    div[data-testid="stTabs"] div[role="tablist"] {
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

tabs = st.tabs(["Create Project", "Quick Chat", "Saved Projects (Coming Soon)"])

# ---------- Tab 0: Create Project Plan ---------- #
with tabs[0]:
    st.markdown("<h2 style='text-align: center;'>Create a Project Plan</h2>", unsafe_allow_html=True)

    # Create 3 columns: empty - form - empty
    col1, col2, col3 = st.columns([1, 2, 1])  # Adjust middle width if needed

    with col2:
        tools = st.text_input("Tools you have:")
        experience = st.selectbox("Experience level:", ["Beginner", "Intermediate", "Advanced"])
        proj_img_file = st.file_uploader("Upload project photo", type=["jpg", "jpeg", "png"], key="proj_img")
        goal = st.text_area("What are you trying to do?")

    st.markdown("")  # Spacer

    # Center the button below
    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        if st.button("Generate Plan") and goal:
            img_bytes = proj_img_file.read() if proj_img_file else None
            
            # Compress image to reduce token usage
            if img_bytes:
                img_bytes = compress_image(img_bytes)
                
            img_b64 = base64.b64encode(img_bytes).decode() if img_bytes else None

            with st.spinner("Building your plan…"):
                plan_md = generate_project_plan(img_b64, tools, experience, goal)
            st.session_state.project_data = {
                "tools": tools,
                "experience": experience,
                "image_bytes": img_bytes,
                "image_b64": img_b64,
                "goal": goal,
                "plan_md": plan_md,
            }
            st.session_state.project_history = [
                {"role": "assistant", "content": plan_md},
            ]
            st.rerun()


# ---------- Tab 1: Quick Chat ---------- #
with tabs[1]:
    st.markdown("<h2 style='text-align: center;'>Quick DIY Chat</h2>", unsafe_allow_html=True)

    if st.session_state.quick_data is None:
        col1, col2, col3 = st.columns([1, 2, 1])  # Center column layout
        with col2:
            q_img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="quick_img")
            q_text = st.text_input("Ask your question:")
            if st.button("Get Answer") and q_text:
                q_bytes = q_img_file.read() if q_img_file else None
                
                # Compress image to reduce token usage
                if q_bytes:
                    q_bytes = compress_image(q_bytes)
                    
                q_b64 = base64.b64encode(q_bytes).decode() if q_bytes else None

                with st.spinner("Thinking…"):
                    first_reply = answer_quick_start(q_b64, q_text)

                st.session_state.quick_data = {
                    "image_bytes": q_bytes,
                    "image_b64": q_b64,
                    "chat_history": [
                        {"role": "user", "content": q_text},
                        {"role": "assistant", "content": first_reply},
                    ],
                }
                st.rerun()

    else:
        qd = st.session_state.quick_data

        if qd.get("image_bytes"):
            st.image(qd["image_bytes"], caption="Reference image", width=80)

        if qd.get("chat_history"):
            for msg in qd["chat_history"]:
                if msg["role"] == "user":
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: center;">
                            <div style="max-width: 700px; text-align: left;">
                                <strong>You:</strong> {msg['content']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif msg["role"] == "user_image":
                    import base64
                    img_bytes = base64.b64decode(msg["content"])
                    st.markdown(
                        "<div style='display:flex;justify-content:center;'>", unsafe_allow_html=True
                    )
                    st.image(img_bytes, width=300, caption="Additional image")
                    st.markdown("</div>", unsafe_allow_html=True)

                else:
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: center;">
                            <div style="max-width: 700px; text-align: left;">
                                {msg['content']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        # 🔥 Moved OUTSIDE the for-loop
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])  # Centered layout
        with col2:
            # 👇 Unique key per rerun to avoid Streamlit errors
            new_img_file = st.file_uploader(
                "Optional: Add another image", type=["jpg", "jpeg", "png"], key=f"follow_img_{len(qd['chat_history'])}"
            )
            new_q = st.text_input("Follow‑up question:", key=f"follow_text_{len(qd['chat_history'])}")

            col_a, col_b = st.columns([1, 1])  # Side-by-side buttons
            if col_a.button("Send") and new_q:
                new_img_bytes = new_img_file.read() if new_img_file else None
                
                # Compress new image to reduce token usage
                if new_img_bytes:
                    new_img_bytes = compress_image(new_img_bytes)
                    
                new_img_b64 = base64.b64encode(new_img_bytes).decode() if new_img_bytes else None

                qd["chat_history"].append({"role": "user", "content": new_q})
                if new_img_bytes:
                    qd["chat_history"].append({
                        "role": "user_image",
                        "content": new_img_b64,
                    })

                with st.spinner("Thinking…"):
                    response = answer_quick_followup(
                        qd["chat_history"],
                        qd.get("image_b64"),  # original image
                        new_q,
                        new_img_b64          # optional follow-up image
                    )

                qd["chat_history"].append({"role": "assistant", "content": response})
                st.rerun()

            if col_b.button("Start New Quick Chat"):
                st.session_state.quick_data = None
                st.rerun()


# ---------- Project Follow‑Up Chat ---------- #
if st.session_state.project_data:
    st.divider()
    pdata = st.session_state.project_data
    st.subheader("📋 Project Plan")
    st.markdown(pdata["plan_md"])
    if pdata.get("image_bytes"):
        st.image(pdata["image_bytes"], caption="Project photo", width=80)

    st.subheader("Project Q&A")
    for i, msg in enumerate(st.session_state.project_history):
        if i == 0 and msg["role"] == "assistant":
            continue  # skip plan reprint

        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(msg["content"])

    st.markdown("---")
    follow_q = st.text_input("Follow‑up question:")
    col1, col2 = st.columns([1,1])
    if col1.button("Send Follow-Up") and follow_q:
        st.session_state.project_history.append({"role": "user", "content": follow_q})
        with st.spinner("Thinking…"):
            reply = answer_project_followup(
                chat_history=st.session_state.project_history,
                image_b64=pdata["image_b64"],
                tools=pdata["tools"],
                experience=pdata["experience"],
                question=follow_q,
            )
        st.session_state.project_history.append({"role": "assistant", "content": reply})
        st.rerun()

    if col2.button("Clear Project"):
        st.session_state.project_data = None
        st.session_state.project_history = []
        st.rerun()