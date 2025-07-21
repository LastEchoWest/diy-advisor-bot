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

st.set_page_config(page_title="Workbench AI", layout="wide")

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

# ------------------ PDF Generation Helper ------------------ #
def create_printable_html(project_data, project_name):
    """Generate print-friendly HTML that browsers can save as PDF"""
    
    # Format the plan content for HTML
    plan_content = project_data.get('plan_md', '').replace('\n', '<br>')
    
    # Handle project image
    image_html = ""
    if project_data.get('image_bytes'):
        img_b64 = base64.b64encode(project_data['image_bytes']).decode()
        image_html = f'''
        <div class="project-image">
            <h2>📷 Project Photo</h2>
            <img src="data:image/jpeg;base64,{img_b64}" alt="Project Photo" style="max-width: 400px; max-height: 300px;">
        </div>
        '''
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{project_name} - DIY Project Plan</title>
        <style>
            @media print {{
                body {{ margin: 0.5in; }}
                .no-print {{ display: none !important; }}
                .page-break {{ page-break-before: always; }}
            }}
            
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 20px; 
                line-height: 1.6; 
                color: #333;
            }}
            
            h1 {{ 
                color: #2c3e50; 
                border-bottom: 3px solid #3498db; 
                padding-bottom: 10px;
                text-align: center;
            }}
            
            h2 {{ 
                color: #34495e; 
                border-bottom: 1px solid #bdc3c7;
                padding-bottom: 5px;
                margin-top: 25px;
            }}
            
            .project-header {{
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 25px;
                border-left: 5px solid #3498db;
            }}
            
            .project-details {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
                margin-bottom: 20px;
            }}
            
            .detail-item {{
                background: white;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #dee2e6;
            }}
            
            .detail-label {{
                font-weight: bold;
                color: #495057;
                margin-bottom: 5px;
            }}
            
            .plan-content {{
                background: white;
                padding: 25px;
                border-radius: 10px;
                border: 1px solid #dee2e6;
                margin-top: 25px;
            }}
            
            .project-image {{
                text-align: center;
                margin: 25px 0;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
            }}
            
            .print-button {{
                background: #3498db;
                color: white;
                border: none;
                padding: 15px 30px;
                font-size: 16px;
                border-radius: 8px;
                cursor: pointer;
                margin: 20px 0;
                display: block;
                margin-left: auto;
                margin-right: auto;
            }}
            
            .print-button:hover {{
                background: #2980b9;
            }}
            
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #dee2e6;
                color: #6c757d;
                font-size: 14px;
            }}
            
            @media screen {{
                body {{ max-width: 800px; margin: 0 auto; }}
            }}
        </style>
    </head>
    <body>
        <div class="no-print">
            <button class="print-button" onclick="window.print()">🖨️ Print / Save as PDF</button>
        </div>
        
        <h1>🔧 {project_name}</h1>
        
        <div class="project-header">
            <h2>📋 Project Overview</h2>
            <p><strong>Goal:</strong> {project_data.get('goal', 'Not specified')}</p>
            
            <div class="project-details">
                <div class="detail-item">
                    <div class="detail-label">🛠️ Tools Available</div>
                    <div>{project_data.get('tools', 'Not specified')}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">📊 Experience Level</div>
                    <div>{project_data.get('experience', 'Not specified')}</div>
                </div>
            </div>
        </div>
        
        {image_html}
        
        <div class="plan-content">
            <h2>📝 Project Plan</h2>
            <div>{plan_content}</div>
        </div>
        
        <div class="footer">
            <p>Generated by Workbench AI• {project_name}</p>
        </div>
        
        <div class="no-print">
            <button class="print-button" onclick="window.print()">🖨️ Print / Save as PDF</button>
            <p style="text-align: center; color: #6c757d; margin-top: 20px;">
                <strong>How to save as PDF:</strong><br>
                • <strong>Windows:</strong> Press Ctrl+P, choose "Save as PDF"<br>
                • <strong>Mac:</strong> Press Cmd+P, choose "Save as PDF"<br>
                • <strong>Mobile:</strong> Use your browser's share button
            </p>
        </div>
    </body>
    </html>
    """
    return html_content

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
    if image_b64:
        prompt = f"""
You are a DIY quick‑help assistant. Look at the image and answer the question concisely.
Question: {question}
"""
    else:
        prompt = f"""
You are a DIY quick‑help assistant. Answer the following question concisely based on your knowledge.
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
    
    # Build system message with context - make images optional
    system_content = "You are a DIY assistant. Keep responses concise and helpful. Images may or may not be provided - answer based on the available information."
    
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


def answer_project_followup_with_image(chat_history: list, original_image_b64: str, tools: str, experience: str, question: str, new_image_b64: str = None) -> str:
    """Enhanced project follow-up with support for new images"""
    # Get conversation summary and recent messages
    summary, recent_messages = create_conversation_summary(chat_history, max_messages_to_keep=8)
    
    # Build enhanced system message - make images optional
    system_content = f"You are a DIY project assistant. User has {tools} tools and {experience} experience. Keep responses concise but thorough. Images may or may not be provided - answer based on available information."
    
    if summary:
        system_content += f"\n\n{summary}"

    system_msg = {
        "role": "system",
        "content": system_content,
    }

    # Build user message content - prioritize new image if provided, otherwise use original
    user_content = [{"type": "text", "text": question}]
    
    if new_image_b64:
        # Use the new image if provided
        user_content.append({
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + new_image_b64}
        })
    elif original_image_b64:
        # Fall back to original project image
        user_content.append({
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + original_image_b64}
        })

    # Include text-only recent messages (filter out image messages to save tokens)
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


# ------------------ Helper Functions ------------------ #
def create_tabs():
    """Create tabs dynamically based on whether a project exists"""
    base_tabs = ["Quick Chat", "Create Project", "Saved Projects (Coming Soon)"]
    
    if st.session_state.project_data and st.session_state.project_name:
        # Insert project tab after "Quick Chat"
        project_tab_name = f"🔧 {st.session_state.project_name}"
        tabs_list = [base_tabs[0], base_tabs[1], project_tab_name] + base_tabs[2:]
        return st.tabs(tabs_list), True  # True = has project tab
    else:
        return st.tabs(base_tabs), False


def extract_project_name(goal_text, max_length=20):
    """Extract a short project name from the goal description"""
    # Simple extraction - take first few words and clean them up
    words = goal_text.strip().split()[:3]  # First 3 words
    name = " ".join(words)
    
    # Clean up and truncate
    name = name.replace("?", "").replace("!", "").strip()
    if len(name) > max_length:
        name = name[:max_length].strip() + "..."
    
    return name if name else "My Project"


# ------------------ Session State ------------------ #
if "project_data" not in st.session_state:
    st.session_state.project_data = None
if "project_history" not in st.session_state:
    st.session_state.project_history = []
if "quick_data" not in st.session_state:
    st.session_state.quick_data = None
# NEW: Track active project and tab state
if "active_project_tab" not in st.session_state:
    st.session_state.active_project_tab = False
if "project_name" not in st.session_state:
    st.session_state.project_name = None

# ------------------ UI ------------------ #
st.markdown("<h1 style='text-align: center;'>Workbench AI</h1>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align: center; font-size: 1.1rem;'>"
    "AI powered help for real‑world repairs. Upload a photo, ask a question, get guidance."
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

# Create tabs dynamically
if st.session_state.project_data and st.session_state.project_name:
    # With project
    tab_names = ["Quick Chat", "Create Project", f"🔧 {st.session_state.project_name}", "Saved Projects", "About"]
    tabs = st.tabs(tab_names)
    has_project_tab = True
else:
    # Without project  
    tab_names = ["Quick Chat", "Create Project", "Saved Projects", "About"]
    tabs = st.tabs(tab_names)
    has_project_tab = False

# Determine tab indices based on whether project tab exists
if has_project_tab:
    quick_tab_idx = 0
    create_tab_idx = 1
    project_tab_idx = 2
    saved_tab_idx = 3
    about_tab_idx = 4
else:
    quick_tab_idx = 0
    create_tab_idx = 1
    project_tab_idx = None
    saved_tab_idx = 2
    about_tab_idx = 3

# ---------- Tab: Create Project Plan ---------- #
with tabs[create_tab_idx]:
    st.markdown("<h2 style='text-align: center;'>Create a Project Plan</h2>", unsafe_allow_html=True)

    # If project exists, show different UI
    if st.session_state.project_data:
        pdata = st.session_state.project_data
        
        # Show generated plan
        st.subheader("📋 Your Project Plan")
        st.markdown(pdata["plan_md"])
        
        if pdata.get("image_bytes"):
            st.image(pdata["image_bytes"], caption="Project photo", width=150)
        
        st.markdown("---")
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("📄 Download PDF", use_container_width=True):
                # Generate printable HTML
                html_content = create_printable_html(pdata, st.session_state.project_name)
                
                # Offer HTML download that opens in browser
                st.download_button(
                    label="📥 Download Printable Plan",
                    data=html_content,
                    file_name=f"{st.session_state.project_name.replace(' ', '_')}_plan.html",
                    mime="text/html",
                    use_container_width=True
                )
                
                st.info("💡 **How to save as PDF:** Download the file, open it in your browser, then press Ctrl+P (or Cmd+P) and choose 'Save as PDF'")
        
        with col2:
            if st.button("🔧 Start Project", use_container_width=True, type="primary"):
                # Switch to project tab
                st.session_state.active_project_tab = True
                st.rerun()
        
        with col3:
            if st.button("🗑️ Clear Project", use_container_width=True):
                st.session_state.project_data = None
                st.session_state.project_history = []
                st.session_state.project_name = None
                st.session_state.active_project_tab = False
                st.rerun()
    
    else:
        # Original project creation form
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            project_name = st.text_input("Project Name:", placeholder="e.g., Fix Kitchen Sink")
            tools = st.text_input("Tools you have:")
            experience = st.selectbox("Experience level:", ["Beginner", "Intermediate", "Advanced"])
            proj_img_file = st.file_uploader("Upload project photo", type=["jpg", "jpeg", "png"], key="proj_img")
            goal = st.text_area("What are you trying to do?")

        st.markdown("")

        _, col_btn, _ = st.columns([1, 2, 1])
        with col_btn:
            if st.button("Generate Plan", use_container_width=True, type="primary") and goal:
                img_bytes = proj_img_file.read() if proj_img_file else None
                
                if img_bytes:
                    img_bytes = compress_image(img_bytes)
                    
                img_b64 = base64.b64encode(img_bytes).decode() if img_bytes else None

                with st.spinner("Building your plan…"):
                    plan_md = generate_project_plan(img_b64, tools, experience, goal)
                
                # Store project data and use custom name or extracted name
                final_project_name = project_name.strip() if project_name.strip() else extract_project_name(goal)
                
                st.session_state.project_data = {
                    "tools": tools,
                    "experience": experience,
                    "image_bytes": img_bytes,
                    "image_b64": img_b64,
                    "goal": goal,
                    "plan_md": plan_md,
                }
                st.session_state.project_name = final_project_name
                st.session_state.project_history = []  # Fresh history for project chat
                st.rerun()

# ---------- Dynamic Project Tab ---------- #
if has_project_tab and project_tab_idx is not None:
    with tabs[project_tab_idx]:
        pdata = st.session_state.project_data
        
        st.markdown(f"<h2 style='text-align: center;'>🔧 {st.session_state.project_name}</h2>", unsafe_allow_html=True)
        
        # Quick access to plan
        with st.expander("📋 View Project Plan", expanded=False):
            st.markdown(pdata["plan_md"])
            if pdata.get("image_bytes"):
                st.image(pdata["image_bytes"], caption="Project photo", width=150)
        
        st.markdown("---")
        
        # Project Chat Interface
        st.subheader("Project Assistant")
        
        # Show chat history
        if st.session_state.project_history:
            for msg in st.session_state.project_history:
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
                    # Display user-uploaded images in project chat
                    img_bytes = base64.b64decode(msg["content"])
                    st.markdown(
                        "<div style='display:flex;justify-content:center;'>", unsafe_allow_html=True
                    )
                    st.image(img_bytes, width=300, caption="Your uploaded image")
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
        else:
            st.info("Ask me anything about your project! I have your plan and context ready.")
        
        st.markdown("---")
        
        # Chat input
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            project_question = st.text_input("Ask about your project:", key="project_chat_input")
            
            # Button layout
            send_col, photo_col = st.columns([2, 1])
            
            project_img_file = None
            with photo_col:
                if st.button("📷 Add Photo", use_container_width=True, key="project_add_photo"):
                    # Show file uploader when button is clicked
                    st.session_state[f"show_uploader_{len(st.session_state.project_history)}"] = True
            
            # Show file uploader if button was clicked
            if st.session_state.get(f"show_uploader_{len(st.session_state.project_history)}", False):
                project_img_file = st.file_uploader(
                    "Select an image:", 
                    type=["jpg", "jpeg", "png"], 
                    key=f"project_img_{len(st.session_state.project_history)}"
                )
            
            with send_col:
                if st.button("Send", use_container_width=True, type="primary") and project_question:
                    # Handle image if uploaded
                    project_img_bytes = None
                    project_img_b64 = None
                    
                    if project_img_file:
                        project_img_bytes = project_img_file.read()
                        if project_img_bytes:
                            project_img_bytes = compress_image(project_img_bytes)
                            project_img_b64 = base64.b64encode(project_img_bytes).decode()
                    
                    # Add user message
                    st.session_state.project_history.append({"role": "user", "content": project_question})
                    
                    # Add image if provided
                    if project_img_b64:
                        st.session_state.project_history.append({
                            "role": "user_image",
                            "content": project_img_b64,
                        })
                    
                    with st.spinner("Thinking…"):
                        # Enhanced project followup to handle new images
                        reply = answer_project_followup_with_image(
                            chat_history=st.session_state.project_history,
                            original_image_b64=pdata["image_b64"],
                            tools=pdata["tools"],
                            experience=pdata["experience"],
                            question=project_question,
                            new_image_b64=project_img_b64
                        )
                    
                    st.session_state.project_history.append({"role": "assistant", "content": reply})
                    
                    # Clear the uploader state after sending
                    if f"show_uploader_{len(st.session_state.project_history)-2}" in st.session_state:
                        del st.session_state[f"show_uploader_{len(st.session_state.project_history)-2}"]
                    
                    st.rerun()

# ---------- Tab: Quick Chat ---------- #
with tabs[quick_tab_idx]:
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

        # Chat input moved outside the for-loop
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])  # Centered layout
        with col2:
            new_q = st.text_input("Follow‑up question:", key=f"follow_text_{len(qd['chat_history'])}")

            # Button layout - Send on left, Add Photo on right
            send_col, photo_col = st.columns([2, 1])
            
            new_img_file = None
            with photo_col:
                if st.button("📷 Add Photo", use_container_width=True, key=f"quick_add_photo_{len(qd['chat_history'])}"):
                    # Show file uploader when button is clicked
                    st.session_state[f"show_quick_uploader_{len(qd['chat_history'])}"] = True
            
            # Show file uploader if button was clicked
            if st.session_state.get(f"show_quick_uploader_{len(qd['chat_history'])}", False):
                new_img_file = st.file_uploader(
                    "Select an image:", 
                    type=["jpg", "jpeg", "png"], 
                    key=f"follow_img_{len(qd['chat_history'])}"
                )

            with send_col:
                if st.button("Send", use_container_width=True, type="primary", key=f"quick_send_{len(qd['chat_history'])}") and new_q:
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
                    
                    # Clear the uploader state after sending
                    if f"show_quick_uploader_{len(qd['chat_history'])-2}" in st.session_state:
                        del st.session_state[f"show_quick_uploader_{len(qd['chat_history'])-2}"]
                    
                    st.rerun()

            # Start new chat button below
            st.markdown("")  # Add some space
            if st.button("Start New Quick Chat", use_container_width=True, key="start_new_quick_chat"):
                st.session_state.quick_data = None
                st.rerun()

# ---------- Tab: Saved Projects ---------- #
if saved_tab_idx < len(tabs):
    with tabs[saved_tab_idx]:
        st.markdown("<h2 style='text-align: center;'>Saved Projects</h2>", unsafe_allow_html=True)
        st.info("Coming soon! Save and manage multiple projects.")

# ---------- Tab: About ---------- #
with tabs[about_tab_idx]:
    st.markdown("<h2 style='text-align: center;'>About Workbench AI</h2>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Introduction
    st.markdown("""
    ### 🤖 What is Workbench AI?
    
    Your smart assistant for DIY projects and repairs! Get instant help with tools, techniques, and step-by-step guidance for any home improvement project and beyond.
    """)
    
    st.markdown("---")
    
    # How to use each feature
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 💬 **Quick Chat**
        **Perfect for:** Quick questions and immediate help
        
        **How to use:**
        1. Upload a photo (optional)
        2. Ask your question
        3. Get instant advice
        4. Continue the conversation with follow-ups
        
        **Examples:**
        - "What's wrong with my sink?"
        - "How do I remove this screw?"
        - "What tool do I need for this job?"
        """)
        
        st.markdown("""
        ### 🔧 **Project Mode**
        **Perfect for:** Complex projects needing step-by-step plans
        
        **How to use:**
        1. Name your project
        2. List your available tools
        3. Select your experience level
        4. Upload a project photo
        5. Describe what you want to accomplish
        6. Get a detailed plan + download as PDF
        7. Use the project chat for guidance while working
        """)
    
    with col2:
        st.markdown("""
        ### 📸 **Using Photos Effectively**
        
        **Best practices:**
        - Take clear, well-lit photos
        - Show the problem area or workspace
        - Include surrounding context
        - Multiple angles can be helpful
        
        **In Quick Chat:** Add photos anytime during conversation
        
        **In Projects:** Upload main photo when creating, add progress photos in chat
        """)
        
        st.markdown("""
        ### 📋 **Getting the Best Results**
        
        **Be specific about:**
        - Your skill level (honest assessment helps!)
        - Available tools and materials
        - Safety concerns or limitations
        - Time constraints
        - Budget considerations
        
        **Ask follow-ups:**
        - "What if I don't have that tool?"
        - "Is there a simpler way?"
        - "What safety precautions should I take?"
        """)
    
    st.markdown("---")
    
    # Tips section
    st.markdown("""
    ### 💡 **Pro Tips**
    
    **🎯 Start with Quick Chat** if you're not sure what you need - you can always create a project later
    
    **🔄 Use both modes together** - Quick Chat for immediate questions, Project Mode for complex work
    
    **📱 Mobile friendly** - Take photos with your phone while working on the project
    
    **🖨️ Save your plans** - Download PDF plans to reference offline or share with others
    
    **🤔 Ask safety questions** - Always prioritize safety, especially with electrical, plumbing, or structural work
    """)
    
    st.markdown("---")
    
    # Safety disclaimer
    st.markdown("""
    ### ⚠️ **Important Safety Notice**
    
    This app provides general DIY guidance and suggestions. Always:
    - Follow local building codes and regulations
    - Consult professionals for electrical, plumbing, or structural work
    - Use proper safety equipment and precautions
    - Know your limits - some jobs require professional expertise
    - Turn off power/water when working on utilities
    
    **When in doubt, consult a professional!**
    """)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #6c757d; margin-top: 30px;'>
        <p>🔧 <strong>Workbench AI</strong> - Your smart companion for home improvement projects</p>
        <p><em>Built to help DIY enthusiasts tackle projects with confidence</em></p>
    </div>
    """, unsafe_allow_html=True)