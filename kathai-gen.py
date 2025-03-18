import streamlit as st
from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO
from together import Together
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import textwrap
import os

# Commands to install the required packages and Run:
# pip install -r requirements.txt 
# streamlit run kathai-gen.py

# Initialize OpenAI client with Together.ai base URL
client = OpenAI(
    api_key=st.secrets['TOGETHER_API_KEY'],
    base_url="https://api.together.xyz/v1"
)

together_client = Together()

# Generate an image using FLUX model
def generate_image(prompt: str):
    try:
        response = client.images.generate(
            model="black-forest-labs/FLUX.1-schnell-Free",
            prompt=prompt,
        )
        # Get image URL from response
        image_url = response.data[0].url
        
        # Load and return the image
        response = requests.get(image_url)
        return Image.open(BytesIO(response.content)), image_url
    except Exception as e:
        st.error(f"Failed to generate image: {str(e)}")
        return None


# A simple wrapper to get the stream content as together api is incompatible with streamlit
def stream_wrapper(stream):
    for chunk in stream:
       if len((chunk.choices)) != 0: # error is thrown as last chunk is empty
            yield chunk.choices[0].delta.content


# Generate story using Llama model
def generate_story(image_url: str, mood: str, size, language: str):
    try:
        story_template = "Create a {size} {mood} story based on the image generated in simple {language}."
        story_prompt = story_template.format(size=size, mood=mood, language=language)

        system_message = "You are a helpful multilingual Story teller and you will answer in {language}."
        system_prompt = system_message.format(language=language)

        st.markdown(f"**Story Prompt:** {story_prompt}")

        stream = together_client.chat.completions.create(
            # model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo", Alt model but not free
            model="meta-llama/Llama-Vision-Free",
            messages=[
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": story_prompt },
                        { "type": "image_url", "image_url": { "url": image_url }},
                    ],
                },
                                {
                    "role": "system",
                    "content": [
                        { "type": "text", "text": system_prompt }
                    ],
                }
            ],
            stream=True
        )
        return stream_wrapper(stream)

    except Exception as e:
        st.error(f"Failed to generate story: {str(e)}")
        return None


def create_pdf(image, story_text, topic):
    # Create a temporary PDF file
    pdf_path = "temp_story.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    
    # Add title
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 50, "AI Generated Story")
    
    # Add topic
    c.setFont("Helvetica", 16)
    c.drawString(50, height - 80, f"Topic: {topic}")
    
    # Add image
    if image:
        # Save image temporarily
        temp_img_path = "temp_image.png"
        image.save(temp_img_path)
        c.drawImage(temp_img_path, 50, height - 400, width=400, height=300)
        os.remove(temp_img_path)
    
    # Add story text
    c.setFont("Helvetica", 12)
    y_position = height - 450
    wrapped_text = textwrap.fill(story_text, width=80)
    for line in wrapped_text.split('\n'):
        if y_position < 50:  # Start new page if we run out of space
            c.showPage()
            y_position = height - 50
            c.setFont("Helvetica", 12)
        c.drawString(50, y_position, line)
        y_position -= 20
    
    c.save()
    return pdf_path


# Main app
st.title("🎨 AI Story Generator")
st.write("Generate a Story from an Image using AI!")
st.write("Note: the story is generated based on the image generated and no other input is passed to AI")

# Get user input
topic = st.text_input("What's your story about?", placeholder="Keanu Reeves riding a Dinosaurs in moon")
mood = st.text_input("Story Mood", value="funny", placeholder="funny, sad, happy, etc")
language = st.text_input("Story Language", value="English", placeholder="English, Tamil, French, etc")
size = st.selectbox("Story Size", ["Short", "Medium", "Long"])

# Create columns for buttons
col1, col2 = st.columns(2)

# Generate button
if col1.button("Generate", type="primary"):
    if topic:
        with st.spinner("Generating your Image..."):
            # Generate and display image
            image, image_url = generate_image(f"An image related to {topic}")
            if image:
                st.image(image, caption="Generated Image")

        with st.spinner("Creating your story..."):       
            # Generate and display story
            story_generator = generate_story(image_url, mood, size, language)
            story_text = ""
            # Create a placeholder for the story
            story_placeholder = st.empty()
            
            for chunk in story_generator:
                if chunk:
                    story_text += chunk
                    # Update the placeholder with the complete story so far
                    story_placeholder.markdown(story_text)
            
            # Store the generated content in session state
            st.session_state.generated_image = image
            st.session_state.generated_story = story_text
            st.session_state.story_generated = True
    else:
        st.warning("Please enter a topic first!")

# Show export button only after story generation
if 'story_generated' in st.session_state and st.session_state.story_generated:
    pdf_path = create_pdf(
        st.session_state.generated_image,
        st.session_state.generated_story,
        topic
    )
    
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    col2.download_button(
        # type="primary",
        label="Export PDF",
        data=pdf_bytes,
        file_name="ai_generated_story.pdf",
        mime="application/pdf"
    )
    
    # Clean up temporary file
    os.remove(pdf_path)
