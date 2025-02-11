import streamlit as st
from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO
from together import Together

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



# Main app
st.title("🎨 AI Story Generator")
st.write("Generate a Story from an Image using AI!")
st.write("Note: the story is generated based on the image generated and no other input is passed to AI")

# Get user input
topic = st.text_input("What's your story about?", placeholder="Keanu Reeves riding a Dinosaurs in moon")
mood = st.text_input("Story Mood", value="funny", placeholder="funny, sad, happy, etc")
language = st.text_input("Story Language", value="English", placeholder="English, Tamil, French, etc")
size = st.selectbox("Story Size", ["Short", "Medium", "Long"])

# Generate button
if st.button("Generate", type="primary"):
    if topic:
        with st.spinner("Generating your Image..."):
            # Generate and display image
            image, image_url = generate_image(f"An image related to {topic}")
            if image:
                st.image(image, caption="Generated Image")

        with st.spinner("Creating your story..."):       
            # Generate and display story
            story = generate_story(image_url, mood, size, language)
        st.write_stream(story)
    else:
        st.warning("Please enter a topic first!")
