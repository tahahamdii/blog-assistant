from langchain_huggingface import HuggingFaceEndpoint
from secret_api_keys import hugging_face_api_key
from langchain.prompts import PromptTemplate

import os
import streamlit as st

os.environ['HUGGINGFACEHUB_API_TOKEN'] = hugging_face_api_key

repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"


llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.6,
    token=hugging_face_api_key,
)


prompt_template_for_title_suggestion = PromptTemplate(
    input_variables=['topic'],
    template =
    '''
    I'm planning a blog post on topic : {topic}.
    The title is informative, or humorous, or persuasive. 
    The target audience is beginners, tech enthusiasts.  
    Suggest a list of ten creative and attention-grabbing titles for this blog post. 
    Don't give any explanation or overview to each title.
    '''
)

title_suggestion_chain = prompt_template_for_title_suggestion | llm # defining the title suggestion chain


prompt_template_for_title = PromptTemplate(
    input_variables=['title', 'keywords', 'blog_length'],
    template=
    '''Write a high-quality, informative, and plagiarism-free blog post on the topic: "{title}". 
    Target the content towards a beginner audience. 
    Use a conversational writing style and structure the content with an introduction, body paragraphs, and a conclusion. 
    Try to incorporate these keywords: {keywords}. 
    Aim for a content length of {blog_length} words. 
    Make the content engaging and capture the reader's attention.'''
)

title_chain = prompt_template_for_title | llm


st.title("blog ass")

st.subheader('Title Generation')
topic_expander = st.expander("Input the topic")

with topic_expander:
    topic_name = st.text_input("", key="topic_name")
    submit_topic = st.button('Submit topic')

if submit_topic: # Handle button click (submit_topic)
    title_selection_text = '' # Initialize an empty string to store title suggestions
    title_suggestion_str = title_suggestion_chain.invoke({topic_name}) # Generate titles using the title suggestion chain
    for sentence in title_suggestion_str.split('\n'):
        title_selection_text += (sentence.strip() + '\n') # Clean up each sentence and add it to the selection text
    st.text(title_selection_text) # Display the generated title suggestions


st.subheader('Blog Generation') # Display a subheader for the blog generation section
title_expander = st.expander("Input the title") # Create an expander for title input


with title_expander: # Create a content block within the title expander
    title_of_the_blog = st.text_input("", key="title_of_the_blog") # Get user input for the blog title
    num_of_words = st.slider('Number of Words', min_value=100, max_value=1000, step=50) # Slider for selecting the desired number of words


    if 'keywords' not in st.session_state: # Manage keyword list in session state
        st.session_state['keywords'] = []  # Initialize empty list on first run
    keyword_input = st.text_input("Enter a keyword:") # Input field for adding keywords
    keyword_button = st.button("Add Keyword") # Button to add keyword to the list
    if keyword_button: # Handle button click for adding keyword
        st.session_state['keywords'].append(keyword_input) # Add the keyword to the session state list
        st.session_state['keyword_input'] = "" # Clear the keyword input field after adding
        for keyword in st.session_state['keywords']:  # Display the current list of keywords
            # Inline styling for displaying keywords
            st.write(f"<div style='display: inline-block; background-color: lightgray; padding: 5px; margin: 5px;'>{keyword}</div>", unsafe_allow_html=True)

    # Button to submit the information for content generation
    submit_title = st.button('Submit Info')

if submit_title: # Handle button click for submitting information
    formatted_keywords = []
    for i in st.session_state['keywords']: # Process and format keywords
        if len(i) > 0:
            formatted_keywords.append(i.lstrip('0123456789 : ').strip('"').strip("'"))
    formatted_keywords = ', '.join(formatted_keywords)

    st.subheader(title_of_the_blog) # Display the blog title as a subheader
    st.write(title_chain.invoke({'title': title_of_the_blog, 'keywords': formatted_keywords, 'blog_length':num_of_words})) # Generate and display the blog content using the title chain
