import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Smart Interview System",
    page_icon="💼",
    layout="wide"
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .score-meter {
        margin: 1rem 0;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .result-box.qualified {
        background-color: #C8E6C9;
        color: #2E7D32;
    }
    .result-box.not-qualified {
        background-color: #FFCDD2;
        color: #C62828;
    }
</style>
""", unsafe_allow_html=True)

# Load data with embeddings
@st.cache_data
def load_data():
    try:
        df = pd.read_pickle('data_with_embeddings.pkl')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return empty DataFrame in case of error
        return pd.DataFrame(columns=['job_role', 'Question', 'Answer', 'embedding'])

# Load SentenceTransformer model
@st.cache_resource
def load_model():
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to calculate similarity between user answer and correct answer
def compare_user_answer(user_answer, correct_answer, model):
    try:
        user_embedding = model.encode(user_answer).reshape(1, -1)
        correct_embedding = model.encode(correct_answer).reshape(1, -1)
        score = cosine_similarity(user_embedding, correct_embedding)[0][0]
        return score
    except Exception as e:
        st.error(f"Error calculating similarity: {e}")
        return 0.0

# Initialize session state
if 'question_history' not in st.session_state:
    st.session_state.question_history = pd.DataFrame(columns=['job_role', 'question', 'user_answer', 'correct_answer', 'score', 'question_num'])
    st.session_state.question_counter = 1
    st.session_state.selected_questions = []
    st.session_state.interview_complete = False

if 'feedback_shown' not in st.session_state:
    st.session_state.feedback_shown = False

# Load data and model
df = load_data()
model = load_model()

# Display main title
st.markdown("<h1 class='main-header'>Smart Interview System</h1>", unsafe_allow_html=True)

# Split UI into columns
main_col1, main_col2 = st.columns([2, 1])

with main_col2:
    st.markdown("<h2 class='sub-header'>Settings</h2>", unsafe_allow_html=True)
    
    # Job role selection
    job_roles = df['job_role'].unique().tolist() if not df.empty else []
    
    if job_roles:
        if 'selected_job' not in st.session_state:
            st.session_state.selected_job = job_roles[0]
            
        selected_job = st.selectbox("Choose Job Role", job_roles)
        
        # Filter questions by selected job role
        filtered_df = df[df['job_role'] == selected_job].reset_index(drop=True)
        
        # Display number of available questions
        st.info(f"Available questions: {len(filtered_df)}")
        
        # Set similarity threshold
        similarity_threshold = 0.8  # Fixed at 80%
        
        # Handle job change
        if 'selected_job' in st.session_state and st.session_state.selected_job != selected_job:
            st.session_state.selected_job = selected_job
            st.session_state.question_counter = 1
            st.session_state.question_history = pd.DataFrame(columns=['job_role', 'question', 'user_answer', 'correct_answer', 'score', 'question_num'])
            st.session_state.selected_questions = []
            st.session_state.interview_complete = False
            st.session_state.feedback_shown = False
            
            # Select 5 random questions
            if len(filtered_df) >= 5:
                st.session_state.selected_questions = filtered_df.sample(5).to_dict('records')
            else:
                st.session_state.selected_questions = filtered_df.to_dict('records')
                
    else:
        st.error("No job roles available. Please check the data file.")

with main_col1:
    if job_roles and not st.session_state.interview_complete:
        # Initialize questions if not already done
        if len(st.session_state.selected_questions) == 0 and len(filtered_df) > 0:
            if len(filtered_df) >= 5:
                st.session_state.selected_questions = filtered_df.sample(5).to_dict('records')
            else:
                st.session_state.selected_questions = filtered_df.to_dict('records')
        
        if st.session_state.question_counter <= len(st.session_state.selected_questions):
            # Interview area
            st.markdown("<h2 class='sub-header'>Interview</h2>", unsafe_allow_html=True)
            
            # Display current job role
            st.markdown(f"<div class='info-box'><b>Current Job Role:</b> {selected_job}</div>", unsafe_allow_html=True)
            
            # Get current question
            current_q = st.session_state.selected_questions[st.session_state.question_counter - 1]
            
            # Display current question
            st.markdown(f"<h3>Question {st.session_state.question_counter} of {len(st.session_state.selected_questions)}:</h3>", unsafe_allow_html=True)
            st.markdown(f"<div class='info-box'>{current_q['Question']}</div>", unsafe_allow_html=True)
            
            # User input for answer
            user_input = st.text_area("Your Answer:", height=150)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                submit_button = st.button("Submit Answer", type="primary", use_container_width=True)
            
            with col2:
                next_button = st.button("Next Question", use_container_width=True)
            
            # Handle answer submission
            if submit_button:
                if user_input.strip() == "":
                    st.warning("Please enter your answer!")
                else:
                    score = compare_user_answer(user_input, current_q['Answer'], model)
                    
                    # Display score meter
                    st.markdown("<div class='score-meter'>", unsafe_allow_html=True)
                    st.markdown(f"**Similarity Score:** {score:.2f}")
                    st.progress(min(score, 1.0))
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Save answer in history
                    new_entry = pd.DataFrame({
                        'job_role': [selected_job],
                        'question': [current_q['Question']],
                        'user_answer': [user_input],
                        'correct_answer': [current_q['Answer']],
                        'score': [score],
                        'question_num': [st.session_state.question_counter]
                    })
                    
                    st.session_state.question_history = pd.concat([st.session_state.question_history, new_entry], ignore_index=True)
                    
                    # Display evaluation
                    if score >= similarity_threshold:
                        st.success(f"✅ Good answer! Similarity score {score:.2f} exceeds the required threshold ({similarity_threshold}).")
                    else:
                        st.error(f"❌ Answer needs improvement. Similarity score {score:.2f} is below the required threshold ({similarity_threshold}).")
                    
                    # Show model answer
                    with st.expander("View Model Answer"):
                        st.write(current_q['Answer'])
                    
                    st.session_state.feedback_shown = True
            
            # Handle next question request
            if next_button:
                if not st.session_state.feedback_shown:
                    st.warning("You haven't submitted an answer yet. Moving to next question anyway.")
                
                # Go to next question or complete interview
                if st.session_state.question_counter < len(st.session_state.selected_questions):
                    st.session_state.question_counter += 1
                    st.session_state.feedback_shown = False
                    st.experimental_rerun()
                else:
                    # Interview complete
                    st.session_state.interview_complete = True
                    st.experimental_rerun()
        
        # Show progress
        progress = st.session_state.question_counter / len(st.session_state.selected_questions)
        st.progress(progress)
        
    elif st.session_state.interview_complete:
        # Display final result
        st.markdown("<h2 class='sub-header'>Interview Results</h2>", unsafe_allow_html=True)
        
        # Calculate average score
        avg_score = st.session_state.question_history['score'].mean()
        
        # Display score chart
        fig = px.line(
            st.session_state.question_history, 
            x="question_num", 
            y="score", 
            title="Score Progression",
            labels={"question_num": "Question Number", "score": "Similarity Score"}
        )
        fig.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="Required Threshold")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display final result
        if avg_score >= 0.8:
            st.markdown("<div class='result-box qualified'>QUALIFIED</div>", unsafe_allow_html=True)
            st.markdown(f"Congratulations! Your average similarity score is {avg_score:.2f}, which meets our requirements.")
        else:
            st.markdown("<div class='result-box not-qualified'>NOT QUALIFIED</div>", unsafe_allow_html=True)
            st.markdown(f"Your average similarity score is {avg_score:.2f}, which is below our threshold of 0.8.")
        
        # Display individual question scores
        st.subheader("Question Scores")
        for idx, row in st.session_state.question_history.iterrows():
            with st.expander(f"Question {row['question_num']}: Score {row['score']:.2f}"):
                st.write(f"**Question:** {row['question']}")
                st.write(f"**Your Answer:** {row['user_answer']}")
                st.write(f"**Model Answer:** {row['correct_answer']}")
                
        # Restart button
        if st.button("Start New Interview", type="primary"):
            st.session_state.question_counter = 1
            st.session_state.question_history = pd.DataFrame(columns=['job_role', 'question', 'user_answer', 'correct_answer', 'score', 'question_num'])
            st.session_state.selected_questions = []
            st.session_state.interview_complete = False
            st.session_state.feedback_shown = False
            
            # Select 5 random questions
            if len(filtered_df) >= 5:
                st.session_state.selected_questions = filtered_df.sample(5).to_dict('records')
            else:
                st.session_state.selected_questions = filtered_df.to_dict('records')
                
            st.experimental_rerun()
            
    else:
        st.warning("Please make sure the data is loaded correctly")

# Add footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>© 2025 Smart Interview System - All Rights Reserved</p>", unsafe_allow_html=True)
