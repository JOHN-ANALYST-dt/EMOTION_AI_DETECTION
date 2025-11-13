# interventions.py

import streamlit as st
import pandas as pd

# --- INTERVENTION DATA ---
EMOTION_INTERVENTIONS = {
    'anger': {
        'title': "🔥 Feeling the Heat? Constructive Ways to Manage Anger",
        'color': "#FF5733", 
        'Mental': [
            "🧠 **The 6-Second Pause:** Wait before reacting. Use 'I' statements to express needs, not blame.",
            "🧠 **Reframing:** Identify the thought that triggered the anger and look for alternative, less hostile interpretations."
        ],
        'Physical': [
            "💪 **Release Tension:** Engage in vigorous physical activity, like a quick walk or clenching/releasing your fists.",
            "🧘 **Breathe:** Use slow, deep, diaphragmatic breathing to regulate your nervous system."
        ],
        'Spiritual': [
            "✨ **Practicing Acceptance:** Reflect on the limits of your control. Let go of the need for others to meet an impossible standard."
        ],
        'Crisis': None
    },
    'sadness': {
        'title': "💧 Acknowledging the Pain: Gentle Steps Through Sadness",
        'color': "#0077B6", 
        'Mental': [
            "🧠 **Non-Judgmental Presence:** Allow the sadness to exist without trying to fix it immediately. Remind yourself this feeling is temporary.",
            "🧠 **Identify the Need:** Ask yourself, 'What am I truly missing or needing right now?'"
        ],
        'Physical': [
            "💪 **Comfort & Care:** Seek warmth (blanket, tea). Ensure adequate hydration and rest, as your body is working hard to process the emotion."
        ],
        'Spiritual': [
            "✨ **Seeking Connection:** Reach out to a trusted friend or community member. Look for small things to be grateful for, acknowledging the goodness that still exists."
        ],
        'Crisis': "If you feel overwhelmed by sadness or despair, please reach out for immediate support."
    },
    'fear': {
        'title': "😨 Navigating Uncertainty: Grounding Against Fear and Anxiety",
        'color': "#7F00FF", 
        'Mental': [
            "🧠 **Fact-Checking:** Differentiate between the actual threat and the catastrophic story your mind is telling you.",
            "🧠 **Focus on NOW:** Bring your attention back to the present moment, away from future worries."
        ],
        'Physical': [
            "💪 **5-4-3-2-1 Grounding:** Name 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste.",
            "🧘 **Controlled Breathing:** Inhale for 4, hold for 4, exhale for 6."
        ],
        'Spiritual': [
            "✨ **Trust and Courage:** Connect with an internal sense of strength or faith that can guide you through uncertainty."
        ],
        'Crisis': "If fear is causing panic or paralyzing distress, take a pause and call someone you trust."
    },
    'disgust': {
        'title': "🤢 Stepping Back: Addressing Disgust and Boundaries",
        'color': "#4C9F38", 
        'Mental': ["🧠 **Boundary Check:** Use this feeling to clarify your personal values and decide if you need to create distance from the source."],
        'Physical': ["💪 **Change Environment:** Step away, open a window, or engage a pleasant sensory input to reset your senses."],
        'Spiritual': ["✨ **Value Clarity:** Reflect on what is ethically and morally acceptable for you. Use this to reinforce positive principles."],
        'Crisis': None
    },
    'joy': {
        'title': "😊 Celebrating the Good! Reinforce Positive Emotions",
        'color': "#FFD700", 
        'Mental': ["🧠 **Savoring:** Mentally re-run the positive experience, focusing on sensory details, to enhance memory consolidation."],
        'Physical': ["💪 **Expression:** Share your joy with positive, energized body language. Go do something fun with that energy!"],
        'Spiritual': ["✨ **Generosity:** Use your good mood to uplift others or practice gratitude for the source of your happiness."],
        'Crisis': None
    },
    'surprise': {
        'title': "😮 The Unexpected: Pausing to Assess Surprise",
        'color': "#87CEEB", 
        'Mental': ["🧠 **Assess & Orient:** Ask yourself: 'Is this surprise positive or negative?' Let that guide your next cognitive step."],
        'Physical': ["💪 **Stop, Look, Listen:** Freeze your movement for one second to fully take in the unexpected sensory data."],
        'Spiritual': ["✨ **Openness:** View the unexpected event as a potential new path or unique moment of learning."],
        'Crisis': None
    },
}

def display_interventions(prediction_results):
    """
    Displays intervention messages based on the highest predicted negative emotion.
    """
    
    # Define negative emotions for targeting interventions
    NEGATIVE_EMOTIONS = ['anger', 'sadness', 'fear', 'disgust']
    
    # Ensure prediction_results is a DataFrame and not empty
    if not isinstance(prediction_results, pd.DataFrame) or prediction_results.empty:
        st.info("No prediction results available for intervention analysis.")
        return

    # Filter for emotions predicted (Predicted=1) or those with high confidence (> 50%)
    target_results = prediction_results[
        (prediction_results['Predicted'] == 1) | (prediction_results['Confidence (%)'] > 50)
    ]
    
    # Find the top predicted negative emotion
    top_negative_emotion = None
    for emotion in NEGATIVE_EMOTIONS:
        if emotion in target_results['Emotion'].values:
            target_row = target_results[target_results['Emotion'] == emotion].iloc[0]
            if target_row['Confidence (%)'] > 50:
                 top_negative_emotion = emotion
                 break
    
    # If a negative emotion is found, display the structured advice
    if top_negative_emotion and top_negative_emotion in EMOTION_INTERVENTIONS:
        intervention_data = EMOTION_INTERVENTIONS.get(top_negative_emotion)
        
        st.markdown(f"## 💡 Personalized Guidance: {top_negative_emotion.title()} Detected")
        st.markdown(f'<div style="border-left: 5px solid {intervention_data["color"]}; padding: 10px; background-color: #f0f2f6; border-radius: 5px;">'
                    f'<h4>{intervention_data["title"]}</h4></div>', 
                    unsafe_allow_html=True)
        
        
        # Display Crisis Warning (if applicable)
        if intervention_data.get('Crisis'):
             st.error(f"⚠️ **IMMEDIATE ATTENTION:** {intervention_data['Crisis']}")

        # Use expanders for clean dropdowns for the advice domains
        st.markdown("##### Here are some helpful strategies to manage this emotion:")

        advice_domains = ['Mental', 'Physical', 'Spiritual']
        for domain in advice_domains:
            advice_list = intervention_data.get(domain, [])
            if advice_list:
                with st.expander(f"**{domain} Advice**"):
                    for item in advice_list:
                        st.markdown(f"* {item}")

        # Final Disclaimer
        st.markdown("""
        ---
        **Disclaimer:** This guidance is AI-generated and for informational support only. It is **not** a substitute for professional counseling or medical advice.
        """)

    # If the only predicted emotion is Joy or Surprise (or no negative is strong)
    elif 'joy' in target_results['Emotion'].values:
        st.success("🎉 **Analysis suggests a positive emotional state (Joy/Happiness)!** Keep up the good work and share your positive energy.")

    else:
        st.info("No strong negative or positive emotion was detected to prompt specific intervention.") 