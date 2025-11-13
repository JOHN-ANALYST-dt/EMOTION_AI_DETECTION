# interventions.py

import streamlit as st
import pandas as pd

# --- INTERVENTION DATA ---
EMOTION_INTERVENTIONS = {
    'anger': {
        'title': "🔥 Feeling the Heat? Constructive Ways to Manage Anger",
        'color': "#FF5733", 
        'Mental': [
                "**Pause Before You React:** When anger rises, give yourself a few seconds of silence before responding. This brief pause activates your brain’s reasoning center and prevents impulsive reactions.",
    
                "**Identify the True Source:** Anger often masks deeper emotions like hurt or fear. Ask yourself, ‘What am I really feeling underneath this anger?’ Recognizing the root emotion promotes understanding and control.",
    
                "**Reframe the Situation:** Instead of focusing on the offense, shift perspective — ‘What can I learn or change in this moment?’ Reframing helps turn confrontation into growth.",
    
                "**Use “I” Statements:** Express your feelings without blame. Say, ‘I feel upset when…’ instead of ‘You always…’. This encourages communication over conflict and preserves relationships.",
    
                "**Reflect After Cooling Down:** Once calm, take time to evaluate what triggered you and what you might do differently next time. Reflection turns anger into emotional intelligence."
    
    
        ],
        'Physical': [
            
            "**Practice Controlled Breathing:** Anger quickens the heartbeat and tightens muscles. Take slow, deep breaths — in through your nose, out through your mouth — to signal your body to relax.",
    
            "**Channel Energy Through Movement:** Go for a brisk walk, do light exercise, or stretch your arms and neck. Physical release helps burn off the adrenaline that fuels anger.",
    
            "**Use Cooling Techniques:** Splash cool water on your face or hold something cold. Cooling the body physiologically lowers emotional intensity and restores calm.",
    
            "**Body Awareness Reset:** Notice where you feel anger — in your fists, chest, or jaw. Relax those muscles and drop your shoulders. This tells your nervous system that the threat has passed.",
    
            "**Ground Yourself Physically:** Feel your feet on the floor, notice your breath, and orient your eyes to the present space. Grounding pulls the mind away from reactive thought loops."
        ],

        
        'Spiritual': [
             " **Pray for Calm and Understanding:** In moments of anger, ask God to quiet your spirit and grant wisdom before you speak. *'My dear brothers and sisters, be quick to listen, slow to speak, and slow to become angry.' — James 1:19*",
    
            " **Meditate on Scriptures of Patience:** Reflect on verses that guide you toward gentleness and peace. *'A gentle answer turns away wrath, but a harsh word stirs up anger.' — Proverbs 15:1*",
    
            " **Surrender the Situation to God:** Release the desire for control or revenge through prayer. *'Do not repay anyone evil for evil... If it is possible, as far as it depends on you, live at peace with everyone.' — Romans 12:17-18*",
    
            " **Speak Blessing Instead of Bitterness:** When tempted to speak harshly, choose words that heal. *'Let all bitterness and wrath and anger and clamor and slander be put away from you... Be kind to one another, tenderhearted, forgiving one another, as God in Christ forgave you.' — Ephesians 4:31-32*",
            " **Reflect on God’s Patience Toward You:** Remember that God is slow to anger and rich in mercy — modeling the same grace brings peace to your heart. *'The Lord is compassionate and gracious, slow to anger, abounding in love.' — Psalm 103:8*"
        ],
        'Diet': [
        "🥑 Avocado, nuts, seeds — healthy fats reduce inflammation",
        "🍌 Bananas, sweet potatoes — potassium for nervous system balance",
        "🥬 Leafy greens — magnesium to calm tension",
        "🌾 Whole grains — stabilize energy",
        "💧 Water 2–3 liters/day — prevent irritability from dehydration"
    ],
        'Crisis': None
    },
    'sadness': {
        'title': "💧 Acknowledging the Pain: Gentle Steps Through Sadness",
        'color': "#0077B6", 
        'Mental': [
            " **Non-Judgmental Presence:** Allow the sadness to exist without trying to fix it immediately. Remind yourself this feeling is temporary.",
            " **Identify the Need:** Ask yourself, 'What am I truly missing or needing right now?'"
            " **Acknowledge and Name the Sadness:** Rather than pushing it away, gently label what you feel — ‘I’m sad right now.’ Naming the emotion helps your brain process it instead of being overwhelmed by it.",
    
            " **Challenge Hopeless Thoughts:** When sadness whispers that ‘nothing will change,’ question that belief. Ask yourself, ‘What evidence do I have that this feeling will last forever?’ This thought reappraisal brings balance and perspective.",
    
            " **Engage in Self-Compassion Talk:** Speak to yourself as you would to a close friend — with kindness, not criticism. Self-compassion activates emotional healing and reduces mental rumination.",
    
            " **Focus on Small Restorative Actions:** When sadness drains motivation, identify one simple step — like opening a window, journaling, or taking a short walk. Small actions reintroduce control and forward movement.",
    
            " **Reframe Loss or Disappointment:** Ask, ‘What might this experience be teaching me?’ Shifting from pain to purpose helps the mind find meaning beyond the sadness."
        ],
        'Physical': [
            " **Comfort & Care:** Seek warmth (blanket, tea). Ensure adequate hydration and rest, as your body is working hard to process the emotion."
            " **Allow the Body to Feel:** Sadness often shows up physically — heaviness in the chest or fatigue. Notice these sensations without judgment and breathe gently through them. Allowing the body to express emotion promotes release.",
    
            " **Gentle Movement:** Engage in mild physical activity — a short walk, stretching, or slow dancing. Movement increases endorphins, the body’s natural mood stabilizers.",
    
            " **Prioritize Rest and Nutrition:** Sadness can disrupt sleep and appetite. Maintain a routine with balanced meals and adequate rest to support emotional recovery.",
    
            " **Seek Natural Light:** Exposure to sunlight helps regulate serotonin and melatonin levels, improving both energy and mood. Even a few minutes outdoors can lift the spirit.",
    
            " **Body Relaxation Practice:** Scan your body for tension. Relax your face, shoulders, and chest as you breathe slowly. This communicates safety and calm to your nervous system."
        ],
        'Spiritual': [
            " **Seeking Connection:** Reach out to a trusted friend or community member. Look for small things to be grateful for, acknowledging the goodness that still exists."
             " **Bring Your Sadness to God in Prayer:** Speak openly with God about what hurts. Honest prayer brings emotional release and invites divine comfort. *'The Lord is close to the brokenhearted and saves those who are crushed in spirit.' — Psalm 34:18*",
    
            " **Reflect on God’s Faithfulness:** When sadness feels heavy, meditate on times God has carried you before. *'He heals the brokenhearted and binds up their wounds.' — Psalm 147:3*",
    
            " **Rest in God’s Presence:** Take a few quiet minutes to sit in stillness, breathing slowly as you imagine being held in God’s peace. *'Come to me, all you who are weary and burdened, and I will give you rest.' — Matthew 11:28*",
    
            " **Speak Words of Hope:** Say aloud verses that remind you of God’s promises. Hearing your own voice affirm truth can restore courage. *'Weeping may last through the night, but joy comes with the morning.' — Psalm 30:5*",
    
            " **Practice Gratitude in Faith:** Even in sadness, name small things you are thankful for — a friend, a moment of light, or breath itself. Gratitude reorients the heart toward hope. *'Give thanks in all circumstances; for this is God’s will for you in Christ Jesus.' — 1 Thessalonians 5:18*"
        ],
        'Diet': [
        "🫐 Berries — antioxidants reduce oxidative stress",
        "🥑 Nuts & seeds — omega-3 for mood support",
        "🥬 Dark leafy greens — folate for dopamine",
        "🌾 Whole grains — stabilize energy",
        "💧 Hydration 2–3 liters/day — supports mood and focus"
    ],
        'Crisis': "If you feel overwhelmed by sadness or despair, please reach out for immediate support."
    },
    'fear': {
        'title': "😨 Navigating Uncertainty: Grounding Against Fear and Anxiety",
        'color': "#7F00FF", 
        'Mental': [
            " **Fact-Checking:** Differentiate between the actual threat and the catastrophic story your mind is telling you.",
            " **Focus on NOW:** Bring your attention back to the present moment, away from future worries."
            " **Pause and Acknowledge the Fear:** When fear arises, don’t rush to suppress it. Instead, take a brief mental pause and simply name what you’re feeling — ‘I’m afraid right now.’ This act of acknowledgment reduces the brain’s alarm response and begins to restore clarity.",
    
            " **Reframe the Thought:** Identify the specific thought that triggered your fear. Ask yourself, ‘Is this thought absolutely true, or is it my mind’s prediction?’ Reframing helps shift your perspective from danger to possibility, giving your mind a sense of control.",
    
             " **Ground in the Present:** Fear often pulls the mind into ‘what-if’ scenarios. Bring yourself back by describing your surroundings — what you can see, hear, and feel. This reorients your brain to safety in the present moment.",
    
             " **Challenge Catastrophic Thinking:** Notice when your mind jumps to the worst-case scenario. Gently counter it with balanced thoughts such as, ‘I may not know what will happen, but I can handle it.’ This reinforces psychological resilience.",
    
        ],
        'Physical': [
            " **5-4-3-2-1 Grounding:** Name 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste.",
            " **Regulate Your Breathing:** When fear arises, your body shifts into a stress state. Inhale slowly through your nose for four seconds, hold for two, and exhale gently through your mouth for six. This simple rhythm signals your nervous system that you are safe.",
    
            " **Engage in Gentle Movement:** Light physical activity — such as a short walk, stretching, or softly shaking out your hands — helps your body release the tension that accumulates during fear. Movement allows the stress hormones to settle naturally.",
    
             " **Perform a Body Check-In:** Notice areas where your body feels tense — the shoulders, neck, or jaw are common. Consciously relax those muscles. Remind yourself that your body no longer needs to stay in 'defense mode.'",
    
            " **Use a Temperature Reset:** Applying cool water to your face or holding something cold activates the vagus nerve, which helps the body transition from a state of alertness to calm. It’s a quick and effective way to ground yourself in the present.",
    
            " **Practice a Body Awareness Scan:** Close your eyes and bring attention to where fear feels strongest — maybe your chest or stomach. Breathe slowly into that area, letting each breath ease the tension and remind your body that you are in control."
        ],
        'Spiritual': [
            " **Trust and Courage:** Connect with an internal sense of strength or faith that can guide you through uncertainty."
            " **Practicing Acceptance:** Reflect on the limits of your control. Let go of the need for others to meet an impossible standard."
            " **Release Fear Through Prayer:** Take a quiet moment to speak to God about what frightens you. You don’t need the perfect words — honesty itself invites peace. *'Cast all your anxiety on Him because He cares for you.' — 1 Peter 5:7*",
    
            " **Meditate on God’s Protection:** Reflect on verses that remind you of divine safety. Repeat them slowly as affirmations of peace. *'Even though I walk through the valley of the shadow of death, I will fear no evil, for You are with me.' — Psalm 23:4*",
    
            " **Practice Faith-Based Visualization:** Picture yourself surrounded by God’s light and protection. Let that image replace the fearful thought. *'The Lord is my light and my salvation—whom shall I fear?' — Psalm 27:1*",
        ],
        'Diet': [
        "🥦 Leafy greens (spinach, kale) — magnesium to calm nerves",
        "🍓 Berries, legumes — stabilize serotonin and mood",
        "🌾 Whole grains — steady energy to reduce jitteriness",
        "🍵 Herbal teas (chamomile, peppermint) — soothing",
        "💧 Drink 2–3 liters water/day — maintain focus and emotional stability"
    ],
        'Crisis': "If fear is causing panic or paralyzing distress, take a pause and call someone you trust."
    },
    'disgust': {
        'title': "🤢 Stepping Back: Addressing Disgust and Boundaries",
        'color': "#4C9F38",

        'Mental': [
                    " **Boundary Check:** Use this feeling to clarify your personal values and decide if you need to create distance from the source."
                    " **Acknowledge the Feeling Without Judgment:** When disgust appears, notice it and name it gently — ‘I feel disgusted right now.’ Recognizing the emotion helps your brain shift from reaction to reflection.",
    
                    " **Explore the Source:** Ask yourself, ‘What exactly triggered this feeling?’ Disgust often signals a boundary being crossed — physical, moral, or emotional. Understanding its origin turns it into useful information rather than rejection.",
    
    " **Reframe the Perception:** Instead of focusing on what repulses you, consider the broader picture — ‘Is there something to learn or forgive here?’ <br>This perspective can soften the harshness of the emotion.",
    
    " **Practice Cognitive Neutralization:** Visualize the situation becoming less vivid or intense. This helps your brain reduce overactivation in the areas linked to disgust.",
    
    " **Redirect Focus Toward Compassion:** When disgust is directed at people or situations, pause and ask, ‘How can I respond with empathy instead of avoidance?’ Compassion restores emotional balance and perspective."
                   ],

        'Physical': [
                    " **Change Environment:** Step away, open a window, or engage a pleasant sensory input to reset your senses."
                    " **Regulate Your Breathing:** Disgust can cause tension in the stomach or face. Take slow, deep breaths to ease muscle tightening and reset your body’s internal calm.",
    
                    " **Release the Tension Physically:** Walk, stretch, or open your posture. Physical movement helps discharge the stress energy associated with disgust.",
    
                    " **Rinse or Wash as Symbolic Renewal:** If the feeling is strong, wash your hands or face mindfully — not as avoidance, but as a physical reminder of cleansing and release.",
    
                    " **Relax Facial Muscles:** Disgust often shows through facial contraction. Soften your face and jaw as you exhale slowly. This signals your nervous system to relax.",
    
                    " **Ground Yourself Through Sensory Reset:** Smell something pleasant, touch a familiar texture, or focus on a neutral sound. Positive sensory input retrains your body’s emotional state."
                     ],

        'Spiritual': [
            " **Value Clarity:** Reflect on what is ethically and morally acceptable for you. Use this to reinforce positive principles."
            "**Pray for Inner Cleansing:** When your heart feels burdened by disgust or resentment, ask God to renew your spirit. *'Create in me a clean heart, O God, and renew a right spirit within me.' — Psalm 51:10*",
    
            " **Reflect on Forgiveness and Grace:** Remember that God often meets human imperfection with compassion, not rejection. *'Bear with each other and forgive one another if any of you has a grievance against someone. Forgive as the Lord forgave you.' — Colossians 3:13*",
    
            " **Meditate on God’s Acceptance:** When disgust is self-directed, remind yourself that God’s love is unconditional. *'Nothing can separate us from the love of God that is in Christ Jesus our Lord.' — Romans 8:38-39*",
    
            " **Speak Words of Renewal:** Say aloud affirmations of spiritual cleansing: ‘I release this burden and receive God’s peace.’ *'If we confess our sins, He is faithful and just to forgive us our sins and to cleanse us from all unrighteousness,AMEN.' — 1 John 1:9*",
    
            " **Replace Judgment with Compassion:** Pray for a heart that sees others through mercy rather than revulsion. *'Be kind and compassionate to one another, forgiving each other, just as in Christ God forgave you.' — Ephesians 4:32*"

            ],
            'Diet': [
        "🍏 Fresh fruits & vegetables — support gut & brain health",
        "🥬 Fermented plant foods — nurture microbiome",
        "🍵 Herbal teas — soothe discomfort",
        "🌾 Whole grains — provide stable energy",
        "💧 Hydration 2–3 liters/day — support body and mind balance"
    ],
        'Crisis': None
    },
    'joy': {
        'title': "😊 Celebrating the Good! Reinforce Positive Emotions",
        'color': "#FFD700", 
        'Mental': [
            " **Savoring:** Mentally re-run the positive experience, focusing on sensory details, to enhance memory consolidation."
            " **Recognize Moments of Joy:** Train your mind to notice small, positive moments — a smile, a kind word, sunlight on your skin. Awareness strengthens the brain’s ability to perceive joy even in ordinary experiences.",
    
            " **Reframe Toward Gratitude:** When challenges arise, gently shift your focus from what’s lacking to what’s still good. Gratitude reframes the mind from scarcity to appreciation, nurturing a steady sense of joy.",
    
            " **Practice Positive Reflection:** At the end of each day, recall three things that went well, no matter how small. This strengthens neural pathways associated with optimism and contentment.",
    
            " **Engage in Purposeful Thinking:** Reflect on how your actions align with your values. Meaningful engagement with life sustains joy beyond fleeting happiness.",

            " **Share Joy Consciously:** Expressing joy to others — through encouragement, humor, or kindness — amplifies your own positive emotion. Joy grows when it’s given away."
            ],
        'Physical': [
            " **Expression:** Share your joy with positive, energized body language. Go do something fun with that energy!"
            " **Move with Energy and Lightness:** Joy is often felt through the body. Stretch, dance, or take a walk in rhythm with music you love. Movement increases endorphins and reinforces emotional vitality.",
    
            " **Smile and Relax the Body:** Even a gentle smile can send signals to your brain to enhance positive mood. Let your shoulders drop, breathe deeply, and let your posture express ease.",
    
            " **Engage with Nature:** Spend time outdoors or near natural light. Sunlight increases serotonin, lifting mood and grounding the sense of joy in the body.",
    
            " **Nourish and Hydrate:** Eat balanced meals and drink enough water. Physical nourishment supports stable mood and emotional balance.",
    
            " **Rest in Delight:** Give your body the rest it deserves. When well-rested, your mind can experience joy more fully and with gratitude."
            ],
        'Spiritual': [
            " **Thank God for the Gift of Joy:** Begin or end your day by thanking God for the blessings — big or small — that bring gladness to your heart. *'This is the day that the Lord has made; let us rejoice and be glad in it.' — Psalm 118:24*",
    
            " **Rejoice in God’s Presence:** Remember that true joy flows not from circumstances but from communion with God. *'In Your presence there is fullness of joy; at Your right hand are pleasures forevermore.' — Psalm 16:11*",
    
            " **Share Joy Through Service:** Acts of kindness amplify divine joy within you. *'The joy of the Lord is your strength.' — Nehemiah 8:10*",
    
            "**Speak Words of Praise:** When you feel uplifted, voice it — sing, pray, or speak words of gratitude. Expressing joy strengthens faith and renews inner peace. *'Rejoice in the Lord always; again I will say, rejoice!' — Philippians 4:4*",

            " **Anchor Joy in Hope:** Even when life feels uncertain, hold to joy as a choice rooted in trust. *'May the God of hope fill you with all joy and peace as you trust in Him.' — Romans 15:13*"
            " **Generosity:** Use your good mood to uplift others or practice gratitude for the source of your happiness."
            ],
            'Diet': [
        "🍇 Colorful fruits & vegetables — vitamin C & antioxidants",
        "🥑 Nuts & seeds — healthy fats for brain & mood",
        "🌾 Whole grains & legumes — steady energy",
        "🍵 Herbal teas & water — hydration supports vitality",
        "💧 Water 2–3 liters/day — maintain energy & focus"
    ],
        'Crisis': None
    },
    'surprise': {
        'title': "😮 The Unexpected: Pausing to Assess Surprise",
        'color': "#87CEEB", 
        'Mental': [
                " **Pause and Observe:** When something unexpected happens, take a moment to notice your thoughts and feelings without judgment. Naming the surprise — ‘I am surprised right now’ — helps your mind process it calmly.",
    
                " **Assess the Situation:** Ask yourself, ‘Is this event a threat, an opportunity, or neutral?’ This cognitive evaluation reduces impulsive reactions and gives clarity.",
    
                " **Reframe the Unexpected:** Look for potential lessons or growth in the surprise. Even challenges can contain insights or opportunities for learning.",
    
                " **Stay Present:** Avoid imagining worst-case scenarios. Focus on what is happening right now, what you can control, and what you can observe objectively.",
    
                "**Reflect After the Moment:** Once the initial surprise fades, consider what this experience teaches you about adaptability and resilience."
                "**Assess & Orient:** Ask yourself: 'Is this surprise positive or negative?' Let that guide your next cognitive step."],
        'Physical': [
            "**Regulate Your Breathing:** Surprises often trigger sudden physiological responses. Take a slow, deep breath to steady your heart rate and relax your muscles.",
    
            "**Ground Yourself Through Movement:** Shift your body — stand up, stretch, or take a few steps. Physical grounding helps process sudden energy spikes.",
    
            " **Release Tension:** Notice any sudden muscle contractions in your jaw, shoulders, or hands. Consciously relax these areas as you exhale.",
    
            " **Engage Your Senses:** Touch something solid, smell a familiar scent, or listen to a calming sound. Sensory focus helps anchor you to the present moment.",
    
            "**Hydrate or Refresh:** Drinking water or washing your hands/face provides a small but effective physiological reset after sudden events."
            "**Stop, Look, Listen:** Freeze your movement for one second to fully take in the unexpected sensory data."
            ],
        'Spiritual': [
            " **Bring the Surprise to God in Prayer:** Share your astonishment, uncertainty, or delight with God. *'Trust in the Lord with all your heart and lean not on your own understanding.' — Proverbs 3:5*",
    
            " **Look for Divine Meaning:** Reflect on whether this unexpected event could have a purpose or lesson. *'And we know that in all things God works for the good of those who love Him.' — Romans 8:28*",
    
            "**Anchor in Faith, Not Fear:** Surprises can be unsettling, but God’s presence remains constant. *'Do not be afraid, for I am with you; do not be dismayed, for I am your God.' — Isaiah 41:10*",
    
            " **Speak Words of Gratitude:** Even when surprised by challenges, thank God for His guidance and protection. *'Give thanks in all circumstances; for this is God’s will for you in Christ Jesus.' — 1 Thessalonians 5:18*",
    
            " **Meditate on God’s Sovereignty:** Remind yourself that life unfolds under His care, which can bring calm and perspective amid unpredictability. *'The Lord has done great things for us, and we are filled with joy.' — Psalm 126:3*"
            " **Openness:** View the unexpected event as a potential new path or unique moment of learning."
            ],
            'Diet': [
        "🥣 Oats, nuts, seeds, bananas — stabilize blood sugar and emotions",
        "🍉 Hydrating fruits & vegetables — reset physiology",
        "🍵 Herbal teas — calm the nervous system",
        "🥬 Leafy greens — magnesium for relaxation",
        "💧 Water 2–3 liters/day — support mental & physical calm"
    ],
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
        
        st.markdown(f"## Personalized Guidance: {top_negative_emotion.title()} Detected")
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
                    <div class="disc">
                    <strong>Disclaimer:</strong> 
                    </div>   

        "💬 **Friendly Reminder:** These interventions are designed to support your emotional, mental, physical, and spiritual well-being. They are helpful tools, but they do not replace professional medical, psychological, or psychiatric care. If your feelings are overwhelming, persistent, or interfere with daily life, please reach out to a licensed healthcare provider for guidance and support.",
                    
        """, unsafe_allow_html=True
                    )
    # If the only predicted emotion is Joy or Surprise (or no negative is strong)
    elif 'joy' in target_results['Emotion'].values:
        st.success(" **Analysis suggests a positive emotional state (Joy/Happiness)!** Keep up the good work and share your positive energy.")

    else:
        st.info("No strong negative or positive emotion was detected to prompt specific intervention.") 