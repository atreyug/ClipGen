import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq()


def chatbot(transcript, specification=""):
    user_query = f"""
    Analyze the provided transcript and extract viral short-form video clips (Shorts / Reels / TikTok).

    USER SPECIFICATION:
    "{specification if specification else 'Extract the most viral, captivating clips.'}"

    TRANSCRIPT DATA:
    {json.dumps(transcript, ensure_ascii=False, indent=2)}
    """

    system_prompt = """
    You are an expert video editor and viral content strategist.
    Your task is to select viral video clips from a transcript.

    CRITICAL RULES FOR CLIP BOUNDARIES (DO NOT VIOLATE):
    1. NEVER cut a clip in the middle of an idea, sentence, or story arc.
    2. A complete clip MUST contain:
       - HOOK (First 2-3 seconds): Attention-grabbing opening.
       - BODY: Full context/explanation.
       - PAYOFF/CONCLUSION: A clear, satisfying ending or punchline.
    3. DURATION REQUIREMENT:
       - Minimum duration: 10 seconds.
       - Maximum duration: 60 seconds.
       - Do not create tiny 5-second or huge 2-minute clips.
    4. TIMESTAMP PRECISION:
       - The `start` time MUST match the EXACT `start` timestamp of the first segment in the clip.
       - The `end` time MUST match the EXACT `end` timestamp of the last segment in the clip.
       - Do NOT invent random float timestamps. Combine whole segments.

    QUALITY OVER QUANTITY:
    - Ensure clips are completely understandable to someone who hasn't watched the full video.

    OUTPUT FORMAT:
    Return strictly valid JSON with no preamble or explanation.

    {
        "clips": [
            {
                "start": float,
                "end": float,
                "viral_score": int,
                "reason": "Why this moment is viral and how it has a complete hook and payoff",
                "text": "Full concatenated text of all included segments"
            }
        ]
    }
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b", 
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_query
            }
        ],
        response_format={
            "type": "json_object"
        }
    )

    return json.loads(
        response.choices[0].message.content
    )