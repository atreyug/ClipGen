from dotenv import load_dotenv
import json
from groq import Groq

load_dotenv()

client = Groq()

def chatbot(transcript, specification):
    user_query = f"""
    I want to generate viral short-form clips from this transcript.

    Find the strongest hooks and moments that an audience would
    likely enjoy.

    You may merge multiple adjacent transcript segments into one clip.

    Quality >>> quantity.

    Return only valid JSON.

    specific detail on which clips are to be generated: {specification} (if this query is related to the transcript then use it otherwise ignore)
    """

    response_fmt = """
    {
        "clips": [
            {
                "start": 0.0,
                "end": 0.0,
                "viral_score": 0,
                "reason": "string"
            }
        ]
    }
    """

    system_prompt = f"""
    You are an AI assistant that identifies high-quality
    short-form video clips from transcripts.

    Analyze ONLY the provided transcript.

    Look for:
    - Strong hooks
    - Curiosity
    - Surprising statements
    - Useful insights
    - Emotional moments
    - Funny moments
    - Controversial or thought-provoking statements
    - Complete standalone thoughts

    Quality is more important than quantity.

    A good clip should:
    1. Start at a natural point.
    2. Contain a complete idea.
    3. Have a strong hook or interesting opening.
    4. Have a satisfying payoff.
    5. Work without requiring too much surrounding context.

    You may combine adjacent transcript segments when necessary.

    The start and end values MUST be timestamps in seconds.
    Use decimal numbers, not MM:SS.

    Return ONLY valid JSON.

    Response format:
    {response_fmt}

    Order clips by decreasing viral_score.

    Transcript:
    {json.dumps(transcript, ensure_ascii=False)}
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
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


