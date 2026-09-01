import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")

MODEL = "@cf/qwen/qwen2.5-coder-32b-instruct"

if not CLOUDFLARE_ACCOUNT_ID:
    raise RuntimeError("CLOUDFLARE_ACCOUNT_ID is missing")

if not CLOUDFLARE_API_TOKEN:
    raise RuntimeError("CLOUDFLARE_API_TOKEN is missing")


def chatbot(transcript, specification=""):
    user_query = f"""
Analyze the provided transcript and extract viral short-form video clips
(Shorts / Reels / TikTok).

USER SPECIFICATION:
"{specification if specification else 'Extract the most viral, captivating clips.'}"

TRANSCRIPT DATA:
{json.dumps(transcript, ensure_ascii=False, indent=2)}
"""

    system_prompt = """
You are an expert video editor and viral content strategist.

Your task is to select viral video clips from a transcript.

Viral score must be between 0 and 10.

CRITICAL RULES FOR CLIP BOUNDARIES:

1. NEVER cut a clip in the middle of an idea, sentence, or story arc.

2. A complete clip MUST contain:
   - HOOK: First 2-3 seconds must grab attention.
   - BODY: Full context/explanation.
   - PAYOFF/CONCLUSION: A clear satisfying ending or punchline.

3. DURATION REQUIREMENT:
   - Minimum duration: 10 seconds.
   - Maximum duration: 60 seconds.
   - Do not create tiny 5-second clips.
   - Do not create clips longer than 60 seconds.

4. TIMESTAMP PRECISION:
   - The "start" time MUST exactly match the "start"
     timestamp of the first segment in the clip.
   - The "end" time MUST exactly match the "end"
     timestamp of the last segment in the clip.
   - NEVER invent random timestamps.
   - ONLY combine complete transcript segments.

5. AT MAX 5 BEST CLIPS SHOULD BE RETURNED.

QUALITY OVER QUANTITY:

- Only select genuinely valuable clips.
- Each clip must be completely understandable
  without watching the full video.
- Prefer:
  - Strong hooks
  - Surprising statements
  - Emotional moments
  - Controversial opinions
  - Useful insights
  - Interesting stories
  - Jokes
  - Memorable conclusions

IMPORTANT:

- Do not select clips just to increase the number of clips.
- Prefer fewer high-quality clips over many weak clips.
- Every clip should have a clear beginning, middle, and ending.
- Avoid clips that require missing context from earlier in the video.
- Follow all the DURATION REQUIREMENT rules without fail.

OUTPUT FORMAT:
Return strictly a valid JSON object matching this schema:
{
  "clips": [
    {
      "start": 0.0,
      "end": 0.0,
      "viral_score": 0,
      "reason": "Why this moment is viral (10 words max)"
    }
  ]
}
Do not include any text outside the JSON object.

Do not include Markdown.
Do not include ```json.
Do not include explanations outside the JSON.
"""

    url = (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{CLOUDFLARE_ACCOUNT_ID}/ai/run/{MODEL}"
    )

    payload = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_query,
            },
        ],

        "response_format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "clips": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start": {
                                    "type": "number",
                                    "description": "Start time in seconds"
                                },
                                "end": {
                                    "type": "number",
                                    "description": "End time in seconds"
                                },
                                "viral_score": {
                                    "type": "integer",
                                    "description": "Viral score out of 10"
                                },
                                "reason": {
                                    "type": "string",
                                    "description": "Why this clip is viral"
                                }
                            },
                            "required": [
                                "start",
                                "end",
                                "viral_score",
                                "reason"
                            ]
                        }
                    }
                },
                "required": [
                    "clips"
                ]
            }
        },

        "temperature": 0.2,
    }

    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=120,
        )
    except requests.RequestException as e:
        raise RuntimeError(
            f"Could not connect to Cloudflare AI: {e}"
        ) from e

    if not response.ok:
        raise RuntimeError(
            f"Cloudflare AI request failed "
            f"({response.status_code}):\n{response.text}"
        )

    try:
        data = response.json()
    except ValueError as e:
        raise RuntimeError(
            f"Cloudflare returned invalid HTTP JSON:\n{response.text}"
        ) from e

    if not data.get("success"):
        raise RuntimeError(
            "Cloudflare AI returned an error:\n"
            + json.dumps(data, indent=2)
        )

    # Cloudflare returns the response here.
    # It may be a JSON string, an already-parsed dict, or None.
    result = data["result"].get("response")

    if result is None:
        # Fallback: OpenAI-compatible choices format
        try:
            result = data["result"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(
                "Cloudflare returned an unexpected response structure:\n"
                + json.dumps(data, indent=2)
            )


    if not isinstance(result, dict) or "clips" not in result:
        raise RuntimeError(
            f"Cloudflare response is missing the 'clips' key. Got:\n{result}"
        )
    
    print(result)

    return result























# import json
# from dotenv import load_dotenv
# from groq import Groq

# load_dotenv()

# client = Groq()


# def chatbot(transcript, specification=""):
#     user_query = f"""
#     Analyze the provided transcript and extract viral short-form video clips (Shorts / Reels / TikTok).

#     USER SPECIFICATION:
#     "{specification if specification else 'Extract the most viral, captivating clips.'}"

#     TRANSCRIPT DATA:
#     {json.dumps(transcript, ensure_ascii=False, indent=2)}
#     """

#     system_prompt = """
#     You are an expert video editor and viral content strategist.
#     Your task is to select viral video clips from a transcript.
#     Viral score should be between 0 and 10.

#     CRITICAL RULES FOR CLIP BOUNDARIES (DO NOT VIOLATE):
#     1. NEVER cut a clip in the middle of an idea, sentence, or story arc.
#     2. A complete clip MUST contain:
#        - HOOK (First 2-3 seconds): Attention-grabbing opening.
#        - BODY: Full context/explanation.
#        - PAYOFF/CONCLUSION: A clear, satisfying ending or punchline.
#     3. DURATION REQUIREMENT:
#        - Minimum duration: 10 seconds.
#        - Maximum duration: 60 seconds.
#        - Do not create tiny 5-second or huge 2-minute clips.
#     4. TIMESTAMP PRECISION:
#        - The `start` time MUST match the EXACT `start` timestamp of the first segment in the clip.
#        - The `end` time MUST match the EXACT `end` timestamp of the last segment in the clip.
#        - Do NOT invent random float timestamps. Combine whole segments.

#     QUALITY OVER QUANTITY:
#     - Ensure clips are completely understandable to someone who hasn't watched the full video.

#     OUTPUT FORMAT:
#     Return strictly valid JSON with no preamble or explanation.

#     {
#         "clips": [
#             {
#                 "start": float,
#                 "end": float,
#                 "viral_score": int,
#                 "reason": "Why this moment is viral and how it has a complete hook and payoff",
#                 "text": "Full concatenated text of all included segments"
#             }
#         ]
#     }
#     """

#     response = client.chat.completions.create(
#         model="llama-3.3-70b-versatile", 
#         messages=[
#             {
#                 "role": "system",
#                 "content": system_prompt
#             },
#             {
#                 "role": "user",
#                 "content": user_query
#             }
#         ],
#         response_format={
#             "type": "json_object"
#         }
#     )

#     return json.loads(
#         response.choices[0].message.content
#     )