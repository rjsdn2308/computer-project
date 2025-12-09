import os
import json
import re
from typing import List, Dict, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

# .env 로드
load_dotenv()

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", "")
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").lower()


# 0. 공통 LLM 호출 함수
def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    system_prompt + user_prompt를 LLM에 보내고 모델의 응답을 문자열로 반환한다.
    """

    if LLM_PROVIDER not in ["openai", "huggingface"]:
        raise ValueError(f"지원하지 않는 LLM_PROVIDER 값입니다: {LLM_PROVIDER}")

    # OpenAI 모델 사용
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

        client = OpenAI(api_key=OPENAI_API_KEY)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
        )
        return response.choices[0].message.content

    # HuggingFace 모델 사용
    if not HUGGINGFACE_API_KEY:
        raise RuntimeError("HUGGINGFACE_API_KEY가 설정되어 있지 않습니다.")

    HF_MODEL = "HuggingFaceH4/zephyr-7b-alpha"

    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": system_prompt + "\n\n" + user_prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.3,
        }
    }

    resp = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_MODEL}",
        headers=headers,
        json=payload,
        timeout=60,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"HuggingFace API 오류: {resp.text}")

    data = resp.json()

    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"]

    return str(data)


# JSON 파싱 유틸
def _extract_json(text: str) -> Optional[str]:
    """
    모델 응답에서 JSON 배열만 추출한다.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    start = cleaned.find("[")
    end = cleaned.rfind("]")

    if start != -1 and end != -1:
        cleaned = cleaned[start:end+1]

    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        return None


# FruitLLM 클래스
class FruitLLM:
    """
    - 어린이를 위한 OX 과일 퀴즈 생성
    - 아이들이 이해할 수 있는 과일 설명 챗봇
    - CV 결과 기반 과일 설명
    """

    def __init__(self):
        pass

    # (1) 어린이용 과일 OX 퀴즈 생성
    def generate_quiz(self, difficulty: str = "Easy", n_questions: int = 5) -> List[Dict]:

        system_prompt = (
            "너는 유치원과 초등학교 저학년 아이들에게 과일 정보를 알려주는 선생님이야. "
            "OX 퀴즈를 만들 때는 문장을 짧고 단순하게 만들고, 어려운 단어나 전문 용어는 쓰지 말아야 해. "
            "문제를 학교 수업처럼 차분한 말투로 출제해 주고, 설명 역시 부드러운 존댓말로 적어 주면 좋겠어."
            "출력은 내가 지정한 JSON 배열 형식만 포함해야 해."
        )

        user_prompt = f"""
난이도: {difficulty}
아이들이 풀 수 있는 짧고 쉬운 과일 OX 퀴즈를 {n_questions}개 만들어 주세요.

반드시 아래 JSON 형식만 출력하세요:

[
  {{
    "question": "문제 문장",
    "answer": "O 또는 X",
    "explanation": "아이들이 이해하기 쉬운 설명",
    "difficulty": "{difficulty}"
  }}
]
"""

        try:
            raw = call_llm(system_prompt, user_prompt)
            json_str = _extract_json(raw)
            if json_str is None:
                raise ValueError("JSON 추출 실패")

            quiz_list = json.loads(json_str)

            if not isinstance(quiz_list, list) or len(quiz_list) == 0:
                raise ValueError("퀴즈 리스트가 비어 있음")

            return quiz_list

        except Exception as e:
            print("[FruitLLM.generate_quiz] 오류:", e)
            print("더미 퀴즈로 대체합니다.")

            return [
                {
                    "question": "사과는 나무에서 자라는 과일이에요.",
                    "answer": "O",
                    "explanation": "사과는 나무에 열리는 과일이에요. 사과나무는 가을에 열매가 많이 열려요.",
                    "difficulty": difficulty
                },
                {
                    "question": "딸기는 물속에서 자라는 과일이에요.",
                    "answer": "X",
                    "explanation": "딸기는 밭의 땅 가까운 곳에서 자라요.",
                    "difficulty": difficulty
                }
            ]

    # (2) 정답 체크
    def check_answer(self, quiz_item: Dict, user_answer: str) -> Dict:
        correct = quiz_item.get("answer", "").strip().upper()
        user = user_answer.strip().upper()

        return {
            "is_correct": (correct == user),
            "correct_answer": correct,
            "explanation": quiz_item.get("explanation", "")
        }

    # (3) 어린이용 과일 Q&A 챗봇
    def chat_about_fruit(self, chat_history: List[Dict[str, str]]) -> str:

        system_prompt = (
            "너는 유치원과 초등학교 저학년 아이들에게 과일을 쉽게 설명하는 선생님이야. "
            "아이들이 이해하기 쉬운 짧은 문장으로 설명해 주고, 어려운 말을 쓰지 않도록 해. "
            "말투는 친절하고 차분한 존댓말로 해 주면 좋아."
        )

        if not chat_history:
            last_user_message = "안녕하세요. 과일에 대해 궁금한 게 있어요."
        else:
            last_user_message = chat_history[-1]["content"]

        user_prompt = (
            f"아이의 질문: {last_user_message}\n"
            "이 아이에게 유치원 선생님처럼 차분하고 쉬운 말로 답해 주세요."
        )

        try:
            answer = call_llm(system_prompt, user_prompt)
        except Exception as e:
            print("[FruitLLM.chat_about_fruit] 오류:", e)
            answer = (
                "지금은 설명을 준비하는 데 조금 어려움이 있어요. "
                "조금 있다가 다시 물어보면 더 자세히 알려줄게요."
            )

        return answer

    # ---------------------------------------------------------------
    # (4) CV 결과 기반 과일 설명
    # ---------------------------------------------------------------
    def explain_fruit_from_cv_result(self, fruit_eng: str, base_description: str = "") -> str:

        system_prompt = (
            "너는 유치원과 초등학교 저학년 아이들에게 과일 이야기를 들려주는 선생님이야. "
            "아이들이 이해하기 쉬운 단어만 사용해서, 과일의 특징, 자라는 곳, 맛, 보관법 등을 "
            "4~7문장 정도로 부드럽게 설명해 줘."
        )

        user_prompt = f"""
과일 이름: {fruit_eng}
기본 설명: {base_description}

위 정보를 참고해서 이 과일을 어린아이에게 설명해 주세요.
"""

        try:
            answer = call_llm(system_prompt, user_prompt)
        except Exception as e:
            print("[FruitLLM.explain_fruit_from_cv_result] 오류:", e)
            answer = (
                f"{fruit_eng}에 대해 설명하고 싶지만 지금은 조금 어려워요. "
                "나중에 다시 시도하면 더 자세히 알려줄 수 있어요."
            )

        return answer