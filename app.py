import os
import numpy as np
import gradio as gr

from cv_model import predict_fruit
from quiz_llm import FruitLLM

# FruitLLM 인스턴스
fruit_llm = FruitLLM()

# 1. LLM: OX 퀴즈 기능
def start_quiz(difficulty: str):
    """
    과일 놀이 퀴즈 시작하기.
    difficulty: "Easy" / "Normal" / "Hard"
    """
    try:
        quiz_list = fruit_llm.generate_quiz(difficulty=difficulty, n_questions=5)
    except Exception as e:
        print("[start_quiz] Error:", e)
        # LLM 실패 시 더미 퀴즈
        quiz_list = [
            {
                "question": "사과는 나무에서 자라는 과일이에요.",
                "answer": "O",
                "explanation": "사과는 나무에서 열리는 과일이에요. 사과나무는 가을에 열매가 많이 익어요.",
                "difficulty": difficulty
            },
            {
                "question": "딸기는 물속에서 자라는 과일이에요.",
                "answer": "X",
                "explanation": "딸기는 밭의 땅 가까운 곳에서 자라요. 그래서 허리를 굽히고 따야 해요.",
                "difficulty": difficulty
            }
        ]

    # 퀴즈 상태
    state = {
        "quiz_list": quiz_list,
        "index": 0,
        "score": 0,
        "difficulty": difficulty
    }

    if quiz_list:
        first_q = quiz_list[0]["question"]
    else:
        first_q = "퀴즈를 가져오지 못했어요. 다시 한 번 눌러 볼까요?"

    feedback = f"난이도 [{difficulty}] 과일 퀴즈를 시작해 볼까요? O 또는 X를 골라 주세요."
    explanation = ""

    return first_q, feedback, explanation, state


def _answer_quiz_internal(user_answer: str, state: dict):
    """
    O / X 버튼 공통 로직.
    """
    if state is None or "quiz_list" not in state:
        return (
            "먼저 퀴즈 시작하기 버튼을 눌러 주세요.",
            "",
            "퀴즈를 시작하려면 먼저 문제를 만들어야 해요.",
            state
        )

    quiz_list = state["quiz_list"]
    idx = state["index"]
    score = state["score"]

    # 이미 다 푼 경우
    if idx >= len(quiz_list):
        final_msg = f"모든 문제를 다 풀었어요. 마지막 점수는 {score}점이에요."
        return "퀴즈가 모두 끝났어요.", final_msg, "", state

    current_q = quiz_list[idx]
    result = fruit_llm.check_answer(current_q, user_answer)

    # 정답/오답 처리
    if result["is_correct"]:
        score += 1
        feedback = f"정답이에요. 지금까지 점수는 {score}점이에요."
    else:
        feedback = f"조금 아쉬워요. 그래도 괜찮아요. 지금까지 점수는 {score}점이에요."

    explanation = result["explanation"]
    idx += 1

    # 다음 문제
    if idx < len(quiz_list):
        next_question = quiz_list[idx]["question"]
    else:
        next_question = f"모든 문제를 다 풀었어요. 최종 점수는 {score}점이에요."

    # 상태 업데이트
    state["index"] = idx
    state["score"] = score

    return next_question, feedback, explanation, state


def answer_quiz_O(state: dict):
    return _answer_quiz_internal("O", state)


def answer_quiz_X(state: dict):
    return _answer_quiz_internal("X", state)


# 2. LLM: 과일 Q&A 챗봇
def chat_respond(user_message, chat_messages):
    """
    과일 Q&A 챗봇 응답 함수.
    chat_messages: Chatbot 컴포넌트의 messages 형식 히스토리
    """
    if not user_message:
        return "", chat_messages

    if chat_messages is None:
        chat_messages = []

    # 유저 메시지 추가
    chat_messages.append({"role": "user", "content": user_message})

    # LLM 호출
    try:
        answer = fruit_llm.chat_about_fruit(chat_messages)
    except Exception as e:
        print("[chat_respond] Error:", e)
        answer = (
            "지금은 과일 선생님이 잠시 쉬고 있어서 자세히 설명하기 어려워요. "
            "나중에 다시 물어보면 더 친절히 알려줄게요."
        )

    # 어시스턴트 답변 추가
    chat_messages.append({"role": "assistant", "content": answer})

    # 입력창 비우고, 히스토리 반환
    return "", chat_messages


# 3. CV: 이미지 분석
def analyze_image(image):
    """
    과일 사진을 분석해서 어떤 과일인지 예측하고
    수확 직전 이미지 경로와 기본 설명 상태를 함께 반환.
    """
    if image is None:
        info_text = "먼저 과일 사진을 올리거나 찍어 볼까요?"
        return info_text, None, "", {"fruit_eng": None, "base_description": ""}

    image_np = np.array(image)

    try:
        result = predict_fruit(image_np)
    except Exception as e:
        print("[analyze_image] Error:", e)
        info_text = "과일을 알아보는 동안 문제가 생겼어요. 다시 한 번 시도해 주세요."
        return info_text, None, "", {"fruit_eng": None, "base_description": ""}

    fruit_ko = result.get("fruit_ko", "알 수 없음")
    fruit_eng = result.get("fruit_eng", "")
    grow_type = result.get("grow_type", "unknown")
    field_img_path = result.get("pre_harvest_image_path", None)
    base_desc = result.get("description", "")

    if grow_type == "tree":
        grow_text = "나무에서 자라는 과일이에요."
    elif grow_type == "ground":
        grow_text = "땅 가까운 곳에서 자라는 과일이에요."
    elif grow_type == "vine":
        grow_text = "덩굴에서 자라는 과일이에요."
    else:
        grow_text = "어디에서 자라는지는 아직 잘 모르겠어요."

    info_text = f"이 과일은 {fruit_ko}로 보이에요.\n{grow_text}"

    state = {
        "fruit_eng": fruit_eng,
        "base_description": base_desc
    }

    # long_explain은 버튼 눌렀을 때 채움
    return info_text, field_img_path, "", state


def explain_from_cv(cv_state):
    """
    CV 인식 결과를 LLM에 전달하여 자세한 설명 생성.
    """
    if cv_state is None or not cv_state.get("fruit_eng"):
        return "먼저 과일 사진을 분석해서 어떤 과일인지 알아봐야 해요."

    fruit_eng = cv_state["fruit_eng"]
    base_description = cv_state.get("base_description", "")

    try:
        detail_text = fruit_llm.explain_fruit_from_cv_result(fruit_eng, base_description)
    except Exception as e:
        print("[explain_from_cv] Error:", e)
        detail_text = (
            "과일에 대해 설명하고 싶지만 지금은 조금 어려워요. "
            "나중에 다시 시도하면 더 자세히 알려줄게요."
        )

    return detail_text


# 4. Gradio UI
def build_app():

    pastel_css = """
    body {
        background-color: #f6f1eb;
    }
    .gradio-container {
        background-color: #f6f1eb;
    }
    .gr-button {
        border-radius: 20px !important;
        font-weight: 600;
    }
    .gr-textbox, .gr-markdown, .gr-radio, .gr-chatbot, .gr-image {
        border-radius: 12px !important;
    }
    """

    with gr.Blocks(title="과일과 친해져요~") as demo:
        # CSS 삽입 (구버전 Gradio에서도 동작)
        gr.HTML(f"<style>{pastel_css}</style>")

        gr.Markdown(
            """
            # 과일과 친해져요~ 
            과일을 쉽고 재미있게 배울 수 있는 공간이에요.  
            퀴즈도 풀고, 과일 선생님에게 질문도 할 수 있어요.
            """
        )

        # 4-1. LLM: 과일 퀴즈 & 챗봇
        with gr.Tab("과일 퀴즈"):
            with gr.Tab("OX 퀴즈"):
                difficulty = gr.Radio(
                    ["Easy", "Normal", "Hard"],
                    value="Easy",
                    label="퀴즈 난이도를 골라 주세요."
                )
                start_btn = gr.Button("과일 퀴즈 시작하기")

                quiz_question = gr.Textbox(
                    label="문제",
                    interactive=False,
                    lines=2
                )
                quiz_feedback = gr.Markdown("답을 맞히면 여기서 알려줄게요.")
                quiz_explanation = gr.Textbox(
                    label="설명",
                    interactive=False,
                    lines=4
                )

                quiz_state = gr.State(value=None)

                with gr.Row():
                    btn_o = gr.Button("O (맞아요)")
                    btn_x = gr.Button("X (아니에요)")

                # 이벤트 연결
                start_btn.click(
                    fn=start_quiz,
                    inputs=difficulty,
                    outputs=[quiz_question, quiz_feedback, quiz_explanation, quiz_state]
                )

                btn_o.click(
                    fn=answer_quiz_O,
                    inputs=quiz_state,
                    outputs=[quiz_question, quiz_feedback, quiz_explanation, quiz_state]
                )

                btn_x.click(
                    fn=answer_quiz_X,
                    inputs=quiz_state,
                    outputs=[quiz_question, quiz_feedback, quiz_explanation, quiz_state]
                )

            with gr.Tab("과일요정봇"):
                gr.Markdown("과일에 대해 궁금한 점을 물어보세요. 과일요정봇이 쉽게 설명해 줄게요.")

                chatbot = gr.Chatbot(
                    label="과일요정봇",
                    height=400
                )
                msg = gr.Textbox(
                    label="궁금한 내용을 적어 주세요.",
                    placeholder="예: 사과는 어디서 자라나요?"
                )
                send_btn = gr.Button("질문 보내기")

                send_btn.click(
                    fn=chat_respond,
                    inputs=[msg, chatbot],
                    outputs=[msg, chatbot]
                )

                msg.submit(
                    fn=chat_respond,
                    inputs=[msg, chatbot],
                    outputs=[msg, chatbot]
                )

        # 4-2. CV: 과일 사진 맞추기
        with gr.Tab("과일 사진 맞추기"):
            gr.Markdown("과일 사진을 올리면 어떤 과일인지 맞춰줄게요.")

            with gr.Row():
                image_input = gr.Image(
                    label="과일 사진을 올려 주세요.",
                    type="numpy"
                )
                field_image = gr.Image(
                    label="과일이 자라는 모습 (예시)",
                    type="filepath"
                )

            info_text = gr.Markdown("과일을 분석한 결과가 여기 나와요.")
            long_explain = gr.Textbox(
                label="과일 선생님 이야기",
                lines=6
            )

            analyze_btn = gr.Button("이 과일은 누구일까요?")
            explain_btn = gr.Button("이 과일 자세히 설명해 주세요")

            cv_state = gr.State(value={"fruit_eng": None, "base_description": ""})

            analyze_btn.click(
                fn=analyze_image,
                inputs=image_input,
                outputs=[info_text, field_image, long_explain, cv_state]
            )

            explain_btn.click(
                fn=explain_from_cv,
                inputs=cv_state,
                outputs=long_explain
            )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch()