from streamlit.testing.v1 import AppTest

FOLLOWUP_APP = """
import streamlit as st

from adl_smart_search.ui.dialog import get_chat_messages, send_followup, set_chat_messages


class FakeProvider:
    def chat(self, dialog):
        return "follow-up answer"


if "t_chat_messages" not in st.session_state:
    set_chat_messages(
        "t",
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ],
    )
    st.session_state["t_dialog_messages"] = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]
    st.session_state["t_rag_context"] = "context"
    st.session_state["t_rag_query"] = "context"
    st.session_state["t_rag_selection_key"] = ""

for i, message in enumerate(get_chat_messages("t")):
    with st.container(key=f"t_chatmsg_{i}_{message['role']}"):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

st.text_input("question", key="t_question")

if st.button("send"):
    send_followup(
        prefix="t",
        question=st.session_state["t_question"] or "context follow-up",
        stale_warning_text="stale",
        llm_setup_msg="",
        llm_provider=FakeProvider(),
        prompts={
            "system_prompt": "test system prompt",
            "answer_user_prompt": "КОНТЕКСТ:\\n{context}\\nВОПРОС:\\n{question}",
            "continuation_prompt": "Продолжи.",
            "query_expand": "",
            "translate": "",
        },
    )
"""


def test_followup_does_not_render_history_twice():
    at = AppTest.from_string(FOLLOWUP_APP, default_timeout=30).run()
    assert [m.name for m in at.chat_message] == ["user", "assistant"]

    at.text_input[0].set_value("context follow-up")
    at.button[0].click().run()

    state = at.session_state["t_chat_messages"]
    assert [m["role"] for m in state] == ["user", "assistant", "user", "assistant"]
    # В DOM ровно столько же сообщений: история не дублируется при отправке уточнения.
    assert [m.name for m in at.chat_message] == ["user", "assistant", "user", "assistant"]
    assert [m.markdown[0].value for m in at.chat_message] == [
        "q1",
        "a1",
        "context follow-up",
        "follow-up answer",
    ]


def test_followup_appends_to_api_dialog_once():
    at = AppTest.from_string(FOLLOWUP_APP, default_timeout=30).run()
    at.text_input[0].set_value("context follow-up")
    at.button[0].click().run()

    dialog = at.session_state["t_dialog_messages"]
    assert [m["role"] for m in dialog] == ["user", "assistant", "user", "assistant"]
    assert dialog[-2]["content"] == "context follow-up"
    assert dialog[-1]["content"] == "follow-up answer"
