from streamlit.testing.v1 import AppTest

PICK_ALL_APP = """
import streamlit as st

from adl_smart_search.ui.search_tab import _apply_pick_all, _sync_pick_all

prefix = "t"
keys = ["a", "b", "c"]
if "t_pick_keys" not in st.session_state:
    st.session_state["t_pick_keys"] = keys
    for key in keys:
        st.session_state[key] = True
    st.session_state["t_pick_all"] = True

st.session_state["t_pick_keys"] = keys
all_selected = all(bool(st.session_state.get(k)) for k in keys)
st.checkbox("all", value=all_selected, key="t_pick_all", on_change=_apply_pick_all, args=(prefix,))
for key in keys:
    st.checkbox(
        key,
        key=key,
        value=bool(st.session_state.get(key, False)),
        on_change=_sync_pick_all,
        args=(prefix,),
    )
"""


def _make_app() -> AppTest:
    return AppTest.from_string(PICK_ALL_APP, default_timeout=30).run()


def test_unchecking_one_result_clears_pick_all():
    at = _make_app()
    assert at.checkbox[0].value is True

    at.checkbox[2].uncheck().run()

    assert at.checkbox[0].value is False
    assert at.session_state["t_pick_all"] is False
    assert at.checkbox[2].value is False


def test_checking_last_result_sets_pick_all():
    at = _make_app()
    at.checkbox[2].uncheck().run()
    assert at.checkbox[0].value is False

    at.checkbox[2].check().run()

    assert at.checkbox[0].value is True
    assert at.session_state["t_pick_all"] is True


def test_pick_all_toggles_every_result():
    at = _make_app()

    at.checkbox[0].uncheck().run()
    assert [c.value for c in at.checkbox] == [False, False, False, False]

    at.checkbox[0].check().run()
    assert [c.value for c in at.checkbox] == [True, True, True, True]
