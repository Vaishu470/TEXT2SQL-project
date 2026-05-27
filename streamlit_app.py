import json
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from text2sql_agent import (
    AgentState,
    guardrails_agent,
    sql_agent,
    execute_sql,
    error_agent,
    analysis_agent,
    decide_graph_need,
    viz_agent,
)


def run_text2sql_pipeline(question: str) -> AgentState:
    state = AgentState(
        question=question,
        sql_query="",
        query_result="",
        final_answer="",
        error="",
        iteration=0,
        needs_graph=False,
        graph_type="",
        graph_json="",
        is_in_scope=True,
    )

    state = guardrails_agent(state)
    if not state["is_in_scope"]:
        return state

    state = sql_agent(state)
    state = execute_sql(state)

    while state.get("error") and state.get("iteration", 0) <= 3:
        state = error_agent(state)
        state = execute_sql(state)

    state = analysis_agent(state)
    state = decide_graph_need(state)
    if state.get("needs_graph"):
        state = viz_agent(state)

    return state


def render_graph(graph_json: str = "", fallback_fig=None):
    if graph_json:
        try:
            fig = pio.from_json(graph_json)
            st.plotly_chart(fig, use_container_width=True)
            return
        except Exception as exc:
            st.warning(f"Could not render visualization from generated graph: {exc}")

    if fallback_fig is not None:
        st.plotly_chart(fallback_fig, use_container_width=True)
        return

    st.info("No visualization available for this query.")


def create_default_chart(query_result: str):
    try:
        result = json.loads(query_result)
    except Exception:
        return None

    if isinstance(result, dict):
        if "query_1" in result and isinstance(result["query_1"], list):
            result = result["query_1"]
        elif len(result) == 1 and isinstance(next(iter(result.values())), list):
            result = next(iter(result.values()))

    if not isinstance(result, list) or len(result) == 0:
        return None

    df = pd.DataFrame(result)
    if df.empty:
        return None

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude="number").columns.tolist()

    if numeric_cols and non_numeric_cols:
        x = non_numeric_cols[0]
        y = numeric_cols[0]
        fig = px.bar(df.sort_values(y, ascending=False).head(20), x=x, y=y, title="Query visualization")
    elif len(numeric_cols) >= 2:
        fig = px.line(df.head(50), x=numeric_cols[0], y=numeric_cols[1], title="Query visualization")
    elif numeric_cols:
        fig = px.bar(df.reset_index().head(20), x=df.index.astype(str), y=numeric_cols[0], title="Query visualization")
    else:
        return None

    fig.update_layout(margin={"t": 40, "b": 40, "l": 20, "r": 20})
    return fig


def main():
    st.set_page_config(page_title="E-commerce Text2SQL Assistant", layout="wide")

    st.title("E-commerce Text2SQL Assistant")
    st.markdown(
        "Ask natural-language questions about the e-commerce database and get SQL, results, and a clear answer."
    )

    question = st.text_input("Ask your question", key="question_input")
    if st.button("Run Query") and question.strip():
        with st.spinner("Generating SQL and analyzing results..."):
            state = run_text2sql_pipeline(question.strip())

        if state.get("error"):
            st.error(state["error"])

        if state.get("sql_query"):
            st.subheader("Generated SQL Query")
            st.code(state["sql_query"], language="sql")

        if state.get("query_result"):
            st.subheader("Query Result")
            try:
                parsed = json.loads(state["query_result"])
                st.json(parsed)
            except Exception:
                st.text(state["query_result"])

        if state.get("final_answer"):
            st.subheader("Answer")
            st.write(state["final_answer"])

        fallback_fig = None
        if not state.get("graph_json") and state.get("query_result") and not state.get("error"):
            fallback_fig = create_default_chart(state["query_result"])

        if state.get("graph_json") or fallback_fig is not None:
            st.subheader("Visualization")
            render_graph(state.get("graph_json", ""), fallback_fig)

    st.markdown("---")
    st.markdown(
        "**Example questions:**\n"
        "- How many orders were delivered?\n"
        "- What are the top 5 product categories by sales?\n"
        "- Show me orders from São Paulo\n"
        "- What's the average review score?\n"
        "- Which sellers have the most orders?"
    )


if __name__ == "__main__":
    main()
