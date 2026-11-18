from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model_engine import (TOPICS, REQUIRED_COLUMNS, generate_demo_data, validate_and_prepare,
                          train_and_evaluate, predict_next_accuracy)
import storage

st.set_page_config(page_title="CodeMentor AI", page_icon="💻", layout="wide")
st.title("💻 CodeMentor AI")
st.caption("Personalized coding performance analysis, prediction and adaptive learning recommendations")
st.markdown("""<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: rgba(128,128,128,.08); padding: 14px; border-radius: 12px;}
</style>""", unsafe_allow_html=True)

storage.init_db()
DEMO_PATH = Path(__file__).parent / "data" / "demo_coding_practice.csv"

@st.cache_data(show_spinner=False)
def get_demo():
    if DEMO_PATH.exists():
        return validate_and_prepare(pd.read_csv(DEMO_PATH))
    return generate_demo_data()


def exercise_for(topic, level):
    bank = {
        "Loops": {"Easy": "Print a multiplication table", "Medium": "Print pattern triangles", "Hard": "Generate the Collatz sequence"},
        "Arrays": {"Easy": "Find the maximum element", "Medium": "Solve Two Sum", "Hard": "Find the maximum subarray sum"},
        "Strings": {"Easy": "Reverse a string", "Medium": "Check for a valid anagram", "Hard": "Find the longest palindromic substring"},
        "Sorting": {"Easy": "Implement bubble sort", "Medium": "Merge two sorted arrays", "Hard": "Implement quick sort"},
        "Linked Lists": {"Easy": "Traverse and print a linked list", "Medium": "Reverse a linked list", "Hard": "Detect a cycle in a linked list"},
        "Recursion": {"Easy": "Calculate factorial recursively", "Medium": "Generate all subsets", "Hard": "Solve the N-Queens problem"},
        "Trees": {"Easy": "Implement inorder traversal", "Medium": "Find the height of a binary tree", "Hard": "Find the lowest common ancestor"},
        "Dynamic Programming": {"Easy": "Fibonacci with memoization", "Medium": "Solve 0/1 Knapsack", "Hard": "Longest common subsequence"},
    }
    return bank.get(topic, {}).get(level, f"Practice a {level.lower()} {topic} problem")


def build_recommendations(history, model=None, weak_threshold=65):
    prereq = {"Arrays": ["Loops"], "Strings": ["Loops"], "Sorting": ["Arrays"],
              "Linked Lists": ["Arrays"], "Recursion": ["Loops"],
              "Trees": ["Recursion", "Linked Lists"], "Dynamic Programming": ["Recursion"]}
    rows = []
    for topic in sorted(history.topic.unique()):
        h = history[history.topic == topic].sort_values("week")
        recent = float(h.accuracy.tail(3).mean())
        predicted = predict_next_accuracy(model, history, topic) if model is not None else recent
        expected = .6 * recent + .4 * predicted
        rows.append({"Topic": topic, "Recent accuracy": recent, "Predicted next": predicted, "Expected": expected})
    if not rows:
        return pd.DataFrame()
    profile = pd.DataFrame(rows).set_index("Topic")
    selected = []
    for topic, row in profile.sort_values("Expected").iterrows():
        target = topic
        for pre in prereq.get(topic, []):
            if pre in profile.index and profile.loc[pre, "Expected"] < weak_threshold:
                target = pre
                break
        if target in [x["Topic"] for x in selected]:
            continue
        score = float(profile.loc[target, "Expected"])
        level = "Easy" if score < 40 else "Medium" if score < weak_threshold else "Hard"
        target_history = history[history.topic == target].sort_values("week")
        latest = target_history.iloc[-1]
        hint_ratio = float(latest.hints_used / max(latest.attempts, 1))
        if score < 40:
            adjustment = "Revisit the fundamentals, then solve two easy problems without a timer."
        elif hint_ratio >= 0.35:
            adjustment = "Try the first 10 minutes independently before using a hint."
        elif float(latest.avg_time_min) >= 30:
            adjustment = "Use short timed practice; focus on reducing solution time without sacrificing correctness."
        else:
            adjustment = "After three correct solutions, move up one difficulty level."
        selected.append({"Priority": len(selected)+1, "Topic": target,
                         "Expected accuracy (%)": round(score, 1), "Level": level,
                         "Suggested exercise": exercise_for(target, level),
                         "Personalized adjustment": adjustment,
                         "Reason": f"Prerequisite for {topic}" if target != topic else "Lowest recent/predicted performance"})
        if len(selected) == 5:
            break
    return pd.DataFrame(selected)


# Sidebar: data source and navigation
st.sidebar.header("Workspace")
page = st.sidebar.radio("Go to", ["Student dashboard", "Practice session", "Learning recommendations", "Model laboratory", "Dataset manager"])
data_mode = st.sidebar.radio("Model training data", ["Demo dataset (synthetic)", "Uploaded dataset", "Saved practice logs"], help="The demo dataset is simulated. Use uploaded or saved records for real-data experiments.")

if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "last_import_name" not in st.session_state:
    st.session_state.last_import_name = None

if data_mode == "Demo dataset (synthetic)":
    model_data = get_demo()
elif data_mode == "Uploaded dataset":
    model_data = st.session_state.uploaded_df
else:
    logs = storage.get_logs()
    model_data = validate_and_prepare(logs) if not logs.empty else None

# Student selection
students = storage.list_students()
if not students.empty:
    labels = {r.student_id: f"{r.display_name} ({r.student_id})" for r in students.itertuples()}
    selected_id = st.sidebar.selectbox("Student profile", students.student_id.tolist(), format_func=lambda x: labels[x])
else:
    selected_id = None

@st.cache_data(show_spinner="Training and comparing machine learning models...")
def cached_train(data):
    return train_and_evaluate(data)

artifacts = None
if model_data is not None and len(model_data) >= 40 and model_data.week.nunique() >= 5:
    try:
        artifacts = cached_train(model_data)
    except Exception as exc:
        st.sidebar.warning(f"Model training unavailable: {exc}")

if page == "Dataset manager":
    st.header("Dataset manager")
    st.write("Upload a CSV containing practice-session records. Accuracy is recalculated from solved / attempted questions to keep the target consistent.")
    st.code("student_id, week, topic, attempts, problems_solved, hints_used, avg_time_min", language="text")
    file = st.file_uploader("Upload coding practice CSV", type=["csv"])
    if file:
        try:
            preview = validate_and_prepare(pd.read_csv(file))
            st.success(f"Validated {len(preview):,} rows across {preview.student_id.nunique()} students and {preview.topic.nunique()} topics.")
            st.dataframe(preview.head(20), use_container_width=True)
            st.caption("Note: an uploaded file is available for model training during this session. Importing it also appends its records to the local SQLite database.")
            if st.button("Use uploaded dataset for model training", type="primary"):
                st.session_state.uploaded_df = preview
                st.session_state.last_import_name = file.name
                st.success("Dataset selected. Choose 'Uploaded dataset' in the sidebar to train on it.")
            if st.button("Import records into saved database"):
                storage.import_logs(preview)
                st.success("Records imported into the local database.")
        except Exception as exc:
            st.error(f"Could not validate this file: {exc}")
    st.subheader("Current model-training data")
    if model_data is None:
        st.info("No dataset selected or available. Upload a CSV or choose the demo dataset in the sidebar.")
    else:
        st.metric("Records", f"{len(model_data):,}")
        st.write({"Students": int(model_data.student_id.nunique()), "Topics": int(model_data.topic.nunique()),
                  "Weeks": int(model_data.week.nunique()), "Missing values": int(model_data.isna().sum().sum())})
        st.dataframe(model_data.head(10), use_container_width=True)

elif page == "Practice session":
    st.header("Log a coding practice session")
    st.write("Create a profile or log a new session. Accuracy is calculated automatically from your attempts and solved questions.")
    with st.expander("Create a student profile", expanded=selected_id is None):
        with st.form("create_student"):
            display_name = st.text_input("Your name")
            new_id = st.text_input("Choose a unique student ID", placeholder="e.g. student_001")
            create = st.form_submit_button("Create profile")
            if create:
                if not display_name.strip() or not new_id.strip():
                    st.error("Enter both a name and a student ID.")
                else:
                    try:
                        storage.create_student(new_id.strip(), display_name.strip())
                        st.success("Profile created. Refresh the page to select it.")
                        st.rerun()
                    except Exception:
                        st.error("That student ID already exists. Choose another one.")
    if selected_id:
        existing = storage.get_logs(selected_id)
        suggested_week = int(existing.week.max()+1) if not existing.empty else 1
        with st.form("practice_log"):
            a, b, c = st.columns(3)
            week = a.number_input("Practice week", min_value=1, value=suggested_week, step=1)
            topic_choice = b.selectbox("Topic", TOPICS + ["Other / custom"])
            custom_topic = st.text_input("Custom topic name") if topic_choice == "Other / custom" else ""
            topic = custom_topic.strip() if topic_choice == "Other / custom" else topic_choice
            attempts = c.number_input("Problems attempted", min_value=1, max_value=500, value=5)
            solved = a.number_input("Problems solved", min_value=0, max_value=int(attempts), value=3)
            hints = b.number_input("Hints used", min_value=0, max_value=int(attempts), value=1)
            avg_time = c.number_input("Average time per problem (minutes)", min_value=.1, max_value=600.0, value=20.0, step=.5)
            submit = st.form_submit_button("Save practice session", type="primary")
            if submit:
                if not topic.strip():
                    st.error("Enter a topic name.")
                elif solved > attempts or hints > attempts:
                    st.error("Solved questions and hints cannot exceed attempted questions.")
                else:
                    storage.add_practice(selected_id, int(week), topic, int(attempts), int(solved), int(hints), float(avg_time))
                    st.success(f"Saved. Session accuracy: {solved/attempts*100:.1f}%")
                    st.rerun()
        st.subheader("Your saved practice history")
        if not existing.empty:
            existing["accuracy"] = existing.problems_solved / existing.attempts * 100
            st.dataframe(existing.sort_values("week", ascending=False), use_container_width=True)
            if artifacts is not None:
                model_history = validate_and_prepare(storage.get_logs(selected_id))
                latest_topic = model_history.sort_values("week").iloc[-1]["topic"]
                predicted = predict_next_accuracy(artifacts["best_regressor"], model_history, latest_topic)
                st.subheader("Your latest session forecast")
                st.metric(f"Predicted next-week accuracy · {latest_topic}", f"{predicted:.1f}%")
                st.caption("This forecast uses the selected model-training dataset. Treat it as an estimate, especially when your own history is small or differs from the training data.")
        else:
            st.info("No sessions yet. Save your first session above.")
    else:
        st.info("Create a student profile to start logging practice sessions.")

elif page == "Student dashboard":
    st.header("Student performance dashboard")
    if selected_id is None:
        st.info("Create a student profile in Practice session to start tracking personal progress.")
    else:
        personal = storage.get_logs(selected_id)
        if personal.empty:
            st.info("No personal practice records yet. Add a session in Practice session, or import a dataset containing this student ID.")
        else:
            personal["accuracy"] = 100 * personal.problems_solved / personal.attempts
            recent = personal.sort_values("week").groupby("topic").tail(3).groupby("topic").accuracy.mean()
            c1,c2,c3 = st.columns(3)
            c1.metric("Practice sessions", len(personal))
            c2.metric("Overall accuracy", f"{personal.accuracy.mean():.1f}%")
            c3.metric("Topics practised", personal.topic.nunique())
            left,right = st.columns(2)
            with left:
                fig=px.bar(recent.sort_values(), orientation="h", labels={"value":"Recent accuracy (%)","topic":"Topic"}, title="Recent accuracy by topic")
                st.plotly_chart(fig,use_container_width=True)
            with right:
                trend=personal.groupby("week").accuracy.mean().reset_index()
                fig=px.line(trend,x="week",y="accuracy",markers=True,title="Accuracy trend",labels={"accuracy":"Accuracy (%)"})
                st.plotly_chart(fig,use_container_width=True)
            st.subheader("Recent sessions")
            st.dataframe(personal.sort_values("week",ascending=False),use_container_width=True)

elif page == "Learning recommendations":
    st.header("Adaptive learning recommendations")
    if selected_id is None:
        st.info("Create a student profile and log practice sessions first.")
    else:
        personal = storage.get_logs(selected_id)
        if personal.empty:
            st.info("Log practice sessions before generating personalized recommendations.")
        else:
            personal = validate_and_prepare(personal)
            threshold=st.slider("Weak performance threshold (%)",40,85,65,5)
            personal_model = artifacts["best_regressor"] if artifacts else None
            recs=build_recommendations(personal,personal_model,threshold)
            if artifacts is None:
                st.warning("A trained next-week prediction model is not available for the selected dataset. Recommendations currently use recent performance only.")
            st.dataframe(recs,use_container_width=True,hide_index=True)
            st.caption("Recommendations prioritize lower-performing topics and check prerequisite skills. Difficulty is selected from the student's current expected performance.")

elif page == "Model laboratory":
    st.header("Model laboratory and honest evaluation")
    if model_data is None:
        st.info("Choose a dataset with at least five distinct weeks in the sidebar.")
    elif artifacts is None:
        st.warning("Not enough data to train. Aim for multiple students, multiple topics and at least five weeks of observations.")
    else:
        st.caption(f"Chronological holdout: training on earlier weeks; final test begins at week {artifacts['test_start_week']}. No random row split.")
        st.subheader("Classification: next-week performance category")
        st.caption("The classifier now predicts the NEXT week's Weak / Average / Strong category using only information available at the end of the current week. This is a forecasting task, not classification of a completed session.")
        st.write(f"Selected model by chronological validation accuracy (macro F1 used as a tie-breaker): **{artifacts['best_classifier_name']}**")
        st.caption(f"Model selection configuration: {artifacts['optimized_classifier_params']}")
        st.dataframe(artifacts["classification_results"].style.format({c:"{:.3f}" for c in ["Validation_Accuracy","Validation_F1_macro","Test_Accuracy","Test_Precision_macro","Test_Recall_macro","Test_F1_macro"]}),use_container_width=True,hide_index=True)
        cm=artifacts["confusion_matrix"]
        fig=px.imshow(cm,text_auto=True,x=["Weak","Average","Strong"],y=["Weak","Average","Strong"],labels={"x":"Predicted","y":"Actual","color":"Count"},title="Confusion matrix for selected classifier")
        st.plotly_chart(fig,use_container_width=True)
        st.subheader("Regression: next-week accuracy prediction")
        st.caption("Regression uses current-week and historical features to forecast the immediately following week's accuracy. Gaps in a student's weekly history are not treated as adjacent weeks.")
        st.write(f"Selected model by lowest MAE: **{artifacts['best_regressor_name']}**")
        st.dataframe(artifacts["regression_results"].style.format({c:"{:.3f}" for c in ["Validation_MAE","Test_MAE","Test_RMSE","Test_R2"]}),use_container_width=True,hide_index=True)
        actual=artifacts["regression_test_actual"]
        pred=artifacts["regression_test_pred"]
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=actual,y=pred,mode="markers",name="Predictions",opacity=.5))
        fig.add_trace(go.Scatter(x=[0,100],y=[0,100],mode="lines",name="Perfect prediction",line=dict(dash="dash")))
        fig.update_layout(title="Actual vs predicted next-week accuracy",xaxis_title="Actual accuracy (%)",yaxis_title="Predicted accuracy (%)")
        st.plotly_chart(fig,use_container_width=True)
        st.info("Accuracy is not the only quality measure. For imbalanced classes, macro precision, recall and F1 reveal whether minority categories are being missed. Regression MAE is in accuracy percentage points.")
        st.download_button("Download classification metrics",artifacts["classification_results"].to_csv(index=False),"classification_metrics.csv","text/csv")
        st.download_button("Download regression metrics",artifacts["regression_results"].to_csv(index=False),"regression_metrics.csv","text/csv")

st.sidebar.divider()
st.sidebar.caption("CodeMentor AI · Local prototype. Student records are stored in a local SQLite file on this computer.")
