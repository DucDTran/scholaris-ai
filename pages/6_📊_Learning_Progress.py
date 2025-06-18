import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="User Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- FUNCTIONS ---
def load_user_scores(user_path):
    """Loads the user's scores from a JSON file."""
    scores_file = os.path.join(user_path, "scores.json")
    if os.path.exists(scores_file):
        try:
            with open(scores_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def generate_calendar_heatmap(df):
    """Generates an HTML/CSS string for a GitHub-style calendar heatmap based on daily average scores."""
    today = datetime.now().date()
    start_date = today - timedelta(days=365)
    
    # Create a DataFrame of all days in the last year
    all_days = pd.to_datetime(pd.date_range(start=start_date, end=today, freq='D'))
    calendar_df = pd.DataFrame({'date': all_days})
    
    # Calculate average percentage score per day
    if not df.empty and 'timestamp' in df.columns:
        # Step 1: Create a new column 'activity_date' containing only the date part of the 'timestamp'.
        # This is the crucial step to group by day, not by the specific time.
        df['activity_date'] = pd.to_datetime(df['timestamp']).dt.normalize()
        
        # Step 2: Group by the new 'activity_date' column and calculate the mean of 'percentage' for each day.
        daily_avg_scores = df.groupby('activity_date')['percentage'].mean().reset_index()

        # Step 3: Rename the column to 'date' to prepare for merging with the main calendar DataFrame.
        daily_avg_scores = daily_avg_scores.rename(columns={'activity_date': 'date'})

        # Step 4: Merge the daily average scores into the main calendar.
        # The merge is done on the 'date' column, ensuring scores are aligned with the correct day.
        calendar_df = pd.merge(calendar_df, daily_avg_scores, on='date', how='left').fillna(0)
    else:
        calendar_df['percentage'] = 0

    # Determine color based on average score
    def get_color(score):
        score = int(score)
        if score == 0:
            return "#f0f2f5"  # Grey for no activity
        elif score < 50:
            return "#FFD700"  # Light orange for low scores
        elif score < 75:
            return "#EDBE10"  # Light green
        elif score < 90:
            return "#E3B218"  # Medium green
        else:
            return "#DAA520"  # Dark green for high scores

    calendar_df['color'] = calendar_df['percentage'].apply(get_color)
    
    # HTML and CSS for the calendar
    html = """
    <style>
        .calendar-grid { display: grid; grid-template-columns: repeat(48, 16px); grid-template-rows: repeat(7, 16px); grid-gap: 4px; }
        .day-cell { width: 16px; height: 16px; border-radius: 2px; }
        .day-cell:hover .tooltip-text { visibility: visible; }
        .tooltip { position: relative; display: inline-block; }
        .tooltip .tooltip-text {
            visibility: hidden; width: 180px; background-color: #555; color: #fff; text-align: center;
            border-radius: 6px; padding: 5px 0; position: absolute; z-index: 1; bottom: 125%; left: 50%;
            margin-left: -90px; opacity: 0; transition: opacity 0.3s;
        }
        .tooltip:hover .tooltip-text { font-size: 12px; visibility: visible; opacity: 1; }
    </style>
    <h3>Learning Performance (Last 365 days)</h3>
    <div class="calendar-grid">
    """
    
    # Create the grid cells
    first_day_of_year = calendar_df['date'].iloc[0]
    html += '<div></div>' * first_day_of_year.weekday()

    for _, row in calendar_df.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        # MODIFICATION: Use average percentage for the tooltip
        avg_score = row['percentage']
        if avg_score > 0:
            tooltip = f"Average Score: {avg_score:.2f}% on {date_str}"
        else:
            tooltip = f"No activity on {date_str}"
            
        html += f'<div class="tooltip"><div class="day-cell" style="background-color: {row["color"]};"></div><span class="tooltip-text">{tooltip}</span></div>'
        
    html += "</div>"

    # Add a legend of the colors, including No activity, Low score, Medium score, High score
    html += """
    <div style="display: flex; justify-content: left; margin-top: 20px; font-size: 12px;">
        <div style="display: flex; align-items: center; margin-right: 1px;">
            <span>No activity</span>
            <div style="width: 16px; height: 16px; background-color: #f0f2f5; border-radius: 2px; margin-right: 5px; margin-left: 5px;"></div>
        </div>
        <div style="display: flex; align-items: center; margin-right: 1px;">
            <div style="width: 16px; height: 16px; background-color: #FFD700; border-radius: 2px; margin-right: 5px;"></div>
        </div>
        <div style="display: flex; align-items: center; margin-right: 1px;">
            <div style="width: 16px; height: 16px; background-color: #E3B218; border-radius: 2px; margin-right: 5px;"></div>
        </div>
        <div style="display: flex; align-items: center; margin-right: 1px;">
            <div style="width: 16px; height: 16px; background-color: #DAA520; border-radius: 2px; margin-right: 5px;"></div>
            <span>High score</span>
        </div>
    """
    return html, calendar_df

def calculate_streaks(df):
    if df.empty:
        return 0, 0
    
    # Use normalize() to keep the column as a datetime64 dtype, which is more robust.
    df['date_col'] = pd.to_datetime(df['timestamp']).dt.normalize()
    df = df.drop_duplicates(subset=['date_col']).sort_values('date_col')
    
    # This calculation is now more reliable as it operates on a proper timedelta object.
    df['date_diff'] = df['date_col'].diff().dt.days.fillna(1)
    
    # Calculate streaks
    df['streak_id'] = (df['date_diff'] > 1).cumsum()
    streaks = df.groupby('streak_id').size()
    
    longest_streak = streaks.max() if not streaks.empty else 0
    
    # Calculate current streak
    today = datetime.now().date()
    if df.empty:
        return 0, 0
        
    # Extract the date part for comparison after all calculations are done.
    last_day = df['date_col'].iloc[-1].date()
    
    current_streak = 0
    if not streaks.empty and (today - last_day).days <= 1:
        current_streak = streaks.iloc[-1]

    return longest_streak, current_streak


# --- UI & LOGIC ---
st.title("📊 Learning Progress")
st.markdown("View your learning progress over time.")

if "user_name" not in st.session_state or not st.session_state.get("user_name"):
    st.warning("Please login to continue.", icon="⚠️")
    st.stop()

user_name = st.session_state.user_name
user_id = st.session_state.user_id
user_data_path = os.path.join("user_data", user_id)
scores = load_user_scores(user_data_path)

st.sidebar.write(f"Welcome, **{user_name}**!")

if not scores:
    st.info("You haven't completed any exercises yet. Complete a quiz or other exercise to see your progress here!")
    st.stop()

df = pd.DataFrame(scores)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# --- Display GitHub-like Calendar ---
html, calendar_df = generate_calendar_heatmap(df.copy())
st.markdown(html, unsafe_allow_html=True)

# --- Display Metrics ---
st.markdown("---")
st.markdown("### Overall Performance")
col1, col2, col3, col4 = st.columns(4)

total_exercises = len(df)
col1.metric("Total Exercises Completed", total_exercises, border=True)

avg_score = df['percentage'].mean()
col2.metric("Average Score", f"{avg_score:.2f}%", border=True)

longest_streak, current_streak = calculate_streaks(df.copy())
col3.metric("Longest Streak", f"{longest_streak} days", border=True)
col4.metric("Current Streak", f"{current_streak} days", border=True)
st.markdown("---")

# # --- Display Chart ---
st.markdown("### Progress Over Time")
st.line_chart(df, x='timestamp', y='percentage', color='type')

# --- Display Score History Table ---
st.markdown("---")
st.markdown("### Detailed History")
display_df = df[['timestamp', 'type', 'score', 'total', 'percentage']].sort_values(by='timestamp', ascending=False)
display_df.columns = display_df.columns.str.capitalize()
st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)
