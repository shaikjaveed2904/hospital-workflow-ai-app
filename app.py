"""
app.py
======

Main Streamlit application for the hospital workflow optimization demo.  This
dashboard showcases how AI and analytics can be used to improve hospital
operations by forecasting patient flow, predicting wait times, balancing
doctor workloads and identifying no-show risks.  Users can explore metrics,
book appointments, view analytics and receive intelligent recommendations.
"""

from __future__ import annotations

import datetime
import math
from typing import List, Dict

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from model_utils import (
    train_flow_model,
    predict_flow,
    train_wait_time_model,
    predict_wait_time,
    train_workload_model,
    predict_workload,
    train_no_show_model,
    predict_no_show_probability,
)


@st.cache_data
def load_data(filepath: str = "data/hospital_data.csv") -> pd.DataFrame:
    """Load appointment data from a CSV file with caching.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the appointment data.
    """
    df = pd.read_csv(filepath)
    return df


@st.cache_resource
def train_models(df: pd.DataFrame) -> Dict[str, object]:
    """Train all predictive models and cache them.

    Parameters
    ----------
    df : pd.DataFrame
        The appointment data.

    Returns
    -------
    Dict[str, object]
        Dictionary containing trained models.
    """
    models: Dict[str, object] = {}
    models["flow"] = train_flow_model(df)
    models["wait"] = train_wait_time_model(df)
    models["workload"] = train_workload_model(df)
    models["noshow"] = train_no_show_model(df)
    return models


def compute_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Compute key performance indicators from appointment data.

    Parameters
    ----------
    df : pd.DataFrame
        Appointment data.

    Returns
    -------
    Dict[str, float]
        Dictionary of KPI values.
    """
    total_appointments = len(df)
    avg_wait = df["wait_time_minutes"].mean()
    # Doctor utilization: total consultation duration / (number of appointments * average available time per appointment)
    avg_consult_duration = df["consultation_duration"].mean()
    # assume each slot is 15 minutes; utilization = (avg duration / 15 min)
    doctor_utilization = min(1.0, avg_consult_duration / 15.0)
    no_show_rate = df["no_show"].mean()
    completion_rate = (df["status"].eq("completed").sum()) / total_appointments
    # Peak traffic hour
    df_time = pd.to_datetime(df["appointment_date"] + " " + df["appointment_time"])
    peak_hour = df_time.dt.hour.mode()[0]
    # Busiest department by number of appointments
    busiest_department = df["department"].value_counts().idxmax()

    return {
        "total_appointments": total_appointments,
        "average_wait_time": avg_wait,
        "doctor_utilization": doctor_utilization,
        "no_show_rate": no_show_rate,
        "completion_rate": completion_rate,
        "peak_traffic_hour": peak_hour,
        "busiest_department": busiest_department,
    }


def display_home(df: pd.DataFrame, models: Dict[str, object]) -> None:
    """Render the home dashboard with KPI cards.

    Parameters
    ----------
    df : pd.DataFrame
        Appointment data.
    """
    st.markdown("## 📊 Home Dashboard")
    kpis = compute_kpis(df)
    # Display KPI cards in columns
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Appointments", f"{kpis['total_appointments']}")
    col1.metric("No‑Show Rate", f"{kpis['no_show_rate'] * 100:.1f}%")
    col1.metric("Completion Rate", f"{kpis['completion_rate'] * 100:.1f}%")

    col2.metric("Average Wait Time", f"{kpis['average_wait_time']:.1f} min")
    col2.metric("Doctor Utilization", f"{kpis['doctor_utilization'] * 100:.1f}%")
    col2.metric("Peak Traffic Hour", f"{int(kpis['peak_traffic_hour']):02d}:00")

    col3.metric("Busiest Dept.", kpis['busiest_department'])
    # Additional description
    st.markdown(
        "The dashboard provides an overview of hospital performance. "
        "High no‑show rates may indicate scheduling issues or patient reminders needs. "
        "Doctor utilization close to 100% suggests efficient use of physician time, while "
        "long wait times highlight congestion during peak hours."
    )


def display_booking(df: pd.DataFrame, models: Dict[str, object]) -> None:
    """Render the appointment booking module.

    Parameters
    ----------
    df : pd.DataFrame
        Appointment data.
    models : Dict[str, object]
        Dictionary of trained models.
    """
    st.markdown("## 📅 Book Appointment")

    # Department selection
    departments = sorted(df["department"].unique().tolist())
    selected_department = st.selectbox("Select Department", departments)

    # Doctor selection filtered by department
    doctors = sorted(df[df["department"] == selected_department]["doctor_name"].unique().tolist())
    selected_doctor = st.selectbox("Select Doctor", doctors)

    # Date input (allow scheduling within the next 30 days)
    today = datetime.date.today()
    selected_date = st.date_input(
        "Choose Date",
        min_value=today,
        max_value=today + datetime.timedelta(days=30)
    )

    # Create available time slots (every 15 minutes between 8:00 and 16:45)
    time_slots = [
        (datetime.time(hour, minute).strftime("%H:%M"))
        for hour in range(8, 17)
        for minute in (0, 15, 30, 45)
    ]

    # Remove already booked slots for that doctor on the selected date
    existing = df[(df["doctor_name"] == selected_doctor) & (df["appointment_date"] == selected_date.isoformat())]
    booked_slots = set(existing["appointment_time"].tolist())
    available_slots = [slot for slot in time_slots if slot not in booked_slots]

    if not available_slots:
        st.warning("No available slots for the selected date. Please choose another date.")
        return

    selected_time = st.selectbox("Choose Time Slot", available_slots)

    # Predict wait time using the model
    # Determine day-of-week and hour
    appt_dt = datetime.datetime.combine(selected_date, datetime.datetime.strptime(selected_time, "%H:%M").time())
    day_of_week = appt_dt.weekday()
    hour = appt_dt.hour
    # Use the patient priority; allow user to select
    priority = st.selectbox("Patient Priority", ["normal", "urgent"])
    predicted_wait = predict_wait_time(
        models["wait"],
        hour=hour,
        day_of_week=day_of_week,
        department=selected_department,
        doctor_name=selected_doctor,
        patient_priority=priority,
    )

    st.info(f"Predicted wait time for this slot: **{predicted_wait:.1f} minutes**")

    # Confirm booking button
    if st.button("Book Appointment"):
        # Append appointment to session state or notify user
        if "appointments" not in st.session_state:
            st.session_state.appointments = []  # type: ignore[attr-defined]
        st.session_state.appointments.append({
            "patient_id": f"NewPatient-{len(st.session_state.appointments) + 1}",
            "appointment_date": selected_date.isoformat(),
            "appointment_time": selected_time,
            "department": selected_department,
            "doctor_name": selected_doctor,
            "doctor_specialty": df[df["doctor_name"] == selected_doctor]["doctor_specialty"].iloc[0],
            "booked_slot": appt_dt.strftime("%Y-%m-%d %H:%M"),
            "actual_arrival_time": "",
            "wait_time_minutes": math.ceil(predicted_wait),
            "consultation_duration": 0,
            "no_show": 0,
            "status": "scheduled",
            "patient_priority": priority,
            "hospital_unit": df[df["doctor_name"] == selected_doctor]["hospital_unit"].mode()[0],
        })
        st.success("Appointment booked successfully! This appointment is stored in session only for demo purposes.")


def display_doctor_selection(df: pd.DataFrame, models: Dict[str, object]) -> None:
    """Render the doctor selection and workload view.

    Parameters
    ----------
    df : pd.DataFrame
        Appointment data.
    models : Dict[str, object]
        Trained predictive models.
    """
    st.markdown("## 👩‍⚕️ Doctor Selection")
    doctors = sorted(df["doctor_name"].unique())
    selected_doctor = st.selectbox("Choose a Doctor", doctors)

    # Show doctor details
    doc_df = df[df["doctor_name"] == selected_doctor]
    specialty = doc_df["doctor_specialty"].iloc[0]
    dept = doc_df["department"].iloc[0]
    st.write(f"**Department:** {dept}")
    st.write(f"**Specialty:** {specialty}")

    # Compute utilization: number of appointments vs available slots
    appointments_per_day = doc_df.groupby(["appointment_date"]).size()
    avg_appointments_day = appointments_per_day.mean() if not appointments_per_day.empty else 0
    max_slots_per_day = len(set(df["appointment_time"].unique()))  # 15 min increments across day
    utilization_pct = min(1.0, avg_appointments_day / max_slots_per_day)
    st.write(f"**Utilization:** {utilization_pct * 100:.1f}%")

    # Show next 10 available slots within next 7 days
    today = datetime.date.today()
    upcoming_slots: List[str] = []
    for offset in range(0, 7):
        day = today + datetime.timedelta(days=offset)
        # generate slots
        for hour in range(8, 17):
            for minute in (0, 15, 30, 45):
                t_str = f"{hour:02d}:{minute:02d}"
                if not ((doc_df["appointment_date"] == day.isoformat()) & (doc_df["appointment_time"] == t_str)).any():
                    upcoming_slots.append(f"{day.isoformat()} {t_str}")
                if len(upcoming_slots) >= 10:
                    break
            if len(upcoming_slots) >= 10:
                break
        if len(upcoming_slots) >= 10:
            break
    st.write("**Next Available Slots:**")
    for slot in upcoming_slots:
        st.write(f"- {slot}")

    # Predicted workload for the coming day (next 24 hours)
    st.write("**Predicted Workload (appointments/hour)**")
    # Determine day-of-week today
    dow = today.weekday()
    hours = list(range(8, 17))
    predicted_counts = [predict_workload(models["workload"], selected_doctor, dow, h) for h in hours]
    fig = px.bar(x=[f"{h:02d}:00" for h in hours], y=predicted_counts, labels={"x": "Hour", "y": "Predicted Appointments"})
    st.plotly_chart(fig, use_container_width=True)


def display_analytics(df: pd.DataFrame, models: Dict[str, object]) -> None:
    """Render the workflow optimization analytics page.

    Parameters
    ----------
    df : pd.DataFrame
        Appointment data.
    models : Dict[str, object]
        Trained predictive models.
    """
    st.markdown("## 📈 Workflow Analytics")
    st.write(
        "Explore patient flow and operational metrics using interactive charts. "
        "Use the filters below to refine the analysis by department, doctor or date range."
    )

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    departments = ["All"] + sorted(df["department"].unique().tolist())
    selected_dept = col_f1.selectbox("Department", departments)
    doctors = ["All"] + sorted(df["doctor_name"].unique().tolist())
    selected_doc = col_f2.selectbox("Doctor", doctors)
    # Date range filter
    min_date = pd.to_datetime(df["appointment_date"]).min().date()
    max_date = pd.to_datetime(df["appointment_date"]).max().date()
    start_date, end_date = col_f3.date_input(
        "Date Range", (min_date, max_date), min_value=min_date, max_value=max_date
    )

    # Apply filters
    filtered_df = df.copy()
    if selected_dept != "All":
        filtered_df = filtered_df[filtered_df["department"] == selected_dept]
    if selected_doc != "All":
        filtered_df = filtered_df[filtered_df["doctor_name"] == selected_doc]
    filtered_df = filtered_df[
        (pd.to_datetime(filtered_df["appointment_date"]).dt.date >= start_date)
        & (pd.to_datetime(filtered_df["appointment_date"]).dt.date <= end_date)
    ]

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return

    # Convert appointment datetime for plotting
    filtered_df["datetime"] = pd.to_datetime(
        filtered_df["appointment_date"] + " " + filtered_df["appointment_time"]
    )

    # Patient arrivals by hour/day (line chart)
    filtered_df["hour"] = filtered_df["datetime"].dt.hour
    arrivals = filtered_df.groupby("hour").size().reset_index(name="count")
    fig1 = px.line(arrivals, x="hour", y="count", markers=True, title="Patient Arrivals by Hour")
    st.plotly_chart(fig1, use_container_width=True)

    # Department-wise patient load (bar chart)
    dept_load = filtered_df.groupby("department").size().reset_index(name="count")
    fig2 = px.bar(dept_load, x="department", y="count", title="Department-wise Patient Load")
    st.plotly_chart(fig2, use_container_width=True)

    # Doctor workload comparison (bar chart)
    doc_load = filtered_df.groupby("doctor_name").size().sort_values(ascending=False).reset_index(name="count")
    fig3 = px.bar(doc_load, x="doctor_name", y="count", title="Doctor Workload Comparison")
    st.plotly_chart(fig3, use_container_width=True)

    # No-show patterns (bar)
    no_show = filtered_df.groupby("doctor_name")["no_show"].mean().reset_index(name="no_show_rate")
    fig4 = px.bar(no_show, x="doctor_name", y="no_show_rate", title="No-Show Rate by Doctor")
    st.plotly_chart(fig4, use_container_width=True)

    # Wait time trends (line chart)
    wait_trends = filtered_df.groupby(filtered_df["datetime"].dt.date)["wait_time_minutes"].mean().reset_index(name="avg_wait")
    fig5 = px.line(wait_trends, x="datetime", y="avg_wait", title="Average Wait Time over Time")
    st.plotly_chart(fig5, use_container_width=True)

    # Peak time heatmap (hour vs day-of-week)
    filtered_df["day_of_week"] = filtered_df["datetime"].dt.dayofweek
    heat = filtered_df.groupby(["day_of_week", "hour"]).size().reset_index(name="count")
    heat_pivot = heat.pivot(index="day_of_week", columns="hour", values="count").fillna(0)
    fig6 = px.imshow(
        heat_pivot,
        labels=dict(x="Hour", y="Day of Week", color="Patient Count"),
        x=[f"{h:02d}:00" for h in heat_pivot.columns],
        y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(heat_pivot.index)],
        aspect="auto",
    )
    fig6.update_layout(title="Peak Time Heatmap")
    st.plotly_chart(fig6, use_container_width=True)

    # Appointment demand forecasting using flow model (for next 24 hours of selected start_date)
    st.markdown("### 📈 Appointment Demand Forecasting")
    forecast_date = start_date  # use start date for demonstration
    dow = forecast_date.weekday()
    hours = list(range(8, 17))
    predicted_counts = [predict_flow(models["flow"], dow, h) for h in hours]
    fig7 = px.line(x=[f"{h:02d}:00" for h in hours], y=predicted_counts, markers=True, labels={"x": "Hour", "y": "Predicted Arrivals"}, title="Forecasted Patient Arrivals")
    st.plotly_chart(fig7, use_container_width=True)


def display_ai_insights(df: pd.DataFrame, models: Dict[str, object]) -> None:
    """Render the AI/Data Science module with prediction tools.

    Parameters
    ----------
    df : pd.DataFrame
        Appointment data.
    models : Dict[str, object]
        Trained models.
    """
    st.markdown("## 🧠 AI & Data Science")
    st.write("Use the trained models to predict patient flow, wait times, workloads and no-show probabilities.")

    # Predict patient flow
    with st.expander("Patient Flow Prediction"):
        day = st.selectbox("Day of Week", [(0, "Monday"), (1, "Tuesday"), (2, "Wednesday"), (3, "Thursday"), (4, "Friday"), (5, "Saturday"), (6, "Sunday")], format_func=lambda x: x[1])
        hour = st.slider("Hour", 0, 23, 9)
        predicted_flow = predict_flow(models["flow"], day_of_week=day[0], hour=hour)
        st.write(f"**Predicted Arrivals:** {predicted_flow:.1f} patients")

    # Predict wait time
    with st.expander("Wait Time Prediction"):
        dept = st.selectbox("Department", sorted(df["department"].unique()))
        doctor = st.selectbox("Doctor", sorted(df[df["department"] == dept]["doctor_name"].unique()))
        # Choose date/time
        pred_date = st.date_input("Date", datetime.date.today())
        pred_time = st.time_input("Time", datetime.time(9, 0))
        priority = st.selectbox("Priority", ["normal", "urgent"])
        dt = datetime.datetime.combine(pred_date, pred_time)
        predicted_wait = predict_wait_time(
            models["wait"],
            hour=dt.hour,
            day_of_week=dt.weekday(),
            department=dept,
            doctor_name=doctor,
            patient_priority=priority,
        )
        st.write(f"**Predicted Wait Time:** {predicted_wait:.1f} minutes")

    # Predict doctor workload
    with st.expander("Doctor Workload Forecasting"):
        doctor = st.selectbox("Select Doctor", sorted(df["doctor_name"].unique()), key="workload_doctor")
        w_date = st.date_input("Date", datetime.date.today(), key="workload_date")
        w_hour = st.slider("Hour", 0, 23, 10, key="workload_hour")
        workload_pred = predict_workload(models["workload"], doctor_name=doctor, day_of_week=w_date.weekday(), hour=w_hour)
        st.write(f"**Predicted Number of Appointments:** {workload_pred:.2f}")

    # Predict no-show probability
    with st.expander("No‑Show Probability Prediction"):
        dept_ns = st.selectbox("Department", sorted(df["department"].unique()), key="noshow_dept")
        doctor_ns = st.selectbox("Doctor", sorted(df[df["department"] == dept_ns]["doctor_name"].unique()), key="noshow_doc")
        ns_date = st.date_input("Date", datetime.date.today(), key="noshow_date")
        ns_time = st.time_input("Time", datetime.time(10, 0), key="noshow_time")
        priority_ns = st.selectbox("Priority", ["normal", "urgent"], key="noshow_priority")
        dt_ns = datetime.datetime.combine(ns_date, ns_time)
        prob_no_show = predict_no_show_probability(
            models["noshow"],
            hour=dt_ns.hour,
            day_of_week=dt_ns.weekday(),
            department=dept_ns,
            doctor_name=doctor_ns,
            patient_priority=priority_ns,
        )
        st.write(f"**Probability of No‑Show:** {prob_no_show * 100:.1f}%")


def display_recommendations(df: pd.DataFrame, models: Dict[str, object]) -> None:
    """Render admin insights and recommendations.

    Parameters
    ----------
    df : pd.DataFrame
        Appointment data.
    models : Dict[str, object]
        Trained models.
    """
    st.markdown("## 🧑‍⚕️ Admin Insights & Recommendations")
    st.write(
        "Leverage the analytics and AI models to derive actionable recommendations for hospital management."
    )

    # Suggest lower crowd times (hours with lowest predicted flow)
    st.markdown("### Suggest Low‑Crowd Appointment Times")
    # Compute predicted arrivals for each hour of the coming weekday (e.g. Monday) using flow model
    today = datetime.date.today()
    dow = today.weekday()
    hours = list(range(8, 17))
    predicted_counts = [(h, predict_flow(models["flow"], dow, h)) for h in hours]
    sorted_counts = sorted(predicted_counts, key=lambda x: x[1])
    low_crowd = sorted_counts[:3]
    for h, count in low_crowd:
        st.write(f"- **{h:02d}:00** – estimated {count:.1f} arrivals")

    # Identify overloaded departments (those with average wait time above threshold)
    st.markdown("### Identify Overloaded Departments")
    dept_wait = df.groupby("department")["wait_time_minutes"].mean().reset_index()
    threshold = df["wait_time_minutes"].mean() * 1.2
    overloaded = dept_wait[dept_wait["wait_time_minutes"] > threshold]
    if overloaded.empty:
        st.write("No departments appear overloaded based on current data.")
    else:
        for _, row in overloaded.iterrows():
            st.write(f"- **{row['department']}** – average wait {row['wait_time_minutes']:.1f} min (above threshold {threshold:.1f} min)")

    # Recommend doctor load balancing (compare workloads between doctors in same department)
    st.markdown("### Recommend Doctor Load Balancing")
    load_summary = df.groupby(["department", "doctor_name"]).size().reset_index(name="appointments")
    for dept, group in load_summary.groupby("department"):
        max_doc = group.loc[group["appointments"].idxmax()]
        min_doc = group.loc[group["appointments"].idxmin()]
        if max_doc["appointments"] > min_doc["appointments"] * 1.5:
            st.write(
                f"- In **{dept}**, Dr. {max_doc['doctor_name']} has {max_doc['appointments']} appointments vs Dr. {min_doc['doctor_name']} with {min_doc['appointments']}. "
                "Consider redistributing appointments to balance workload."
            )

    # Predict congestion periods (peak demand hours)
    st.markdown("### Predict Upcoming Congestion Periods")
    # Forecast next day demand for all hours
    next_day = today + datetime.timedelta(days=1)
    dow_next = next_day.weekday()
    forecast_counts = [(h, predict_flow(models["flow"], dow_next, h)) for h in hours]
    forecast_counts_sorted = sorted(forecast_counts, key=lambda x: x[1], reverse=True)
    top_congestion = forecast_counts_sorted[:3]
    for h, count in top_congestion:
        st.write(f"- **{h:02d}:00** – predicted {count:.1f} arrivals tomorrow")

    # Before vs After optimization comparison (estimated metrics)
    st.markdown("### Before vs After Optimization")
    # Baseline
    baseline_wait = df["wait_time_minutes"].mean()
    baseline_util = df["consultation_duration"].mean() / 15.0
    baseline_peak = df.groupby([pd.to_datetime(df["appointment_date"]).dt.dayofweek, pd.to_datetime(df["appointment_time"]).dt.hour]).size().max()
    # Assume improvements after recommendations
    improved_wait = baseline_wait * 0.8  # 20% reduction
    improved_util = min(1.0, baseline_util * 1.1)  # 10% increase
    improved_peak = baseline_peak * 0.85  # 15% reduction
    comparison_df = pd.DataFrame({
        "Metric": ["Average Wait Time", "Doctor Utilization", "Peak Patient Count"],
        "Before": [f"{baseline_wait:.1f} min", f"{baseline_util*100:.1f}%", f"{baseline_peak:.0f} patients"],
        "After": [f"{improved_wait:.1f} min", f"{improved_util*100:.1f}%", f"{improved_peak:.0f} patients"],
    })
    st.table(comparison_df)


def main() -> None:
    """Main entry point for the Streamlit app."""
    st.set_page_config(
        page_title="AI for Hospital Workflow Optimization",
        page_icon="🏥",
        layout="wide",
    )
    st.title("AI‑Driven Hospital Workflow Optimization")

    # Load data
    df = load_data()
    # Train models
    models = train_models(df)

    # Sidebar navigation
    pages = {
        "Home": display_home,
        "Book Appointment": display_booking,
        "Doctor Selection": display_doctor_selection,
        "Analytics": display_analytics,
        "AI Insights": display_ai_insights,
        "Admin Recommendations": display_recommendations,
    }
    choice = st.sidebar.radio("Navigation", list(pages.keys()))

    # Call selected page function
    if choice in pages:
        pages[choice](df if choice != "Book Appointment" else df, models)


if __name__ == "__main__":
    main()