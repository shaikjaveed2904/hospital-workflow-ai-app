# AI‑Driven Hospital Workflow Optimization

This project is a demo‑ready Streamlit application that showcases how data
analytics and machine learning can optimize hospital operations. The goal
is to provide administrators, clinicians and students with an interactive
dashboard that illustrates common challenges—such as long wait times,
inefficient scheduling and uneven doctor workloads—and how AI can be used
to improve them.

## Features

- **Home Dashboard** – Key performance indicators (KPIs) including total
  appointments, average wait time, doctor utilization, no‑show rate,
  completion rate, peak traffic hour and busiest department.
- **Appointment Booking** – Select a department and doctor, pick a date
  and time slot, view predicted wait time and book an appointment. This
  module uses the wait time model to estimate how long a patient will wait.
- **Doctor Selection** – Inspect doctor details (specialty, department),
  current utilization, next available slots and predicted workload for the
  coming hours.
- **Workflow Analytics** – Interactive charts for patient arrivals by
  hour/day, department workload, doctor comparisons, no‑show patterns,
  wait time trends, peak time heatmap and demand forecasting.
- **AI / Data Science Module** – Predict patient flow, wait times,
  doctor workload and no‑show probability using trained models.
- **Admin Recommendations** – Smart suggestions such as low‑crowd times,
  overloaded departments, doctor load balancing and congestion forecasts
  plus a before‑vs‑after optimization comparison.

## Project Structure

```
hospital_app/
├── app.py                # Streamlit application
├── generate_data.py       # Script to generate a synthetic dataset
├── model_utils.py         # Helper functions for model training and prediction
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── data/
    └── hospital_data.csv  # Generated synthetic data (after running generate_data.py)
```

## Getting Started

1. **Clone or download** this repository and navigate into the `hospital_app`
   folder.  All commands below assume you are in this directory.
2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Generate the synthetic dataset** (only needed the first time):

   ```bash
   python generate_data.py --output data/hospital_data.csv --rows 2000
   ```

   The script creates a realistic dataset of hospital appointments with
   departments, doctors, wait times and no‑show flags.  Feel free to
   adjust the `--rows`, `--start-date` or `--end-date` parameters.

5. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

   Open the URL shown in the terminal (usually http://localhost:8501) to
   interact with the dashboard.

## Page Overview

### Home Dashboard
This landing page displays summary metrics.  Use the cards to quickly
understand how many appointments are scheduled, how long patients wait on
average, how efficiently doctors are being utilized and which department
receives the most traffic.

### Appointment Booking
Simulate booking an appointment by selecting a department and doctor and
choosing a date and time.  The app prevents double‑booking by removing
occupied slots and shows the predicted wait time based on historical data.

### Doctor Selection
Learn more about each physician.  View their specialty, department, current
utilization level and upcoming availability.  A bar chart shows the
predicted number of appointments per hour for the chosen doctor.

### Workflow Analytics
Explore rich visualisations with interactive filters.  You can filter by
department, doctor or date range to refine the analysis.  Charts include:

- **Patient arrivals by hour** – identifies rush periods.
- **Department workload** – shows which departments are busiest.
- **Doctor workload comparison** – compares appointment counts among doctors.
- **No‑show patterns** – highlights which doctors experience higher no‑show rates.
- **Wait time trends** – tracks how wait times change over days.
- **Peak time heatmap** – maps demand across days of the week and hours of the day.
- **Demand forecasting** – uses a linear regression model to predict arrivals for the selected day.

### AI & Data Science
Interact with the underlying models directly:

- **Patient flow prediction** – estimate the number of arrivals given a day of the week and hour.
- **Wait time prediction** – forecast the wait time for a selected department, doctor and appointment time.
- **Doctor workload forecasting** – predict how many patients a doctor will see at a specific time.
- **No‑show probability** – estimate the likelihood that a patient will miss an appointment.

### Admin Insights & Recommendations
Derive actionable insights:

- **Low‑crowd times** – suggests appointment slots with the lowest predicted demand.
- **Overloaded departments** – flags departments where wait times exceed acceptable thresholds.
- **Doctor load balancing** – points out imbalances in appointments among doctors within the same department.
- **Congestion periods** – forecasts high‑demand hours for the coming days.
- **Before vs After optimization** – compares key metrics (average wait time, doctor utilization and peak patient count) before and after hypothetical operational improvements.  In our synthetic data, we assume a 20% reduction in wait times, a 10% increase in doctor utilization and a 15% reduction in peak congestion after applying the recommendations.

## Demo Walkthrough

1. **Home Dashboard** – Observe current KPIs.  Note the average wait time and busiest department.
2. **Book Appointment** – Choose a department (e.g., Cardiology), select a doctor, pick a date and time.  The predicted wait time updates automatically.  Click “Book Appointment” to see how the new booking is added (in session memory only).
3. **Doctor Selection** – Select a doctor to view their utilization, next available slots and predicted workload.  Use this to identify over‑ or under‑booked physicians.
4. **Workflow Analytics** – Filter to a specific department or doctor to explore demand patterns.  Use the heatmap to spot the busiest days and hours.  The demand forecasting chart illustrates projected patient arrivals for a selected day.
5. **AI & Data Science** – Try predicting flows and wait times under different scenarios.  For example, choose a Saturday at 10 AM to see how demand differs from a Wednesday afternoon.
6. **Admin Insights** – Read the recommendations and examine the before‑and‑after comparison table.  Notice how strategic scheduling and load balancing can reduce wait times and even out doctor workloads.

## Extending the App

- Replace the synthetic dataset with real appointment data stored in a database (e.g. SQLite or PostgreSQL).
- Enhance the predictive models by incorporating additional features such as patient demographics or weather conditions.
- Integrate authentication to allow patients and staff to log in, view and manage their own appointments.
- Deploy the application to a cloud platform (e.g. Heroku, AWS, Azure) for broader access.

## License

This project is provided for educational purposes only and is not intended for production use without further testing and validation.  Feel free to modify and extend it under the terms of the MIT license.