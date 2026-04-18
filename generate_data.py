"""
generate_data.py
=================

This script generates a synthetic hospital appointment dataset for the hospital workflow
optimization app. It simulates appointment bookings across multiple departments and
doctors, including features such as wait times, no‑show flags and consultation
durations.  The resulting CSV can be used both for demonstration purposes and to
train simple predictive models.

Usage:
    python generate_data.py --output data/hospital_data.csv --rows 2000

The script creates a CSV file at the specified location.  If the directory does
not exist it will be created.
"""

import argparse
import os
from datetime import datetime, timedelta
import random
import numpy as np
import pandas as pd


def generate_synthetic_dataset(rows: int = 2000,
                               start_date: str = "2024-01-01",
                               end_date: str = "2024-12-31",
                               seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic dataset of hospital appointments.

    Parameters
    ----------
    rows : int
        Number of appointment records to generate.
    start_date : str
        Earliest appointment date in YYYY-MM-DD format.
    end_date : str
        Latest appointment date in YYYY-MM-DD format.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the synthetic appointment data.
    """
    rng = np.random.default_rng(seed)

    # Define hospital departments and associated doctors and specialties
    departments = {
        "Cardiology": [("Dr. Alice Smith", "Cardiologist"), ("Dr. Bob Johnson", "Cardiologist")],
        "Neurology": [("Dr. Carol Lee", "Neurologist"), ("Dr. Daniel Kim", "Neurologist")],
        "Oncology": [("Dr. Emma Clark", "Oncologist"), ("Dr. Frank Wright", "Oncologist")],
        "Pediatrics": [("Dr. Grace Lewis", "Pediatrician"), ("Dr. Henry Patel", "Pediatrician")],
        "General Medicine": [("Dr. Irene Martinez", "General Practitioner"), ("Dr. Jason Lee", "General Practitioner")],
        "Orthopedics": [("Dr. Karen Walker", "Orthopedic Surgeon"), ("Dr. Liam Young", "Orthopedic Surgeon")],
    }

    # Generate appointment dates within range
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    date_range = (end_dt - start_dt).days

    records = []
    for i in range(rows):
        # Choose department and doctor
        department = rng.choice(list(departments.keys()))
        doctor_name, doctor_specialty = rng.choice(departments[department])

        # Random appointment date and time
        appt_date = start_dt + timedelta(days=int(rng.integers(0, date_range)))
        # Schedule times between 8am and 5pm
        hour = int(rng.integers(8, 17))
        minute = int(rng.choice([0, 15, 30, 45]))
        appointment_datetime = datetime(
            year=appt_date.year, month=appt_date.month, day=appt_date.day,
            hour=hour, minute=minute
        )

        # Simulate patient arrival time (may arrive early/late)
        arrival_offset = int(rng.normal(loc=0, scale=10))  # in minutes
        actual_arrival_time = appointment_datetime + timedelta(minutes=arrival_offset)

        # Compute wait time (longer during peak hours and for busy doctors)
        base_wait = max(0, rng.normal(loc=15, scale=10))
        # Increase wait times during midday (11am-2pm)
        if 11 <= appointment_datetime.hour <= 14:
            base_wait *= 1.3
        # Adjust wait by doctor workload (random factor per doctor)
        doctor_factor = 1 + 0.1 * rng.random()
        wait_time_minutes = round(base_wait * doctor_factor)

        # Consultation duration (minutes)
        consultation_duration = round(max(5, rng.normal(loc=20, scale=5)))

        # Determine no-show (higher chance for early morning/late afternoon)
        no_show_prob = 0.05 + 0.02 * (appointment_datetime.hour < 9 or appointment_datetime.hour > 15)
        no_show = rng.random() < no_show_prob

        # Status (completed, cancelled, no_show)
        if no_show:
            status = "no_show"
        else:
            status = rng.choice(["completed", "cancelled"], p=[0.9, 0.1])

        # Patient priority (normal, urgent)
        patient_priority = rng.choice(["normal", "urgent"], p=[0.85, 0.15])

        # Hospital unit within department (e.g., Ward A, Room B)
        hospital_unit = rng.choice(["Unit A", "Unit B", "Unit C"])

        # Build record dictionary
        records.append({
            "patient_id": f"P{100000 + i}",
            "appointment_date": appointment_datetime.date().isoformat(),
            "appointment_time": appointment_datetime.time().strftime("%H:%M"),
            "department": department,
            "doctor_name": doctor_name,
            "doctor_specialty": doctor_specialty,
            "booked_slot": appointment_datetime.strftime("%Y-%m-%d %H:%M"),
            "actual_arrival_time": actual_arrival_time.strftime("%Y-%m-%d %H:%M"),
            "wait_time_minutes": int(wait_time_minutes),
            "consultation_duration": int(consultation_duration),
            "no_show": int(no_show),
            "status": status,
            "patient_priority": patient_priority,
            "hospital_unit": hospital_unit,
        })

    df = pd.DataFrame(records)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic hospital appointment data")
    parser.add_argument(
        "--output", type=str, default="data/hospital_data.csv",
        help="Output CSV file path (default: data/hospital_data.csv)"
    )
    parser.add_argument(
        "--rows", type=int, default=2000,
        help="Number of rows/appointments to generate (default: 2000)"
    )
    parser.add_argument(
        "--start-date", type=str, default="2024-01-01",
        help="Start date for appointments in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end-date", type=str, default="2024-12-31",
        help="End date for appointments in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    # Generate dataset
    df = generate_synthetic_dataset(
        rows=args.rows,
        start_date=args.start_date,
        end_date=args.end_date,
        seed=args.seed
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Generated synthetic dataset with {len(df)} rows at {args.output}")


if __name__ == "__main__":
    main()