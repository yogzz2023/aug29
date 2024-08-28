import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

# Define lists to store results
r = []
el = []
az = []
time_list = []

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))  # Predicted state vector
        self.Pp = np.eye(6)  # Predicted state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.prev_Time = 0
        self.Q = np.eye(6)
        self.Phi = np.eye(6)
        self.Z = np.zeros((3, 1)) 
        self.Z1 = np.zeros((3, 1)) # Measurement vector
        self.Z2 = np.zeros((3, 1)) 
        self.first_rep_flag = False
        self.second_rep_flag = False
        self.gate_threshold = 9.21  # 95% confidence interval for Chi-square distribution with 3 degrees of freedom

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        if not self.first_rep_flag:
            self.Z1=np.array([[x],[y],[z]])
            self.Meas_Time=time
            self.prev_Time=self.Meas_Time
            self.first_rep_flag = True
        elif self.first_rep_flag and not self.second_rep_flag:
            self.Z2=np.array([[x],[y],[z]])
            self.prev_Time=self.Meas_Time
            self.Meas_Time=time
            dt=self.Meas_Time - self.prev_Time
            self.vx =(self.Z2[0] - self.Z1[0]) / dt
            self.vy =(self.Z2[1] - self.Z1[1]) / dt
            self.vz =(self.Z2[2] - self.Z1[2]) / dt

            self.Meas_Time = time
            self.second_rep_flag = True
        else:
            self.Z=np.array([[x],[y],[z]])
            self.prev_Time=self.Meas_Time
            self.Meas_Time=time

    def predict_step(self, current_time):
        dt = current_time - self.prev_Time
        T_2=(dt*dt)/2.0
        T_3=(dt*dt*dt)/3.0
        self.Phi[0, 3] = dt
        self.Phi[1, 4] = dt
        self.Phi[2, 5] = dt
              
        self.Q[0, 0] = T_3
        self.Q[1, 1] = T_3
        self.Q[2, 2] = T_3
        self.Q[0, 3] = T_2
        self.Q[1, 4] = T_2
        self.Q[2, 5] = T_2
        self.Q[3, 0] = T_2
        self.Q[4, 1] = T_2
        self.Q[5, 2] = T_2
        self.Q[3, 3] = dt
        self.Q[4, 4] = dt
        self.Q[5, 5] = dt
        self.Q = self.Q * self.plant_noise
        self.Sp = np.dot(self.Phi, self.Sf)
        self.Pp = np.dot(np.dot(self.Phi, self.Pf), self.Phi.T) + self.Q
        self.Meas_Time = current_time

    def update_step(self, Z):
        Inn = Z - np.dot(self.H, self.Sp)
        S = np.dot(self.H, np.dot(self.Pp, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pp, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sp + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pp)

def form_measurement_groups(measurements, max_time_diff=50):
    measurement_groups = []
    current_group = []
    base_time = measurements[0][3]

    for measurement in measurements:
        if measurement[3] - base_time <= max_time_diff:
            current_group.append(measurement)
        else:
            measurement_groups.append(current_group)
            current_group = [measurement]
            base_time = measurement[3]

    if current_group:
        measurement_groups.append(current_group)

    return measurement_groups

def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[7])  # MR column
            ma = float(row[8])  # MA column
            me = float(row[9])  # ME column
            mt = float(row[10])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            r, az, el = cart2sph(x, y, z)  # Convert Cartesian to spherical coordinates
            measurements.append((r, az, el, mt))
    return measurements

def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
    az = math.atan2(y, x)

    if x > 0.0:
        az = np.pi / 2 - az
    else:
        az = 3 * np.pi / 2 - az

    az = az * 180 / np.pi

    if az < 0.0:
        az = 360 + az

    if az > 360:
        az = az - 360

    return r, az, el

def cart2sph2(x:float, y:float, z:float, filtered_values_csv):
    r = []
    az = []
    el = []
    for i in range(len(filtered_values_csv)):
        r.append(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2))
        el.append(math.atan(z[i]/np.sqrt(x[i]**2 + y[i]**2))*180/np.pi)
        az.append(math.atan(y[i]/x[i]))
         
        if x[i] > 0.0:                
            az[i] = np.pi / 2 - az[i]
        else:
            az[i] = 3 * np.pi / 2 - az[i]       
        
        az[i] = az[i] * 180 / np.pi 

        if az[i] < 0.0:
            az[i] = 360 + az[i]
    
        if az[i] > 360:
            az[i] = az[i] - 360   

    return r, az, el

def create_cost_matrix(track_positions, report_positions):
    """
    Create a cost matrix where cost is the Euclidean distance between track and report positions.

    Args:
        track_positions (list of tuples): List of (track_id, (x, y, z)).
        report_positions (list of tuples): List of (report_id, (x, y, z)).

    Returns:
        np.ndarray: Cost matrix.
    """
    num_tracks = len(track_positions)
    num_reports = len(report_positions)
    cost_matrix = np.zeros((num_tracks, num_reports))

    for i, (track_id, track_pos) in enumerate(track_positions):
        for j, (report_id, report_pos) in enumerate(report_positions):
            cost_matrix[i, j] = np.linalg.norm(np.array(track_pos) - np.array(report_pos))

    return cost_matrix

def munkres_algorithm(cost_matrix):
    """
    Apply the Munkres (Hungarian) algorithm to find the optimal assignment.

    Args:
        cost_matrix (np.ndarray): The cost matrix.

    Returns:
        list of tuples: List of (track_index, report_index) tuples for the optimal assignment.
    """
    track_indices, report_indices = linear_sum_assignment(cost_matrix)
    return list(zip(track_indices, report_indices))

def main():
    # Load measurements from CSV file
    measurements = read_measurements_from_csv('ttk_84_test.csv')
    
    # Form measurement groups based on time intervals
    measurement_groups = form_measurement_groups(measurements, max_time_diff=50)
    
    # Initialize the Kalman filter
    kalman_filter = CVFilter()

    # Placeholder for track initialization
    track_positions = []
    report_positions = []

    # Process each measurement group
    for group in measurement_groups:
        for measurement in group:
            x, y, z, mt = measurement
            if not kalman_filter.first_rep_flag:
                kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
            elif kalman_filter.first_rep_flag and not kalman_filter.second_rep_flag:
                kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
            else:
                kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)

                # Predict and update Kalman filter with the measurement
                kalman_filter.predict_step(mt)
                kalman_filter.update_step(np.array([[x], [y], [z]]))
                
                # Track position for cost matrix
                track_positions.append((len(track_positions), kalman_filter.Sf[:3].flatten()))
                report_positions.append((len(report_positions), [x, y, z]))

    # Generate the cost matrix and find optimal assignment using Munkres algorithm
    cost_matrix = create_cost_matrix(track_positions, report_positions)
    assignment = munkres_algorithm(cost_matrix)
    
    # Print best assignments
    print("Optimal Assignment (track_index, report_index):")
    print(assignment)
    print("Best Assignments:")
    for track_index, report_index in assignment:
        track_id, track_pos = track_positions[track_index]
        report_id, report_pos = report_positions[report_index]
        print(f"Track ID {track_id} (Position {track_pos}) is assigned to Report ID {report_id} (Position {report_pos})")

if __name__ == "__main__":
    main()
