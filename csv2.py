import numpy as np
import math
import csv
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
from scipy.stats import chi2

# Define lists to store results
r = []
el = []
az = []

class CVFilter:
    # ... (CVFilter class remains unchanged) ...

def read_measurements_from_csv(file_path):
    # ... (read_measurements_from_csv function remains unchanged) ...

def sph2cart(az, el, r):
    # ... (sph2cart function remains unchanged) ...

def cart2sph(x, y, z):
    # ... (cart2sph function remains unchanged) ...

def cart2sph2(x: float, y: float, z: float, filtered_values_csv):
    # ... (cart2sph2 function remains unchanged) ...

def form_measurement_groups(measurements, max_time_diff=0.050):
    # ... (form_measurement_groups function remains unchanged) ...

def generate_hypotheses(tracks, reports):
    # ... (generate_hypotheses function remains unchanged) ...

def calculate_probabilities(hypotheses, tracks, reports, cov_inv):
    # ... (calculate_probabilities function remains unchanged) ...

def get_association_weights(hypotheses, probabilities, num_tracks):
    # ... (get_association_weights function remains unchanged) ...

def mahalanobis_distance(track, report, cov_inv):
    # ... (mahalanobis_distance function remains unchanged) ...

def form_clusters_via_association(tracks, reports, kalman_filter, chi2_threshold):
    # ... (form_clusters_via_association function remains unchanged) ...

def select_best_report(cluster_tracks, cluster_reports, kalman_filter):
    # ... (select_best_report function remains unchanged) ...

def main():
    # File path for measurements CSV
    file_path = 'ttk.csv'

    # CSV file to save updated Kalman filter states
    output_csv_file = 'updated_filter_states.csv'
    
    # Read measurements from CSV
    measurements = read_measurements_from_csv(file_path)
    
    kalman_filter = CVFilter()
    
    csv_file_predicted = "ttk.csv"
    df_predicted = pd.read_csv(csv_file_predicted)
    filtered_values_csv = df_predicted[['F_TIM', 'F_X', 'F_Y', 'F_Z']].values
    measured_values_csv = df_predicted[['MT', 'MR', 'MA', 'ME']].values

    A = cart2sph2(filtered_values_csv[:, 1], filtered_values_csv[:, 2], filtered_values_csv[:, 3], filtered_values_csv)
    number = 1000

    result = np.divide(A[0], number)              

    # Form measurement groups based on time intervals less than 50 milliseconds
    measurement_groups = form_measurement_groups(measurements, max_time_diff=0.050)

    t = []
    rnge = []
    azme = []
    elem = []

    # Prepare the output CSV file for writing
    with open(output_csv_file, mode='w', newline='') as output_file:
        writer = csv.writer(output_file)
        
        # Write the header
        writer.writerow(['Time', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ'])

        # Process each group of measurements
        for group_idx, group in enumerate(measurement_groups):
            print(f"Processing measurement group {group_idx + 1}...")

            # List of tracks and reports (in Cartesian coordinates)
            tracks = []
            reports = []

            for i, (rng, azm, ele, mt) in enumerate(group):
                print(f"\nMeasurement {i + 1}: (az={azm}, el={ele}, r={rng}, t={mt})\n")

                x, y, z = sph2cart(azm, ele, rng)

                if not kalman_filter.first_rep_flag:
                    kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
                    print("Initialized Filter state:", kalman_filter.Sf.flatten())
                    continue

                elif kalman_filter.first_rep_flag and not kalman_filter.second_rep_flag:
                    kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
                    print("Initialized Filter state 2nd M:", kalman_filter.Sf.flatten())
                else:
                    kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
                    
                    kalman_filter.predict_step(mt)
                    reports.append((x, y, z))  # Append as a tuple of Cartesian coordinates

                    # Use the Kalman filter state as the track
                    tracks.append(kalman_filter.Sf[:3].flatten())

            # Form clusters using the association logic
            clusters = form_clusters_via_association(tracks, reports, kalman_filter, chi2_threshold=kalman_filter.gate_threshold)
            print("Clusters formed:", clusters)  # Print clusters
            print("Number of clusters:", len(clusters))

            # Process each cluster and select the best report for updating
            for cluster_tracks, cluster_reports in clusters:
                if cluster_tracks and cluster_reports:
                    best_track_idx, best_report = select_best_report(cluster_tracks, cluster_reports, kalman_filter)

                    if best_report is not None:
                        print(f"Selected Best Report for Track {best_track_idx + 1}: {best_report}")

                        # Prepare the measurement vector Z for the Kalman filter update
                        Z = np.array([[best_report[0]], [best_report[1]], [best_report[2]]])
                        print("Measurement Vector Z:", Z)

                        # Perform the Kalman filter update step with the selected report
                        kalman_filter.update_step(Z)
                        print("Updated filter state:", kalman_filter.Sf.flatten())

                        # Save the updated state to the CSV file
                        writer.writerow([mt] + kalman_filter.Sf.flatten().tolist())

                        # Save the updated state for plotting or further processing
                        r_val, az_val, el_val = cart2sph(kalman_filter.Sf[0], kalman_filter.Sf[1], kalman_filter.Sf[2])
                        t.append(mt)
                        rnge.append(r_val)
                        azme.append(az_val)
                        elem.append(el_val)
                    else:
                        print("No valid report found for this cluster.")

    # Plotting remains unchanged
    plt.figure(figsize=(12, 6))
    plt.subplot(facecolor="white")
    plt.scatter(t, rnge, label='filtered range (code)', color='green', marker='*')
    plt.scatter(filtered_values_csv[:, 0], result, label='filtered range (track id 31)', color='red', marker='*')
    plt.scatter(measured_values_csv[:, 0], measured_values_csv[:, 1], label='measured range (code)', color='blue', marker='o')
    plt.xlabel('Time', color='black')
    plt.ylabel('Range (r)', color='black')
    plt.title('Range vs. Time', color='black')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.tight_layout()
    mplcursors.cursor(hover=True)
    plt.show()

    # Plot azimuth (az) vs. time
    plt.figure(figsize=(12, 6))
    plt.subplot(facecolor="white")
    plt.scatter(t, azme, label='filtered azimuth (code)', color='green', marker='*')
    plt.scatter(filtered_values_csv[:, 0], A[1], label='filtered azimuth (track id 31)', color='red', marker='*')
    plt.scatter(measured_values_csv[:, 0], measured_values_csv[:, 2], label='measured azimuth (code)', color='blue', marker='o')
    plt.xlabel('Time', color='black')
    plt.ylabel('Azimuth (az)', color='black')
    plt.title('Azimuth vs. Time', color='black')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.tight_layout()
    mplcursors.cursor(hover=True)
    plt.show()

    # Plot elevation (el) vs. time
    plt.figure(figsize=(12, 6))
    plt.subplot(facecolor="white")
    plt.scatter(t, elem, label='filtered elevation (code)', color='green', marker='*')
    plt.scatter(filtered_values_csv[:, 0], A[2], label='filtered elevation (track id 31)', color='red', marker='*')
    plt.scatter(measured_values_csv[:, 0], measured_values_csv[:, 3], label='measured elevation (code)', color='blue', marker='o')
    plt.xlabel('Time', color='black')
    plt.ylabel('Elevation (el)', color='black')
    plt.title('Elevation vs. Time', color='black')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.tight_layout()
    mplcursors.cursor(hover=True)
    plt.show()

if __name__ == "__main__":
    main()
