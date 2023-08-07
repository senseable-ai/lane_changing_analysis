
import pandas as pd
import matplotlib.pyplot as plt

# Reading the data file
def load_data(file_path):
    column_names = [
        "frame_number", "id", "bbox_left", "bbox_top", "bbox_w", "bbox_h", 
        "indicator_on", "distance", "distance_indicator", "video_time"
    ]
    df = pd.DataFrame(pd.read_csv(file_path, sep=' ', header=None).values, columns=column_names)

    for col in ["frame_number", "id", "bbox_left", "bbox_top", "bbox_w", "bbox_h", "indicator_on", "distance_indicator"]:
        df[col] = df[col].astype(int)

    df["distance"] = df["distance"].astype(float)
    
    return df

# Function to determine if a lane change occurred based on the given criteria
def lane_change_occurred(vehicle_data, OBSERVATION_FRAME_COUNT, DISTANCE_CHANGE_THRESHOLD):
    indicator_on_frames = vehicle_data[vehicle_data["indicator_on"] == 1]["frame_number"].tolist()
    
    for frame in indicator_on_frames:
        subset_data = vehicle_data[(vehicle_data["frame_number"] > frame) & 
                                   (vehicle_data["frame_number"] <= frame + OBSERVATION_FRAME_COUNT)]
        
        if not subset_data.empty and (subset_data["distance"].max() - subset_data["distance"].min()) > DISTANCE_CHANGE_THRESHOLD:
            return True
    return False

def main():
    # Load the data
    file_path = "C:\\Users\\user\\Desktop\\im_jg_DeepSort_Pytorch_lane_detection\\NOR_20230612_093659_FT_yellow_coordinates.txt"
    df = load_data(file_path)
    
    # Define the observation frame count and distance change threshold
    OBSERVATION_FRAME_COUNT = 30
    DISTANCE_CHANGE_THRESHOLD = 2 * df["distance"].std()

    # Print the standard deviation of the distance
    print(f"Standard deviation of distance: {df['distance'].std():.4f}")

    # Determine lane change for each vehicle ID with indicator on
    vehicle_ids = df[df["indicator_on"] == 1]["id"].unique()
    lane_change_results = {vehicle_id: lane_change_occurred(df[df["id"] == vehicle_id], OBSERVATION_FRAME_COUNT, DISTANCE_CHANGE_THRESHOLD) 
                           for vehicle_id in vehicle_ids}

    # Count the number of vehicles that changed lanes and that didn't
    changed_lanes = sum(lane_change_results.values())
    did_not_change = len(lane_change_results) - changed_lanes

    # Calculate the probability of lane change
    lane_change_probability = changed_lanes / (changed_lanes + did_not_change)
    
    print(f"Probability of lane change: {lane_change_probability*100:.2f}%")
    
    # Plot distance values for each vehicle ID
    plt.figure(figsize=(15, 10))
    for vehicle_id in vehicle_ids:
        vehicle_data = df[df["id"] == vehicle_id]
        plt.plot(vehicle_data["frame_number"], vehicle_data["distance"], label=f'ID: {vehicle_id}')

    plt.title("Distance value over frames for vehicles with indicators on")
    plt.xlabel("Frame Number")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
