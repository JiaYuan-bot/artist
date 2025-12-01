#!/bin/bash

# Define the start and end port for the range
START_PORT=55050
END_PORT=55100

# Array to hold the Process IDs (PIDs) of the background jobs
PIDS=()

# Function to clean up and kill all child processes when the script exits
cleanup() {
    echo "Stopping all servers..."
    # Using pkill to be more robust, targeting child processes of this script's PID
    pkill -P $$
    echo "All servers stopped."
}

# Set the trap. This calls the 'cleanup' function when you press Ctrl+C (SIGINT)
trap cleanup SIGINT

echo "Starting gRPC servers from port $START_PORT to $END_PORT..."

# Loop through the port range
for (( port=$START_PORT; port<=$END_PORT; port++ ))
do
    # Run the python server in the background (&) and pass the current port
    # NOTE: Make sure 'main.py' is the correct name of your Python server script.
    echo "-> Starting server on port $port"
    python3 main.py --port $port &
    
    # Store the PID of the last command run in the background
    PIDS+=($!)
done

echo "All servers are running in the background."
echo "Press Ctrl+C to stop all servers."

# Wait for all background processes to finish (which they won't until killed)
wait