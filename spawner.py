from Game import Game, _unique_paths
import multiprocessing
import time
import sys

def manage_processes():
    # Get the list of active processes
    g = Game()
    active_processes = multiprocessing.active_children()
	
    # If there are less than 16 processes, spawn new ones
    while len(active_processes) < 16:
        process = multiprocessing.Process(target=_unique_paths)
        process.start()
        active_processes.append(process)
        print(f"Started new process. Total active processes: {len(active_processes)}")
        time.sleep(1)  # Optional: adjust the frequency of the checks

    # Wait for all processes to complete
    for process in active_processes:
        process.join()

if __name__ == "__main__":
    manage_processes(int(sys.argv[1]), int(sys.argv[2]))
