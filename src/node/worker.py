import requests
import time
import random

# Command and control server URL
SERVER_URL = 'http://localhost:5000'

def register_node(node_id):
    """
    Registers the node with the command and control server.
    """
    response = requests.post(f'{SERVER_URL}/register', json={'node_id': node_id})
    if response.status_code == 200:
        print(response.json()['message'])
    else:
        print(response.json()['error'])

def assign_task(node_id):
    """
    Assigns a training task to the node from the command and control server.
    """
    response = requests.post(f'{SERVER_URL}/assign_task', json={'task': 'train_model'})
    if response.status_code == 200:
        print(response.json()['message'])
        # Simulate task execution time
        time.sleep(random.uniform(1, 5))
    else:
        print(response.json()['error'])

def main():
    # Generate a unique node ID (replace this with your own logic)
    node_id = 'node123'

    # Register the node with the command and control server
    register_node(node_id)

    # Continuously check for tasks and execute them
    while True:
        assign_task(node_id)
        time.sleep(2)  # Wait before checking for new tasks again

if __name__ == '__main__':
    main()
