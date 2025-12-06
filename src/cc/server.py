from flask import Flask, request, jsonify

app = Flask(__name__)

# Define a dictionary to store node information
nodes = {}

@app.route('/register', methods=['POST'])
def register_node():
    """
    Endpoint for nodes to register themselves with the command and control server.
    """
    node_id = request.json.get('node_id')
    if node_id:
        nodes[node_id] = {
            'status': 'active',
            'tasks_completed': 0
        }
        return jsonify({'message': f'Node {node_id} registered successfully'}), 200
    else:
        return jsonify({'error': 'Node ID not provided'}), 400

@app.route('/assign_task', methods=['POST'])
def assign_task():
    """
    Endpoint to assign a training task to a registered node.
    """
    task = request.json.get('task')
    if not task:
        return jsonify({'error': 'Task not provided'}), 400
    
    # Find an available node to assign the task
    assigned_node = None
    for node_id, node_info in nodes.items():
        if node_info['status'] == 'active':
            assigned_node = node_id
            break
    
    if assigned_node:
        # Increment task count for the assigned node
        nodes[assigned_node]['tasks_completed'] += 1
        return jsonify({'message': f'Task assigned to Node {assigned_node}'}), 200
    else:
        return jsonify({'error': 'No active nodes available for task assignment'}), 404

@app.route('/status', methods=['GET'])
def get_node_status():
    """
    Endpoint to retrieve the status of all registered nodes.
    """
    return jsonify(nodes), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
