from flask import Flask, render_template, request, jsonify, Response
import json
import threading
import time
import queue
from typing import Dict, List
from test_organization import create_test_agent, mock_api
from agent_simplified import Message
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

class OrganizationScenarios:
    """Define different organizational scenarios."""
    
    def __init__(self):
        self.scenarios = {
            "tech_startup": {
                "name": "Tech Startup",
                "description": "Small engineering team working on new features",
                "employees": {
                    "ENG001": {"name": "Alice Chen", "role": "software engineers", "team": "Backend"},
                    "ENG002": {"name": "Bob Smith", "role": "software engineers", "team": "Frontend"}, 
                    "SEC001": {"name": "Frank Security", "role": "security engineers", "team": "Security"},
                    "LEAD001": {"name": "David Johnson", "role": "managers", "team": "Engineering"}
                },
                "tasks": [
                    "implementing OAuth2 authentication",
                    "adding real-time notifications", 
                    "optimizing database queries",
                    "setting up CI/CD pipeline"
                ]
            },
            "enterprise": {
                "name": "Enterprise Corporation",
                "description": "Large organization with multiple departments",
                "employees": {
                    "ENG001": {"name": "Alice Chen", "role": "software engineers", "team": "Platform"},
                    "ENG002": {"name": "Bob Smith", "role": "software engineers", "team": "Mobile"},
                    "ENG003": {"name": "Carol Davis", "role": "software engineers", "team": "Web"},
                    "SEC001": {"name": "Frank Security", "role": "security engineers", "team": "InfoSec"},
                    "SEC002": {"name": "Grace Shield", "role": "security engineers", "team": "DevSec"},
                    "RES001": {"name": "Henry Research", "role": "researchers", "team": "AI/ML"},
                    "EXEC001": {"name": "Eve Wilson", "role": "execs", "team": "Executive"},
                    "LEAD001": {"name": "David Johnson", "role": "managers", "team": "Engineering"}
                },
                "tasks": [
                    "migrating to microservices architecture",
                    "implementing zero-trust security",
                    "building ML recommendation system",
                    "planning quarterly roadmap"
                ]
            },
            "research_lab": {
                "name": "Research Laboratory", 
                "description": "Academic research environment with collaboration focus",
                "employees": {
                    "RES001": {"name": "Dr. Henry Research", "role": "researchers", "team": "AI"},
                    "RES002": {"name": "Dr. Iris Discovery", "role": "researchers", "team": "Data Science"},
                    "ENG001": {"name": "Alice Chen", "role": "software engineers", "team": "Infrastructure"},
                    "LEAD001": {"name": "Prof. David Johnson", "role": "managers", "team": "Administration"}
                },
                "tasks": [
                    "publishing research on neural networks",
                    "collaborating on data pipeline",
                    "presenting at conference",
                    "applying for research grants"
                ]
            }
        }

# Global variables for simulation state
scenarios = OrganizationScenarios()
simulation_queue = queue.Queue()
simulation_active = False

@app.route('/')
def index():
    """Main page showing organization selector."""
    return render_template('index.html', scenarios=scenarios.scenarios)

@app.route('/organization/<org_id>')
def organization_detail(org_id):
    """Organization detail page with simulation controls."""
    if org_id not in scenarios.scenarios:
        return "Organization not found", 404
    
    org = scenarios.scenarios[org_id]
    return render_template('organization.html', org_id=org_id, organization=org)

@app.route('/api/start_simulation', methods=['POST'])
def start_simulation():
    """Start agent simulation for selected organization."""
    global simulation_active
    
    data = request.get_json()
    org_id = data.get('org_id')
    task = data.get('task', 'general collaboration')
    initiator = data.get('initiator')
    
    if org_id not in scenarios.scenarios:
        return jsonify({'error': 'Invalid organization'}), 400
    
    if simulation_active:
        return jsonify({'error': 'Simulation already running'}), 400
    
    # Check if API key is available
    if not os.getenv('GEMINI_API_KEY'):
        return jsonify({'error': 'GEMINI_API_KEY not set'}), 500
    
    # Start simulation in background thread
    simulation_active = True
    thread = threading.Thread(
        target=run_simulation, 
        args=(org_id, task, initiator)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'Simulation started'})

@app.route('/api/simulation_events')
def simulation_events():
    """Server-sent events endpoint for real-time simulation updates."""
    def event_stream():
        while True:
            try:
                # Get event from queue with timeout
                event = simulation_queue.get(timeout=1)
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                # Send keep-alive ping
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"
            
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/api/stop_simulation', methods=['POST'])
def stop_simulation():
    """Stop the current simulation."""
    global simulation_active
    simulation_active = False
    
    # Clear the queue
    while not simulation_queue.empty():
        try:
            simulation_queue.get_nowait()
        except queue.Empty:
            break
    
    simulation_queue.put({
        'type': 'simulation_stopped',
        'message': 'Simulation stopped by user'
    })
    
    return jsonify({'status': 'Simulation stopped'})

def run_simulation(org_id: str, task: str, initiator: str = None):
    """Run the agent simulation and emit events."""
    global simulation_active
    
    try:
        org = scenarios.scenarios[org_id]
        
        # Send simulation start event
        simulation_queue.put({
            'type': 'simulation_started',
            'organization': org['name'],
            'task': task,
            'timestamp': time.time()
        })
        
        # Create agents for this organization
        agents = {}
        for emp_id, emp_info in org['employees'].items():
            try:
                agent = create_test_agent(emp_id)
                agents[emp_id] = agent
                
                simulation_queue.put({
                    'type': 'agent_created',
                    'agent_id': emp_id,
                    'name': emp_info['name'],
                    'role': emp_info['role'],
                    'team': emp_info['team'],
                    'social_capital': agent.social_capital
                })
                
                time.sleep(0.5)  # Small delay for visual effect
                
            except Exception as e:
                simulation_queue.put({
                    'type': 'error',
                    'message': f"Failed to create agent {emp_id}: {str(e)}"
                })
                continue
        
        if not agents:
            simulation_queue.put({
                'type': 'error',
                'message': 'No agents could be created'
            })
            return
        
        # Choose initiator if not specified
        if not initiator or initiator not in agents:
            initiator = list(agents.keys())[0]
        
        current_agent = agents[initiator]
        available_agents = list(agents.keys())
        available_agents.remove(initiator)
        
        # Send task assignment
        current_agent.write_scratch_pad(f"Working on: {task}. Need to coordinate with team.")
        
        simulation_queue.put({
            'type': 'task_assigned',
            'agent_id': initiator,
            'agent_name': org['employees'][initiator]['name'],
            'task': task,
            'scratch_pad': current_agent.scratch_pad
        })
        
        # Run simulation rounds
        for round_num in range(5):
            if not simulation_active:
                break
                
            simulation_queue.put({
                'type': 'round_started',
                'round': round_num + 1,
                'current_agent': initiator,
                'agent_name': org['employees'][initiator]['name']
            })
            
            # Agent chooses who to contact
            try:
                chosen = current_agent.choose_contact(
                    available_agents=available_agents,
                    task_context=task
                )
                
                if not chosen:
                    simulation_queue.put({
                        'type': 'no_contact_chosen',
                        'agent_id': initiator,
                        'reason': 'Insufficient social capital or no suitable contacts'
                    })
                    break
                
                simulation_queue.put({
                    'type': 'contact_chosen',
                    'from_agent': initiator,
                    'to_agent': chosen,
                    'from_name': org['employees'][initiator]['name'],
                    'to_name': org['employees'][chosen]['name']
                })
                
                time.sleep(2)  # Pause for dramatic effect
                
                # Compose and send message
                message_content = current_agent.compose_message(
                    to_id=chosen,
                    task_context=task
                )
                
                distance = current_agent._calculate_distance_tool(initiator, chosen)
                cost = current_agent._calculate_communication_cost(chosen, distance)
                
                message = current_agent.send_message(
                    to_id=chosen,
                    content=message_content
                )
                
                if message:
                    simulation_queue.put({
                        'type': 'message_sent',
                        'from_agent': initiator,
                        'to_agent': chosen,
                        'from_name': org['employees'][initiator]['name'],
                        'to_name': org['employees'][chosen]['name'],
                        'content': message_content,
                        'cost': cost,
                        'distance': distance,
                        'remaining_capital': current_agent.social_capital
                    })
                    
                    # Simulate response
                    time.sleep(3)
                    
                    if chosen in agents:
                        response_content = f"Thanks for reaching out about {task}. I'd be happy to collaborate on this."
                        response = Message(
                            from_id=chosen,
                            to_id=initiator,
                            content=response_content
                        )
                        
                        current_agent.receive_message(response)
                        agents[chosen].receive_message(message)
                        
                        simulation_queue.put({
                            'type': 'response_received',
                            'from_agent': chosen,
                            'to_agent': initiator,
                            'from_name': org['employees'][chosen]['name'],
                            'to_name': org['employees'][initiator]['name'],
                            'content': response_content
                        })
                    
                    # Switch to the contacted agent for next round
                    initiator = chosen
                    current_agent = agents[chosen]
                    available_agents = [aid for aid in agents.keys() if aid != chosen]
                    
                else:
                    simulation_queue.put({
                        'type': 'message_failed',
                        'agent_id': initiator,
                        'reason': 'Insufficient social capital'
                    })
                    break
                    
            except Exception as e:
                simulation_queue.put({
                    'type': 'error',
                    'message': f"Error in round {round_num + 1}: {str(e)}"
                })
                break
            
            time.sleep(1)
        
        # Send final status
        simulation_queue.put({
            'type': 'simulation_completed',
            'total_rounds': round_num + 1,
            'final_agent_states': {
                aid: {
                    'social_capital': agent.social_capital,
                    'conversations': len(agent.memory)
                }
                for aid, agent in agents.items()
            }
        })
        
    except Exception as e:
        simulation_queue.put({
            'type': 'error',
            'message': f"Simulation error: {str(e)}"
        })
    
    finally:
        simulation_active = False

if __name__ == '__main__':
    app.run(debug=True, threaded=True)