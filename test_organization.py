from agent_simplified import Agent, Message
import json
from typing import Dict, List, Tuple
import os


class MockOrgAPI:
    """Mock organizational API for testing."""
    
    def __init__(self):
        # Define a realistic org structure
        self.employees = {
            # Engineering Team
            "ENG001": {
                "name": "Alice Chen",
                "employee_class": "software engineers",
                "neighbors": ["ENG002", "ENG003", "LEAD001", "SEC001"],
                "department": "Engineering"
            },
            "ENG002": {
                "name": "Bob Smith",
                "employee_class": "software engineers", 
                "neighbors": ["ENG001", "ENG003", "LEAD001"],
                "department": "Engineering"
            },
            "ENG003": {
                "name": "Carol Davis",
                "employee_class": "software engineers",
                "neighbors": ["ENG001", "ENG002", "LEAD001", "RES001"],
                "department": "Engineering"
            },
            
            # Leadership
            "LEAD001": {
                "name": "David Johnson",
                "employee_class": "managers",
                "neighbors": ["ENG001", "ENG002", "ENG003", "EXEC001"],
                "department": "Engineering"
            },
            "EXEC001": {
                "name": "Eve Wilson",
                "employee_class": "execs",
                "neighbors": ["LEAD001", "LEAD002", "CFO001"],
                "department": "Executive"
            },
            
            # Security Team
            "SEC001": {
                "name": "Frank Security",
                "employee_class": "security engineers",
                "neighbors": ["ENG001", "SEC002", "LEAD002"],
                "department": "Security"
            },
            "SEC002": {
                "name": "Grace Shield", 
                "employee_class": "security engineers",
                "neighbors": ["SEC001", "LEAD002"],
                "department": "Security"
            },
            
            # Research Team
            "RES001": {
                "name": "Henry Research",
                "employee_class": "researchers",
                "neighbors": ["RES002", "ENG003", "LEAD003"],
                "department": "Research"
            },
            "RES002": {
                "name": "Iris Discovery",
                "employee_class": "researchers",
                "neighbors": ["RES001", "LEAD003"],
                "department": "Research"
            }
        }
        
        # Calculate distances between all employees (simplified)
        self.distances = {}
        for emp1 in self.employees:
            for emp2 in self.employees:
                if emp1 == emp2:
                    self.distances[(emp1, emp2)] = 0
                elif emp2 in self.employees[emp1]["neighbors"]:
                    self.distances[(emp1, emp2)] = 1
                else:
                    # Simple heuristic: 2 if same department, 3 otherwise
                    dept1 = self.employees[emp1]["department"]
                    dept2 = self.employees[emp2]["department"]
                    self.distances[(emp1, emp2)] = 2 if dept1 == dept2 else 3
    
    def query_employee(self, employee_id: str) -> Dict:
        """Query employee information."""
        if employee_id not in self.employees:
            raise ValueError(f"Employee {employee_id} not found")
        
        emp = self.employees[employee_id]
        return {
            "employee_class": emp["employee_class"],
            "neighbors": emp["neighbors"],
            "name": emp["name"],
            "department": emp["department"]
        }
    
    def get_distance(self, from_id: str, to_id: str) -> int:
        """Get distance between two employees."""
        return self.distances.get((from_id, to_id), 4)  # Default to 4 if not found


# Global mock API instance
mock_api = MockOrgAPI()


# Monkey patch the agent methods to use our mock API
def mock_query_org_graph(self, employee_id: str) -> Dict:
    """Override to use mock API."""
    return mock_api.query_employee(employee_id)


def mock_calculate_distance(self, from_id: str, to_id: str) -> int:
    """Override to use mock API."""
    return mock_api.get_distance(from_id, to_id)


def create_test_agent(employee_id: str, agent_name: str = None) -> Agent:
    """Create an agent with mocked API calls."""
    
    # Define comprehensive prompts
    prompt_registry = {
        "software engineers": """You are a software engineer. Your responsibilities include:
- Writing and reviewing code
- Collaborating on technical architecture
- Debugging and solving technical problems
- Mentoring junior developers
- Ensuring code quality and best practices

When communicating:
- Be specific about technical details
- Share code examples when relevant
- Ask clarifying questions about requirements
- Report blockers and dependencies clearly""",
        
        "managers": """You are an engineering manager. Your responsibilities include:
- Leading and supporting your team
- Planning sprints and deliverables
- Removing blockers for your team
- Coordinating with other departments
- Reporting progress to leadership

When communicating:
- Check on team progress regularly
- Offer help and resources
- Escalate issues when necessary
- Provide clear direction and priorities""",
        
        "execs": """You are an executive. Your responsibilities include:
- Setting strategic direction
- Making high-level decisions
- Allocating resources across teams
- Managing stakeholder relationships
- Ensuring organizational alignment

When communicating:
- Focus on business impact and ROI
- Request metrics and KPIs
- Provide strategic guidance
- Make decisions efficiently""",
        
        "security engineers": """You are a security engineer. Your responsibilities include:
- Reviewing code for security vulnerabilities
- Implementing security best practices
- Responding to security incidents
- Educating teams on security
- Maintaining security infrastructure

When communicating:
- Explain security risks clearly
- Provide actionable recommendations
- Share security best practices
- Be available for security reviews""",
        
        "researchers": """You are a researcher. Your responsibilities include:
- Exploring new technologies and approaches
- Running experiments and analyzing results
- Publishing findings and insights
- Collaborating with engineering teams
- Staying current with industry trends

When communicating:
- Share data and evidence
- Explain methodologies clearly
- Discuss potential applications
- Be open to feedback and iteration"""
    }
    
    # Create agent
    agent = Agent(employee_id, prompt_registry, agent_name)
    
    # Monkey patch the methods
    agent._query_org_graph_tool = lambda eid: mock_query_org_graph(agent, eid)
    agent._calculate_distance_tool = lambda fid, tid: mock_calculate_distance(agent, fid, tid)
    
    return agent


def simulate_scenario():
    """Simulate a realistic scenario in the organization."""
    print("=== Organizational Communication Simulation ===\n")
    
    # Create an engineer who needs to implement a security feature
    alice = create_test_agent("ENG001")
    print(f"Created agent: Alice Chen (ENG001)")
    print(f"Role: {alice.employee_class}")
    print(f"Department: Engineering")
    print(f"Direct connections: {', '.join(alice.neighbors)}")
    print(f"Social capital: {alice.social_capital}\n")
    
    # Alice writes her thoughts
    alice.write_scratch_pad(
        "Working on implementing OAuth2 authentication for our API. "
        "Need to ensure it's secure and follows best practices. "
        "Should probably get input from security team before proceeding."
    )
    
    print("Alice's scratch pad:")
    print(f'"{alice.scratch_pad}"\n')
    
    # Get available contacts
    all_employees = list(mock_api.employees.keys())
    all_employees.remove(alice.employee_id)  # Remove self
    
    # Alice decides who to contact
    print("Alice is deciding who to contact about OAuth2 implementation...")
    chosen = alice.choose_contact(
        available_agents=all_employees,
        task_context="implementing OAuth2 authentication with security best practices"
    )
    print(f"Alice chose to contact: {chosen} ({mock_api.employees[chosen]['name']})\n")
    
    # Alice composes a message
    if chosen:
        print("Alice is composing a message...")
        message_content = alice.compose_message(
            to_id=chosen,
            task_context="implementing OAuth2 authentication and need security review"
        )
        print(f"Message content: \"{message_content}\"\n")
        
        # Send the message
        distance = alice._calculate_distance_tool(alice.employee_id, chosen)
        cost = alice._calculate_communication_cost(chosen, distance)
        print(f"Communication cost: {cost} tokens (distance: {distance})")
        
        message = alice.send_message(
            to_id=chosen,
            content=message_content
        )
        
        if message:
            print(f"Message sent successfully!")
            print(f"Remaining social capital: {alice.social_capital}\n")
            
            # Simulate a response
            if chosen.startswith("SEC"):
                response_content = (
                    "Hi Alice! I'd be happy to help with the OAuth2 implementation. "
                    "Here are the key security considerations:\n"
                    "1. Use PKCE flow for public clients\n"
                    "2. Implement proper token rotation\n" 
                    "3. Set appropriate token expiration times\n"
                    "Let's schedule a code review once you have the initial implementation."
                )
            else:
                response_content = "Sure, I'd be happy to help with that. Let me know what you need."
            
            response = Message(
                from_id=chosen,
                to_id=alice.employee_id,
                content=response_content
            )
            
            alice.receive_message(response)
            print(f"Received response from {mock_api.employees[chosen]['name']}:")
            print(f'"{response_content}"\n')
            
            # Check conversation history
            history = alice.get_conversation_history(chosen)
            print(f"Conversation history with {chosen}:")
            for i, msg in enumerate(history, 1):
                sender_name = mock_api.employees.get(msg.from_id, {}).get('name', msg.from_id)
                print(f"{i}. {sender_name}: {msg.content[:80]}...")


def test_social_capital_constraints():
    """Test how agents handle social capital constraints."""
    print("\n=== Testing Social Capital Constraints ===\n")
    
    # Create a researcher with limited social capital
    researcher = create_test_agent("RES001")
    researcher.social_capital = 50  # Limited capital
    
    print(f"Researcher: {mock_api.employees['RES001']['name']} (RES001)")
    print(f"Social capital: {researcher.social_capital}")
    print(f"Needs to contact executive team about research findings\n")
    
    # Calculate costs to different people
    contacts = ["RES002", "ENG003", "LEAD001", "EXEC001"]
    print("Communication costs:")
    for contact in contacts:
        distance = researcher._calculate_distance_tool(researcher.employee_id, contact)
        cost = researcher._calculate_communication_cost(contact, distance)
        print(f"  {contact} ({mock_api.employees[contact]['name']}): {cost} tokens (distance: {distance})")
    
    print(f"\nWith {researcher.social_capital} tokens, researcher chooses...")
    chosen = researcher.choose_contact(
        available_agents=contacts,
        task_context="share breakthrough research findings that could impact product strategy"
    )
    
    if chosen:
        print(f"Chosen: {chosen} ({mock_api.employees[chosen]['name']})")
    else:
        print("Cannot afford to contact anyone!")


if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv('GEMINI_API_KEY'):
        print("Please set GEMINI_API_KEY environment variable")
        print("export GEMINI_API_KEY='your-api-key-here'")
        exit(1)
    
    try:
        # Run main scenario
        simulate_scenario()
        
        # Test constraints
        test_social_capital_constraints()
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("\nMake sure you have:")
        print("1. Installed google-genai: pip3 install google-genai")
        print("2. Set GEMINI_API_KEY environment variable")