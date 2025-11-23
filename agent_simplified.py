from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
from google import genai

import os

from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env


@dataclass
class Message:
    """Represents a message between agents."""
    from_id: str
    to_id: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    cost: int = 0


class Agent:
    """LLM-based agent that represents an employee in the organization."""
    
    def __init__(self, employee_id: str, prompt_registry: Dict[str, str], gemini_api_key: Optional[str] = None):
        """
        Initialize an LLM agent with Gemini.
        
        Args:
            employee_id: Unique identifier for the employee
            prompt_registry: Dictionary mapping employee classes to system prompts
            gemini_api_key: Optional Gemini API key (will use GEMINI_API_KEY env var if not provided)
        """
        self.employee_id = employee_id
        self.prompt_registry = prompt_registry
        
        # Store API key for later use
        self.api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY environment variable or pass gemini_api_key parameter.")
        
        # First, query org API to get employee information (without Gemini)
        org_info = self._query_org_graph_tool(employee_id)
        
        # Extract employee class and neighbors
        self.employee_class = org_info.get('employee_class')
        self.neighbors = org_info.get('neighbors', [])  # Direct connections
        
        # Set system prompt based on employee class
        self.system_prompt = prompt_registry.get(self.employee_class, 
                                                "You are an employee in the organization.")
        
        # Define available tools
        self.tools = [
            {
                "name": "query_org_graph",
                "description": "Query the organizational graph API for employee information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "employee_id": {
                            "type": "string",
                            "description": "Employee ID to query"
                        }
                    },
                    "required": ["employee_id"]
                }
            },
            {
                "name": "calculate_distance",
                "description": "Calculate graph distance between two employees",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from_id": {
                            "type": "string",
                            "description": "Source employee ID"
                        },
                        "to_id": {
                            "type": "string",
                            "description": "Target employee ID"
                        }
                    },
                    "required": ["from_id", "to_id"]
                }
            }
        ]
        
        # NOW initialize Gemini client with system prompt and tools
        self.client = genai.Client()
        self.model_id = 'gemini-2.5-pro'  # Using the latest model
        
        # Initialize memory and scratch pad
        self.memory: Dict[str, List[Message]] = {}  # Conversation history per agent
        self.scratch_pad: str = ""
        
        # Starting social capital
        self.social_capital: int = 1000
        
    def _query_org_graph_tool(self, employee_id: str) -> Dict:
        """
        Tool: Query the organizational graph for employee information.
        """
        # This represents a tool call that the LLM would invoke
        # In production, this would be handled by the LLM framework
        
        # Simulated tool response
        return {
            'employee_class': 'software engineers',
            'neighbors': ['EMP002', 'EMP003', 'EMP004']
        }
    
    def _calculate_distance_tool(self, from_id: str, to_id: str) -> int:
        """
        Tool: Calculate graph distance between two employees.
        """
        # This represents a tool call that the LLM would invoke
        # In production, this would be handled by the LLM framework
        
        # Simulated response - in practice would query actual org graph
        if to_id in self.neighbors:
            return 1
        else:
            return 2  # Simplified for demo
    
    def _calculate_communication_cost(self, target_id: str, distance: int) -> int:
        """
        Calculate social capital cost based on distance.
        
        Args:
            target_id: ID of the target employee
            distance: Graph distance to the target
            
        Returns:
            Cost in social capital tokens
        """
        base_cost = 10
        
        # Neighbors (distance 1) cost base_cost
        if target_id in self.neighbors:
            return base_cost
        
        # Cost increases exponentially with distance
        return base_cost * (2 ** (distance - 1))
    
    def write_scratch_pad(self, content: str):
        """Update scratch pad with current thoughts and plans."""
        self.scratch_pad += content
    
    def clear_scratch_pad(self):
        """Clear the scratch pad."""
        self.scratch_pad = ""
    
    def remember_message(self, message: Message):
        """Store a message in conversation history."""
        # Determine which agent this conversation is with
        other_agent = message.to_id if message.from_id == self.employee_id else message.from_id
        
        if other_agent not in self.memory:
            self.memory[other_agent] = []
        
        self.memory[other_agent].append(message)
    
    def get_conversation_history(self, other_agent_id: str) -> List[Message]:
        """Get conversation history with a specific agent."""
        return self.memory.get(other_agent_id, [])
    
    def choose_contact(self, available_agents: List[str], task_context: str) -> Optional[str]:
        """
        Use LLM to choose which agent to contact based on task and history.
        
        Args:
            available_agents: List of agent IDs that can be contacted
            task_context: Current task or reason for communication
            
        Returns:
            Chosen agent ID or None
        """
        # Build context for LLM decision
        conversation_summaries = {}
        for agent_id in available_agents:
            if agent_id in self.memory:
                recent_msgs = self.memory[agent_id][-3:]  # Last 3 messages
                conversation_summaries[agent_id] = [
                    f"{msg.from_id}: {msg.content[:50]}..." for msg in recent_msgs
                ]
        
        # Calculate costs for each option
        agent_costs = {}
        for agent_id in available_agents:
            distance = self._calculate_distance_tool(self.employee_id, agent_id)
            cost = self._calculate_communication_cost(agent_id, distance)
            if cost <= self.social_capital:
                agent_costs[agent_id] = cost
        
        if not agent_costs:
            return None
        
        # Prepare prompt for Gemini
        prompt = f"""
{self.system_prompt}

Current task: {task_context}

Scratch pad: {self.scratch_pad}

Available contacts and costs:
{json.dumps(agent_costs, indent=2)}

Past conversations:
{json.dumps(conversation_summaries, indent=2)}

Your current social capital: {self.social_capital}

Based on the current task and your role, which agent should you contact? 
Consider the cost, relevance to task, and past interactions.
Respond with just the agent ID.
"""
        
        # Call Gemini to make decision
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
        
            chosen_agent = response.text.strip()
            
            # Validate the response is a valid agent ID
            if chosen_agent in agent_costs:
                return chosen_agent
            else:
                # Fallback to lowest cost if LLM returns invalid ID
                return min(agent_costs.items(), key=lambda x: x[1])[0]
        except Exception as e:
            # Fallback to lowest cost on error
            print(f"Gemini error: {e}. Using fallback selection.")
            return min(agent_costs.items(), key=lambda x: x[1])[0]
    
    def compose_message(self, to_id: str, task_context: str) -> str:
        """
        Use LLM to compose a message to another agent.
        
        Args:
            to_id: Recipient agent ID
            task_context: Current task or reason for communication
            
        Returns:
            Composed message content
        """
        # Get conversation history
        history = self.get_conversation_history(to_id)
        history_str = "\n".join([
            f"{msg.from_id}: {msg.content}" 
            for msg in history[-5:]  # Last 5 messages
        ])
        
        prompt = f"""
{self.system_prompt}

You need to communicate with {to_id} about: {task_context}

Your scratch pad notes: {self.scratch_pad}

Previous conversation:
{history_str if history_str else "No previous conversation"}

Compose a professional message that:
1. Is relevant to your role and the task
2. Builds on any previous conversations
3. Is clear and actionable

Message:
"""
        
        # Call Gemini to compose message
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            # Fallback message on error
            print(f"Gemini error: {e}. Using fallback message.")
            return f"Regarding {task_context}, I'd like to discuss this further with you."
    
    def send_message(self, to_id: str, content: str, distance: Optional[int] = None) -> Optional[Message]:
        """
        Send a message to another agent.
        
        Args:
            to_id: Recipient agent ID
            content: Message content
            distance: Graph distance to recipient (optional, will calculate if not provided)
            
        Returns:
            Message object if sent successfully, None if insufficient capital
        """
        # Get distance if not provided
        if distance is None:
            distance = self._calculate_distance_tool(self.employee_id, to_id)
        
        cost = self._calculate_communication_cost(to_id, distance)
        
        if cost > self.social_capital:
            return None
        
        # Deduct social capital
        self.social_capital -= cost
        
        # Create message
        message = Message(
            from_id=self.employee_id,
            to_id=to_id,
            content=content,
            cost=cost
        )
        
        # Remember the message
        self.remember_message(message)
        
        return message
    
    def receive_message(self, message: Message):
        """Receive and process a message from another agent."""
        self.remember_message(message)
        
        # Here you could add logic to process the message
        # and potentially update scratch pad or trigger a response