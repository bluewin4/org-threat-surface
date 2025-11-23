from agent_simplified import Agent, Message
import os


def main():
    """Example usage of the LLM-based Agent class with Gemini."""
    
    # Define prompts for different employee classes
    prompt_registry = {
        "software engineers": """You are a software engineer. Focus on:
- Technical implementation details
- Code quality and best practices
- Collaboration on technical challenges
- Sharing knowledge about systems""",
        
        "researchers": """You are a researcher. Focus on:
- Exploring new ideas and approaches
- Data-driven insights
- Experimental results
- Knowledge discovery""",
        
        "execs": """You are an executive. Focus on:
- Strategic decisions
- Resource allocation
- High-level coordination
- Business outcomes"""
    }
    
    # Create an agent (will use GEMINI_API_KEY from environment)
    try:
        agent1 = Agent(
            employee_id="ENG001",
            prompt_registry=prompt_registry
        )
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your GEMINI_API_KEY environment variable:")
        print("export GEMINI_API_KEY='your-api-key-here'")
        return
    
    print(f"Agent {agent1.employee_id} created")
    print(f"Class: {agent1.employee_class}")
    print(f"System prompt: {agent1.system_prompt[:50]}...")
    print(f"Direct neighbors: {agent1.neighbors}")
    print(f"Social capital: {agent1.social_capital}")
    
    # Agent writes to scratch pad
    agent1.write_scratch_pad(
        "Need to collaborate on authentication system. "
        "Should reach out to security team or another engineer."
    )
    print(f"\nScratch pad: {agent1.scratch_pad}")
    
    # List of available agents (would come from org graph tool in production)
    available_agents = ["EMP002", "SEC001", "EMP010"]
    
    # Agent chooses who to contact using Gemini
    chosen = agent1.choose_contact(
        available_agents=available_agents,
        task_context="implementing authentication system with security best practices"
    )
    print(f"\nAgent chose to contact: {chosen}")
    
    # Compose message using Gemini
    if chosen:
        message_content = agent1.compose_message(
            to_id=chosen,
            task_context="implementing authentication system with security best practices"
        )
        print(f"\nComposed message: {message_content}")
        
        # Send the message
        message = agent1.send_message(
            to_id=chosen,
            content=message_content
        )
        
        if message:
            print(f"\nMessage sent!")
            print(f"From: {message.from_id}")
            print(f"To: {message.to_id}")
            print(f"Content: {message.content[:100]}...")
            print(f"Cost: {message.cost} social capital")
            print(f"Remaining capital: {agent1.social_capital}")
    
    # Simulate receiving a response
    response = Message(
        from_id="SEC001",
        to_id=agent1.employee_id,
        content="Sure! Here are the key security considerations for authentication..."
    )
    
    agent1.receive_message(response)
    
    # Check conversation history
    history = agent1.get_conversation_history("SEC001")
    print(f"\nConversation history with SEC001:")
    for msg in history:
        print(f"  {msg.from_id} -> {msg.to_id}: {msg.content[:50]}...")


def demonstrate_llm_decision_making():
    """Show how the agent uses Gemini for decision making."""
    
    print("\n=== Gemini-Based Decision Making ===")
    
    prompt_registry = {"software engineers": "You are an engineer focused on technical excellence."}
    
    try:
        agent = Agent("ENG001", prompt_registry)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Set up some conversation history
    past_messages = [
        Message("ENG001", "EMP002", "Can you help with the database schema?"),
        Message("EMP002", "ENG001", "Sure, let's discuss the requirements."),
        Message("ENG001", "SEC001", "Need security review for API endpoints."),
        Message("SEC001", "ENG001", "I'll review them tomorrow."),
    ]
    
    for msg in past_messages:
        agent.remember_message(msg)
    
    # Update scratch pad with current thinking
    agent.write_scratch_pad(
        "Working on API authentication. Already discussed with SEC001 yesterday. "
        "Need to follow up or get implementation help from another engineer."
    )
    
    # Agent makes decision
    available = ["EMP002", "SEC001", "EMP003", "ARCH001"]
    chosen = agent.choose_contact(available, "implementing JWT authentication for REST API")
    
    print(f"Given the context and history, agent chose: {chosen}")
    print(f"This decision was based on:")
    print(f"- System prompt (role: {agent.employee_class})")
    print(f"- Current task: implementing JWT authentication")
    print(f"- Scratch pad notes: {agent.scratch_pad[:50]}...")
    print(f"- Past conversations with available contacts")
    print(f"- Social capital costs")
    print(f"- Gemini's analysis of the situation")


def setup_example():
    """Show how to set up the environment."""
    print("=== Setup Instructions ===")
    print("1. Install the Google Gen AI package:")
    print("   pip3 install google-genai")
    print("\n2. Set your Gemini API key:")
    print("   export GEMINI_API_KEY='your-api-key-here'")
    print("\n3. Get your API key from:")
    print("   https://makersuite.google.com/app/apikey")


if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv('GEMINI_API_KEY'):
        setup_example()
        print("\n" + "="*50 + "\n")
    
    main()
    demonstrate_llm_decision_making()