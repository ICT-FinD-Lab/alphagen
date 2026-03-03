
"""
Test script for the feedback loop fixing functionality.
"""
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from alphagen.data.parser import ExpressionParser
from alphagen.data.expression import Expression, Operators, Greater, Less, Sub
from alphagen_llm.feedback import run_feedback_loop_on_alphas, call_fixer_llm
from alphagen_llm.client.base import ChatClient
from unittest.mock import Mock, MagicMock


def build_parser() -> ExpressionParser:
    """Build parser matching the one used in rl.py"""
    return ExpressionParser(
        Operators,
        ignore_case=True,
        non_positive_time_deltas_allowed=False,
        additional_operator_mapping={
            "Max": [Greater],
            "Min": [Less],
            "Delta": [Sub]
        }
    )


class MockChatClient(ChatClient):
    """Mock chat client for testing."""
    
    def __init__(self):
        # Initialize without calling parent __init__ to avoid logger setup issues
        self.messages = []
        self.responses = {
            "fix": "Ref(Log(Close), 2)",  # Simple valid alpha for testing
            "critique": "[Weak Signal] IC is too low.",
            "improve": "Mean(Close, 5)"
        }
        
    def chat_complete(self, prompt: str) -> str:
        """Mock chat completion that logs and returns canned responses."""
        self.messages.append(prompt)
        print(f"[MOCK CHAT] Prompt: {prompt[:100]}...")
        
        # Return different responses based on prompt content
        if "fix" in prompt.lower() or "syntax" in prompt.lower():
            response = self.responses["fix"]
        elif "critique" in prompt.lower():
            response = self.responses["critique"]
        elif "refine" in prompt.lower() or "improve" in prompt.lower():
            response = self.responses["improve"]
        else:
            response = self.responses["fix"]
        
        print(f"[MOCK CHAT] Response: {response}")
        return response
    
    def log_message(self, msg):
        """Mock log message."""
        print(f"[LOG] {msg}")


class MockInteractionSession:
    """Mock interaction session."""
    
    def __init__(self, chat_client):
        self.client = chat_client
        self.logger = chat_client


def test_fixer_llm_with_logging():
    """Test that fixer LLM logs properly."""
    print("\n" + "="*60)
    print("TEST: Fixer LLM with Logging")
    print("="*60)
    
    chat_client = MockChatClient()
    mock_session = MockInteractionSession(chat_client)
    
    # Test invalid alpha that needs fixing
    invalid_alpha = "Ref(Log(Close, 2"  # Missing closing paren
    
    print(f"\nInput alpha: {invalid_alpha}")
    print(f"Calling call_fixer_llm...")
    
    fixed = call_fixer_llm(mock_session, invalid_alpha)
    
    print(f"\nFixed alpha: {fixed}")
    print(f"Total chat messages logged: {len(chat_client.messages)}")
    for i, msg in enumerate(chat_client.messages, 1):
        print(f"  Message {i}: {msg[:80]}...")


def test_feedback_loop_fix_only():
    """Test feedback loop in fix-only mode."""
    print("\n" + "="*60)
    print("TEST: Feedback Loop (fix-only mode)")
    print("="*60)
    
    parser = build_parser()
    chat_client = MockChatClient()
    mock_session = MockInteractionSession(chat_client)
    
    # Create mock pool
    mock_pool = MagicMock()
    mock_pool.calculator = MagicMock()
    mock_pool.size = 0
    
    # Test alphas with some invalid syntax
    test_alphas = [
        "Ref(Log(Close), 2)",           # Valid
        "Ref(Log(Close, 2",              # Invalid - missing paren
        "Mean(Close, 5",                 # Invalid - missing paren
    ]
    
    print(f"\nInput alphas: {test_alphas}")
    print(f"Running feedback loop in fix-only mode...")
    
    final_exprs = run_feedback_loop_on_alphas(
        initial_alphas=test_alphas,
        pool=mock_pool,
        chat_session=mock_session,
        feedback_iter=2,
        parser=parser,
        feedback_mode="fix-only"
    )
    
    print(f"\nFinal expressions: {[str(e) for e in final_exprs]}")
    print(f"Total chat messages logged: {len(chat_client.messages)}")
    for i, msg in enumerate(chat_client.messages, 1):
        print(f"  Message {i}: {msg[:80]}...")


def test_direct_chat_logging():
    """Test direct chat client logging."""
    print("\n" + "="*60)
    print("TEST: Direct Chat Client Logging")
    print("="*60)
    
    chat_client = MockChatClient()
    
    # Test direct chat_complete calls
    prompts = [
        "Fix this: Ref(Log(Close, 2",
        "Critique this: Mean(Close, 5)",
        "Improve this: Ref(Close, 1)"
    ]
    
    for prompt in prompts:
        print(f"\nCalling chat_complete with: {prompt[:50]}...")
        response = chat_client.chat_complete(prompt)
        print(f"Response: {response}")
    
    print(f"\nTotal messages in chat_client: {len(chat_client.messages)}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("FEEDBACK LOOP LOGGING TESTS")
    print("="*60)
    
    try:
        test_direct_chat_logging()
    except Exception as e:
        print(f"ERROR in test_direct_chat_logging: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_fixer_llm_with_logging()
    except Exception as e:
        print(f"ERROR in test_fixer_llm_with_logging: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_feedback_loop_fix_only()
    except Exception as e:
        print(f"ERROR in test_feedback_loop_fix_only: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
