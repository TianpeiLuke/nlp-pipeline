import unittest
import time

# Assuming CSAdapter is in the same module as CSChatSplitterProcessor
from src.processing.cs_processor import CSChatSplitterProcessor, CSAdapter


class TestCSChatSplitterProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = CSChatSplitterProcessor()

    # ---------- core functionality ----------
    def test_basic_chat_splitting(self):
        txt = "[bot]:Hello there[bot]:How can I help?[customer]:I need help[agent]:I'm here to assist"
        expected = [
            {"role": "bot",       "content": "Hello there"},
            {"role": "bot",       "content": "How can I help?"},
            {"role": "customer",  "content": "I need help"},
            {"role": "agent",     "content": "I'm here to assist"},
        ]
        self.assertEqual(self.processor.process(txt), expected)

    def test_empty_input(self):
        self.assertEqual(self.processor.process(""), [])

    # ---------- whitespace handling ----------
    def test_whitespace_handling(self):
        # internal multiple spaces are **preserved** by the processor
        txt = "[bot]:  Hello   there  [customer]:    Test    message    "
        expected = [
            {"role": "bot",      "content": "Hello   there"},
            {"role": "customer", "content": "Test    message"},
        ]
        self.assertEqual(self.processor.process(txt), expected)

    # ---------- consecutive messages ----------
    def test_multiple_consecutive_messages(self):
        txt = "[agent]:First message[agent]:Second message[agent]:Third message"
        expected = [
            {"role": "agent", "content": "First message"},
            {"role": "agent", "content": "Second message"},
            {"role": "agent", "content": "Third message"},
        ]
        self.assertEqual(self.processor.process(txt), expected)

    def test_message_order_preservation(self):
        txt = "[bot]:First[customer]:Second[agent]:Third[bot]:Fourth"
        roles = [m["role"] for m in self.processor.process(txt)]
        self.assertEqual(roles, ["bot", "customer", "agent", "bot"])

    # ---------- empty segments ----------
    def test_empty_messages(self):
        txt = "[bot]:[customer]:Real message[agent]:"
        expected = [{"role": "customer", "content": "Real message"}]
        self.assertEqual(self.processor.process(txt), expected)

    # ---------- complex content ----------
    def test_complex_message_content(self):
        txt = (
            "[bot]:Message with [special] characters"
            "[customer]:Message with numbers 123"
            "[agent]:Message with \n newlines"
        )
        expected = [
            {"role": "bot",      "content": "Message with [special] characters"},
            {"role": "customer", "content": "Message with numbers 123"},
            {"role": "agent",    "content": "Message with \n newlines"},
        ]
        self.assertEqual(self.processor.process(txt), expected)

    # ---------- New tests for embedded messages (Improved Coverage) ----------

    def test_chat_with_single_embedded_message(self):
        """Tests splitting when one message contains another."""
        txt = "[customer]:Here is my main message. [bot]: And here is an embedded one."
        expected = [
            {"role": "customer", "content": "Here is my main message."},
            {"role": "bot", "content": "And here is an embedded one."}
        ]
        self.assertEqual(self.processor.process(txt), expected)

    def test_chat_with_multiple_embedded_messages(self):
        """Tests splitting with multiple embedded messages from different roles."""
        txt = "[agent]:Ok, I see. [bot]:First embedded. [customer]:Second embedded."
        expected = [
            {"role": "agent", "content": "Ok, I see."},
            {"role": "bot", "content": "First embedded."},
            {"role": "customer", "content": "Second embedded."}
        ]
        self.assertEqual(self.processor.process(txt), expected)

    def test_chat_with_only_embedded_messages(self):
        """Tests when a turn consists only of embedded messages and has no primary content."""
        txt = "[customer]:[bot]:Embedded one.[agent]:Embedded two."
        expected = [
            {"role": "bot", "content": "Embedded one."},
            {"role": "agent", "content": "Embedded two."}
        ]
        self.assertEqual(self.processor.process(txt), expected)

    def test_embedded_messages_mixed_with_normal_flow(self):
        """Tests a more complex flow with both normal and embedded messages."""
        txt = "[bot]:Hello![customer]:I have an issue. [agent]:Let me look that up for you. [bot]:I found your order.[customer]:Thanks!"
        expected = [
            {"role": "bot", "content": "Hello!"},
            {"role": "customer", "content": "I have an issue."},
            {"role": "agent", "content": "Let me look that up for you."},
            {"role": "bot", "content": "I found your order."},
            {"role": "customer", "content": "Thanks!"}
        ]
        self.assertEqual(self.processor.process(txt), expected)
        
    def test_embedded_message_without_main_content(self):
        """Tests a turn that has no main content, only an embedded message."""
        txt = "[customer]:[bot]:I am an embedded message."
        expected = [{"role": "bot", "content": "I am an embedded message."}]
        self.assertEqual(self.processor.process(txt), expected)

    def test_embedded_message_with_empty_content(self):
        """Tests that an embedded message with no content is ignored."""
        txt = "[agent]:Main content here. [customer]:"
        expected = [{"role": "agent", "content": "Main content here."}]
        self.assertEqual(self.processor.process(txt), expected)
        
    def test_whitespace_only_main_content_before_embedded(self):
        """Tests a turn where the main content is only whitespace before an embedded message."""
        txt = "[agent]:   [bot]:This is embedded."
        expected = [{"role": "bot", "content": "This is embedded."}]
        self.assertEqual(self.processor.process(txt), expected)

    # ---------- edge cases ----------
    def test_invalid_format(self):
        self.assertEqual(self.processor.process("Invalid format"), [])

    def test_mixed_case_roles(self):
        # The current regex is case-sensitive, so uppercase tags are ignored.
        self.assertEqual(self.processor.process("[BOT]:x[Customer]:y"), [])

    def test_real_world_example(self):
        txt = (
            "[bot]:Hi, you're in the right place for customer service support.\n"
            "[bot]:Item displayed, asin:B0CFNV34T3\n"
            "[customer]:I need help with my order\n"
            "[agent]:Hello, how can I assist you today?\n"
            "[customer]:The item is damaged\n"
            "[agent]:I understand, let me help you with that"
        )
        out = self.processor.process(txt)
        self.assertEqual(len(out), 6)
        self.assertEqual(out[0]["role"], "bot")
        self.assertEqual(out[2]["role"], "customer")
        self.assertEqual(out[3]["role"], "agent")

    def test_malformed_tags(self):
        # The processor should correctly extract the valid message and ignore malformed parts.
        txt = "[bot:bad[customer]:ok[agent]bad"
        # The current processor implementation is greedy and captures content until the next valid
        # [role]: marker or the end of the string.
        expected = [{"role": "customer", "content": "ok[agent]bad"}]
        self.assertEqual(self.processor.process(txt), expected)

    def test_text_after_last_tag(self):
        """Tests that text after the final valid tag is captured."""
        txt = "[agent]:Final message. and some trailing text"
        expected = [{"role": "agent", "content": "Final message. and some trailing text"}]
        self.assertEqual(self.processor.process(txt), expected)

    # ---------- performance ----------
    def test_performance_with_large_input(self):
        big = "[bot]:M[customer]:M[agent]:M" * 1000
        t0 = time.time()
        res = self.processor.process(big)
        # Ensure processing completes within a reasonable time (e.g., 1 second)
        self.assertLess(time.time() - t0, 1.0)
        self.assertEqual(len(res), 3000)


class TestCSAdapter(unittest.TestCase):
    """Unit tests for the CSAdapter processor."""

    def test_basic_formatting(self):
        """Test standard message list is formatted correctly."""
        adapter = CSAdapter()
        messages = [
            {"role": "customer", "content": "Hello"},
            {"role": "agent", "content": "Hi, how can I help?"}
        ]
        # The expected output should match the processor's actual format.
        expected = [
            "[customer]: Hello",
            "[agent]: Hi, how can I help?"
        ]
        self.assertEqual(adapter.process(messages), expected)

    def test_empty_message_list(self):
        """Test that an empty list of messages produces an empty list."""
        adapter = CSAdapter()
        self.assertEqual(adapter.process([]), [])

    def test_single_message_formatting(self):
        """Test adapter functionality for a single message."""
        adapter = CSAdapter()
        messages = [{"role": "bot", "content": "Test"}]
        expected = ["[bot]: Test"]
        self.assertEqual(adapter.process(messages), expected)

    def test_message_with_missing_keys_raises_error(self):
        """Test that a message with a missing key raises a KeyError."""
        adapter = CSAdapter()
        messages_missing_content = [{"role": "customer"}]
        messages_missing_role = [{"content": "Just content"}]
        
        # Test that the processor raises a KeyError, which is its current behavior.
        with self.assertRaises(KeyError):
            adapter.process(messages_missing_content)
        with self.assertRaises(KeyError):
            adapter.process(messages_missing_role)

    def test_message_with_special_characters_and_newlines(self):
        """Test that special characters and newlines in content are preserved."""
        adapter = CSAdapter()
        messages = [
            {"role": "bot", "content": "A message\nwith newlines & [special] chars."}
        ]
        expected = ["[bot]: A message\nwith newlines & [special] chars."]
        self.assertEqual(adapter.process(messages), expected)


if __name__ == "__main__":
    unittest.main()
