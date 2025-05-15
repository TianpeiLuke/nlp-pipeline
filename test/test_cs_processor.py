import unittest
import time

from src.processing.cs_processor import CSChatSplitterProcessor


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

    # ---------- edge cases ----------
    def test_invalid_format(self):
        self.assertEqual(self.processor.process("Invalid format"), [])

    def test_mixed_case_roles(self):
        # regex is case-sensitive → uppercase tags ignored
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
        txt = "[bot:bad[customer]:ok[agent]bad"
        self.assertEqual(len(self.processor.process(txt)), 1)

    # ---------- performance ----------
    def test_performance_with_large_input(self):
        big = "[bot]:M[customer]:M[agent]:M" * 1000
        t0 = time.time()
        res = self.processor.process(big)
        self.assertLess(time.time() - t0, 1.0)        # within 1 s
        self.assertEqual(len(res), 3000)


if __name__ == "__main__":
    unittest.main()
