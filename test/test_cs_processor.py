import unittest
from src.processing.cs_processor import CSChatSplitterProcessor

class TestCSChatSplitterProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        self.processor = CSChatSplitterProcessor()
        
    def test_basic_chat_splitting(self):
        """Test basic chat splitting with all roles"""
        input_text = """[bot]:Hello there[bot]:How can I help?[customer]:I need help[agent]:I'm here to assist"""
        expected = [
            {'role': 'bot', 'content': 'Hello there'},
            {'role': 'bot', 'content': 'How can I help?'},
            {'role': 'customer', 'content': 'I need help'},
            {'role': 'agent', 'content': "I'm here to assist"}
        ]
        result = self.processor.process(input_text)
        self.assertEqual(result, expected)

    def test_empty_input(self):
        """Test handling of empty input"""
        input_text = ""
        result = self.processor.process(input_text)
        self.assertEqual(result, [])

    def test_whitespace_handling(self):
        """Test handling of extra whitespace"""
        input_text = """[bot]:  Hello   there  [customer]:    Test    message    """
        expected = [
            {'role': 'bot', 'content': 'Hello there'},
            {'role': 'customer', 'content': 'Test message'}
        ]
        result = self.processor.process(input_text)
        self.assertEqual(result, expected)

    def test_multiple_consecutive_messages(self):
        """Test handling of multiple consecutive messages from same role"""
        input_text = """[agent]:First message[agent]:Second message[agent]:Third message"""
        expected = [
            {'role': 'agent', 'content': 'First message'},
            {'role': 'agent', 'content': 'Second message'},
            {'role': 'agent', 'content': 'Third message'}
        ]
        result = self.processor.process(input_text)
        self.assertEqual(result, expected)

    def test_message_order_preservation(self):
        """Test if message order is preserved"""
        input_text = """[bot]:First[customer]:Second[agent]:Third[bot]:Fourth"""
        result = self.processor.process(input_text)
        roles = [msg['role'] for msg in result]
        expected_roles = ['bot', 'customer', 'agent', 'bot']
        self.assertEqual(roles, expected_roles)

    def test_empty_messages(self):
        """Test handling of empty messages"""
        input_text = """[bot]:[customer]:Real message[agent]:"""
        expected = [
            {'role': 'customer', 'content': 'Real message'}
        ]
        result = self.processor.process(input_text)
        self.assertEqual(result, expected)

    def test_complex_message_content(self):
        """Test handling of complex message content"""
        input_text = """[bot]:Message with [special] characters[customer]:Message with numbers 123[agent]:Message with \n newlines"""
        expected = [
            {'role': 'bot', 'content': 'Message with [special] characters'},
            {'role': 'customer', 'content': 'Message with numbers 123'},
            {'role': 'agent', 'content': 'Message with newlines'}
        ]
        result = self.processor.process(input_text)
        self.assertEqual(result, expected)

    def test_invalid_format(self):
        """Test handling of invalid format"""
        input_text = "Invalid format without proper tags"
        result = self.processor.process(input_text)
        self.assertEqual(result, [])

    def test_mixed_case_roles(self):
        """Test handling of mixed case in role tags"""
        input_text = """[BOT]:Message 1[Customer]:Message 2[AGENT]:Message 3"""
        result = self.processor.process(input_text)
        self.assertEqual(len(result), 0)  # Should not match case-sensitive tags

    def test_real_world_example(self):
        """Test with a real-world chat example"""
        input_text = """[bot]:Hi, you're in the right place for customer service support.
        [bot]:Item displayed, asin:B0CFNV34T3
        [customer]:I need help with my order
        [agent]:Hello, how can I assist you today?
        [customer]:The item is damaged
        [agent]:I understand, let me help you with that"""
        
        result = self.processor.process(input_text)
        
        self.assertEqual(len(result), 6)
        self.assertEqual(result[0]['role'], 'bot')
        self.assertEqual(result[2]['role'], 'customer')
        self.assertEqual(result[3]['role'], 'agent')

    def test_malformed_tags(self):
        """Test handling of malformed tags"""
        input_text = """[bot:Message 1[customer]:Message 2[agent]Message 3"""
        result = self.processor.process(input_text)
        self.assertEqual(len(result), 1)  # Should only match properly formatted tags

    def test_performance_with_large_input(self):
        """Test performance with large input"""
        large_input = "[bot]:Message[customer]:Message[agent]:Message" * 1000
        start_time = time.time()
        result = self.processor.process(large_input)
        end_time = time.time()
        
        self.assertLess(end_time - start_time, 1.0)  # Should process in less than 1 second
        self.assertEqual(len(result), 3000)

if __name__ == '__main__':
    unittest.main()
