from unittest.mock import MagicMock, patch
from tests.test_utils import TestUtils
from nodes.qa import clean_state_node, domanda_node, critique_node
from langchain_core.messages import AIMessage


class TestNodes:

    def setup_method(self):
        self.utils = TestUtils()

    
    @patch("nodes.qa.search_all")
    @patch("nodes.qa.qa_chain")
    def test_domanda_node_response_test_answer(self, mock_qa_chain, mock_search_all):
        
        """
            This test check the domanda_node function
        """ 
        mock_state = self.utils._make_state(
            messages= [AIMessage(content="")],
            original_text="test domanda",
            threshold=0
        )

        mock_qa_chain.invoke.return_value = MagicMock(needs_tools=False, answer="test answer")
        mock_search_all.return_value = MagicMock(return_value="")
        
        result = domanda_node(mock_state)
        
        assert "messages" in result, "Expected 'messages' key in the result."
        assert "threshold" in result, "Expected 'threshold' key in the result."
        assert result.get("threshold") == 1, "Expected threshold to be incremented by 1."
    
    @patch("nodes.qa.search_all")
    @patch("nodes.qa.qa_chain")
    def test_domanda_node_response_tools_needed(self, mock_qa_chain, mock_search_all):
        
        """
            This test check the domanda_node function when the response needs tools
        """
        state = self.utils._make_state(
            messages= [AIMessage(content="")],
            original_text="test domanda",
            threshold=0
        )
        mock_qa_chain.invoke.return_value = MagicMock(needs_tools=True, answer="")
        mock_search_all.return_value = MagicMock(return_value="")

        result = domanda_node(state)
        assert "messages" in result, "Expected 'messages' key in the result."
        assert "threshold" in result, "Expected 'threshold' key in the result."
        assert result.get("threshold") == 1, "Expected threshold to be incremented by 1."

    
    @patch("nodes.qa.critique_chain")
    def test_critique_node_tools_needed(self, mock_critique_chain): 
        
        """
            Test to check the critique_node function when the previous response is tools_needed
        """
        state = self.utils._make_state(
            messages=[AIMessage(content="tools_needed")]
        )
        mock_critique_chain.invoke.return_value = MagicMock(approvato=True, punti_da_correggere=[])
        result = critique_node(state)
        assert result.get("critique_approvata") is True, "Expected critique to be approved."
        assert result.get("critique_punti") == [], "Expected critique_punti to be an empty list."
    

    @patch("nodes.qa.critique_chain")
    def test_critique_node_not_tools_needed_and_approved(self, mock_critique_chain):
        """
            Test to check the critique node function when the previouse response is not tools_needed and is approved.
        """

        state = self.utils._make_state(
            messages=[AIMessage(content="test response")]
        )
        mock_critique_chain.invoke.return_value = MagicMock(approvato=True, punti_da_correggere=[])
        result = critique_node(state)
        assert result.get("critique_approvata") is True, "Expected critique to be approved."
        assert result.get("critique_punti") == [], "Expected critique_punti to be an empty list."
    
    @patch("nodes.qa.critique_chain")
    def test_test_critique_node_not_approved_and_no_points(self, mock_critique_chain):
        """
            Test to check the critique node function when the previouse response is not tools_needed and is not approved, but no points to correct.
        """

        state = self.utils._make_state(
            messages=[AIMessage(content="test response")]
        )
        mock_critique_chain.invoke.return_value = MagicMock(approvato=False, punti_da_correggere=[])
        result = critique_node(state)
        assert result.get("critique_approvata") is True, "Expected critique to be not approved."
        assert result.get("critique_punti") == [], "Expected critique_punti to be an empty list."

    @patch("nodes.qa.save_conversation")    
    def test_clean_state_node(self, mock_save_conversation):

        state = self.utils._make_state(
            messages=[AIMessage(content="test response")],
            session_id="test_session"
        )
        mock_save_conversation.return_value = MagicMock(return_value=None)

        result = clean_state_node(state)
        assert result.get("threshold") == 0, "Expected threshold to be reset to 0."
        assert result.get("critique_approvata") is False, "Expected critique_approvata to be reset to False."
        assert result.get("critique_punti") == [], "Expected critique_punti to be reset to an empty list."
        assert result.get("plan") == [], "Expected plan to be reset to an empty list."
        assert result.get("past_steps") == [], "Expected past_steps to be reset to an empty list."
        assert result.get("response") == "", "Expected response to be reset to an empty string."
        
    