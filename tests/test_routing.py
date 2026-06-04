from nodes.qa import reflection_routing
from nodes.qa import exit_or_plan_router
from langchain_core.messages import AIMessage
from tests.test_utils import TestUtils


class TestRouting: 

    def setup_method(self):
        self.utils = TestUtils()
        
    
    def test_reflection_routing_approved(self):
        """
            Test to check the reflection routing when the critique is approved.    
        """    
        state = self.utils._make_state(
            critique_approvata=True,
            threshold=0
        )
        result = reflection_routing(state)
        assert result == "clean_and_exit", "Expected to route to 'clean_and_exit' when critique is approved."
    
    def test_reflaction_routing_threshold_exceeded_5(self):
        """
            Test to check the reflection routing when the threshold is exceeded.    
        """    
        state = self.utils._make_state(
            critique_approvata=False,
            threshold=5
        )
        result = reflection_routing(state)
        assert result == "clean_and_exit", "Expected to route to 'clean_and_exit' when threshold is exceeded."
    
    def test_reflaction_routing_threshold_not_exceeded_4(self):
        """
            Test to check the reflaction routng when the threshold is not exceeded.
        """
        state = self.utils._make_state(
            critique_approvata=False,
            threshold=4
        )
        result = reflection_routing(state)
        assert result == "correggi_domanda", "Expected to route to 'correggi_domanda' when threshold is not exceeded."

    def test_reflaction_routing_threshold_impossible_value_big_value(self):
        """"
            This test check the reflaction_routing in an impossible case: when the threshold is greater than the maximum allowed value.
        """
        state = self.utils._make_state(
            critique_approvata=False,
            threshold=6
        )
        result = reflection_routing(state)
        assert result == "clean_and_exit", "Expected to route to 'clean_and_exit' when threshold is greater than the maximum allowed value."

    def test_reflaction_routing_threshold_impossible_negative_value(self):
        """"
            This test check the reflaction_routing in an impossible case: when the threshold is greater than the maximum allowed value.
        """
        state = self.utils._make_state(
            critique_approvata=False,
            threshold=-1
        )
        result = reflection_routing(state)
        assert result == "correggi_domanda", "Expected to route to 'correggi_domanda' when threshold is negative."

    def test_reflection_routing_needs_correction(self):
        """
            Test to check the reflection routing when the critique is not approved and threshold is not exceeded.    
        """    
        state = self.utils._make_state(
            critique_approvata=False,
            threshold=3
        )
        result = reflection_routing(state)
        assert result == "correggi_domanda", "Expected to route to 'correggi_domanda' when critique is not approved and threshold is not exceeded."

    def test_exit_or_plan_router_tools_needed(self):

        """
            Test to check the exit_or_plan_router function.
        """
        state = self.utils._make_state(
            messages=[AIMessage(content="tools_needed")],
        )

        result = exit_or_plan_router(state=state)
        assert result == "go_to_planning", "Expected to route 'go_to_planning' when question node response tools_needed"

    def test_exit_or_plan_router_tools_needed_message_error(self):

        """
            Test to check the exit_or_plan_router function.
        """
        state = self.utils._make_state(
            messages=[AIMessage(content="tools_needed ")],
        )

        result = exit_or_plan_router(state=state)
        assert result == "go_to_planning", "Expected to route 'go_to_planning' when question node response tools_needed"


    def test_exit_or_plan_router_go_to_end(self):
        """
            Test to check the exit_or_plan_router function when the response is not tools_needed.
        """
        state = self.utils._make_state(
            messages=[AIMessage(content="test response")],
        )

        result = exit_or_plan_router(state=state)
        assert result == "go_to_end", "Expected to route 'go_to_end' when question node response is not tools_needed"
        
