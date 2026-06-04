class TestUtils: 

    def _make_state(self, session_id="test_session", original_text="", messages=None, message_summary="", message_buffer="", critique_approvata=False, critique_punti=None, threshold=0, plan="", past_steps=None, response=""):
        """
            Method to setup the agent state for the test classes.
        """
        return{
            "session_id": session_id, 
            "original_text": original_text,
            "messages": messages,
            "message_summary": message_summary,
            "message_buffer": message_buffer,

            "critique_approvata": critique_approvata,
            "critique_punti": critique_punti,
            "threshold": threshold, 

            "plan": plan,
            "past_steps": past_steps,
            "response": response
        }