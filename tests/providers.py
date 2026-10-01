"""Scripted Slick provider shared by optimizer tests."""

from pydantic import BaseModel


class ScriptedProvider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    async def acall(self, context, *, tools=None, tool_results=None):
        self.calls.append(context)
        try:
            response = next(self.responses)
        except StopIteration as exc:
            raise RuntimeError("Scripted provider exhausted") from exc
        if isinstance(response, Exception):
            raise response
        return response.model_dump_json() if isinstance(response, BaseModel) else response, []
