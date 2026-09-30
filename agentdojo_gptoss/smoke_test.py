"""Quick check that the vLLM endpoint returns a proper tool call for gpt-oss-20b."""
import json, sys, openai
c = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
tools = [{"type": "function", "function": {"name": "get_weather",
          "description": "Get weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
r = c.chat.completions.create(model="gpt-oss-20b", temperature=0.0,
        messages=[{"role": "user", "content": "What's the weather in Zurich? Use the tool."}],
        tools=tools, tool_choice="auto")
m = r.choices[0].message
print("content:", repr(m.content))
print("reasoning:", repr(getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None))[:300])
print("tool_calls:", json.dumps([t.model_dump() for t in (m.tool_calls or [])], indent=1))
sys.exit(0 if m.tool_calls else 1)
