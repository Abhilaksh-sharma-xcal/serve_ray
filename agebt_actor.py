import ray 
# sdk/ray_controller.py
import ray
from dotenv import load_dotenv
from ray import serve
import os
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import time
import uuid

load_dotenv()
class RayController:
    def __init__(self):
        self.initialized = False

    def init(self, namespace="agents"):
        if not self.initialized:
            ray.init(ignore_reinit_error=True, namespace=namespace, runtime_env={
                "env_vars": dict(os.environ)  # sends everything
            })
            self.initialized = True

ray_controller = RayController()



app = FastAPI()

@serve.deployment
@serve.ingress(app)
class ChatService:
    def __init__(self, actor_name: str, model_name: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"):
        self.actor_name = actor_name
        self.agent_actor = ray.get_actor(actor_name)
        self.model_name = model_name

    async def _stream_completions(self, message):
        """
        Stream tokens in OpenAI-compatible SSE format.
        Assumes `self.agent_actor.stream_run.remote(prompt)` yields tokens.
        """
        stream_id = f"chatcmpl-{uuid.uuid4().hex}"

        # Example: streaming chunks from the actor
        for token in message:
            chunk = {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        # End of stream signal
        yield "data: [DONE]\n\n"

    @app.post("/v1/chat/completions")
    async def chat_completions(self, request: Request):
        payload = await request.json()
        
        print(payload)

        # Required fields
        stream = payload.get("stream", False)
        messages = payload.get("messages", [])

        if not messages:
            return {"error": "No messages provided"}

        # Call the agent's run method with the messages
        response_message = await self.agent_actor.run.remote(messages[-1]['content'])

        if stream:
            return StreamingResponse(
                self._stream_completions(response_message),
                media_type="text/event-stream"
            )

        # Build OpenAI-compatible response
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_message
                    },
                    "finish_reason": "stop"
                }
            ],
            # Usage tracking is optional unless you have token counting
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }


def task_node(name: str = None):
    def decorator(fn):
        ray_controller.init()
        remote_fn = ray.remote(fn)

        def wrapper(*args, **kwargs):
            return remote_fn.options(name=name).remote(*args, **kwargs)

        return wrapper
    return decorator


def service_node(name: str = None):
    def decorator(cls):
        ray_controller.init()
        actor_name = name or cls.__name__
        actor_cls = ray.remote(name=actor_name, lifetime="detached")(cls)

        class WrappedService:
            def __init__(self, *args, **kwargs):
                print("INIT Called")
                try:
                    self._actor = ray.get_actor(actor_name)
                    print(f"[service_node] Retrieved existing actor: {actor_name}")
                except ValueError:
                    self._actor = actor_cls.remote(*args, **kwargs)
                    print(f"[service_node] Created new detached actor: {actor_name}")

            def __getattr__(self, attr):
                return getattr(self._actor, attr)
            
            def serve(self):
                serve.start(detached=True)
                serve.run(ChatService.bind(actor_name), route_prefix="/")
                print(f"🚀 OpenAI-compatible Chat API running for actor '{actor_name}' at /v1/chat/completions")


        return WrappedService
    return decorator


from openai import OpenAI
from typing import List, Dict, Optional, Any

@task_node("call_llm")
def call_llm(message_array: List[Dict], tools: List[Dict]) -> Dict:
    logging.basicConfig(
            filename="LLm_call.log",
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            force=True 
        )
    logger = logging.getLogger(f"LLM_call_Logger.log")
    # tools = sanitize_for_json(tools) if tools else None

    # message_array = sanitize_for_json(message_array)
    logger.info(f"{message_array},,,,{tools}")
    client = OpenAI(
        base_url="https://agentx-litellm.xcaliberhealth.io/",  
        api_key="sk-8_SG-SBqhGKexRmDUnf7lw"
    )
    try:
        response = client.chat.completions.create(
            model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            messages=message_array,
            tools=tools,
            tool_choice="auto",
        )
        
        message = response.choices[0].message
        logger.info(f"LLM {response}////{message}")
        return message

    except APIStatusError as e:
        # Extract useful info without passing the raw exception
        status = getattr(e.response, "status_code", "unknown")
        body = getattr(e, "body", "unknown")
        logger.info(f"call_llm failed: status={status}, body={body}")
        raise RuntimeError(
            f"call_llm failed: status={status}, body={body}"
        ) from None




@task_node("call_tool")
def call_tool(message: Dict, tools: List) -> Dict:
    print(message)
    logging.basicConfig(
            filename="tool_call.log",
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            force=True # Use force=True to re-configure logging in the new Ray process
        )
    logger = logging.getLogger(f"AgentLogger.tool") 
    
    tool_call = message["tool_calls"][0]
    tool_name = tool_call["function"]["name"]
    tool_args = json.loads(tool_call["function"]["arguments"])

    try:
        
            result = f"Unknown tool: {tool_name}"
        
        # Sanitize result to handle Decimal objects before returning
        sanitized_result = sanitize_for_json(result)
        logger.info(f"full result role: tool, tool_call_id: {tool_call['id']},name: {tool_name},content: {sanitized_result}")    
        return {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "name": tool_name,
            # OpenAI schema expects tool content to be a string
            "content": json.dumps(sanitized_result) if not isinstance(sanitized_result, str) else sanitized_result
        }
        
    except Exception as e:
        logger.info(f"full result role: tool, tool_call_id: {tool_call['id']},name: {tool_name},content: {str(e)}")
        return {
            "role": "tool",
            "tool_call_id": tool_call["id"], 
            "name": tool_name,
            "content": f"Error: {str(e)}"
        }







def add(num1, num2):
    return num1*2+num2 + 1

import logging
  
@service_node(name="CQMAgent")
class CQMAgent:
    def __init__(self, name: str, sys_prompt: str, tools: list):
        self.name = name
        self.tools = tools
        # self.tool_schemas = self._create_tool_schemas(self.tools)
        self.tool_schemas = [{
                    "type": "function",
                    "function": {
                        "name": "call_sql_tool",
                        "description": "Executes a natural language query against the clinical database.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "natural_language_query": {
                                    "type": "string",
                                    "description": "The natural language query to run."
                                }
                            },
                            "required": ["natural_language_query"],
                        },
                    },
                } 
        ]
        self.messages = [{'role': 'system', 'content': sys_prompt}]
        self.max_iterations = 7
        logging.basicConfig(
            filename="agent_1.log",
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            force=True # Use force=True to re-configure logging in the new Ray process
        )
        self.logger = logging.getLogger(f"AgentLogger.{name}") 
        

    def _create_tool_schemas(self, tools: List) -> List[Dict]:
        return

    def run(self, query: str):
        self.messages.append({'role': 'user', 'content': query})
        ready_for_final = False
        final_context = {}
        for i in range(self.max_iterations):
            self.logger.debug(f"--- Iteration {i+1}/{self.max_iterations} ---{self.messages} {self.tool_schemas}")
            llm_message = call_llm(self.messages, self.tool_schemas)
            llm_message = ray.get(llm_message)
            llm_message = llm_message.model_dump()
            self.messages.append(llm_message)
            self.logger.debug(f"LLM Response received: {llm_message}")

            if llm_message.get("tool_calls"):
                self.logger.info(f"LLM requested tool call: {llm_message.get('tool_calls')} ")
                
                tool_response = call_tool(llm_message, self.tools)
                tool_response = ray.get(tool_response)  
                self.logger.info(f"Tool execution result: {tool_response}")
                self.messages.append(tool_response)
                # if "content" not in tool_response or tool_response["content"] is None:
                #     tool_response["content"] = ""

                # ✅ Check if tool_response has SQL results (or exclusions)
                if tool_response.get("name") == "call_sql_tool" and tool_response.get("content"):
                    final_context["sql_result"] = tool_response["content"]

                if tool_response.get("name") == "retriever_tool" and tool_response.get("content"):
                    final_context["exclusions"] = tool_response["content"]

                # stop if we have both 
                if "sql_result" in final_context and "exclusions" in final_context:
                    ready_for_final = True
                    self.logger.info("It exited because of rready for result ")
                    # break
                    continue
                continue

            
            content = llm_message.get("content", "").strip()

            self.logger.info(f"No tool call detected. Treating as final answer: {content}")
            print("\n[agent]:", content, "\n")
            return content

        if ready_for_final:
            self.messages.append({
                'role': 'user', 
                'content': 'Now generate the final clinical analysis with visualization using the collected data.'
            })
            # Ensure messages are normalized before final call
            self.messages = clean_messages(self.messages)
            final_response = call_llm(self.messages, self.tool_schemas)
            final_response = ray.get(final_response)
            return final_response

        return "FINAL_ANSWER: max turns reached."


import ray
if __name__ == "__main__":
    

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, namespace="agents")
    print("hii")

    agent_tools = [call_sql_tool, retriever_tool]
    # sys_prompt = prompt_store.read("cqm_agent")
    sys_prompt = llm_sys_prompt
    cqm_agent = CQMAgent(
        name="CQM Agent",
        sys_prompt=sys_prompt,
        tools=agent_tools,
    )
    # user_query = "Give me a count of all the patients with a moderately high HbA1c value greater than 9 with Medicaid payor, compared by ethnicity as a quality measure with valid exclusions. Visualize this information."
    user_query = "Give me a count of all the patients with a moderately high HbA1c value greater than 9 with Medicaid payor, compared by ethnicity as a quality measure with valid exclusions. Visualize this information."
    user_query = "Give me a count of all the patients with a moderately high HbA1c value greater than 9 with Medicaid payor, compared by ethnicity as a quality measure with valid exclusions. Visualize this information."
    final_answer = cqm_agent.run.remote(user_query)
    
    # print(final_answer)
    try:
        xx = ray.get(final_answer)
        print("Final answer:", xx)
    except Exception as e:
        import traceback
        print("Ray task failed with error:", e)
        traceback.print_exc()
    
    
