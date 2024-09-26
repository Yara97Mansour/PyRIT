# Copyright (c) UnifyAI
# Licensed under the MIT license.

import json
import logging
from typing import Optional, Union, List, Dict, Iterable

from openai._types import Headers
from unify import Unify, AsyncUnify, MultiLLM, MultiLLMAsync

from pyrit.common import default_values
from pyrit.exceptions import pyrit_target_retry
from pyrit.memory import MemoryInterface
from pyrit.models import ChatMessage, PromptRequestPiece, PromptRequestResponse
from pyrit.models import construct_response_from_request
from pyrit.prompt_target import PromptChatTarget, limit_requests_per_minute

logger = logging.getLogger(__name__)


class UnifyChatTarget(PromptChatTarget):

    _top_p: float
    _deployment_name: str
    _temperature: float
    _frequency_penalty: float
    _presence_penalty: float
    _client: Unify
    _async_client: AsyncUnify
    _Multi_client: MultiLLM
    _Multi_async_client: MultiLLMAsync

    API_KEY_ENVIRONMENT_VARIABLE: str = "UNIFY_CHAT_KEY"
    DEPLOYMENT_ENVIRONMENT_VARIABLE: str = "UNIFY_CHAT_DEPLOYMENT"


    def __init__(
        self,
        *,
        deployment_name: Optional[Union[str,Iterable[str]]] = None,
        api_key: str = None,
        memory: MemoryInterface = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.5,
        presence_penalty: float = 0.5,
        headers: Optional[Headers] = None,
        max_requests_per_minute: Optional[int] = None,
    ) -> None:
        """
        Class that initializes an openai chat target

        Args:
            deployment_name (str, optional): The name of the deployment. Defaults to the
                DEPLOYMENT_ENVIRONMENT_VARIABLE environment variable.
            api_key (str, optional): The API key for accessing the Unify service.
                Defaults to the API_KEY_ENVIRONMENT_VARIABLE environment variable.
            memory (MemoryInterface, optional): An instance of the MemoryInterface class
                for storing conversation history. Defaults to None.
            max_tokens (int, optional): The maximum number of tokens to generate in the response.
                Defaults to 1024.
            temperature (float, optional): The temperature parameter for controlling the
                randomness of the response. Defaults to 1.0.
            top_p (float, optional): The top-p parameter for controlling the diversity of the
                response. Defaults to 1.0.
            frequency_penalty (float, optional): The frequency penalty parameter for penalizing
                frequently generated tokens. Defaults to 0.5.
            presence_penalty (float, optional): The presence penalty parameter for penalizing
                tokens that are already present in the conversation history. Defaults to 0.5.
            max_requests_per_minute (int, optional): Number of requests the target can handle per
                minute before hitting a rate limit. The number of requests sent to the target
                will be capped at the value provided.
        """
        PromptChatTarget.__init__(self, memory=memory, max_requests_per_minute=max_requests_per_minute)

        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._frequency_penalty = frequency_penalty
        self._presence_penalty = presence_penalty

        self._deployment_name = default_values.get_required_value(
            env_var_name=self.DEPLOYMENT_ENVIRONMENT_VARIABLE, passed_value=deployment_name
        )
        api_key = default_values.get_required_value(
            env_var_name=self.API_KEY_ENVIRONMENT_VARIABLE, passed_value=api_key
        )
        self._client = Unify(
            endpoint=self._deployment_name,
            api_key=api_key,
            extra_headers=headers,
        )
        self._async_client = AsyncUnify(
            endpoint=self._deployment_name,
            api_key=api_key,
            extra_headers=headers,
        )


    @limit_requests_per_minute
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        self._validate_request(prompt_request=prompt_request)
        request: PromptRequestPiece = prompt_request.request_pieces[0]

        messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=request.conversation_id)
        messages.append(request.to_chat_message())

        logger.info(f"Sending the following prompt to the prompt target: {request}")

        resp_text = await self._complete_chat_async(messages=messages)

        if not resp_text:
            raise ValueError("The chat returned an empty response.")

        logger.info(f'Received the following response from the prompt target "{resp_text}"')
        response_entry = construct_response_from_request(request=request, response_text_pieces=[resp_text])

        return response_entry

    @pyrit_target_retry
    async def _complete_chat_async(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.5,
        presence_penalty: float = 0.5,
        ) -> str:
        """
        Completes asynchronous chat request.

        Sends a chat message to the OpenAI chat model and retrieves the generated response.

        Args:
            messages (List[ChatMessage]): The chat message objects containing the role and content.
            max_tokens (int, optional): The maximum number of tokens to generate.
                Defaults to 1024.
            temperature (float, optional): Controls randomness in the response generation.
                Defaults to 1.0.
            top_p (float, optional): Probability mass for nucleus sampling (an alternative to
                temperature). Tokens are sampled from the rescaled output probability distribution
                whose support is the smallest set of tokens whose cumulative probability mass exceeds
                top_p. So, with top_p = 0.1 only the top 10% of the tokens will be considered.
                This technique helps increase output diversity and fluency.
                It is recommended to set top_p or temperature, but not both.
                Defaults to 1.0.
            frequency_penalty (float, optional): Controls the frequency of generating the same lines of text.
                Defaults to 0.5.
            presence_penalty (float, optional): Controls the likelihood to talk about new topics.
                Defaults to 0.5.

        Returns:
            str: The generated response message.
        """

        response= await self._async_client.generate(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            n=1,
            stream=False,
            messages=[{"role": msg.role, "content": msg.content} for msg in messages],
        )

        return response

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")
