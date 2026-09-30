"""Minimal stub so modules that `import litellm` are importable in unit tests.
Real LLM calls are monkeypatched in the tests; this is never invoked."""


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Usage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _Resp:
    """`usage` is only set when given, so a response without one (some
    providers omit it) can be modelled too."""
    def __init__(self, content, usage=None):
        self.choices = [_Choice(content)]
        if usage is not None:
            self.usage = usage


def completion(**kwargs):
    return _Resp("{}")
