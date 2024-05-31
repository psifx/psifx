from typing import Union

from psifx.tool import Tool
from langchain_community.llms import Ollama
from psifx.io.json import JSONReader
import json
from psifx.text.llm.ollama.tool import get_ollama
from psifx.text.llm.hf.tool import get_lc_hf
class LLMTool(Tool):
    """
    Base class for text tools.
    """

    def __init__(self, model,
                 overwrite: bool = False,
                 verbose: Union[bool, int] = True):

        super().__init__(
            device="cuda",
            overwrite=overwrite,
            verbose=verbose,
        )

        try:
            data = JSONReader.read(model)
        except NameError:
            try:
                data = json.loads(model)
            except json.JSONDecodeError:
                data = {'provider': 'ollama', 'model': model}
        assert 'provider' in data, 'Please give a provider'
        assert data['provider'] in ['hf', 'ollama'], 'provider should be hf, or ollama'
        if data['provider'] == 'hf':
            data.pop('provider')
            self.llm = get_lc_hf(**data)
        elif data['provider'] == 'ollama':
            data.pop('provider')
            self.llm = get_ollama(**data)