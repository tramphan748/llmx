# llmx-gemini

`llmx-gemini` is a customized version of the original [llmx](https://github.com/victordibia/llmx) library developed by [Victor Dibia](https://github.com/victordibia). This version has been extended to include support for the **Gemini API** because the **PaLM API** is decomissioned.

---

## Introduction

The original `llmx` library is a powerful API for working with fine-tuned language models such as OpenAI, Cohere, Hugging Face, and more. `llmx-gemini` builds upon this foundation by integrating **Gemini API** functionality, allowing users to leverage Google Generative AI for text generation and conversational AI tasks.

---

## New Features in `llmx-gemini`

- **Gemini API Support**: This version introduces classes and methods to interact with Google's Gemini models.
- Maintains compatibility with all features of the original `llmx`.
- Easily integrates with existing applications using `llmx`.

---

## Installation

Install directly from this GitHub repository:

```bash
pip install git+https://github.com/tramphan748/llmx-gemini.git
```
---
## Usage
```python
from llmx import llm

# Initialize llm for Gemini API
gen = llm(provider="gemini", api_key="YOUR_GEMINI_API_KEY")
```

## References
This customized version is based on the original library:
```bibtex
@software{victordibiallmx,
author = {Victor Dibia},
license = {MIT},
month =  {10},
title = {LLMX - An API for Chat Fine-Tuned Language Models},
url = {https://github.com/victordibia/llmx},
year = {2023}
}
```

For more information, visit the [llmx](https://github.com/victordibia/llmx) repository.

---
## License
llmx-gemini is developed under the MIT license inherited from the original library. Please see the LICENSE file for more details.
