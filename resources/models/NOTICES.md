# Local model notices

The application uses three unmodified, revision-pinned ONNX exports downloaded separately. The two
`intfloat` embedding repositories declare the MIT license; the Russian to English
translation model is a conversion of a Helsinki-NLP model published under
CC BY 4.0. The exact source revision and SHA-256 of every model file are
recorded in the application's model specifications.

| Model | Pinned source | Downloaded files |
| --- | --- | --- |
| multilingual-e5-small | https://huggingface.co/intfloat/multilingual-e5-small/tree/614241f622f53c4eeff9890bdc4f31cfecc418b3 | `model.onnx`, `tokenizer.json`, `MODEL_CARD.md` (the upstream README) |
| e5-small-v2 | https://huggingface.co/intfloat/e5-small-v2/tree/ffb93f3bd4047442299a41ebb6fa998a38507c52 | `model.onnx`, `tokenizer.json` |
| opus-mt-ru-en | https://huggingface.co/Xenova/opus-mt-ru-en/tree/afe8c6c738ec81b6d033fd8f44f9678a639a7c67 | `encoder_model_quantized.onnx`, `decoder_model_quantized.onnx`, `tokenizer.json` |

## Russian to English translation model

The downloaded ONNX export is https://huggingface.co/Xenova/opus-mt-ru-en at the
pinned revision above. That export page declares no license of its own. It is a
format conversion of https://huggingface.co/Helsinki-NLP/opus-mt-ru-en, whose
model card declares **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/),
and the original training work is the Helsinki-NLP OPUS-MT project
(https://github.com/Helsinki-NLP/Opus-MT).

Attribution, as CC BY 4.0 requires: Helsinki-NLP OPUS-MT, model `opus-mt-ru-en`,
licensed CC BY 4.0. Changes made: none to the downloaded files, which are stored
byte for byte as published and verified by SHA-256 before loading. At load time
the application replaces one unusable field of the tokenizer description in
memory — `"precompiled_charsmap": null` becomes an NFKC normalizer — because the
published value is meant for a different runtime; the file on disk is unchanged.

This model produces a draft search wording that the user reviews before any
search runs. It does not label, score or judge any scientific evidence.

## License texts

The upstream model cards and license declarations can be inspected at these
revision-pinned links. The full MIT license terms for the two embedding models
follow. This notice does not claim that the application or any unrelated
component has the same license.

MIT License

Copyright (c) the respective copyright holders of the pinned intfloat models

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
