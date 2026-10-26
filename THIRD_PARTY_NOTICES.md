# Third-party software and models

The MIT license in this repository covers ARGUS's original source code,
documentation, and original banner artwork. External libraries, model weights,
services, and user-supplied footage retain their respective licenses and terms.
No third-party detector source or model weights are distributed here.

| Component | Role | Upstream license / terms |
| --- | --- | --- |
| [Pillow](https://github.com/python-pillow/Pillow) | Core image processing | HPND; see upstream LICENSE |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | Optional YOLO / tracking adapter | AGPL-3.0 or Enterprise; model terms also apply |
| [OpenCV](https://github.com/opencv/opencv) | Optional video decoding | Apache-2.0 for current 4.x releases; see upstream notices |
| [lap](https://github.com/gatagat/lap) | Optional track assignment dependency | BSD-2-Clause |
| [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) | Optional Claude adapter | MIT; service terms apply separately |
| [OpenAI Python SDK](https://github.com/openai/openai-python) | Optional compatible API adapter | Apache-2.0; endpoint terms apply separately |

Installing an optional dependency does not change that dependency's license.
ARGUS's MIT grant does not make an Ultralytics-based combination MIT-only or
exempt it from AGPL obligations. Review the terms of your complete application,
including distribution, hosted use, detector weights, and transitive dependencies.

See [Ultralytics licensing](https://www.ultralytics.com/license), its
[source license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE),
and the notices in each installed dependency. A custom `FramePacket` producer
can connect a detector or tracker under terms appropriate to your project.
