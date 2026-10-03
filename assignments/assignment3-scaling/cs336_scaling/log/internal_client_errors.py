from http import HTTPStatus

import requests

MAX_ERROR_BODY_CHARS = 4096


class InternalAPIError(requests.HTTPError):
    def __init__(self, response: requests.Response) -> None:
        super().__init__(
            _format_response_error(response),
            request=response.request,
            response=response,
        )


def _format_response_error(response: requests.Response) -> str:
    request = response.request
    method = request.method if request is not None else "UNKNOWN"
    url = response.url
    status_code = response.status_code
    try:
        reason = HTTPStatus(status_code).phrase
    except ValueError:
        reason = response.reason

    body = response.text
    if len(body) > MAX_ERROR_BODY_CHARS:
        body = f"{body[:MAX_ERROR_BODY_CHARS]}... <truncated>"

    return (
        f"internal API request failed: {method} {url} returned "
        f"{status_code} {reason}; response_body={body!r}"
    )
