import threading
import urllib.parse
import urllib.request


def send(url_template, title, body):
    """Fire-and-forget GET to a push service.

    url_template uses {title} and {body}, e.g. Bark:
      https://api.day.app/<key>/{title}/{body}
    """
    if not url_template:
        return
    url = url_template.format(title=urllib.parse.quote(title), body=urllib.parse.quote(body))

    def _go():
        try:
            urllib.request.urlopen(url, timeout=5).read()
        except Exception as exc:
            print("notify failed:", exc)

    threading.Thread(target=_go, daemon=True).start()
