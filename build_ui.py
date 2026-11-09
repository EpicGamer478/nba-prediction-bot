"""Bundle data/picks.json into picks.html (the picks page)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "data", "picks.json")) as f:
    data = json.load(f)
with open(os.path.join(HERE, "ui_template.html")) as f:
    html = f.read()
with open(os.path.join(HERE, "picks.html"), "w") as f:
    f.write(html.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")))
print("wrote picks.html")
