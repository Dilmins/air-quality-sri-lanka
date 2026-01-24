import os

port = os.environ.get("PORT", "8080")
bind = f"0.0.0.0:{port}"

workers = 2
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 5

accesslog = "-"
errorlog = "-"
loglevel = "info"

proc_name = "air-quality-system"

daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

keyfile = None
certfile = None