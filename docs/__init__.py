import asyncio
import os

import aiofiles
from aiohttp import web

from config_parser.config import DOCS

try:
    import uvloop
except ImportError:
    print("[Warn] no install uvloop package")
else:
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
finally:
    loop = asyncio.get_event_loop()


routes = web.RouteTableDef()
base = DOCS['base']
docs_dir = DOCS['docs_dir']
host = DOCS['host']
port = DOCS['port']


@routes.get('%s{path:.*}' % base)
async def all_handler(request):
    relative_path = request.path.replace(base, '', 1)
    print(relative_path)
    if relative_path == '':
        relative_path = 'index.html'
    file_path = os.path.join(docs_dir, relative_path)
    if not os.path.exists(file_path):
        file_path = os.path.join(docs_dir, '404.html')
    async with aiofiles.open(file_path, 'rb') as f:
        content = await f.read()
    return web.Response(body=content, headers={'Content-type': 'text/html'})


def docs_dev():
    app = web.Application()
    app.add_routes(routes)
    web.run_app(app, host=host, port=port)
