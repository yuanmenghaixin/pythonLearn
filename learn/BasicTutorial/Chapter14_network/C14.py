import re
from urllib.request import urlopen
webpage = urlopen('http://www.python.org')
text = webpage.read()
m = re.search(b'<a href="([^"]+)" .*?>about</a>', text, re.IGNORECASE)
m.group(1)