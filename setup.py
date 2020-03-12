from setuptools import setup, find_packages
import logging
logger = logging.getLogger(__name__)

name = 'quantum-state-database'
package_name = 'quad'
version = '0.1.0'

try:
    with open('README.md', 'r') as f:
        long_desc = f.read()
except:
    logger.warning('Could not open README.md.  long_description will be set to None.')
    long_desc = None

setup(
    name = package_name,
    packages = find_packages(),
    version = version,
    description = 'A database for storing and querying quantum state vectors.',
    long_description = long_desc,
    long_description_content_type = 'text/markdown',
    author = 'Casey Duckering',
    #author_email = '',
    url = f'https://github.com/cduck/{name}',
    download_url = f'https://github.com/cduck/{name}/archive/{version}.tar.gz',
    keywords = ['quantum computing', 'databases', 'locality sensitive hashing'],
    classifiers = [
        'License :: OSI Approved :: MIT License',
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
    install_requires = [
        'numpy',
        'cirq-unstable==0.8.0.dev20200312000748',
        'zarr~=2.4',
        'numcodecs~=0.6.4',
    ],
    extras_require = {
        'dev': [
            'pytest',
            'pytest-benchmark',
        ]
    },
)

