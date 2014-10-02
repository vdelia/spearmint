from setuptools import setup, find_packages

setup(
    name='value_search_optimizer',
    version = '0.1',
    description='Spearmint',
    author='Vincenzo D\'Elia',
    author_email='vince.delia@gmail.com',
    packages = find_packages(),

    scripts = ["bin/spearmint", "bin/cleanup"],
    package_data = {
        # If any package contains *.txt or *.rst files, include them:
        # avro is for fixtures in test env
        '': ['*.txt', '*.rst', '*.cfg', '*.md', '*.avsc', '*avro'],
    },
)

