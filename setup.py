from setuptools import setup

setup(
    name='demucslib',
    author="Dion Timmer",
    author_email="diontimmer@live.nl",
    version='0.1',
    description='Library for easily controlling demucs stem separation through Python.',
    scripts=['demucslib.py'],
    install_requires=[
        'demucs',
    ],
)

