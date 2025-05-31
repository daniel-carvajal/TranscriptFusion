from setuptools import setup, find_packages

setup(
    name="transcriptfusion",
    version="1.0.0",
    description="Precision word-level timing enrichment for human-curated transcripts",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        line.strip() 
        for line in open('requirements.txt', 'r').readlines() 
        if line.strip() and not line.startswith('#')
    ],
    entry_points={
        'console_scripts': [
            'transcriptfusion=transcriptfusion.cli:main',
            'tf-download=transcriptfusion.cli:download_audio',
            'tf-fetch=transcriptfusion.cli:fetch_transcript',
            'tf-transcribe=transcriptfusion.cli:transcribe_audio',
            'tf-enrich=transcriptfusion.cli:enrich_transcript',
        ],
    },
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)