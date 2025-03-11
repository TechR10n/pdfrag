#! /usr/bin/env python3

import sys
import subprocess
from pathlib import Path

puml_path = 'docs/puml/'
svg_path = 'png/'

PLANTUML_JAR = '/opt/homebrew/Cellar/plantuml/1.2025.2/libexec/plantuml.jar'

if not Path(PLANTUML_JAR).exists():
    print(f"Error: {PLANTUML_JAR} does not exist.")
    sys.exit(1)

for puml_file in Path(puml_path).glob('*.puml'):
    svg_file = svg_path / puml_file.with_suffix('.svg')

# Run PlantUML to generate the SVG file
subprocess.run([
    'java', '-jar', PLANTUML_JAR,
    '-tpng',
    str(puml_path),
    '-o', str(svg_path)
])