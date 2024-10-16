#!/usr/bin/env python3

import yaml
import os.path as op
from glob import glob

for workshop in glob('workshops/*'):
    md_file = {}
    md_file['base_url'] = workshop
    md_file['name'] = op.split(workshop)[-1].replace('-', ' ')
    md_file['external_url'] = ''
    md_file['pulls'] = [op.split(p)[-1] for p in glob(op.join(workshop, 'pulls/*'))]
    md_file['releases'] = [op.split(p)[-1] for p in glob(op.join(workshop, 'releases/*'))]
    with open(f'site/_workshops/{op.split(workshop)[-1]}.md', 'w') as f:
        f.write('---\n')
        yaml.safe_dump(md_file, f)
        f.write('---')
