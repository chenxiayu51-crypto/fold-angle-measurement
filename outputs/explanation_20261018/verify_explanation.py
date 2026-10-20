"""Validate teaching numbers, local document links and generated image files."""
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
doc = ROOT/'docs/手机铰链轴与IMU角度计算_六个问题图解.md'
text = doc.read_text(encoding='utf-8')
targets = re.findall(r'\]\(([^)]+)\)', text)
missing = [s for s in targets if not Path(s).exists()]
assert not missing, missing
assert text.count('```') % 2 == 0, 'Unbalanced code fences'
assert text.count('$$') % 2 == 0, 'Unbalanced math fences'
assert '\ufffd' not in text, 'Invalid UTF-8 replacement character'
p = json.loads((ROOT/'outputs/experiment_20261016_110453/csv_to_angle_parameters.json').read_text(encoding='utf-8'))
k = p['first_three_updates'][1]
delta = -.5*(p['first_three_updates'][0]['axial_rate_rad_s']+k['axial_rate_rad_s'])*k['dt_s']*180/np.pi
assert abs(delta-k['gyro_increment_deg']) < 1e-12, 'Actual-sample trapezoid differs'
fused = k['prediction_deg']+(1-np.exp(-k['dt_s']))*(k['acc_angle_deg']-k['prediction_deg'])
assert abs(fused-k['final_fused_deg']) < 1e-12, 'Actual-sample correction differs'
images = list(OUT.glob('*.png'))
assert len(images) == 7, 'Missing figure'
for f in images:
    with Image.open(f) as image:
        image.verify()
ver = json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
ver.update(document_links_checked=len(targets), missing_document_links=missing,
           balanced_fences=True, utf8_no_replacement_character=True,
           trapezoid_actual_sample_verified=True, complementary_actual_sample_verified=True,
           document_characters=len(text))
(OUT/'verification.json').write_text(json.dumps(ver, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(ver, ensure_ascii=False, indent=2))
