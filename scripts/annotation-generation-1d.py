import json
import os.path as osp
from glob import glob

import pandas as pd

# 1. N - Normal
# 2. V - PVC (Premature ventricular contraction)
# 3. \ - PAB (Paced beat)
# 4. R - RBB (Right bundle branch)
# 5. L - LBB (Left bundle branch)
# 6. A - APB (Atrial premature beat)
# 7. ! - AFW (Ventricular flutter wave)
# 8. E - VEB (Ventricular escape beat)

classes = ['N', 'V', '\\', 'R', 'L', 'A', '!', 'E']
lead = 'MLII'
extension = 'npy'  # or `png` for 2D
data_path = osp.abspath('../data/*/*/*/*/*.{}'.format(extension))
val_size = 0.1  # [0, 1]

output_path = '/'.join(data_path.split('/')[:-5])
random_state = 7

if __name__ == '__main__':
    dataset = []
    files = glob(data_path)

    for file in glob(data_path):
        *_, name, lead, label, filename = file.split('/')
        dataset.append({
            "name": name,
            "lead": lead,
            "label": label,
            "filename": osp.splitext(filename)[0],
            "path": file
        })

    data = pd.DataFrame(dataset)
    data = data[data['lead'] == lead]
    data = data[data['label'].isin(classes)]
    data = data.sample(frac=1, random_state=random_state)

    val_ids = []
    for cl in classes:
        val_ids.extend(data[data['label'] == cl].sample(frac=val_size, random_state=random_state).index)

    val = data.loc[val_ids, :]
    train = data[~data.index.isin(val.index)]

    train.to_json(osp.join(output_path, 'train.json'), orient='records')
    val.to_json(osp.join(output_path, 'val.json'), orient='records')

    d = {}
    for label in train.label.unique():
        d[label] = len(d)

    with open(osp.join(output_path, 'class-mapper.json'), 'w') as file:
        file.write(json.dumps(d, indent=1))
