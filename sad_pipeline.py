import torch
from matplotlib import pyplot as plt
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Segment, notebook

sad = torch.hub.load('pyannote/pyannote-audio', 'sad')

name = 'anatoliy_1'
test_file = {'uri': f'{name}.wav',
             'audio': f'./uisrnn_tests/integration/fixtures/rtk/{name}/{name}.wav'}

sad_scores = sad(test_file)
binarize = Binarize(offset=0.52, onset=0.52, log_scale=True,
                    min_duration_off=0.1, min_duration_on=0.1)
speech = binarize.apply(sad_scores, dimension=1)

notebook.crop = Segment(0, 55)

nrows = 2
fig, ax = plt.subplots(nrows=nrows, ncols=1)
fig.set_figwidth(20)
fig.set_figheight(nrows * 2)

notebook.plot_timeline(speech, ax=ax[0], time=True)
ax[0].text(notebook.crop.start + 0.5, 0.1, 'speech activity detection', fontsize=14)

plt.show()
